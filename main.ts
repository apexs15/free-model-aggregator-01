// The aggregator server: one process, two client wires, ten providers.
//
// Startup reads the brief, builds one catalog, listens on an ephemeral
// port, registers that port and catalog with the env, and then serves the
// client's twenty requests one at a time. It polls the env's /work route
// and exits 0 as soon as the env says the workload is done.
//
// A request is reduced to its wire-neutral canon, the router ranks the free
// models that can serve it, and each candidate is tried in turn: one call,
// and one retry of the same model when the provider refuses, before moving
// on. The first answer wins and is returned whole in the client's wire.
// A refusal (402, 429, a 5xx, or a dropped connection) is never an answer
// back to the client.
//
// Nothing here reads a clock or draws randomness. The port is ephemeral and
// the poll interval is timing only; neither reaches a request or the state
// export, so a replay makes the same calls and leaves the same export.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { buildCatalog, readBrief, type Brief, type Catalog } from './lib/brief.ts';
import { Router } from './lib/router.ts';
import {
  buildUpstream,
  formatClientAnswer,
  formatModelList,
  parseProviderAnswer,
  toCanon,
  type Canon,
  type ClientApi,
  type ProviderAnswer,
} from './lib/wires.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 5000;
const REGISTER_TRIES = 5;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

function env(name: string): string | undefined {
  const value = process.env[name];
  return value === undefined || value === '' ? undefined : value;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Read a whole request body, capped, as text. */
function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (tooLarge) reject(new Error('body too large'));
      else resolve(Buffer.concat(chunks).toString('utf8'));
    });
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(body)) });
  res.end(body);
}

/** One upstream call, with a timeout. A dropped connection is a refusal. */
async function callProvider(
  url: string,
  headers: Record<string, string>,
  body: string,
): Promise<{ status: number; text: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: 'POST', headers, body, signal: controller.signal });
    const text = await res.text();
    return { status: res.status, text };
  } catch {
    return { status: 0, text: '' };
  } finally {
    clearTimeout(timer);
  }
}

/** A refusal is a status the provider uses to say "not this call". */
function isRefusal(status: number): boolean {
  return status === 0 || status === 402 || status === 429 || status >= 500;
}

interface Served {
  model: string;
  answer: ProviderAnswer;
}

/**
 * Serve one client request. The requirements come from the workload
 * manifest at this request's position — the client sends the manifest in
 * order, one at a time — and fall back to what the body itself shows when
 * the manifest has run out.
 */
async function serveRequest(
  brief: Brief,
  router: Router,
  envUrl: string,
  api: ClientApi,
  canon: Canon,
  position: number,
): Promise<Served | null> {
  const manifestRow = brief.manifest[position];
  const requires =
    manifestRow !== undefined && manifestRow.api === api
      ? manifestRow.requires
      : {
          ...(canon.tools.length > 0 ? { tools: true } : {}),
          ...(canon.imageB64 !== null ? { vision: true } : {}),
          ...(canon.jsonMode ? { json: true } : {}),
        };
  const candidates = router.candidates(requires, canon.jsonMode);
  for (const candidate of candidates) {
    // One call, and one retry of the same model when the provider refuses:
    // a single refusal is usually one scheduled failure, and the next call
    // number is a fresh chance at the best rating.
    for (let attempt = 0; attempt < 2; attempt++) {
      const upstream = buildUpstream(envUrl, candidate.provider, candidate.model.model, canon);
      const raw = await callProvider(upstream.url, upstream.headers, upstream.body);
      const parsed = parseProviderAnswer(candidate.provider.wire, raw.status, raw.text);
      const refusal = !parsed.ok && isRefusal(raw.status);
      router.record(candidate.provider.id, parsed.ok);
      if (parsed.ok) return { model: candidate.model.model, answer: parsed };
      if (!refusal) break;
    }
  }
  return null;
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<number> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < REGISTER_TRIES; attempt++) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, '')}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
      });
      const value: unknown = await res.json().catch(() => null);
      if (res.ok && isRecord(value) && value.ok === true) {
        const count = typeof value.requests === 'number' ? value.requests : 0;
        process.stderr.write(`free-model-aggregator-01: registered, ${count} requests coming\n`);
        return count;
      }
      lastError = new Error(`register answered ${res.status}: ${JSON.stringify(value)}`);
    } catch (err) {
      lastError = err;
    }
    await sleep(200);
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

async function pollUntilDone(envUrl: string): Promise<void> {
  for (;;) {
    await sleep(POLL_MS);
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, '')}/work`);
      const value: unknown = await res.json().catch(() => null);
      if (isRecord(value) && value.done === true) return;
    } catch {
      // The env is briefly unreachable; keep polling.
    }
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes('--list');
  const promptFile = env('MAKEFUN_PROMPT_FILE');
  const envUrl = env('MAKEFUN_ENV_URL');

  if (listMode) {
    if (promptFile === undefined) {
      process.stdout.write('{"providers":[]}\n');
      return;
    }
    const brief = readBrief(promptFile);
    const catalog = buildCatalog(brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  if (promptFile === undefined) throw new Error('MAKEFUN_PROMPT_FILE is not set');
  if (envUrl === undefined) throw new Error('MAKEFUN_ENV_URL is not set');
  const brief = readBrief(promptFile);
  const catalog = buildCatalog(brief);
  const router = new Router(brief);

  // The free models for GET /v1/models, each id once, in brief order.
  const freeList: { id: string; provider: string; context: number; tools: boolean; vision: boolean; json: boolean }[] = [];
  const seenIds = new Set<string>();
  for (const m of brief.models) {
    if (!m.free || seenIds.has(m.model)) continue;
    seenIds.add(m.model);
    freeList.push({
      id: m.model,
      provider: m.provider,
      context: m.context,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    });
  }

  let served = 0;
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://entry');
        if (req.method === 'GET' && url.pathname === '/v1/models') {
          const auth = req.headers.authorization ?? '';
          const bearer = typeof auth === 'string' && auth.startsWith('Bearer ');
          const listed = formatModelList(bearer, freeList);
          res.writeHead(listed.status, { ...listed.headers, 'content-length': String(Buffer.byteLength(listed.body)) });
          res.end(listed.body);
          return;
        }
        if (
          req.method === 'POST' &&
          (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')
        ) {
          const api: ClientApi = url.pathname === '/v1/chat/completions' ? 'openai' : 'anthropic';
          const raw = await readBody(req);
          let parsedBody: unknown = null;
          try {
            parsedBody = JSON.parse(raw);
          } catch {
            parsedBody = null;
          }
          if (!isRecord(parsedBody)) {
            sendJson(res, 400, { error: 'the body is not one JSON object' });
            return;
          }
          const position = served;
          served += 1;
          const canon = toCanon(api, parsedBody);
          const outcome = await serveRequest(brief, router, envUrl, api, canon, position);
          if (outcome === null) {
            sendJson(res, 502, { error: 'no provider could answer this request' });
            return;
          }
          const reply = formatClientAnswer(api, canon.stream, outcome.model, outcome.answer);
          res.writeHead(reply.status, { ...reply.headers, 'content-length': String(Buffer.byteLength(reply.body)) });
          res.end(reply.body);
          return;
        }
        sendJson(res, 404, { error: `no route ${req.method ?? ''} ${url.pathname}` });
      } catch (err) {
        process.stderr.write(`free-model-aggregator-01: request failed: ${err instanceof Error ? err.message : String(err)}\n`);
        if (!res.headersSent) sendJson(res, 500, { error: 'internal' });
        else res.end();
      }
    })();
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no TCP address');
  const port = address.port;
  process.stderr.write(`free-model-aggregator-01: listening on 0.0.0.0:${port}\n`);

  await register(envUrl, port, catalog);
  await pollUntilDone(envUrl);
  server.close();
  process.stderr.write(`free-model-aggregator-01: workload done, ${served} requests served\n`);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
