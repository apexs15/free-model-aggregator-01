// Reading the trial brief and building the catalog it implies.
//
// The brief is prose with three JSON documents embedded in it. Each document
// names itself with a `schema` member, so this file finds them by scanning
// for balanced braces and checking the schema, instead of assuming any
// layout. That keeps the parse working whether the documents are packed on
// one line or pretty-printed.
//
// From those documents this file builds the catalog the server registers:
// one entry per provider, in the directory's order, with the provider's own
// path as base_url, its key_env, its wire, and its free models in the order
// the quality table lists them, each with its id, its free flag and its
// context length. The catalog is built once, from the brief alone, and the
// same object is printed by --list and posted at register time, so the two
// can never drift apart.
//
// The brief's quality table lists free models only. A paid model the brief
// never names is simply absent from the catalog, which is fine: the score
// checks that the free sets match exactly and that no paid model is marked
// free, and an absent model is neither.

import { readFileSync } from 'node:fs';

export type WireKind = 'openai' | 'anthropic' | 'gemini';
export type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export interface ProviderRow {
  id: string;
  name: string;
  wire: WireKind;
  auth: AuthKind;
  /** The provider's own path, e.g. "/p/groq" — never joined to the env URL here. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface ModelRow {
  model: string;
  provider: string;
  /** Public rating in milli-points; higher is stronger. */
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RequestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: {
    tools?: boolean;
    vision?: boolean;
    json?: boolean;
    min_context?: number;
  };
}

export interface Brief {
  providers: ProviderRow[];
  models: ModelRow[];
  manifest: RequestRow[];
}

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const asString = (v: unknown, fallback = ''): string =>
  typeof v === 'string' ? v : fallback;

const asBool = (v: unknown): boolean => v === true;

const asInt = (v: unknown, fallback = 0): number =>
  typeof v === 'number' && Number.isSafeInteger(v) ? v : fallback;

/**
 * Pull the balanced JSON object starting at `open` out of `text`, honouring
 * string escapes so a brace inside a string does not end the scan early.
 * Returns the parsed value plus the index just past its closing brace, or
 * null when the braces never balance or the text is not JSON.
 */
function objectAt(text: string, open: number): { value: unknown; end: number } | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          return { value: JSON.parse(text.slice(open, i + 1)), end: i + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/** Every top-level JSON object in the text, in the order it appears. */
function documentsIn(text: string): unknown[] {
  const found: unknown[] = [];
  let at = 0;
  while (at < text.length) {
    const open = text.indexOf('{', at);
    if (open === -1) break;
    const parsed = objectAt(text, open);
    if (parsed === null) {
      at = open + 1;
      continue;
    }
    found.push(parsed.value);
    at = parsed.end;
  }
  return found;
}

function parseDirectory(doc: Record<string, unknown>): ProviderRow[] {
  const rows: ProviderRow[] = [];
  const list = Array.isArray(doc.providers) ? doc.providers : [];
  for (const raw of list) {
    if (!isRecord(raw)) continue;
    const wire = asString(raw.wire);
    if (wire !== 'openai' && wire !== 'anthropic' && wire !== 'gemini') continue;
    const auth = asString(raw.auth);
    if (auth !== 'bearer' && auth !== 'x-api-key' && auth !== 'x-goog-api-key' && auth !== 'none') continue;
    const base = asString(raw.path) || asString(raw.base_path) || asString(raw.base_url) || asString(raw.base);
    if (!base) continue;
    rows.push({
      id: asString(raw.id),
      name: asString(raw.name),
      wire,
      auth,
      base,
      keyEnv: asString(raw.key_env),
      key: asString(raw.key),
    });
  }
  return rows;
}

function parseRatings(doc: Record<string, unknown>): ModelRow[] {
  const rows: ModelRow[] = [];
  const list = Array.isArray(doc.ratings) ? doc.ratings : [];
  for (const raw of list) {
    if (!isRecord(raw)) continue;
    rows.push({
      model: asString(raw.model),
      provider: asString(raw.provider),
      rating: asInt(raw.rating_milli, 0),
      context: asInt(raw.context_length, 0),
      tools: asBool(raw.tools),
      vision: asBool(raw.vision),
      json: asBool(raw.json),
      free: asBool(raw.free),
    });
  }
  return rows;
}

function parseManifest(doc: Record<string, unknown>): RequestRow[] {
  const rows: RequestRow[] = [];
  const list = Array.isArray(doc.manifest) ? doc.manifest : [];
  for (const raw of list) {
    if (!isRecord(raw)) continue;
    const api = asString(raw.api);
    if (api !== 'openai' && api !== 'anthropic') continue;
    const req = isRecord(raw.requires) ? raw.requires : {};
    rows.push({
      id: asString(raw.id),
      api,
      stream: raw.stream === true,
      weight: asInt(raw.weight, 1),
      requires: {
        ...(req.tools === true ? { tools: true } : {}),
        ...(req.vision === true ? { vision: true } : {}),
        ...(req.json === true ? { json: true } : {}),
        ...(typeof req.min_context === 'number' ? { min_context: req.min_context } : {}),
      },
    });
  }
  return rows;
}

/** Read the brief at `path` and return its three documents, parsed. */
export function readBrief(path: string): Brief {
  const text = readFileSync(path, 'utf8');
  let providers: ProviderRow[] = [];
  let models: ModelRow[] = [];
  let manifest: RequestRow[] = [];
  for (const doc of documentsIn(text)) {
    if (!isRecord(doc) || typeof doc.schema !== 'string') continue;
    if (doc.schema === PROVIDERS_SCHEMA) providers = parseDirectory(doc);
    else if (doc.schema === RATINGS_SCHEMA) models = parseRatings(doc);
    else if (doc.schema === WORKLOAD_SCHEMA) manifest = parseManifest(doc);
  }
  if (providers.length === 0) throw new Error(`no provider directory in ${path}`);
  if (models.length === 0) throw new Error(`no quality table in ${path}`);
  return { providers, models, manifest };
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: WireKind;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/**
 * The catalog: one entry per provider in directory order, each model in the
 * order the quality table lists it. Only free models appear — the brief
 * names no paid ones — each with its id, its free flag and its context
 * length, exactly the fields the score reads.
 */
export function buildCatalog(brief: Brief): Catalog {
  const byProvider = new Map<string, ModelRow[]>();
  for (const row of brief.models) {
    const list = byProvider.get(row.provider) ?? [];
    list.push(row);
    byProvider.set(row.provider, list);
  }
  return {
    providers: brief.providers.map((p) => ({
      id: p.id,
      base_url: p.base,
      key_env: p.keyEnv,
      wire: p.wire,
      models: (byProvider.get(p.id) ?? [])
        .filter((m) => m.free)
        .map((m) => ({ id: m.model, free: true, context_length: m.context })),
    })),
  };
}
