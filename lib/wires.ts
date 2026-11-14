// The three wires: turning a client request into a provider call, and a
// provider's answer back into the client's wire.
//
// Every request is first reduced to one wire-neutral shape (the "canon"):
// the system text, the user text, one image if the request carries one, the
// tools in the order they were declared, max_tokens, the stop sequences,
// whether JSON mode was asked for, and the temperature. Nothing else in a
// client body reaches a provider, and nothing the provider echoes back is
// allowed to differ from what the client sent — the env compares the echo
// with the client's own body field by field, so the canon is built to keep
// every echoed field byte-identical.
//
// Two rules the canon exists to enforce:
//   - JSON mode only travels on a wire that can carry it. The Anthropic
//     wire has no JSON-mode field, so a JSON-mode request never goes to an
//     Anthropic-wire provider; the router is told this through `jsonWireOk`.
//   - Upstream calls are always plain JSON, never streamed. The provider's
//     text, usage and stop reason are identical either way, and a JSON
//     answer is one object to parse instead of three different event
//     streams. Streaming is a property of the answer back to the client,
//     not of the provider call, so the server re-streams the text itself.
//
// The provider's answer is returned whole: its text verbatim, its usage
// counts exactly as reported, and its stop reason mapped across wires by
// family (stop/end_turn/STOP are one family, length/max_tokens/MAX_TOKENS
// another, and so on).

import type { ProviderRow, WireKind } from './brief.ts';

export type ClientApi = 'openai' | 'anthropic';

export interface CanonTool {
  name: string;
  description: string;
  parameters: unknown;
}

export interface Canon {
  api: ClientApi;
  stream: boolean;
  system: string;
  userText: string;
  /** Raw base64 of the one image, without any data: prefix; null when none. */
  imageB64: string | null;
  imageMime: string;
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const textOf = (v: unknown): string => (typeof v === 'string' ? v : '');

/** The text of a message content: a string as-is, or its text parts joined. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/**
 * Split a data: URL into its mime type and base64 payload. Returns null
 * when the URL is not a base64 data URL.
 */
function splitDataUrl(url: string): { mime: string; b64: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], b64: m[2] };
}

/** Reduce an OpenAI Chat Completions body to the canon. */
function fromOpenAI(body: Record<string, unknown>): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageB64: string | null = null;
  let imageMime = 'image/png';
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
          const url = textOf(block.image_url.url);
          const split = splitDataUrl(url);
          if (split !== null) {
            imageB64 = split.b64;
            imageMime = split.mime;
          }
        }
      }
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const fn = tool.function;
      if (typeof fn.name !== 'string') continue;
      tools.push({
        name: fn.name,
        description: textOf(fn.description),
        parameters: fn.parameters ?? { type: 'object', properties: {} },
      });
    }
  }
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : null;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: 'openai',
    stream: body.stream === true,
    system: systemParts.join('\n'),
    userText,
    imageB64,
    imageMime,
    tools,
    maxTokens,
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
  };
}

/** Reduce an Anthropic Messages body to the canon. */
function fromAnthropic(body: Record<string, unknown>): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageB64: string | null = null;
  let imageMime = 'image/png';
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          imageB64 = src.data;
          imageMime = textOf(src.media_type) || 'image/png';
        }
      }
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: textOf(tool.description),
        parameters: tool.input_schema ?? { type: 'object', properties: {} },
      });
    }
  }
  return {
    api: 'anthropic',
    stream: body.stream === true,
    system: blocksText(body.system),
    userText,
    imageB64,
    imageMime,
    tools,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    stops: stringList(body.stop_sequences),
    // The Anthropic wire carries no JSON-mode field, so a client request on
    // this wire never asks for JSON mode in a way the echo can see.
    jsonMode: false,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
  };
}

export function toCanon(api: ClientApi, body: Record<string, unknown>): Canon {
  return api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
}

/**
 * Whether a provider's wire can carry a JSON-mode request. The Anthropic
 * wire cannot: its parser always reports json_mode false, so the echo check
 * would fail.
 */
export function jsonWireOk(wire: WireKind): boolean {
  return wire === 'openai' || wire === 'gemini';
}

export interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

function authHeader(provider: ProviderRow): Record<string, string> {
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function openAIUserContent(canon: Canon): unknown {
  if (canon.imageB64 === null) return canon.userText;
  return [
    { type: 'text', text: canon.userText },
    { type: 'image_url', image_url: { url: `data:${canon.imageMime};base64,${canon.imageB64}` } },
  ];
}

function anthropicUserContent(canon: Canon): unknown {
  if (canon.imageB64 === null) return canon.userText;
  return [
    { type: 'text', text: canon.userText },
    { type: 'image', source: { type: 'base64', media_type: canon.imageMime, data: canon.imageB64 } },
  ];
}

/**
 * Build the upstream call for one provider and model. The URL is the env
 * origin joined to the provider's own path; the body is plain JSON, never a
 * stream, in the provider's own wire.
 */
export function buildUpstream(
  envUrl: string,
  provider: ProviderRow,
  model: string,
  canon: Canon,
): UpstreamCall {
  const origin = envUrl.replace(/\/$/, '');
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeader(provider),
  };
  if (provider.wire === 'openai') {
    const messages: Record<string, unknown>[] = [];
    if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
    messages.push({ role: 'user', content: openAIUserContent(canon) });
    const body: Record<string, unknown> = {
      model,
      messages,
      ...(canon.maxTokens !== null ? { max_tokens: canon.maxTokens } : {}),
      ...(canon.temperature !== null ? { temperature: canon.temperature } : {}),
      ...(canon.tools.length > 0
        ? {
            tools: canon.tools.map((t) => ({
              type: 'function',
              function: { name: t.name, description: t.description, parameters: t.parameters },
            })),
            tool_choice: 'auto',
          }
        : {}),
      ...(canon.jsonMode ? { response_format: { type: 'json_object' } } : {}),
      ...(canon.stops.length > 0 ? { stop: canon.stops } : {}),
    };
    return { url: `${origin}${provider.base}/v1/chat/completions`, headers, body: JSON.stringify(body) };
  }
  if (provider.wire === 'anthropic') {
    const body: Record<string, unknown> = {
      model,
      ...(canon.maxTokens !== null ? { max_tokens: canon.maxTokens } : {}),
      ...(canon.system !== '' ? { system: canon.system } : {}),
      messages: [{ role: 'user', content: anthropicUserContent(canon) }],
      ...(canon.temperature !== null ? { temperature: canon.temperature } : {}),
      ...(canon.tools.length > 0
        ? {
            tools: canon.tools.map((t) => ({
              name: t.name,
              description: t.description,
              input_schema: t.parameters,
            })),
          }
        : {}),
      ...(canon.stops.length > 0 ? { stop_sequences: canon.stops } : {}),
    };
    // anthropic-version travels as a header, never in the body.
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${origin}${provider.base}/v1/messages`, headers, body: JSON.stringify(body) };
  }
  // Gemini wire: the model rides in the path, the rest in generationConfig.
  const parts: Record<string, unknown>[] = [{ text: canon.userText }];
  if (canon.imageB64 !== null) {
    parts.push({ inlineData: { mimeType: canon.imageMime, data: canon.imageB64 } });
  }
  const config: Record<string, unknown> = {
    ...(canon.maxTokens !== null ? { maxOutputTokens: canon.maxTokens } : {}),
    ...(canon.stops.length > 0 ? { stopSequences: canon.stops } : {}),
    ...(canon.jsonMode ? { responseMimeType: 'application/json' } : {}),
    ...(canon.temperature !== null ? { temperature: canon.temperature } : {}),
  };
  const body: Record<string, unknown> = {
    contents: [{ role: 'user', parts }],
    ...(canon.system !== ''
      ? { systemInstruction: { parts: [{ text: canon.system }] } }
      : {}),
    ...(canon.tools.length > 0
      ? {
          tools: [
            {
              functionDeclarations: canon.tools.map((t) => ({
                name: t.name,
                description: t.description,
                parameters: t.parameters,
              })),
            },
          ],
        }
      : {}),
    generationConfig: config,
  };
  return {
    url: `${origin}${provider.base}/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    headers,
    body: JSON.stringify(body),
  };
}

export interface ProviderAnswer {
  ok: boolean;
  /** The provider's answer text, whole, when ok. */
  text: string;
  usage: { input: number; output: number } | null;
  /** The provider's own stop reason string ("stop", "end_turn" or "STOP"). */
  stop: string | null;
  status: number;
  code: string | null;
}

/** Join the text parts of a Gemini candidate's content. */
function geminiText(candidate: Record<string, unknown>): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/**
 * Parse one provider's JSON answer. A 2xx with the expected shape yields
 * the text, the usage and the stop reason; anything else is a refusal, and
 * its status and error code are what the router records.
 */
export function parseProviderAnswer(wire: WireKind, status: number, raw: string): ProviderAnswer {
  let value: unknown = null;
  try {
    value = JSON.parse(raw);
  } catch {
    value = null;
  }
  if (status < 200 || status >= 300 || !isRecord(value)) {
    let code: string | null = null;
    if (isRecord(value)) {
      const err = isRecord(value.error) ? value.error : null;
      if (err !== null && typeof err.code === 'string') code = err.code;
      else if (typeof value.code === 'string') code = value.code;
      else if (isRecord(value.error) && typeof err.type === 'string') code = err.type;
    }
    return { ok: false, text: '', usage: null, stop: null, status, code };
  }
  if (wire === 'openai') {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (first === null) return { ok: false, text: '', usage: null, stop: null, status, code: 'bad_shape' };
    const message = isRecord(first.message) ? first.message : null;
    const text =
      message !== null
        ? typeof message.content === 'string'
          ? message.content
          : blocksText(message.content)
        : '';
    const usage = isRecord(value.usage)
      ? {
          input: typeof value.usage.prompt_tokens === 'number' ? value.usage.prompt_tokens : 0,
          output: typeof value.usage.completion_tokens === 'number' ? value.usage.completion_tokens : 0,
        }
      : null;
    return {
      ok: true,
      text,
      usage,
      stop: typeof first.finish_reason === 'string' ? first.finish_reason : null,
      status,
      code: null,
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(value.content) ? value.content : [];
    const texts: string[] = [];
    for (const block of content) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    }
    const usage = isRecord(value.usage)
      ? {
          input: typeof value.usage.input_tokens === 'number' ? value.usage.input_tokens : 0,
          output: typeof value.usage.output_tokens === 'number' ? value.usage.output_tokens : 0,
        }
      : null;
    return {
      ok: true,
      text: texts.join(''),
      usage,
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : null,
      status,
      code: null,
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return { ok: false, text: '', usage: null, stop: null, status, code: 'bad_shape' };
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : null;
  const usage =
    meta !== null
      ? {
          input: typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0,
          output: typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0,
        }
      : null;
  return {
    ok: true,
    text: geminiText(first),
    usage,
    stop: typeof first.finishReason === 'string' ? first.finishReason : null,
    status,
    code: null,
  };
}

// ---------------------------------------------------------------------------
// Answers back to the client, in the wire it used.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_AT = '2026-01-01T00:00:00Z';
const DELTA_CHARS = 13;

/** Stop-reason families, shared by all three wires. */
const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

const FAMILY_STOP: Record<number, { openai: string; anthropic: string }> = {
  0: { openai: 'stop', anthropic: 'end_turn' },
  1: { openai: 'length', anthropic: 'max_tokens' },
  2: { openai: 'tool_calls', anthropic: 'tool_use' },
  3: { openai: 'content_filter', anthropic: 'refusal' },
};

/** Map a provider's stop reason into the client's wire, by family. */
export function mapStop(providerStop: string | null, api: ClientApi): string {
  const family = providerStop === null ? 0 : (STOP_FAMILY.get(providerStop) ?? 0);
  return FAMILY_STOP[family][api];
}

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

export interface ClientAnswer {
  status: number;
  headers: Record<string, string>;
  body: string;
}

function openAIJson(model: string, answer: ProviderAnswer, api: ClientApi): ClientAnswer {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const value = {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: mapStop(answer.stop, api),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
  return { status: 200, headers: { 'content-type': 'application/json' }, body: `${JSON.stringify(value)}\n` };
}

function openAIStream(model: string, answer: ProviderAnswer): ClientAnswer {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    `data: ${JSON.stringify({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  const frames: string[] = [chunk({ role: 'assistant', content: '' }, null)];
  for (const piece of splitPieces(answer.text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, 'stop'));
  frames.push(
    `data: ${JSON.stringify({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' },
    body: frames.join(''),
  };
}

function anthropicJson(model: string, answer: ProviderAnswer, api: ClientApi): ClientAnswer {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const value = {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: mapStop(answer.stop, api),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
  return { status: 200, headers: { 'content-type': 'application/json' }, body: `${JSON.stringify(value)}\n` };
}

function anthropicStream(model: string, answer: ProviderAnswer, api: ClientApi): ClientAnswer {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frame = (event: string, data: Record<string, unknown>): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    frame('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      frame('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(frame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    frame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(answer.stop, api), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  frames.push(frame('message_stop', { type: 'message_stop' }));
  return {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' },
    body: frames.join(''),
  };
}

/**
 * Format one provider answer for the client that asked: the client's wire,
 * streamed exactly as it asked, with the provider's text whole, its usage
 * counts as reported, and its stop reason in the client's family.
 */
export function formatClientAnswer(
  api: ClientApi,
  stream: boolean,
  model: string,
  answer: ProviderAnswer,
): ClientAnswer {
  if (api === 'openai') {
    return stream ? openAIStream(model, answer) : openAIJson(model, answer, api);
  }
  return stream ? anthropicStream(model, answer, api) : anthropicJson(model, answer, api);
}

/** The model list for GET /v1/models, in the shape the auth header names. */
export function formatModelList(
  bearer: boolean,
  freeIds: { id: string; provider: string; context: number; tools: boolean; vision: boolean; json: boolean }[],
): ClientAnswer {
  if (bearer) {
    const value = {
      object: 'list',
      data: freeIds.map((m) => ({
        id: m.id,
        object: 'model',
        created: FIXED_CREATED,
        owned_by: m.provider,
        free: true,
        context_length: m.context,
        tools: m.tools,
        vision: m.vision,
        json: m.json,
      })),
    };
    return { status: 200, headers: { 'content-type': 'application/json' }, body: `${JSON.stringify(value)}\n` };
  }
  const value = {
    data: freeIds.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.id,
      created_at: FIXED_ANTHROPIC_AT,
      free: true,
      context_length: m.context,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    })),
    has_more: false,
  };
  return { status: 200, headers: { 'content-type': 'application/json' }, body: `${JSON.stringify(value)}\n` };
}
