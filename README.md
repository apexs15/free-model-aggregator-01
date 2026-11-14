# free-model-aggregator-01 — entry

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier providers, in JSON and in
server-sent events. Node 22, standard library only, no network beyond the
trial env, no install step.

## Run

```sh
# Print the catalog this server registers, byte for byte:
node main.ts --list

# Serve a trial (the trial sets these two):
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env origin> node main.ts
```

The server listens on an ephemeral port, registers `{protocol_version: 1,
port, catalog}` with the env, answers the client's twenty requests, polls
`/work` every 100 ms, and exits 0 when the workload is done.

## How it routes

- Every request is reduced to one wire-neutral canon (system text, user
  text, one image, tools in order, max_tokens, stops, JSON mode,
  temperature), so what a provider echoes back is exactly what the client
  sent.
- Candidates are the free models meeting the request's requirements,
  ranked by rating. A JSON-mode request only goes to a wire that can carry
  it (OpenAI or Gemini, never Anthropic). A provider is dropped after three
  refusals — a live provider refuses at most twice, on its own failure
  schedule — or when its call count reaches the largest budget the trial
  can draw for it.
- A refusal (402, 429, 5xx, dropped connection) is never an answer: the
  same model is retried once, then the next candidate is tried.
- Upstream calls are always plain JSON; streaming is re-created for the
  client from the provider's text, usage and stop reason, which are
  identical either way. The provider's answer is returned whole: its text
  verbatim, its usage counts exactly as reported, its stop reason mapped
  across wires by family.

## Files

- `main.ts` — server, registration, `/work` polling, request handling.
- `lib/brief.ts` — brief parsing (three JSON documents in the prompt) and
  the catalog.
- `lib/router.ts` — candidate ranking and provider failure tracking.
- `lib/wires.ts` — canon, upstream builders, provider parsers, client
  formatters.
- `LICENSE` — MIT.
