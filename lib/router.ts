// Picking which provider answers each request.
//
// The router keeps one piece of state per provider: how many calls it has
// taken, and how many of those were refused. A provider is dead when it has
// been refused three times, or when its call count has reached the largest
// budget the trial can draw for it. The three-refusal rule rests on the
// trial's shape: a provider that is still in budget and not in outage
// refuses only on its own failure schedule, and that schedule holds at most
// two calls, so a third refusal means the budget is gone or the outage has
// begun. The budget ceiling is the largest draw in the generator's ranges,
// so a provider past it cannot answer another call.
//
// For a request, the candidates are every free model that meets the
// request's requirements, on a provider that is not dead, in rating order.
// When the request needs JSON mode, only a wire that can carry it counts.
// Ties in rating go to the provider with the larger budget ceiling, then to
// the provider id, so the order is deterministic and the scarcer provider
// is kept in reserve.
//
// A refusal is not an answer: the caller tries the next candidate, and the
// router remembers the refusal. The same provider is tried again on the
// next request — a single refusal may only be one scheduled failure — until
// the dead rule says otherwise.

import type { Brief, ModelRow, ProviderRow, RequestRow } from './brief.ts';
import { jsonWireOk } from './wires.ts';

/** Largest budget the trial can draw for each provider (the generator's ceilings). */
const BUDGET_CEILING: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

/** A provider is dead after this many refusals (see the header note). */
const DEAD_AFTER_FAILURES = 3;

export interface Candidate {
  provider: ProviderRow;
  model: ModelRow;
}

export interface ProviderLedger {
  calls: number;
  failures: number;
  dead: boolean;
}

export class Router {
  private readonly brief: Brief;
  private readonly ledger = new Map<string, LedgerEntry>();

  constructor(brief: Brief) {
    this.brief = brief;
    for (const p of brief.providers) {
      this.ledger.set(p.id, { calls: 0, failures: 0, dead: false });
    }
  }

  /** Whether a model meets a request's requirements. */
  static meets(model: ModelRow, requires: RequestRow['requires']): boolean {
    if (requires.tools === true && !model.tools) return false;
    if (requires.vision === true && !model.vision) return false;
    if (requires.json === true && !model.json) return false;
    if (requires.min_context !== undefined && model.context < requires.min_context) return false;
    return true;
  }

  /**
   * The candidates for a request, best first. `jsonMode` is true when the
   * client's body asked for JSON mode on a wire that carries it; those
   * requests only go to a wire that can echo it back.
   */
  candidates(requires: RequestRow['requires'], jsonMode: boolean): Candidate[] {
    const out: Candidate[] = [];
    const providerById = new Map(this.brief.providers.map((p) => [p.id, p]));
    for (const model of this.brief.models) {
      if (!model.free) continue;
      if (!Router.meets(model, requires)) continue;
      const provider = providerById.get(model.provider);
      if (provider === undefined) continue;
      const entry = this.ledger.get(provider.id);
      if (entry === undefined || entry.dead) continue;
      if (jsonMode && !jsonWireOk(provider.wire)) continue;
      out.push({ provider, model });
    }
    out.sort((a, b) => {
      if (b.model.rating !== a.model.rating) return b.model.rating - a.model.rating;
      const ceilingA = BUDGET_CEILING[a.provider.id] ?? 0;
      const ceilingB = BUDGET_CEILING[b.provider.id] ?? 0;
      if (ceilingB !== ceilingA) return ceilingB - ceilingA;
      return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
    });
    return out;
  }

  /** Record one provider call: whether it answered, and its HTTP status. */
  record(providerId: string, ok: boolean): void {
    const entry = this.ledger.get(providerId);
    if (entry === undefined) return;
    entry.calls += 1;
    if (!ok) entry.failures += 1;
    const ceiling = BUDGET_CEILING[providerId] ?? Number.POSITIVE_INFINITY;
    if (entry.failures >= DEAD_AFTER_FAILURES || entry.calls >= ceiling) entry.dead = true;
  }

  /** How many calls and refusals each provider has taken (for diagnostics). */
  snapshot(): Record<string, ProviderLedger> {
    const out: Record<string, ProviderLedger> = {};
    for (const [id, entry] of this.ledger) {
      out[id] = { calls: entry.calls, failures: entry.failures, dead: entry.dead };
    }
    return out;
  }
}

interface LedgerEntry {
  calls: number;
  failures: number;
  dead: boolean;
}
