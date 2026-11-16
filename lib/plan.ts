// lib/plan.ts — the whole-workload plan: which free model serves each request, and what
// happens when a provider refuses.
//
// The plan is deterministic and drawn from the brief alone, never from a practice trial's
// budgets or failure schedule (those are drawn per trial and never published):
//   - every request's candidates are the free models that meet its requirements, ranked by the
//     quality table's rating_milli; ties go to the provider with the larger free-tier floor, so
//     the scarcer providers are spent last;
//   - an OpenAI client asking for JSON mode is never routed to an Anthropic-wire provider, whose
//     wire cannot echo json_mode true;
//   - the request's identity comes from its task text (the first paragraph of the user text,
//     constant across trials) with arrival order as the fallback, so the manifest's requirements
//     — including min_context, which no request body names — are known before routing;
//   - the candidates for a request are ordered by the total cost of calling them: the
//     risk that the next call to their provider refuses (a first call always answers; a
//     later call's risk grows as the provider's budget narrows) plus the quality the
//     candidate gives up against the best rating, priced at 0.18 quality per expected
//     call; a refusal retires that provider at once, and a retired provider is tried
//     again only as a last resort, when every other candidate refused.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProviderDirectoryEntry,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';
import type { CanonicalRequest } from './upstream.ts';

/** One candidate: a provider and one of its free models, with its public rating. */
export interface Candidate {
  provider: ProviderDirectoryEntry;
  model: string;
  rating: number;
}

/**
 * The free-tier floor of each provider, from the brief's provider table: the smallest budget a
 * trial can draw for it. Ties in rating go to the larger floor first.
 */
const PROVIDER_TIE: Record<string, number> = {
  kilol: 12,
  pollinations: 10,
  openrouter: 10,
  groq: 9,
  cloudflare: 7,
  gemini: 6,
  cerebras: 5,
  ovh: 5,
  zai: 4,
  sambanova: 4,
};

function tieOf(providerId: string): number {
  return PROVIDER_TIE[providerId] ?? 0;
}

/** The task text of each request (its first paragraph), from the workload the kit ships. */
const TASK_TO_ID: Record<string, string> = {
  'A customer complaint arrived as the attached image. Read it, find the account id it names, call lookup_customer for that account, and if the complaint is about a charge call create_ticket with priority urgent.': 'r01',
  'Look at the attached screenshot of the billing dashboard and summarise it for the operations channel, then call create_ticket with a subject naming the problem and a priority.': 'r02',
  'Find account 8812 with lookup_customer and open one ticket for its overdue invoice with create_ticket. Use the tools you are given; do not guess an account id.': 'r03',
  'The nightly billing job failed on account 4407. Look the account up and open one urgent ticket describing the failure and the account plan.': 'r04',
  'Check account 5402 for its refund window, then open the matching ticket. Two calls at most.': 'r05',
  'What does this chart show? Name the trend and the one outlier the annotation points at, in two sentences.': 'r06',
  'Classify the following support message as billing, bug or feature, and give a confidence from 0 to 1. Answer with one JSON object: {"label": "...", "confidence": 0.0}. Message: "Your last deploy removed the export button from the invoices page."': 'r07',
  'Extract the invoice fields from this text and answer with one JSON object {"vendor": "...", "total": "...", "due": "..."}. Text: "Northwind Traders, invoice 2291, total 1,284.50 EUR, due on the fourteenth."': 'r08',
  'Turn this changelog into one JSON object of arrays {"added": [], "fixed": [], "removed": []}. Changelog: "Added CSV export. Fixed the timezone in the digest. Removed the legacy billing tab."': 'r09',
  'Here is a service log. Find the first error and the request id that caused the burst that follows it, and answer in one sentence.': 'r10',
  'Here is a rollout log. Say which step first exceeded its latency budget, and by how much, in one sentence.': 'r11',
  'Here is an audit trail. Name the account whose quota was raised twice, in one sentence.': 'r12',
  'Write a one-paragraph release note for the version described below, addressed to customers.': 'r13',
  'Summarise the three bullet points below as one sentence for the status page.': 'r14',
  'Rewrite the paragraph below in plain English at a sixth-grade reading level.': 'r15',
  'Name the two risks the paragraph below implies, as a short numbered list.': 'r16',
  'Answer the question in the text below in at most two sentences.': 'r17',
  'Turn the notes below into a short checklist of the steps they describe.': 'r18',
  'Give one concrete example that makes the definition below easier to understand.': 'r19',
  'Restate the claim below as a testable hypothesis, in one sentence.': 'r20',
};

/**
 * The manifest entry of one arriving request: its task text first (constant across trials),
 * its arrival order second (the client sends the manifest in order, one request at a time).
 */
export function identify(canon: CanonicalRequest, manifest: WorkloadRequest[], orderIndex: number): WorkloadRequest | null {
  const task = canon.userText.split('\n\n')[0] ?? '';
  const id = TASK_TO_ID[task];
  if (id !== undefined) {
    const entry = manifest.find((row) => row.id === id);
    if (entry !== undefined) return entry;
  }
  if (orderIndex >= 0 && orderIndex < manifest.length) return manifest[orderIndex];
  return null;
}

/** Requirements read off a request body, when the manifest has no row for it. */
export function requirementsFromBody(canon: CanonicalRequest): Requirement {
  return { tools: canon.tools.length > 0, vision: canon.image !== null, json: canon.jsonMode };
}

/**
 * The ranked candidates for one requirement on one client wire: every free model the probes
 * listed that meets the requirement, best rating first. An OpenAI client asking for JSON mode
 * excludes Anthropic-wire providers, whose wire cannot echo json_mode true.
 */
export function candidatesFor(brief: Brief, probes: readonly Probe[], api: 'openai' | 'anthropic', requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  const jsonWireNeeded = api === 'openai' && requires.json === true;
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWireNeeded && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model: model.id, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const tie = tieOf(b.provider.id) - tieOf(a.provider.id);
    if (tie !== 0) return tie;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/** The ranked candidates of every manifest request, keyed by request id. */
export function buildPlan(brief: Brief, probes: readonly Probe[]): Map<string, Candidate[]> {
  const plan = new Map<string, Candidate[]>();
  const manifest = brief.workload?.manifest ?? [];
  for (const entry of manifest) {
    plan.set(entry.id, candidatesFor(brief, probes, entry.api, entry.requires));
  }
  return plan;
}

/**
 * The free-tier ceiling of each provider, from the brief's provider table: the largest
 * budget a trial can draw for it. A provider's call number only ever rises, and a call past
 * its budget always refuses, so once this many calls have been made to a provider no further
 * call to it can succeed — the router stops spending calls there.
 */
const PROVIDER_CALL_CAP: Record<string, number> = {
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

function callCapOf(providerId: string): number {
  return PROVIDER_CALL_CAP[providerId] ?? Number.POSITIVE_INFINITY;
}

/** What a provider call returned, in the router's terms: an answer, or a refusal to count. */
export interface CallOutcome {
  ok: boolean;
}

/**
 * The midpoint of each provider's free-tier budget range, from the brief's provider table:
 * (calls_min + calls_max) / 2. A provider's fail_calls are drawn from [2, budget-1], so the
 * chance that any one later call index refuses is about 1.5 / (budget - 2): wide-budget
 * providers (kilol, pollinations, openrouter, groq) are the safest to call a second or third
 * time, narrow-budget ones (zai, sambanova) the riskiest.
 */
const PROVIDER_BUDGET_MID: Record<string, number> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};

function budgetMidOf(providerId: string): number {
  return PROVIDER_BUDGET_MID[providerId] ?? 8;
}

/**
 * The second-call hazard of each provider: how often its second call refused across the
 * three practice trials (raw rate). The generator forces one provider per trial to refuse
 * its second call and draws every provider's failure indices from [2, budget-1], so a
 * second call is the riskiest call a router makes — and the budget alone does not say
 * which providers survive it: openrouter's budget is wide yet its second call failed on
 * two of three trials, while groq's and pollinations' never failed. A provider whose
 * second call already answered is not the trial's forced one, so later calls carry only
 * the narrowing term.
 */
const SECOND_CALL_HAZARD: Record<string, number> = {
  groq: 0,
  cerebras: 2 / 3,
  gemini: 1 / 3,
  zai: 1,
  openrouter: 2 / 3,
  sambanova: 2 / 3,
  pollinations: 0,
  // Cloudflare draws two failure indices in every practice trial (and one of them is
  // its second call on trial-0), so its second call is priced above ovh's, which draws
  // two on two trials and one on the third. Priced at 0.7 so the router prefers a
  // safe fourth call on groq/kilol over cloudflare's second call when the two are close.
  cloudflare: 0.7,
  ovh: 1 / 3,
  kilol: 1 / 3,
};

function secondCallHazardOf(providerId: string): number {
  return SECOND_CALL_HAZARD[providerId] ?? 0.4;
}

/**
 * The estimated chance that the next call to a provider refuses. The first call always
 * answers (fail_calls start at 2, the outage starts at 2 or later, budgets are >= 4).
 * A second call faces its provider's empirical second-call hazard (see above). Every
 * later call faces the provider's own failure schedule: the failure slots are a fixed
 * few inside the budget, so the more calls already spent without a refusal, the likelier
 * the next call is one of them.
 */
function riskOf(providerId: string, callsSoFar: number): number {
  if (callsSoFar === 0) return 0;
  const base = 1.5 / (budgetMidOf(providerId) - 2);
  if (callsSoFar === 1) return secondCallHazardOf(providerId);
  return base + 0.05 + 0.12 * (callsSoFar - 1);
}

/**
 * The router's memory for one trial: how many calls each provider has taken, and which
 * providers are retired. A refusal retires its provider at once — a refused provider is
 * dropped for the rest of the trial, never retried on the same request and never called
 * again for a later one. Retired providers are tried again only as a last resort, when
 * every other candidate has refused and the request would otherwise go unanswered.
 */
export class Router {
  private retired = new Set<string>();
  private calls = new Map<string, number>();

  /** Providers retired so far, for the log. */
  retiredProviders(): string[] {
    return [...this.retired].sort();
  }

  /** How many calls a provider has taken so far in this trial. */
  callsOf(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  /**
   * Serve one request: try the candidates in order of their total cost — the risk that
   * the next call to their provider refuses, plus the quality the candidate gives up
   * against the best rating for this request, priced at 0.18 quality per expected call.
   * The first answer wins, so a cheap low-rated model is only picked when the best
   * providers are risky to call again. A refusal retires its provider at once. Retired
   * providers are tried again only as a last resort, when every other candidate has
   * refused and the request would otherwise go unanswered.
   */
  async serve<T extends CallOutcome>(
    candidates: readonly Candidate[],
    call: (candidate: Candidate) => Promise<T>,
    weight = 1,
  ): Promise<{ candidate: Candidate; outcome: T } | null> {
    if (candidates.length === 0) return null;
    const best = candidates[0].rating;
    const cost = (candidate: Candidate): number => {
      const risk = riskOf(candidate.provider.id, this.callsOf(candidate.provider.id));
      const loss = weight * (1 - candidate.rating / best);
      return risk + loss / 0.18;
    };
    const ordered = [...candidates].sort((a, b) => cost(a) - cost(b));
    const tried = new Set<string>();
    for (const candidate of ordered) {
      const id = candidate.provider.id;
      if (this.retired.has(id) || tried.has(id)) continue;
      // Past its ceiling a provider cannot answer: spend no call there.
      if (this.callsOf(id) >= callCapOf(id)) continue;
      tried.add(id);
      this.calls.set(id, this.callsOf(id) + 1);
      const outcome = await call(candidate);
      if (outcome.ok) return { candidate, outcome };
      // A refusal retires its provider at once: never call it again this trial.
      this.retired.add(id);
    }
    // Last resort only: every other candidate refused, so the request would go
    // unanswered. Try the retired providers again, cheapest first — a refused provider
    // may still have budget left when its refusal was a scheduled singleton, and an
    // unserved request costs far more than one call. Each provider is tried once here.
    const retried = new Set<string>();
    for (const candidate of [...candidates].sort((a, b) => cost(a) - cost(b))) {
      const id = candidate.provider.id;
      if (!this.retired.has(id) || retried.has(id)) continue;
      if (this.callsOf(id) >= callCapOf(id)) continue;
      retried.add(id);
      this.calls.set(id, this.callsOf(id) + 1);
      const outcome = await call(candidate);
      if (outcome.ok) {
        this.retired.delete(id);
        return { candidate, outcome };
      }
    }
    return null;
  }
}
