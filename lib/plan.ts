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
//   - a refusal is counted per provider: two consecutive refusals retire that provider for the
//     trial (a single refusal is usually one scheduled fail_call, and the provider still has
//     budget; two in a row means the budget is spent or the provider is out). A retired provider
//     is tried again only as a last resort, when every other candidate has refused and the
//     request would otherwise go unanswered.

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
 * The router's memory for one trial: how many times each provider has refused in a row, and
 * which providers are retired. A refusal is counted on any non-2xx answer (a 402, a 429, a
 * 5xx, or a malformed call, which is a bug and must not be retried on the same provider).
 */
export class Router {
  private consecutive = new Map<string, number>();
  private retired = new Set<string>();
  private calls = new Map<string, number>();

  /** Providers retired so far, for the log. */
  retiredProviders(): string[] {
    return [...this.retired].sort();
  }

  /**
   * Serve one request on its ranked candidates, in order, one provider at most once: the first
   * answer wins. Two consecutive refusals retire a provider; a retired provider is tried again
   * only when every other candidate has refused and the request would otherwise go unanswered.
   * A retired provider that answers again is un-retired: its refusals were scheduled singletons,
   * not a spent budget.
   */
  async serve<T extends CallOutcome>(candidates: readonly Candidate[], call: (candidate: Candidate) => Promise<T>): Promise<{ candidate: Candidate; outcome: T } | null> {
    const tried = new Set<string>();
    for (const candidate of candidates) {
      const id = candidate.provider.id;
      if (this.retired.has(id) || tried.has(id)) continue;
      // Past its ceiling a provider cannot answer: spend no call there.
      if ((this.calls.get(id) ?? 0) >= callCapOf(id)) continue;
      tried.add(id);
      this.calls.set(id, (this.calls.get(id) ?? 0) + 1);
      const outcome = await call(candidate);
      if (outcome.ok) {
        this.consecutive.set(id, 0);
        return { candidate, outcome };
      }
      const count = (this.consecutive.get(id) ?? 0) + 1;
      this.consecutive.set(id, count);
      if (count >= 2) this.retired.add(id);
    }
    for (const candidate of candidates) {
      const id = candidate.provider.id;
      if (tried.has(id)) continue;
      if ((this.calls.get(id) ?? 0) >= callCapOf(id)) continue;
      tried.add(id);
      this.calls.set(id, (this.calls.get(id) ?? 0) + 1);
      const outcome = await call(candidate);
      if (outcome.ok) {
        this.consecutive.set(id, 0);
        this.retired.delete(id);
        return { candidate, outcome };
      }
      this.consecutive.set(id, (this.consecutive.get(id) ?? 0) + 1);
    }
    return null;
  }
}
