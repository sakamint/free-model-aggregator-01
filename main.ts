// main.ts — one server, two client wires, ten free providers.
//
// The shape of a trial: the brief at $MAKEFUN_PROMPT_FILE carries the provider directory,
// the quality table and the workload manifest; the env at $MAKEFUN_ENV_URL serves the providers
// and the control routes. This server probes every provider's catalog, registers the one
// catalog it built, then answers the client's twenty requests one at a time — each on the best
// free model that meets it, failing over when a provider refuses — and exits 0 the moment the
// env says the workload is done. A process still running at the wall scores zero, so the poll
// is the only thing the server waits on.
//
// `node main.ts --list` prints the same catalog JSON the register body carries and exits 0
// before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  probeProvider,
  ratingOfModel,
  readBrief,
  modelFree,
  type CatalogReport,
  type Json,
  type Probe,
} from './lib/providers.ts';
import {
  anthropicJsonResponse,
  anthropicStreamResponse,
  callAnthropic,
  callGemini,
  callOpenAI,
  canonicalize,
  openAIJsonResponse,
  openAIStreamResponse,
  type CanonicalRequest,
  type UpstreamResult,
} from './lib/upstream.ts';
import {
  buildPlan,
  candidatesFor,
  identify,
  requirementsFromBody,
  Router,
  type Candidate,
} from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan = new Map<string, Candidate[]>();
const router = new Router();
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Upstream provider calls made so far, for the log. */
let upstreamCalls = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog and the whole-workload plan. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = buildPlan(brief, probes);
}

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** POST one request to one candidate's provider, in that provider's own wire. */
async function callCandidate(candidate: Candidate, canon: CanonicalRequest): Promise<UpstreamResult> {
  upstreamCalls += 1;
  const wire = candidate.provider.wire;
  if (wire === 'openai') return callOpenAI(ENV_URL, candidate.provider, candidate.model, canon);
  if (wire === 'anthropic') return callAnthropic(ENV_URL, candidate.provider, candidate.model, canon);
  return callGemini(ENV_URL, candidate.provider, candidate.model, canon);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') {
    sendJson(res, status, { type: 'error', error: { type, message } });
    return;
  }
  sendJson(res, status, { error: { message, type, param: null, code: null } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((model) => ({ type: 'model', id: model.id, display_name: model.display_name, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: list.map((model) => ({ id: model.id, object: 'model', created: 1767225600, owned_by: model.provider })),
  });
}

/**
 * Serve one client request: identify it, take its ranked candidates, and let the router spend
 * them until one provider answers. The answer is rendered back in the client's wire.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const canon = canonicalize(api, body);
  if (canon === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one request');
  const orderIndex = manifestIndex;
  manifestIndex += 1;
  const manifest = brief.workload?.manifest ?? [];
  const entry = identify(canon, manifest, orderIndex);
  const requires = entry?.requires ?? requirementsFromBody(canon);
  const candidates = entry !== null ? (plan.get(entry.id) ?? []) : candidatesFor(brief, probes, api, requires);
  const label = entry?.id ?? `?${orderIndex}`;
  if (candidates.length === 0) {
    log(`${label}: no free model meets the request`);
    return sendError(res, api, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const served = await router.serve(candidates, (candidate) => callCandidate(candidate, canon));
  if (served === null) {
    log(`${label}: every candidate refused (${upstreamCalls} upstream calls so far)`);
    return sendError(res, api, 502, 'upstream_unreachable', 'every provider that could serve this request refused it');
  }
  const { candidate, outcome } = served;
  if (!outcome.ok) {
    log(`${label}: the router returned a refusal as an answer`);
    return sendError(res, api, 502, 'upstream_unreachable', 'every provider that could serve this request refused it');
  }
  log(
    `${label}: ${candidate.provider.id}/${candidate.model} rating ${candidate.rating}` +
      ` (upstream call ${upstreamCalls}, retired: ${router.retiredProviders().join(',') || 'none'})`,
  );
  const answer = outcome.answer;
  if (api === 'openai') {
    if (canon.stream) sendSse(res, openAIStreamResponse(candidate.model, answer));
    else sendJson(res, 200, openAIJsonResponse(candidate.model, answer));
    return;
  }
  if (canon.stream) sendSse(res, anthropicStreamResponse(candidate.model, answer));
  else sendJson(res, 200, anthropicJsonResponse(candidate.model, answer));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
