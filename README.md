# Free model aggregator

One server (Node 22, standard library only) serving an OpenAI-compatible API
(`POST /v1/chat/completions`) and an Anthropic-compatible API (`POST /v1/messages`),
JSON and SSE streaming, plus `GET /v1/models` in the shape of the auth header it got.
It routes each request to the best free model across ten simulated free-tier providers
(groq, cerebras, gemini, zai, openrouter, sambanova, pollinations, cloudflare, ovh, kilol),
fails over when a provider refuses, and registers the catalog it read from them.

## How it routes

- Candidates are the free models that meet a request's requirements, ranked by the
  quality table's `rating_milli`; ties go to the provider with the larger free-tier floor.
- A request is identified by its task text (constant across trials) with arrival order as
  the fallback, so the manifest's requirements — including `min_context`, which no request
  body names — are known before routing.
- A refusal is counted per provider: two consecutive refusals retire that provider for the
  trial (one refusal is usually a single scheduled fail_call; two in a row means the budget
  is spent or the provider is out). A retired provider is tried again only as a last resort,
  when every other candidate has refused.
- All three upstream wires are spoken (OpenAI, Anthropic, Gemini) and both client wires,
  with the system prompt, user text, image, tool names, `max_tokens`, stops and JSON mode
  passed through exactly, so the provider's echo matches the client's.

## Run

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE and $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog JSON and exit 0
```

The server listens on an OS-chosen port, registers with the env, serves the workload,
polls `GET /work` and exits 0 when the env says done.

## Layout

- `main.ts` — the server: brief, probe, catalog, register, routes, poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe
  (house scaffold).
- `lib/upstream.ts` — the canonical request, the three upstream callers, the two client
  response builders.
- `lib/plan.ts` — request identification, the ranked candidate table, the router.
