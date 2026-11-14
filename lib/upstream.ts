// lib/upstream.ts — the three provider wires and the two client wires, written for this entry.
//
// A client request (OpenAI or Anthropic) is first reduced to one canonical form that keeps
// everything the providers echo back verbatim: the system prompt, the user text, the image, the
// tool names, max_tokens, the stop sequences and JSON mode. Each upstream wire then renders that
// form in its own shape, and each client wire renders the provider's answer back in the shape the
// client asked for. Nothing here depends on a wall clock or on randomness: the same trial replays
// byte for byte.
//
// The echo rules this file honours (they are what the wire score checks):
//   - the provider echoes the system text, the user text, the tool names, max_tokens, the stops,
//     the image bytes and the JSON-mode flag of the UPSTREAM body, and the client compares that
//     echo with the one its own body would give — so every one of those fields is passed through
//     exactly, in the upstream wire's spelling;
//   - an Anthropic wire cannot express JSON mode (its echo always says json_mode false), so an
//     OpenAI client asking for JSON is never routed to an Anthropic-wire provider (see plan.ts);
//   - a Gemini wire carries the model in the URL path, the system prompt in systemInstruction, the
//     user text and image in contents parts, the tools as functionDeclarations and max_tokens,
//     stops and JSON mode in generationConfig;
//   - the provider's answer text is relayed whole, its usage is relayed whole, and its stop is
//     rendered in the client's wire (stop / end_turn), so the usage and stop checks see the
//     provider's own numbers.

import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';

/** One image, as the client sent it: a data URL's bytes, or a plain URL. */
export interface ImageRef {
  mimeType: string;
  data: string;
  url: string | null;
}

/** One tool, as the client named it: only the name is echoed, the rest is carried faithfully. */
export interface ToolDef {
  name: string;
  description: string;
  parameters: unknown;
}

/** One client request, either wire, reduced to the fields a provider echoes. */
export interface CanonicalRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  systemText: string;
  userText: string;
  image: ImageRef | null;
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

/** A provider's answer: the text, the usage and the stop, all the provider's own. */
export interface UpstreamAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

export type UpstreamResult =
  | { ok: true; status: 200; answer: UpstreamAnswer }
  | { ok: false; status: number; error: string };

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a content field: a string as it is, text blocks joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The image of a content field: the last image block, as a data URL or a plain URL. */
function imageOf(content: unknown): ImageRef | null {
  if (!Array.isArray(content)) return null;
  let found: ImageRef | null = null;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      found = imageRefOfUrl(block.image_url.url);
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        found = { mimeType: stringOf(source.media_type) !== '' ? stringOf(source.media_type) : 'image/png', data: source.data, url: null };
      } else if (source.type === 'url' && typeof source.url === 'string') {
        found = imageRefOfUrl(source.url);
      }
    }
  }
  return found;
}

function imageRefOfUrl(url: string): ImageRef {
  const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { mimeType: dataUrl[1], data: dataUrl[2], url: null };
  return { mimeType: '', data: '', url };
}

/** The tools of a body, in either wire's spelling, reduced to name/description/parameters. */
function toolsOf(body: Json): ToolDef[] {
  const out: ToolDef[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (isRecord(tool.function) && typeof tool.function.name === 'string') {
      out.push({ name: tool.function.name, description: stringOf(tool.function.description), parameters: tool.function.parameters ?? undefined });
    } else if (typeof tool.name === 'string') {
      out.push({ name: tool.name, description: stringOf(tool.description), parameters: tool.input_schema ?? undefined });
    }
  }
  return out;
}

/** Reduce one client body, either wire, to the canonical form. Null when it is not one request. */
export function canonicalize(api: 'openai' | 'anthropic', body: Json): CanonicalRequest | null {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  if (api === 'openai') {
    const systemParts: string[] = [];
    let userText = '';
    let image: ImageRef | null = null;
    for (const message of messages) {
      if (message.role === 'system') {
        const text = blocksText(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') {
        userText = blocksText(message.content);
        image = imageOf(message.content) ?? image;
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      api,
      stream: body.stream === true,
      systemText: systemParts.join('\n'),
      userText,
      image,
      tools: toolsOf(body),
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      jsonMode: format === 'json_object' || format === 'json_schema',
      temperature: numberOrNull(body.temperature),
    };
  }
  const systemText = blocksText(body.system);
  let userText = '';
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    image = imageOf(message.content) ?? image;
  }
  return {
    api,
    stream: body.stream === true,
    systemText,
    userText,
    image,
    tools: toolsOf(body),
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
  };
}

/** One SSE frame: its event name (null for a bare data frame) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an SSE body into frames, the way the client's parser does. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function usageOf(value: Record<string, unknown>): { input: number; output: number } | null {
  if (!isRecord(value.usage)) return null;
  const input = numberOrNull(value.usage.prompt_tokens) ?? numberOrNull(value.usage.input_tokens);
  const output = numberOrNull(value.usage.completion_tokens) ?? numberOrNull(value.usage.output_tokens);
  if (input === null || output === null) return null;
  return { input, output };
}

/** The answer text of an OpenAI JSON body or stream chunk: the message content, or a delta. */
function openAIText(value: Record<string, unknown>): { text: string; stop: string | null; usage: { input: number; output: number } | null } {
  const usage = usageOf(value);
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return { text: '', stop: null, usage };
  const message = isRecord(choice.message) ? choice.message : null;
  const delta = isRecord(choice.delta) ? choice.delta : null;
  const text = typeof message?.content === 'string' ? message.content : typeof delta?.content === 'string' ? delta.content : '';
  const stop = typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
  return { text, stop, usage };
}

/** The answer text of an Anthropic JSON body or stream event. */
function anthropicText(value: Record<string, unknown>): { text: string; stop: string | null; usage: { input: number; output: number } | null } {
  if (isRecord(value.message) && isRecord(value.message.usage)) {
    return { text: '', stop: null, usage: usageOf({ usage: value.message.usage }) };
  }
  if (isRecord(value.delta) && typeof value.delta.text === 'string') {
    return { text: value.delta.text, stop: null, usage: null };
  }
  if (isRecord(value.delta) && typeof value.delta.stop_reason === 'string') {
    const usage = isRecord(value.usage) ? usageOf(value) : null;
    return { text: '', stop: value.delta.stop_reason, usage };
  }
  const content = Array.isArray(value.content) ? value.content : [];
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return { text: parts.join(''), stop: typeof value.stop_reason === 'string' ? value.stop_reason : null, usage: usageOf(value) };
}

/** The answer text of a Gemini JSON body or stream frame. */
function geminiText(value: Record<string, unknown>): { text: string; stop: string | null; usage: { input: number; output: number } | null } {
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  let text = '';
  if (candidate !== null && isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
  }
  const stop = candidate !== null && typeof candidate.finishReason === 'string' ? candidate.finishReason : null;
  let usage: { input: number; output: number } | null = null;
  if (isRecord(value.usageMetadata)) {
    const input = numberOrNull(value.usageMetadata.promptTokenCount);
    const output = numberOrNull(value.usageMetadata.candidatesTokenCount);
    if (input !== null && output !== null) usage = { input, output };
  }
  return { text, stop, usage };
}

async function readUpstream(res: Response, wire: 'openai' | 'anthropic' | 'gemini', stream: boolean): Promise<UpstreamResult> {
  const bodyText = await res.text();
  if (!res.ok) return { ok: false, status: res.status, error: bodyText.slice(0, 500) };
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (value === null) return { ok: false, status: 502, error: 'the provider answered a body that is not JSON' };
    const parsed = wire === 'openai' ? openAIText(value) : wire === 'anthropic' ? anthropicText(value) : geminiText(value);
    text = parsed.text;
    usage = parsed.usage;
    stop = parsed.stop;
  } else {
    for (const frame of parseSse(bodyText)) {
      if (frame.data.trim() === '[DONE]') break;
      const value = jsonOrNull(frame.data);
      if (value === null) continue;
      const parsed = wire === 'openai' ? openAIText(value) : wire === 'anthropic' ? anthropicText(value) : geminiText(value);
      text += parsed.text;
      if (parsed.usage !== null) usage = parsed.usage;
      if (parsed.stop !== null) stop = parsed.stop;
    }
  }
  if (usage === null) return { ok: false, status: 502, error: 'the provider answered no usage' };
  return { ok: true, status: 200, answer: { text, usage, stop: stop ?? 'stop' } };
}

function headersFor(provider: ProviderDirectoryEntry, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  return headers;
}

/** The user message of an OpenAI upstream body: text alone, or text plus the image. */
function openAIUserContent(canon: CanonicalRequest): unknown {
  if (canon.image === null) return canon.userText;
  const content: Record<string, unknown>[] = [{ type: 'text', text: canon.userText }];
  if (canon.image.url !== null) content.push({ type: 'image_url', image_url: { url: canon.image.url } });
  else content.push({ type: 'image_url', image_url: { url: `data:${canon.image.mimeType};base64,${canon.image.data}` } });
  return content;
}

/** The user content of an Anthropic upstream body: text alone, or text plus the image. */
function anthropicUserContent(canon: CanonicalRequest): unknown {
  if (canon.image === null) return canon.userText;
  const content: Record<string, unknown>[] = [{ type: 'text', text: canon.userText }];
  if (canon.image.url !== null) content.push({ type: 'image', source: { type: 'url', url: canon.image.url } });
  else content.push({ type: 'image', source: { type: 'base64', media_type: canon.image.mimeType, data: canon.image.data } });
  return content;
}

/** The user parts of a Gemini upstream body: the text, plus the image as inline data. */
function geminiUserParts(canon: CanonicalRequest): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [{ text: canon.userText }];
  if (canon.image !== null) {
    if (canon.image.url !== null) parts.push({ fileData: { fileUri: canon.image.url } });
    else parts.push({ inlineData: { mimeType: canon.image.mimeType, data: canon.image.data } });
  }
  return parts;
}

function toolParam(tool: ToolDef): Record<string, unknown> {
  const out: Record<string, unknown> = { name: tool.name };
  if (tool.description !== '') out.description = tool.description;
  if (tool.parameters !== undefined) out.parameters = tool.parameters;
  return out;
}

/** POST one request to an OpenAI-wire provider and read its answer. */
export async function callOpenAI(envUrl: string, provider: ProviderDirectoryEntry, model: string, canon: CanonicalRequest): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const messages: Record<string, unknown>[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  messages.push({ role: 'user', content: openAIUserContent(canon) });
  const body: Record<string, unknown> = { model, messages };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
    body.tool_choice = 'auto';
  }
  if (canon.stops.length > 0) body.stop = [...canon.stops];
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (canon.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  let res: Response;
  try {
    res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: headersFor(provider, canon.stream),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
    });
  } catch (err) {
    return { ok: false, status: 502, error: (err as Error).message };
  }
  return readUpstream(res, 'openai', canon.stream);
}

/** POST one request to an Anthropic-wire provider and read its answer. */
export async function callAnthropic(envUrl: string, provider: ProviderDirectoryEntry, model: string, canon: CanonicalRequest): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const body: Record<string, unknown> = { model, messages: [{ role: 'user', content: anthropicUserContent(canon) }] };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.systemText !== '') body.system = canon.systemText;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((tool) => {
      const out: Record<string, unknown> = { name: tool.name };
      if (tool.description !== '') out.description = tool.description;
      if (tool.parameters !== undefined) out.input_schema = tool.parameters;
      return out;
    });
  }
  if (canon.stops.length > 0) body.stop_sequences = [...canon.stops];
  if (canon.stream) body.stream = true;
  const headers = headersFor(provider, canon.stream);
  headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
    });
  } catch (err) {
    return { ok: false, status: 502, error: (err as Error).message };
  }
  return readUpstream(res, 'anthropic', canon.stream);
}

/** POST one request to a Gemini-wire provider and read its answer. */
export async function callGemini(envUrl: string, provider: ProviderDirectoryEntry, model: string, canon: CanonicalRequest): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const body: Record<string, unknown> = { contents: [{ role: 'user', parts: geminiUserParts(canon) }] };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) body.tools = [{ functionDeclarations: canon.tools.map(toolParam) }];
  const config: Record<string, unknown> = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.stops.length > 0) config.stopSequences = [...canon.stops];
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  body.generationConfig = config;
  const method = canon.stream ? 'streamGenerateContent' : 'generateContent';
  let res: Response;
  try {
    res = await fetch(`${base}/v1beta/models/${encodeURIComponent(model)}:${method}`, {
      method: 'POST',
      headers: headersFor(provider, canon.stream),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
    });
  } catch (err) {
    return { ok: false, status: 502, error: (err as Error).message };
  }
  return readUpstream(res, 'gemini', canon.stream);
}

/** The provider's stop, rendered in the client's wire. The provider only ever stops cleanly. */
export function stopForClient(upstreamStop: string, api: 'openai' | 'anthropic'): string {
  const family: Record<string, number> = {
    stop: 0,
    end_turn: 0,
    STOP: 0,
    length: 1,
    max_tokens: 1,
    MAX_TOKENS: 1,
    tool_calls: 2,
    function_call: 2,
    tool_use: 2,
    content_filter: 3,
    refusal: 3,
    SAFETY: 3,
  };
  const f = family[upstreamStop] ?? 0;
  if (api === 'openai') return f === 0 ? 'stop' : f === 1 ? 'length' : f === 2 ? 'tool_calls' : 'content_filter';
  return f === 0 ? 'end_turn' : f === 1 ? 'max_tokens' : f === 2 ? 'tool_use' : 'refusal';
}

/** Split an answer into the 13-character pieces the providers stream. */
function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

const CREATED = 1767225600;

/** An OpenAI JSON completion carrying the provider's text, usage and stop. */
export function openAIJsonResponse(model: string, answer: UpstreamAnswer): string {
  return JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stopForClient(answer.stop, 'openai'), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  });
}

/** An OpenAI event stream: role chunk, content pieces, finish chunk, usage chunk, [DONE]. */
export function openAIStreamResponse(model: string, answer: UpstreamAnswer): string {
  const stop = stopForClient(answer.stop, 'openai');
  const chunk = (delta: Record<string, unknown>, finish: string | null): Record<string, unknown> => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`];
  for (const piece of splitPieces(answer.text)) frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  frames.push(`data: ${JSON.stringify(chunk({}, stop))}\n\n`);
  frames.push(
    `data: ${JSON.stringify({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
    })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic JSON message carrying the provider's text, usage and stop. */
export function anthropicJsonResponse(model: string, answer: UpstreamAnswer): string {
  return JSON.stringify({
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopForClient(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  });
}

/** An Anthropic event stream in order: start, block start, deltas, block stop, delta, stop. */
export function anthropicStreamResponse(model: string, answer: UpstreamAnswer): string {
  const stop = stopForClient(answer.stop, 'anthropic');
  const frame = (event: string, data: Record<string, unknown>): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
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
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(frame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    frame('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } }),
  );
  frames.push(frame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
