export const DEFAULT_CHARS = 5000;
export const MAX_CHARS = 5500;
const MAX_BYTES = 2_000_000;
const TIMEOUT_MS = 30_000;
const MAX_URL_CHARS = 2048;
const USER_AGENT = "Mozilla/5.0 (compatible; pi-strata-web-fetch)";
const BLOCKED_NOTICE =
  "This site answers programs with a browser check (Cloudflare, HTTP {status}), so its page cannot be read here. Do not guess what it contains: say so to the user, or look for the same information on another site.";

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "-", mdash: "-", hellip: "...", laquo: "«", raquo: "»", euro: "€", pound: "£", copy: "(c)" };
const MAIN_MIN_CHARS = 500;
const SHORT_ITEM_CHARS = 32;
const MIN_ITEM_RUN = 8;
const DROPPED_ELEMENTS = new Set(["script", "style", "noscript", "svg", "template", "head", "nav", "footer", "aside"]);
const LINE_BREAK_ELEMENTS = new Set(["p", "div", "section", "article", "header", "footer", "main", "nav", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "form", "figure", "dl", "dt", "dd", "br", "hr"]);
const CHALLENGE_TITLE = /^(just a moment\.\.\.|attention required! \| cloudflare)$/i;
const CHALLENGE_SCRIPT = /\/cdn-cgi\/challenge-platform\//i;

export interface FetchOptions {
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
  maxBytes?: number;
  timeoutMs?: number;
}

export interface Page {
  url: string;
  status: number;
  contentType: string;
  body: string;
  cut: boolean;
  blocked: boolean;
}

export function decodeEntities(text: string): string {
  return text.replace(/&(?:#(\d{1,7})|#x([0-9a-f]{1,6})|([a-z]{2,8}));/gi, (whole, decimal, hex, name) => {
    if (name !== undefined) return ENTITIES[name.toLowerCase()] ?? whole;
    const code = decimal !== undefined ? Number(decimal) : Number.parseInt(hex, 16);
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
  });
}

// Every step below moves forward through the page, so a hostile page of many unclosed tags or comments costs no more than a
// normal one; regular expressions over `<...>` take quadratic time on such input.
const asciiLower = (value: string): string => value.replace(/[A-Z]/g, (letter) => String.fromCharCode(letter.charCodeAt(0) + 32));
const isNameCharacter = (code: number): boolean => (code >= 97 && code <= 122) || (code >= 48 && code <= 57) || code === 45 || code === 58;

function findElementStart(lower: string, tag: string): number {
  let from = lower.indexOf(`<${tag}`);
  while (from !== -1 && isNameCharacter(lower.charCodeAt(from + tag.length + 1))) from = lower.indexOf(`<${tag}`, from + 1);
  return from;
}

// `</script` also begins `</scripture`: the name must end there
function findClosingTag(lower: string, tag: string, from: number): number {
  let at = lower.indexOf(`</${tag}`, from);
  while (at !== -1 && isNameCharacter(lower.charCodeAt(at + tag.length + 2))) at = lower.indexOf(`</${tag}`, at + 1);
  return at;
}

function findLastClosingTag(lower: string, tag: string): number {
  let at = lower.lastIndexOf(`</${tag}`);
  while (at > 0 && isNameCharacter(lower.charCodeAt(at + tag.length + 2))) at = lower.lastIndexOf(`</${tag}`, at - 1);
  return at !== -1 && isNameCharacter(lower.charCodeAt(at + tag.length + 2)) ? -1 : at;
}

function titleOf(html: string, lower: string): string {
  const from = findElementStart(lower, "title");
  const open = from === -1 ? -1 : html.indexOf(">", from);
  const close = open === -1 ? -1 : findClosingTag(lower, "title", open);
  return close === -1 ? "" : decodeEntities(html.slice(open + 1, close)).replace(/\s+/g, " ").trim();
}

function elementBounds(html: string, lower: string, tag: string): [number, number] | undefined {
  const from = findElementStart(lower, tag);
  const open = from === -1 ? -1 : html.indexOf(">", from);
  const end = findLastClosingTag(lower, tag);
  return open === -1 || end <= open ? undefined : [open + 1, end];
}

function scanText(html: string, lower: string, from: number, to: number): string {
  const out: string[] = [];
  const unclosed = new Set<string>();
  let position = from;
  while (position < to) {
    const open = html.indexOf("<", position);
    if (open === -1 || open >= to) {
      out.push(html.slice(position, to));
      break;
    }
    out.push(html.slice(position, open));
    if (lower.startsWith("<!--", open)) {
      const end = lower.indexOf("-->", open + 4);
      position = end === -1 ? to : end + 3;
      continue;
    }
    const next = html[open + 1];
    if (next === "!" || next === "?") {
      const end = html.indexOf(">", open + 2);
      if (end === -1) break;
      position = end + 1;
      continue;
    }
    const closing = next === "/";
    const nameStart = open + (closing ? 2 : 1);
    let nameEnd = nameStart;
    while (nameEnd < to && isNameCharacter(lower.charCodeAt(nameEnd))) nameEnd++;
    if (nameEnd === nameStart) {
      out.push("<");
      position = open + 1;
      continue;
    }
    const name = lower.slice(nameStart, nameEnd);
    const tagEnd = html.indexOf(">", nameEnd);
    if (tagEnd === -1) break;
    position = tagEnd + 1;
    if (!closing && DROPPED_ELEMENTS.has(name) && html[tagEnd - 1] !== "/") {
      if (!unclosed.has(name)) {
        const close = findClosingTag(lower, name, position);
        if (close === -1) unclosed.add(name);
        else position = html.indexOf(">", close) === -1 ? to : html.indexOf(">", close) + 1;
      }
    } else if (closing) {
      if (name === "td" || name === "th") out.push(" | ");
      else if (LINE_BREAK_ELEMENTS.has(name)) out.push("\n");
    } else if (name === "li") {
      out.push("\n- ");
    } else if (LINE_BREAK_ELEMENTS.has(name)) {
      out.push("\n");
    }
  }
  return out.join("");
}

function stripTrailingPipes(line: string): string {
  let end = line.length;
  for (;;) {
    while (end > 0 && line[end - 1] === " ") end--;
    if (end > 0 && line[end - 1] === "|") end--;
    else break;
  }
  return line.slice(0, end);
}

// a run of short list items (a language list, a menu) costs a line each: it is joined into one line
function joinItemRuns(lines: string[]): string[] {
  const out: string[] = [];
  let run: string[] = [];
  const flush = () => {
    if (run.length >= MIN_ITEM_RUN) out.push(`- ${run.map((item) => item.slice(2)).join(", ")}`);
    else out.push(...run);
    run = [];
  };
  for (const line of lines) {
    if (line.length <= SHORT_ITEM_CHARS && line.startsWith("- ")) {
      run.push(line);
    } else {
      flush();
      out.push(line);
    }
  }
  flush();
  return out;
}

function tidy(raw: string): string {
  const lines = decodeEntities(raw)
    .split("\n")
    .map((line) => stripTrailingPipes(line.replace(/[ \t\u00a0]+/g, " ")).trim())
    .filter((line) => line !== "");
  return joinItemRuns(lines).join("\n");
}

export function htmlToText(html: string): string {
  const lower = asciiLower(html);
  let body: string | undefined;
  for (const tag of ["main", "article"]) {
    const bounds = elementBounds(html, lower, tag);
    const region = bounds === undefined ? "" : tidy(scanText(html, lower, bounds[0], bounds[1]));
    if (region.length >= MAIN_MIN_CHARS) {
      body = region;
      break;
    }
  }
  body ??= tidy(scanText(html, lower, 0, html.length));
  const heading = titleOf(html, lower);
  return heading === "" ? body : `Title: ${heading}\n\n${body}`;
}

export function pageSlice(text: string, offset: number, maxChars: number): string {
  const start = Math.min(Math.max(Math.trunc(offset) || 0, 0), text.length);
  const size = Math.min(Math.max(Math.trunc(maxChars) || DEFAULT_CHARS, 200), MAX_CHARS);
  if (text.length === 0) return "[the page has no readable text]";
  if (start >= text.length) return `[end of the page: it has ${text.length} characters]`;
  const end = Math.min(start + size, text.length);
  const piece = text.slice(start, end);
  return end >= text.length ? piece : `${piece}\n[characters ${start}-${end} of ${text.length}; call web_fetch again with offset=${end} for the next part]`;
}

export function parseWebUrl(raw: string): URL {
  if (raw.length > MAX_URL_CHARS) throw new Error("the URL is too long");
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    throw new Error(`not a valid URL: ${raw.slice(0, 100)}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error(`only http and https URLs can be fetched, not ${url.protocol}`);
  if (url.username !== "" || url.password !== "") throw new Error("a URL with a user name or password is not accepted");
  return url;
}

function decode(bytes: Uint8Array, contentType: string): string {
  const label = /charset\s*=\s*"?([\w.:-]+)/i.exec(contentType)?.[1] ?? "utf-8";
  try {
    return new TextDecoder(label).decode(bytes);
  } catch {
    return new TextDecoder("utf-8").decode(bytes);
  }
}

async function readLimited(response: Response, maxBytes: number): Promise<{ bytes: Uint8Array; cut: boolean }> {
  if (response.body === null) return { bytes: new Uint8Array(), cut: false };
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let cut = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    if (total > maxBytes) {
      cut = true;
      await reader.cancel();
      break;
    }
  }
  const bytes = new Uint8Array(Math.min(total, maxBytes));
  let position = 0;
  for (const chunk of chunks) {
    const room = bytes.length - position;
    if (room <= 0) break;
    bytes.set(chunk.subarray(0, room), position);
    position += Math.min(chunk.length, room);
  }
  return { bytes, cut };
}

export async function fetchPage(rawUrl: string, options: FetchOptions = {}): Promise<Page> {
  const url = parseWebUrl(rawUrl);
  const signals = [AbortSignal.timeout(options.timeoutMs ?? TIMEOUT_MS)];
  if (options.signal !== undefined) signals.push(options.signal);
  const response = await (options.fetchImpl ?? fetch)(url, {
    signal: AbortSignal.any(signals),
    redirect: "follow",
    headers: { "user-agent": USER_AGENT, accept: "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5", "accept-language": "en" },
  });
  const contentType = response.headers.get("content-type") ?? "";
  const { bytes, cut } = await readLimited(response, options.maxBytes ?? MAX_BYTES);
  const body = decode(bytes, contentType);
  const head = body.slice(0, 8000);
  const challenged =
    response.headers.get("cf-mitigated") === "challenge" ||
    ([403, 429, 503].includes(response.status) && (CHALLENGE_TITLE.test(titleOf(head, asciiLower(head))) || CHALLENGE_SCRIPT.test(head)));
  return { url: response.url || url.href, status: response.status, contentType, body, cut, blocked: challenged };
}

const TEXTUAL = /^(text\/|application\/(json|xml|xhtml\+xml|javascript|ld\+json|rss\+xml|atom\+xml)|[\w.+-]+\/[\w.+-]*\+(json|xml))/i;

export function pageText(page: Page): string {
  if (page.blocked) return BLOCKED_NOTICE.replace("{status}", String(page.status));
  const type = page.contentType.split(";")[0]?.trim().toLowerCase() ?? "";
  if (type !== "" && !TEXTUAL.test(type)) return `[not text: ${type}, ${page.body.length} bytes read]`;
  const text = /html/.test(type) || (type === "" && /<html|<!doctype html/i.test(page.body.slice(0, 500))) ? htmlToText(page.body) : page.body.trim();
  const note = page.cut ? "\n[the page was longer than the limit and is cut]" : "";
  return `${text}${note}`;
}

export function describeFetchError(error: unknown): string {
  if (error instanceof DOMException && (error.name === "TimeoutError" || error.name === "AbortError")) return "the request timed out or was cancelled";
  const message = error instanceof Error ? error.message : String(error);
  const cause = error instanceof Error && error.cause instanceof Error ? error.cause.message : "";
  const detail = `${message} ${cause}`;
  // undici shows a refused proxy tunnel as a cancelled request, so the site closing the connection looks the same
  if (/tunnel|proxy response \(403\)|\b403\b.*proxy|proxy.*\b403\b|request was cancelled/i.test(detail)) {
    return "the connection could not be made: the web proxy refused this address (it is not a public internet address, the name does not resolve, or the port is not allowed), or the site closed the connection";
  }
  if (/ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(detail)) return "the host name does not resolve";
  if (/ECONNREFUSED|ECONNRESET|UND_ERR_SOCKET|other side closed/i.test(detail)) return `the connection was refused or closed (${cause || message})`;
  return cause === "" ? message : `${message}: ${cause}`;
}
