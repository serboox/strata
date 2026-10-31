export interface DoorConfig {
  enabled: boolean;
  refusalsBeforeBlock: number;
  blockMinutes: number;
  block: boolean;
}

// By default a refusal only gets advice appended to its result; with `block` the calls to a site that refused
// `refusalsBeforeBlock` times are stopped as well
export const DEFAULT_DOOR: DoorConfig = { enabled: true, refusalsBeforeBlock: 2, blockMinutes: 30, block: false };

// Ways round a site that is closed to programs, each one checked to work from the sandbox (a duckduckgo.com search is not: it
// asks for a human). The place that refused is never offered back to the model.
function otherWays(host: string): string {
  const ways = [
    ...(host === "bing.com" ? [] : ["a web search (web_fetch https://www.bing.com/search?q=WORDS)"]),
    "the maker's own site or another shop",
    "a price-comparison site",
    "a public API or feed of the data (JSON, RSS, sitemap.xml)",
    ...(host === "web.archive.org" || host === "archive.org" ? [] : ["the web archive (web_fetch https://web.archive.org/web/2024/ADDRESS)"]),
  ];
  const last = ways.pop();
  return `Get the same information from another place: ${ways.join(", ")}, or ${last}. If nothing works, tell the user what you tried and ask for the page text.`;
}

// What to say to the model after a refusal: that knocking again is pointless, and where else to look
export function adviceFor(host: string, reason: string, refusals: number): string {
  const search = host === "bing.com" ? "" : " with a web search (web_fetch https://www.bing.com/search?q=WORDS)";
  if (reason === "the name does not resolve") {
    return `[strata: ${host} does not exist: its name does not resolve. Guessing other spellings of it rarely works. Find the real address${search}, or ask the user for it.]`;
  }
  if (reason === "refused by the web proxy") {
    // the proxy answers the same for a name that does not exist and for an address that is not public, and a site that drops the
    // connection looks the same to the tool, so all three are named
    return `[strata: the connection to ${host} failed: the web proxy refused it (the name does not exist, it is not an address on the public internet, or its port is not allowed) or the site closed the connection. Another spelling will not change that: find the real address${search}, or ask the user.]`;
  }
  if (refusals <= 1) {
    return `[strata: ${host} turns programs away (${reason}). Every page of it will do the same, so retrying or fetching other pages of it will not help. ${otherWays(host)}]`;
  }
  return `[strata: ${host} has turned you away ${refusals} times (${reason}) and will keep doing so. Do not knock again; take another way. ${otherWays(host)}]`;
}

const URL_HOST = /\bhttps?:\/\/(?:[^\s/@'"`]*@)?(\[[0-9a-fA-F:.]+\]|[\p{L}\p{N}._-]+)/giu;
const MINUTE_MS = 60_000;
const SCAN_CHARS = 4000;

// The address of a site as one spelling: lower case, IDN in punycode, no trailing dot, no leading www.
export function normalizeHost(host: string): string {
  let name = host.toLowerCase();
  try {
    name = new URL(`http://${host}`).hostname;
  } catch {
    // an odd name keeps its lower-case spelling
  }
  return name.replace(/\.$/, "").replace(/^www\./, "");
}

export function hostsIn(text: string): string[] {
  const hosts = new Set<string>();
  for (const match of text.matchAll(URL_HOST)) hosts.add(normalizeHost(match[1] as string));
  return [...hosts];
}

// The status a reply is about: web_fetch names it on its first line; in curl output it is the first status line that is not the
// proxy's "Connection established" and not a redirect. What the page says after that cannot change it.
const STATUS_LINE = /^(?:< )?HTTP\/[\d.]+ (\d{3})\b(.*)$/gm;
function decisiveStatus(head: string): number | undefined {
  const fetched = /^\S+ \(HTTP (\d{3})[,)]/.exec(head.split("\n", 1)[0] ?? "");
  if (fetched !== null) return Number(fetched[1]);
  for (const match of head.matchAll(STATUS_LINE)) {
    const code = Number(match[1]);
    if (/connection established/i.test(match[2] ?? "") || (code >= 300 && code < 400)) continue;
    return code;
  }
  return undefined;
}

// What marks a refusal when the reply carries no status line (a curl without -i, a failure of the tool itself). Each pattern is
// tied to the start of a line or to the page title, so an article that merely talks about 403 or a captcha does not match;
// 401 (a login request) and 429 (a pause) are not doors.
const REFUSALS: { pattern: RegExp; reason: string }[] = [
  { pattern: /^(?:status|code|http_code)[ :=]+403\s*$/im, reason: "HTTP 403" },
  { pattern: /^\s*403\s*$/, reason: "HTTP 403" },
  { pattern: /^curl: \(56\) CONNECT tunnel failed, response 403/m, reason: "refused by the web proxy" },
  { pattern: /^web_fetch: .*web proxy refused/m, reason: "refused by the web proxy" },
  { pattern: /^curl: \(6\) Could not resolve host/m, reason: "the name does not resolve" },
  { pattern: /^web_fetch: the host name does not resolve/m, reason: "the name does not resolve" },
  { pattern: /<title>\s*(?:Just a moment\.\.\.|Access Denied|Attention Required! \| Cloudflare)\s*<\/title>/i, reason: "a browser check or an access-denied page" },
  { pattern: /Enable JavaScript and cookies to continue/i, reason: "a browser check or an access-denied page" },
];

// What a tool result says about the door: a short reason when the site (or the proxy in front of it) turned the request away
export function refusalReason(text: string): string | undefined {
  const head = text.slice(0, SCAN_CHARS);
  const status = decisiveStatus(head);
  if (status === 403) return "HTTP 403";
  if (status !== undefined) return undefined;
  for (const { pattern, reason } of REFUSALS) if (pattern.test(head)) return reason;
  return undefined;
}

const FETCHER = /\b(?:curl|wget)\b/;

// A call that goes to the web: web_fetch, or a shell command that runs curl or wget. Other commands may mention an address
// without fetching it (echo, git clone) and are left alone.
export function webCall(toolName: string, input: unknown): { hosts: string[]; text: string } | undefined {
  const fields = input as { url?: unknown; command?: unknown };
  if (toolName === "web_fetch" && typeof fields.url === "string") return { hosts: hostsIn(fields.url), text: fields.url };
  if ((toolName === "bash" || toolName === "powershell") && typeof fields.command === "string" && FETCHER.test(fields.command)) {
    return { hosts: hostsIn(fields.command), text: fields.command };
  }
  return undefined;
}

// The sites a result speaks for: web_fetch names the final address, after redirects, on its first line; a command with one
// address speaks for it; with several, which of them answered is unknown.
export function sitesOfResult(toolName: string, result: string, call: { hosts: string[] }): string[] {
  if (toolName === "web_fetch") {
    const named = hostsIn(result.split("\n", 1)[0] ?? "").slice(0, 1);
    if (named.length > 0) return named;
  }
  return call.hosts.length === 1 ? call.hosts : [];
}

const escapeForRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

interface Door {
  refusals: number;
  reason: string;
  last: number;
}

export class DoorGuard {
  private readonly doors = new Map<string, Door>();
  private readonly config: DoorConfig;
  private readonly now: () => number;

  constructor(config: DoorConfig = DEFAULT_DOOR, now: () => number = Date.now) {
    this.config = config;
    this.now = now;
  }

  private live(host: string): Door | undefined {
    const door = this.doors.get(host);
    if (door === undefined) return undefined;
    if (this.now() - door.last > this.config.blockMinutes * MINUTE_MS) {
      this.doors.delete(host);
      return undefined;
    }
    return door;
  }

  // Books a refusal and returns how many this site has given lately
  refused(host: string, reason: string): number {
    const refusals = (this.live(host)?.refusals ?? 0) + 1;
    this.doors.set(host, { refusals, reason, last: this.now() });
    return refusals;
  }

  opened(host: string): void {
    this.doors.delete(host);
  }

  // For the block mode: the refusal to give the model when a call goes to a door that stayed shut, or undefined. A call
  // goes to a shut site when its URLs name it, or, when no URL names a site, when its text does: `host=shop.test; curl
  // "https://$host/"` hides the URL from the parser but not the name.
  check(hosts: string[], text = ""): string | undefined {
    if (this.config.refusalsBeforeBlock <= 0) return undefined;
    for (const [host, door] of [...this.doors]) {
      if (this.live(host) === undefined || door.refusals < this.config.refusalsBeforeBlock) continue;
      const named = hosts.includes(host) || (hosts.length === 0 && new RegExp(`(?<![\\w-])${escapeForRegExp(host)}(?![\\w-]|\\.[\\w])`, "i").test(text));
      if (!named) continue;
      return `Stop: ${host} turned this session away ${door.refusals} times (${door.reason}); another try gets the same answer. ${otherWays(host)} It may open again after ${this.config.blockMinutes} minutes.`;
    }
    return undefined;
  }

  reset(): void {
    this.doors.clear();
  }
}
