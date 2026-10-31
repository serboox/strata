import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { DEFAULT_CHARS, MAX_CHARS, decodeEntities, describeFetchError, fetchPage, htmlToText, pageSlice, pageText, parseWebUrl } from "../src/web.ts";

const PAGE = `<!DOCTYPE html><html><head><title> Washing &amp; drying </title><style>.x{color:red}</style>
<script>var secret = "do not show";</script></head><body>
<!-- a comment --><nav><a href="/">Home</a></nav>
<h1>Washers</h1><p>Best for a family&nbsp;of four.<br>Price: 499&nbsp;USD</p>
<ul><li>Front load</li><li>8 kg &#8211; 1200 rpm</li></ul>
<table><tr><th>Model</th><th>Price</th></tr><tr><td>A1</td><td>$450</td></tr></table>
<noscript>enable scripts</noscript><svg><text>icon</text></svg></body></html>`;

let base = "";
const server = createServer((request: IncomingMessage, response: ServerResponse) => {
  const url = request.url ?? "/";
  if (url === "/page") return void response.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(PAGE);
  if (url === "/json") return void response.writeHead(200, { "content-type": "application/json" }).end('{"price":499}');
  if (url === "/redirect") return void response.writeHead(302, { location: "/page" }).end();
  if (url === "/cyrillic") {
    const bytes = Buffer.from([0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2]);
    return void response.writeHead(200, { "content-type": "text/plain; charset=windows-1251" }).end(bytes);
  }
  if (url === "/big") return void response.writeHead(200, { "content-type": "text/plain" }).end("x".repeat(50_000));
  if (url === "/image") return void response.writeHead(200, { "content-type": "image/png" }).end(Buffer.from([1, 2, 3]));
  if (url === "/challenge") return void response.writeHead(403, { "content-type": "text/html" }).end("<html><head><title>Just a moment...</title></head><body>Checking</body></html>");
  if (url === "/mitigated") return void response.writeHead(403, { "cf-mitigated": "challenge", "content-type": "text/html" }).end("<html>x</html>");
  if (url === "/phrase403") return void response.writeHead(403, { "content-type": "text/html" }).end("<html><body>Just a moment... The requested account is forbidden.</body></html>");
  if (url === "/script403") return void response.writeHead(403, { "content-type": "text/html" }).end('<html><head><title>Checking</title></head><body><script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></script></body></html>');
  if (url === "/plain403") return void response.writeHead(403, { "content-type": "text/html" }).end("<html><body>Forbidden for you</body></html>");
  if (url === "/slow") return void setTimeout(() => response.end("late"), 2000);
  if (url === "/echo") return void response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ agent: request.headers["user-agent"] }));
  response.writeHead(404, { "content-type": "text/plain" }).end("not here");
});

before(async () => {
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => {
  server.closeAllConnections();
  server.close();
});

test("htmlToText keeps the readable text and drops scripts, styles and comments", () => {
  const text = htmlToText(PAGE);
  assert.match(text, /^Title: Washing & drying\n\n/);
  assert.match(text, /Washers\nBest for a family of four\.\nPrice: 499 USD/);
  assert.match(text, /\n- Front load\n- 8 kg – 1200 rpm/);
  assert.match(text, /Model \| Price\nA1 \| \$450/);
  for (const hidden of ["secret", "color:red", "a comment", "enable scripts", "icon", "<", "Home"]) assert.ok(!text.includes(hidden), hidden);
});

test("htmlToText prefers the main content when it holds real text", () => {
  const article = "The washing machine article body. ".repeat(30);
  const html = `<html><head><title>T</title></head><body><div>Jump to content Main menu Donate Log in</div><main><h1>Washing machine</h1><p>${article}</p></main><div>Edit links Privacy</div></body></html>`;
  const text = htmlToText(html);
  assert.match(text, /^Title: T\n\nWashing machine\nThe washing machine article body\./);
  for (const noise of ["Jump to content", "Donate", "Privacy"]) assert.ok(!text.includes(noise), noise);
});

test("htmlToText keeps the whole body when the main element is nearly empty", () => {
  const text = htmlToText("<body><div>Price list</div><main>menu</main><div>Footer note</div></body>");
  assert.match(text, /Price list\nmenu\nFooter note/);
});

test("htmlToText drops navigation, footer and aside blocks", () => {
  assert.equal(htmlToText("<nav>Menu one</nav><p>Content</p><aside>Related</aside><footer>Legal</footer>"), "Content");
});

test("htmlToText joins a long run of short list items into one line and keeps short lists as they are", () => {
  const items = ["Afrikaans", "Deutsch", "Español", "Français", "Italiano", "日本語", "Polski", "Русский"];
  const list = (names: string[]) => `<ul>${names.map((name) => `<li>${name}</li>`).join("")}</ul>`;
  assert.equal(htmlToText(`<p>Languages</p>${list(items)}<p>Article text</p>`), `Languages\n- ${items.join(", ")}\nArticle text`);
  assert.equal(htmlToText(list(items.slice(0, 3))), "- Afrikaans\n- Deutsch\n- Español");
  const long = "x".repeat(40);
  assert.equal(htmlToText(list(items.map(() => long))), items.map(() => `- ${long}`).join("\n"));
});

test("htmlToText leaves no blank lines, which only cost tokens", () => {
  assert.equal(htmlToText("<p>a</p>\n\n\n<div></div><div></div><p>b</p>"), "a\nb");
  assert.equal(htmlToText(""), "");
});

test("htmlToText handles case, declarations, self-closing tags and a lone angle bracket", () => {
  assert.equal(htmlToText("<!DOCTYPE html><HTML><BODY><DIV>One<BR/>Two</DIV><P>a < b</P><script src=x /></BODY></HTML>"), "One\nTwo\na < b");
});

test("htmlToText treats an unclosed comment as running to the end and an unfinished tag as the end", () => {
  assert.equal(htmlToText("<p>kept</p><!-- never closed <p>lost</p>"), "kept");
  assert.equal(htmlToText("<p>kept</p><p lost forever"), "kept");
});

test("a closing tag must end its name: </scripture> does not close a script", () => {
  assert.equal(htmlToText('<script>const x="</scripture>LEAK";</script><p>Visible</p>'), "Visible");
  assert.equal(htmlToText("<main><p>" + "words ".repeat(120) + "</mainframe>tail</p></main><p>outside</p>").includes("tail"), true);
  assert.equal(htmlToText("<head><title>A</titles>B</title></head><p>x</p>"), "Title: A</titles>B\n\nx");
});

test("htmlToText keeps the text of an unclosed script-like element", () => {
  assert.equal(htmlToText("<p>a</p><nav>menu text</p><p>b</p>"), "a\nmenu text\nb");
});

test("htmlToText takes linear time on hostile pages of 600 KB", () => {
  const cases: Record<string, string> = {
    "unclosed tags": "<p ".repeat(200_000),
    "unclosed comments": "<!--".repeat(150_000),
    "unclosed scripts": "<script>".repeat(75_000),
    "unclosed titles": "<title>".repeat(85_000),
    "main lookalikes": "<mainframe ".repeat(55_000),
    "pipes": " | ".repeat(200_000) + "x",
    "entities": "&".repeat(600_000),
    "angle brackets": "<".repeat(600_000),
    "nested lists": "<ul><li>".repeat(60_000),
    "table cells": "<td>x</td>".repeat(60_000),
  };
  for (const [name, html] of Object.entries(cases)) {
    const started = performance.now();
    htmlToText(html);
    assert.ok(performance.now() - started < 1500, `${name} took ${Math.round(performance.now() - started)} ms`);
  }
});

test("decodeEntities handles named, decimal and hex references and leaves the rest", () => {
  assert.equal(decodeEntities("a &amp; b &lt;c&gt; &#65;&#x42; &unknown; &#0; &euro;"), "a & b <c> AB &unknown; &#0; €");
});

test("pageSlice returns a part and names the offset of the next one", () => {
  const text = "a".repeat(12000);
  const first = pageSlice(text, 0, 5000);
  assert.ok(first.startsWith("a".repeat(5000)));
  assert.match(first, /\[characters 0-5000 of 12000; call web_fetch again with offset=5000 for the next part\]$/);
  assert.match(pageSlice(text, 10000, 5000), /^a{2000}$/);
  assert.equal(pageSlice(text, 12000, 5000), "[end of the page: it has 12000 characters]");
});

test("pageSlice clamps its arguments", () => {
  const text = "b".repeat(20000);
  assert.ok(pageSlice(text, -5, 999999).length <= MAX_CHARS + 200);
  assert.ok(pageSlice(text, Number.NaN, Number.NaN).startsWith("b".repeat(DEFAULT_CHARS)));
  assert.equal(pageSlice("", 0, 100), "[the page has no readable text]");
});

test("parseWebUrl accepts http and https only, without credentials", () => {
  assert.equal(parseWebUrl(" https://example.com/a?b=1 ").hostname, "example.com");
  for (const bad of ["file:///etc/passwd", "ftp://example.com", "javascript:alert(1)", "example.com", "", "https://user:pw@example.com/", `https://example.com/${"a".repeat(3000)}`]) {
    assert.throws(() => parseWebUrl(bad), Error, bad.slice(0, 40));
  }
});

test("fetchPage returns a page as text, through redirects, with the final address", async () => {
  const page = await fetchPage(`${base}/redirect`);
  assert.equal(page.url, `${base}/page`);
  assert.equal(page.status, 200);
  assert.match(pageText(page), /^Title: Washing & drying/);
});

test("fetchPage gives JSON as it is and honest error statuses", async () => {
  assert.equal(pageText(await fetchPage(`${base}/json`)), '{"price":499}');
  const missing = await fetchPage(`${base}/nowhere`);
  assert.equal(missing.status, 404);
  assert.equal(pageText(missing), "not here");
});

test("fetchPage decodes the charset the server names", async () => {
  assert.equal(pageText(await fetchPage(`${base}/cyrillic`)), "Привет");
});

test("fetchPage cuts a body at the byte limit and says so", async () => {
  const page = await fetchPage(`${base}/big`, { maxBytes: 10_000 });
  assert.equal(page.body.length, 10_000);
  assert.equal(page.cut, true);
  assert.match(pageText(page), /\n\[the page was longer than the limit and is cut\]$/);
});

test("a body that is not text is named, not dumped", async () => {
  assert.match(pageText(await fetchPage(`${base}/image`)), /^\[not text: image\/png, \d+ bytes read\]$/);
});

test("a Cloudflare browser check is recognised and reported honestly", async () => {
  for (const path of ["/challenge", "/mitigated"]) {
    const page = await fetchPage(`${base}${path}`);
    assert.equal(page.blocked, true, path);
    assert.match(pageText(page), /^This site answers programs with a browser check \(Cloudflare, HTTP 403\).*Do not guess/s);
  }
  const plain = await fetchPage(`${base}/plain403`);
  assert.equal(plain.blocked, false);
  assert.match(pageText(plain), /Forbidden for you/);
  const phrase = await fetchPage(`${base}/phrase403`);
  assert.equal(phrase.blocked, false);
  assert.match(pageText(phrase), /The requested account is forbidden/);
  const script = await fetchPage(`${base}/script403`);
  assert.equal(script.blocked, true);
});

test("fetchPage sends a plain identifying user agent and stops at the timeout", async () => {
  assert.match(pageText(await fetchPage(`${base}/echo`)), /pi-strata-web-fetch/);
  const started = Date.now();
  await assert.rejects(fetchPage(`${base}/slow`, { timeoutMs: 200 }), (error: unknown) => describeFetchError(error) === "the request timed out or was cancelled");
  assert.ok(Date.now() - started < 1500);
});

test("fetchPage stops when the caller aborts", async () => {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 100);
  await assert.rejects(fetchPage(`${base}/slow`, { signal: controller.signal }), (error: unknown) => describeFetchError(error) === "the request timed out or was cancelled");
});

test("describeFetchError explains the cases an agent meets", () => {
  const withCause = (message: string, cause: string) => new TypeError(message, { cause: new Error(cause) });
  assert.match(describeFetchError(withCause("fetch failed", "Proxy response (403) !== 200 when HTTP Tunneling")), /web proxy refused this address/);
  assert.match(describeFetchError(withCause("fetch failed", "Request was cancelled.")), /web proxy refused this address.*or the site closed the connection/);
  assert.equal(describeFetchError(withCause("fetch failed", "getaddrinfo ENOTFOUND nowhere.test")), "the host name does not resolve");
  assert.match(describeFetchError(withCause("fetch failed", "connect ECONNREFUSED 127.0.0.1:9")), /refused or closed/);
  assert.equal(describeFetchError(new Error("odd")), "odd");
  assert.equal(describeFetchError("plain"), "plain");
});
