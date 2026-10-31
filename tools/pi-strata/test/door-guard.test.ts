import assert from "node:assert/strict";
import { test } from "node:test";
import { DEFAULT_DOOR, DoorGuard, adviceFor, hostsIn, normalizeHost, refusalReason, sitesOfResult, webCall } from "../src/door-guard.ts";

test("hostsIn finds the sites of a command, once each, without www and port", () => {
  assert.deepEqual(hostsIn("curl -sL https://www.beko.com/en/x?y=1 -o /tmp/a"), ["beko.com"]);
  assert.deepEqual(hostsIn("curl https://beko.com/a http://www.beko.com:8080/b https://Example.org./c"), ["beko.com", "example.org"]);
  assert.deepEqual(hostsIn("curl -u user:pw https://user:pw@shop.example/a"), ["shop.example"]);
  assert.deepEqual(hostsIn("curl 'http://[2001:db8::1]:80/x'"), ["[2001:db8::1]"]);
  assert.deepEqual(hostsIn("ls -la && echo done"), []);
  assert.deepEqual(hostsIn("git clone git@github.com:o/r.git"), []);
});

test("one site has one spelling, an IDN included", () => {
  assert.equal(normalizeHost("WWW.Beko.COM."), "beko.com");
  assert.equal(normalizeHost("www2.beko.com"), "www2.beko.com");
  assert.deepEqual(hostsIn("curl https://пример.рф/x"), ["xn--e1afmkfd.xn--p1ai"]);
  assert.deepEqual(hostsIn("curl https://xn--e1afmkfd.xn--p1ai/y"), ["xn--e1afmkfd.xn--p1ai"]);
});

test("refusalReason recognises the lines of a turned-away request", () => {
  const cases: [string, string][] = [
    ["HTTP/2 403 \ncontent-type: text/html\n\n<html>x</html>", "HTTP 403"],
    ["HTTP/1.1 200 Connection established\n\nHTTP/2 403 \nserver: AkamaiGHost", "HTTP 403"],
    ["starting download\nHTTP/2 403\nserver: x\n\n" + "body ".repeat(3000), "HTTP 403"],
    ["* Request completed\n< HTTP/2 403\n< server: x", "HTTP 403"],
    ["status 403", "HTTP 403"],
    ["403", "HTTP 403"],
    ["https://www.beko.com/ (HTTP 403, text/html)\nThis site answers programs with a browser check", "HTTP 403"],
    ["curl: (56) CONNECT tunnel failed, response 403", "refused by the web proxy"],
    ["web_fetch: the connection could not be made: the web proxy refused this address (it is not a public internet address)", "refused by the web proxy"],
    ["curl: (6) Could not resolve host: www.beko.ge", "the name does not resolve"],
    ["web_fetch: the host name does not resolve", "the name does not resolve"],
    ["<html><head><title>Just a moment...</title></head></html>", "a browser check or an access-denied page"],
    ["<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD>", "a browser check or an access-denied page"],
  ];
  for (const [text, reason] of cases) assert.equal(refusalReason(text), reason, text.slice(0, 60));
});

test("the first significant status decides, not what the page says later", () => {
  const body = "\nThe reply of a refused request looks like this:\nHTTP/2 403\nserver: x\n";
  assert.equal(refusalReason(`HTTP/2 200\ncontent-type: text/html\n${body}`), undefined);
  assert.equal(refusalReason(`https://docs.example/http (HTTP 200, text/html)\nTitle: Errors${body}`), undefined);
  assert.equal(refusalReason("HTTP/2 200\n\n<html>Enable JavaScript and cookies to continue</html>"), undefined);
  assert.equal(refusalReason("HTTP/2 200\n\n<title>Access Denied</title> (an article)"), undefined);
  assert.equal(refusalReason("HTTP/1.1 200 Connection established\n\nHTTP/2 403\nserver: x"), "HTTP 403");
  assert.equal(refusalReason("HTTP/2 301\nlocation: /next\n\nHTTP/2 403\nserver: x"), "HTTP 403");
  assert.equal(refusalReason(`HTTP/2 301\nlocation: /next\n\nHTTP/2 200\n${body}`), undefined);
  assert.equal(refusalReason("HTTP/1.1 200 Connection established\n\nHTTP/2 404\n\n<title>Access Denied</title>"), undefined);
  assert.equal(refusalReason("<html><title>Access Denied</title></html>"), "a browser check or an access-denied page");
  assert.equal(refusalReason("Enable JavaScript and cookies to continue"), "a browser check or an access-denied page");
});

test("refusalReason leaves ordinary replies, logins and pauses alone", () => {
  const cases = [
    "",
    "https://example.com/ (HTTP 200, text/html)\nTitle: Example",
    "https://x.test/ (HTTP 404, text/html)\nnot here",
    "HTTP/2 200\nThis article explains HTTP 403 and Access Denied pages and the captcha.",
    "HTTP/2 401\nwww-authenticate: Bearer realm=api",
    "HTTP/2 429\nretry-after: 30",
    "status 200",
    "total 12\ndrwxr-xr-x 2 u u 4096 .",
    "retry in 403 seconds",
    "found 403 files",
    "a list:\n403\n404\n500",
    "Status code 403 means the server refuses. An Access Denied page and a captcha are common. ".repeat(200),
  ];
  for (const text of cases) assert.equal(refusalReason(text), undefined, text.slice(0, 60));
});

test("webCall covers web_fetch and curl or wget, and leaves other commands alone", () => {
  assert.deepEqual(webCall("web_fetch", { url: "https://www.beko.com/a" }), { hosts: ["beko.com"], text: "https://www.beko.com/a" });
  assert.deepEqual(webCall("bash", { command: "curl -sI https://beko.com/ | head" })?.hosts, ["beko.com"]);
  assert.deepEqual(webCall("bash", { command: "wget -q https://beko.com/x" })?.hosts, ["beko.com"]);
  assert.deepEqual(webCall("bash", { command: "host=beko.com; curl \"https://$host/x\"" })?.hosts, []);
  for (const command of ["echo 'https://beko.com/docs'", "git clone https://beko.com/repo.git", "cat <<EOF\nhttps://beko.com\nEOF", "# https://beko.com"]) {
    assert.equal(webCall("bash", { command }), undefined, command);
  }
  assert.equal(webCall("read", { path: "https://beko.com" }), undefined);
  assert.equal(webCall("web_fetch", {}), undefined);
});

test("sitesOfResult blames the site that answered", () => {
  const one = { hosts: ["good.example"] };
  assert.deepEqual(sitesOfResult("web_fetch", "https://bad.example/page (HTTP 403, text/html)\nblocked", one), ["bad.example"]);
  assert.deepEqual(sitesOfResult("web_fetch", "https://a.example/redirect?next=https://b.example/ (HTTP 403, text/html)\nblocked", one), ["a.example"]);
  assert.deepEqual(sitesOfResult("web_fetch", "web_fetch: the host name does not resolve", one), ["good.example"]);
  assert.deepEqual(sitesOfResult("bash", "HTTP/2 403", one), ["good.example"]);
  assert.deepEqual(sitesOfResult("bash", "HTTP/2 403", { hosts: ["bad.example", "good.example"] }), []);
  assert.deepEqual(sitesOfResult("bash", "HTTP/2 403", { hosts: [] }), []);
});

test("by default a refusal only advises: the guard does not block", () => {
  assert.equal(DEFAULT_DOOR.enabled, true);
  assert.equal(DEFAULT_DOOR.block, false);
});

test("a refusal is booked with a running count", () => {
  const guard = new DoorGuard();
  assert.equal(guard.refused("beko.com", "HTTP 403"), 1);
  assert.equal(guard.refused("beko.com", "HTTP 403"), 2);
  assert.equal(guard.refused("shop.test", "HTTP 403"), 1);
  guard.opened("beko.com");
  assert.equal(guard.refused("beko.com", "HTTP 403"), 1);
});

test("the advice says knocking again is pointless and names ways that work", () => {
  const first = adviceFor("beko.com", "HTTP 403", 1);
  assert.match(first, /^\[strata: beko\.com turns programs away \(HTTP 403\)\. Every page of it will do the same/);
  assert.match(first, /web_fetch https:\/\/www\.bing\.com\/search\?q=WORDS/);
  assert.match(first, /web_fetch https:\/\/web\.archive\.org\/web\/2024\/ADDRESS/);
  assert.match(first, /another shop/);
  assert.match(first, /ask for the page text\.\]$/);
  assert.ok(!first.includes("duckduckgo"));
  const again = adviceFor("beko.com", "a browser check or an access-denied page", 3);
  assert.match(again, /^\[strata: beko\.com has turned you away 3 times \(a browser check or an access-denied page\) and will keep doing so\. Do not knock again; take another way\./);
  assert.match(again, /web\.archive\.org/);
});

test("the advice differs for a name that does not exist and for an address the proxy refused", () => {
  const missing = adviceFor("beko.ge", "the name does not resolve", 1);
  assert.match(missing, /^\[strata: beko\.ge does not exist: its name does not resolve\. Guessing other spellings of it rarely works\./);
  assert.match(missing, /web_fetch https:\/\/www\.bing\.com/);
  assert.ok(!missing.includes("archive"));
  const refused = adviceFor("10.0.0.1", "refused by the web proxy", 2);
  assert.match(refused, /^\[strata: the connection to 10\.0\.0\.1 failed: the web proxy refused it \(the name does not exist, it is not an address on the public internet, or its port is not allowed\) or the site closed the connection\./);
  assert.match(refused, /find the real address with a web search \(web_fetch https:\/\/www\.bing\.com/);
});

test("the advice does not send the model back to the place that refused", () => {
  assert.doesNotMatch(adviceFor("bing.com", "HTTP 403", 1), /bing\.com\/search/);
  assert.match(adviceFor("bing.com", "HTTP 403", 1), /web\.archive\.org/);
  assert.doesNotMatch(adviceFor("web.archive.org", "HTTP 403", 2), /web\.archive\.org\/web/);
  assert.match(adviceFor("web.archive.org", "HTTP 403", 2), /bing\.com\/search/);
  assert.doesNotMatch(adviceFor("bing.com", "the name does not resolve", 1), /bing\.com\/search/);
  assert.doesNotMatch(adviceFor("bing.com", "refused by the web proxy", 1), /bing\.com\/search/);
  assert.match(adviceFor("beko.com", "HTTP 403", 1), /a web search \(web_fetch https:\/\/www\.bing\.com\/search\?q=WORDS\), the maker's own site or another shop, a price-comparison site, a public API or feed of the data \(JSON, RSS, sitemap\.xml\), or the web archive/);
});

test("the advice never suggests getting round a block", () => {
  for (const reason of ["HTTP 403", "a browser check or an access-denied page", "the name does not resolve", "refused by the web proxy"]) {
    for (const count of [1, 2, 5]) assert.doesNotMatch(adviceFor("beko.com", reason, count), /get around|bypass|circumvent|spoof/i, `${reason} ${count}`);
  }
});

test("a door shuts after the configured number of refusals and says why, for the block mode", () => {
  const guard = new DoorGuard();
  guard.refused("beko.com", "HTTP 403");
  assert.equal(guard.check(["beko.com"]), undefined);
  guard.refused("beko.com", "HTTP 403");
  const message = guard.check(["beko.com", "other.test"]);
  assert.match(message ?? "", /^Stop: beko\.com turned this session away 2 times \(HTTP 403\); another try gets the same answer\./);
  assert.match(message ?? "", /web_fetch https:\/\/www\.bing\.com\/search\?q=WORDS/);
  assert.match(message ?? "", /web\.archive\.org/);
  assert.match(message ?? "", /after 30 minutes/);
  assert.doesNotMatch(message ?? "", /get around|bypass|circumvent/i);
});

test("a shut site is recognised by its name in the command text, not only by its URLs", () => {
  const guard = new DoorGuard();
  guard.refused("bad.example", "HTTP 403");
  guard.refused("bad.example", "HTTP 403");
  assert.match(guard.check([], 'host=bad.example; curl "https://$host/page"') ?? "", /^Stop: bad\.example/);
  assert.match(guard.check(hostsIn("curl -s https://www.bad.example/x"), "curl -s https://www.bad.example/x") ?? "", /^Stop: bad\.example/);
  assert.equal(guard.check(hostsIn("curl https://shop.bad.example/x"), "curl https://shop.bad.example/x"), undefined, "a subdomain is a door of its own");
  assert.match(guard.check([], "curl https://BAD.EXAMPLE/x") ?? "", /^Stop: bad\.example/);
  assert.equal(guard.check(["good.example"], "curl https://good.example/?next=bad.example"), undefined);
  assert.equal(guard.check(["good.example"], "echo bad.example; curl https://good.example/"), undefined);
  for (const text of ["curl https://notbad.example/x", "curl https://bad.example.org/x", "curl https://bad.examples.test/", "curl https://good.example/bad-example"]) {
    assert.equal(guard.check([], text), undefined, text);
  }
});

test("doors are separate per site, and an open door resets the count", () => {
  const guard = new DoorGuard();
  guard.refused("beko.com", "HTTP 403");
  guard.refused("shop.test", "HTTP 403");
  assert.equal(guard.check(["beko.com"]), undefined);
  assert.equal(guard.check(["shop.test"]), undefined);
  guard.opened("beko.com");
  guard.refused("beko.com", "HTTP 403");
  assert.equal(guard.check(["beko.com"]), undefined);
});

test("a shut door opens again after the block time, counted from the last refusal", () => {
  let now = 0;
  const guard = new DoorGuard({ ...DEFAULT_DOOR, blockMinutes: 10 }, () => now);
  guard.refused("beko.com", "HTTP 403");
  guard.refused("beko.com", "HTTP 403");
  assert.notEqual(guard.check(["beko.com"]), undefined);
  now = 9 * 60_000;
  guard.refused("beko.com", "HTTP 403");
  now = 18 * 60_000;
  assert.match(guard.check(["beko.com"]) ?? "", /3 times/);
  now = 29 * 60_000;
  assert.equal(guard.check(["beko.com"]), undefined);
  guard.refused("beko.com", "HTTP 403");
  assert.equal(guard.check(["beko.com"]), undefined);
});

test("zero refusals before the block means the block mode never stops a call", () => {
  const guard = new DoorGuard({ ...DEFAULT_DOOR, refusalsBeforeBlock: 0 });
  for (let attempt = 0; attempt < 5; attempt++) guard.refused("beko.com", "HTTP 403");
  assert.equal(guard.check(["beko.com"], "curl https://beko.com/"), undefined);
});

test("reset forgets every door", () => {
  const guard = new DoorGuard();
  guard.refused("beko.com", "HTTP 403");
  guard.refused("beko.com", "HTTP 403");
  guard.reset();
  assert.equal(guard.check(["beko.com"]), undefined);
});

test("scanning stays fast on large tool output and odd commands", () => {
  const started = performance.now();
  refusalReason("x".repeat(5_000_000));
  refusalReason("HTTP/".repeat(400_000));
  hostsIn("http://".repeat(80_000));
  hostsIn(`curl https://${"a".repeat(200_000)}.test/`);
  const guard = new DoorGuard();
  guard.refused("bad.example", "HTTP 403");
  guard.refused("bad.example", "HTTP 403");
  guard.check([], "bad.exampl ".repeat(30_000));
  assert.ok(performance.now() - started < 8000, `took ${Math.round(performance.now() - started)} ms`);
});
