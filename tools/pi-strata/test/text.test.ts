import assert from "node:assert/strict";
import { test } from "node:test";
import { SAFE_NORMALIZE, capOutput, commandShape, countLines, estimateTokens, normalizeOutput } from "../src/text.ts";

test("normalizeOutput strips escape sequences and progress rewrites", () => {
  const text = "\x1b[31mred\x1b[0m line\nprogress 10%\rprogress 60%\rprogress 100%\ndone  \n";
  assert.equal(normalizeOutput(text), "red line\nprogress 100%\ndone");
});

test("normalizeOutput collapses blank runs and identical line runs", () => {
  const text = ["a", "", "", "", "b", "x", "x", "x", "x", "x", "c", "y", "y", "d"].join("\n");
  assert.equal(
    normalizeOutput(text),
    ["a", "", "b", "x", "[the line above repeated 4 more times]", "c", "y", "y", "d"].join("\n"),
  );
});

test("normalizeOutput is idempotent", () => {
  const text = "\x1b[1mhead\x1b[0m\n\n\n\nfoo   \nfoo\nfoo\nfoo\nbar\r\nbaz";
  const once = normalizeOutput(text);
  assert.equal(normalizeOutput(once), once);
});

test("normalizeOutput caps one very long line", () => {
  const out = normalizeOutput("x".repeat(5000));
  assert.ok(out.length < 1300);
  assert.match(out, /3800 more characters/);
});

test("capOutput leaves a short output alone", () => {
  assert.equal(capOutput("one\ntwo", { maxChars: 100, hint: "h" }), "one\ntwo");
});

test("capOutput keeps the start and the end and names what was left out", () => {
  const lines = Array.from({ length: 200 }, (_, i) => `line ${i} ${"y".repeat(20)}`);
  const out = capOutput(lines.join("\n"), { maxChars: 1500, hint: "use sed -n" });
  assert.ok(out.length <= 1500 + 120, `length ${out.length}`);
  assert.ok(out.startsWith("line 0 "));
  assert.ok(out.endsWith(lines[199]));
  assert.match(out, /\[\.\.\. \d+ lines \(~\d+ tokens\) omitted; use sed -n\]/);
  const omitted = Number(/\[\.\.\. (\d+) lines/.exec(out)?.[1]);
  assert.equal(countLines(out) - 1 + omitted, 200);
});

test("capOutput splits the budget by headShare", () => {
  const lines = Array.from({ length: 100 }, (_, i) => `row ${String(i).padStart(3, "0")}`);
  const out = capOutput(lines.join("\n"), { maxChars: 200, headShare: 0.5, hint: "h" });
  const kept = out.split("\n");
  const marker = kept.findIndex((l) => l.startsWith("[..."));
  assert.ok(marker > 0 && marker < kept.length - 1);
  assert.ok(Math.abs(marker - (kept.length - 1 - marker)) <= 2);
});

test("commandShape ignores numbers and spacing", () => {
  const a = commandShape("bash", { command: "cd /x && git diff | grep -B 3 -A 5 cursor | head -50" });
  const b = commandShape("bash", { command: "cd /x  &&  git diff | grep -B 4 -A 9 cursor | head -120" });
  assert.equal(a, b);
  assert.notEqual(a, commandShape("bash", { command: "cd /x && git diff | grep cursor | wc -l" }));
});

test("commandShape of read depends on the path only", () => {
  assert.equal(commandShape("read", { path: "a.go", offset: 1 }), commandShape("read", { path: "a.go", offset: 900 }));
  assert.notEqual(commandShape("read", { path: "a.go" }), commandShape("read", { path: "b.go" }));
});

test("estimateTokens grows with length", () => {
  assert.ok(estimateTokens("a".repeat(2800)) === 1000);
});

test("safe mode keeps whitespace, blank lines and short repeats exactly", () => {
  const text = "def f():\n    return 1   \n\n\n\n    pass\n}\n}\n}\n";
  assert.equal(normalizeOutput(text, SAFE_NORMALIZE), text);
});

test("safe mode still removes escape codes and folds long repeats", () => {
  const text = `\x1b[32mok\x1b[0m\n${"spam\n".repeat(8)}end`;
  assert.equal(normalizeOutput(text, SAFE_NORMALIZE), "ok\nspam\n[the line above repeated 7 more times]\nend");
});

test("commandShape keeps digits inside names and masks whole numbers and ranges", () => {
  assert.notEqual(commandShape("bash", { command: "cat file1" }), commandShape("bash", { command: "cat file2" }));
  assert.equal(commandShape("bash", { command: "sed -n '10,20p' f.py" }), commandShape("bash", { command: "sed -n '30,90p' f.py" }));
  assert.equal(commandShape("bash", { command: "git diff | head -50" }), commandShape("bash", { command: "git diff | head -120" }));
  assert.equal(commandShape("bash", { command: "grep -B 3 -A 5 x f" }), commandShape("bash", { command: "grep -B 9 -A 1 x f" }));
});

test("safe mode keeps CRLF line endings", () => {
  assert.equal(normalizeOutput("a\r\nb\r\n", SAFE_NORMALIZE), "a\r\nb\r\n");
});

test("safe mode still collapses a progress bar that ends a CRLF line", () => {
  assert.equal(normalizeOutput("10%\r50%\r100%\r\ndone", SAFE_NORMALIZE), "100%\r\ndone");
});
