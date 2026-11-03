import assert from "node:assert/strict";
import { test } from "node:test";
import { buildDigest, touchedFiles, type DigestMessage } from "../src/digest.ts";
import { buildSummaryPrompt, composeSummary, splitFileLists } from "../src/summary.ts";

function call(id: string, name: string, args: Record<string, unknown>, text = ""): DigestMessage {
  const content: Record<string, unknown>[] = [];
  if (text) content.push({ type: "text", text });
  content.push({ type: "toolCall", id, name, arguments: args });
  return { role: "assistant", content };
}

function result(id: string, name: string, text: string, isError = false): DigestMessage {
  return { role: "toolResult", toolCallId: id, toolName: name, isError, content: [{ type: "text", text }] };
}

const conversation: DigestMessage[] = [
  { role: "user", content: [{ type: "text", text: "Rename DEFAULT_MODE to DEFAULT_DRAFTING_MODE in strata_control." }] },
  call("1", "bash", { command: "grep -rn DEFAULT_MODE strata_control" }, "Looking for every use of the constant first."),
  result("1", "bash", "strata_control/drafting.py:24:DEFAULT_MODE = 'auto'\nstrata_control/drafting.py:137: return DEFAULT_MODE"),
  call("2", "read", { path: "strata_control/drafting.py", offset: 120, limit: 40 }),
  result("2", "read", "def normalize(value):\n    ..."),
  call("3", "edit", { path: "strata_control/drafting.py" }),
  result("3", "edit", "Edited"),
  call("4", "bash", { command: "python3 -m py_compile strata_control/drafting.py" }),
  result("4", "bash", "boom: SyntaxError", true),
];

test("the digest names the request, the actions and what each returned", () => {
  const digest = buildDigest(conversation, { maxChars: 6000 });
  assert.match(digest, /## Original request\nRename DEFAULT_MODE/);
  assert.match(digest, /1\. bash: grep -rn DEFAULT_MODE strata_control -> strata_control\/drafting.py:24.* \[2 lines\]/);
  assert.match(digest, /2\. read strata_control\/drafting.py:120\+40 -> def normalize\(value\): \[2 lines\]/);
  assert.match(digest, /4\. bash: python3 -m py_compile .* -> ERROR boom: SyntaxError/);
  assert.match(digest, /## Assistant notes\n- Looking for every use/);
});

test("identical consecutive actions are collapsed", () => {
  const messages: DigestMessage[] = [{ role: "user", content: "go" }];
  for (let i = 0; i < 5; i++) {
    messages.push(call(`c${i}`, "bash", { command: "git diff | head -50" }), result(`c${i}`, "bash", "same output here\nsecond line"));
  }
  const digest = buildDigest(messages, { maxChars: 6000 });
  assert.equal((digest.match(/git diff/g) ?? []).length, 1);
  assert.match(digest, /git diff \| head -50 x5 ->/);
});

test("the oldest actions are dropped first when the digest is over budget", () => {
  const messages: DigestMessage[] = [{ role: "user", content: "do the thing" }];
  for (let i = 0; i < 60; i++) {
    messages.push(call(`c${i}`, "bash", { command: `echo step number ${i} ${"padding ".repeat(6)}` }), result(`c${i}`, "bash", `out ${i}`));
  }
  const digest = buildDigest(messages, { maxChars: 2500 });
  assert.ok(digest.length <= 2500);
  assert.match(digest, /earlier actions omitted/);
  assert.match(digest, /step number 59/);
  assert.doesNotMatch(digest, /step number 0 /);
  assert.match(digest, /## Original request\ndo the thing/);
});

test("the previous summary is carried and later user messages are kept", () => {
  const messages: DigestMessage[] = [
    { role: "user", content: "first" },
    { role: "user", content: "second request" },
  ];
  const digest = buildDigest(messages, { maxChars: 4000, previousSummary: "## Goal\nOld goal" });
  assert.match(digest, /## Later user messages\n- second request/);
  assert.match(digest, /## Earlier summary\n## Goal\nOld goal/);
});

test("touchedFiles tells reads from edits", () => {
  const touched = touchedFiles(conversation);
  assert.deepEqual([...touched.read], ["strata_control/drafting.py"]);
  assert.deepEqual([...touched.modified], ["strata_control/drafting.py"]);
});

test("file lists: a file that was modified is not listed as only read", () => {
  const { readFiles, modifiedFiles } = splitFileLists(new Set(["a.py", "b.py"]), new Set(["b.py"]));
  assert.deepEqual(readFiles, ["a.py"]);
  assert.deepEqual(modifiedFiles, ["b.py"]);
});

test("composeSummary appends the file blocks in pi's tags", () => {
  const text = composeSummary("## Goal\nx\n", ["a.py"], ["b.py"]);
  assert.equal(text, "## Goal\nx\n\n<read-files>\na.py\n</read-files>\n\n<modified-files>\nb.py\n</modified-files>");
  assert.equal(composeSummary("body", [], []), "body");
});

test("the summary prompt carries the digest and the section list", () => {
  const prompt = buildSummaryPrompt("DIGEST TEXT", "focus on tests");
  assert.match(prompt, /<digest>\nDIGEST TEXT\n<\/digest>/);
  assert.match(prompt, /## Goal\n## Done\n## Findings\n## Next steps/);
  assert.match(prompt, /Extra focus requested by the user: focus on tests/);
});

test("the previous summary is carried without its file lists, and longer than before", () => {
  const previous = `## Goal\n${"detail ".repeat(500)}end\n\n<read-files>\na.py\n</read-files>\n\n<modified-files>\nb.py\n</modified-files>`;
  const digest = buildDigest([{ role: "user", content: "go" }], { maxChars: 9000, previousSummary: previous });
  assert.doesNotMatch(digest, /read-files|modified-files|a\.py/);
  assert.match(digest, /detail detail detail[\s\S]*end/);
});

test("messages excluded from the context are not digested", () => {
  const digest = buildDigest(
    [
      { role: "user", content: "keep this request" },
      { role: "bashExecution", command: "secret command", output: "x", excludeFromContext: true },
    ],
    { maxChars: 4000 },
  );
  assert.doesNotMatch(digest, /secret command/);
});
