import assert from "node:assert/strict";
import { test } from "node:test";
import { DEFAULT_GUARD, LoopGuard } from "../src/loop-guard.ts";

const BODY = "diff line with some content\n".repeat(40);

function result(turn: number, text: string, shape = "bash:git diff | head -N") {
  return { turn, tool: "bash", shape, text, isError: false };
}

test("a unique result passes through untouched", () => {
  const guard = new LoopGuard();
  assert.deepEqual(guard.onResult(result(1, BODY)), { duplicates: 0 });
});

test("an identical result becomes a stub that points at the first one", () => {
  const guard = new LoopGuard();
  guard.onResult(result(3, BODY));
  const verdict = guard.onResult(result(4, BODY));
  assert.equal(verdict.duplicates, 1);
  assert.match(verdict.replacement as string, /identical to the result of turn 3 \(40 lines/);
  assert.ok((verdict.replacement as string).length < BODY.length / 4);
  assert.doesNotMatch(verdict.replacement as string, /number \d+ in a row/);
});

test("the second identical repeat carries advice", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, BODY));
  guard.onResult(result(2, BODY));
  const verdict = guard.onResult(result(3, BODY));
  assert.equal(verdict.duplicates, 2);
  assert.match(verdict.replacement as string, /identical result number 3 in a row/);
});

test("a different result ends the run", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, BODY));
  guard.onResult(result(2, BODY));
  guard.onResult(result(3, "something else entirely\n".repeat(20)));
  assert.equal(guard.run, 0);
  assert.equal(guard.beforeCall("bash", "bash:git diff | head -N"), undefined);
});

test("the same command is refused after enough identical results, others are not", () => {
  const guard = new LoopGuard();
  for (let turn = 1; turn <= 1 + DEFAULT_GUARD.blockAt; turn++) guard.onResult(result(turn, BODY));
  const refusal = guard.beforeCall("bash", "bash:git diff | head -N");
  assert.ok(refusal);
  assert.match(refusal.reason, /already run 4 times with identical output/);
  assert.equal(refusal.abort, false);
  assert.equal(guard.beforeCall("bash", "bash:ls -la"), undefined);
});

test("the run is aborted when refused calls keep coming", () => {
  const guard = new LoopGuard();
  for (let turn = 1; turn <= 4; turn++) guard.onResult(result(turn, BODY));
  const verdicts = [1, 2, 3].map(() => guard.beforeCall("bash", "bash:git diff | head -N"));
  assert.deepEqual(
    verdicts.map((v) => v?.abort),
    [false, false, true],
  );
});

test("short identical results are counted but never replaced", () => {
  const guard = new LoopGuard();
  const verdicts = [1, 2, 3, 4].map((turn) => guard.onResult(result(turn, "No such file or directory", "bash:cat missing")));
  assert.equal(verdicts[0].duplicates, 0);
  assert.equal(verdicts[3].duplicates, 3);
  assert.equal(verdicts[3].replacement, undefined);
  assert.ok(guard.beforeCall("bash", "bash:cat missing"));
});

test("short identical results from different commands are not a run", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, "OK", "bash:a"));
  const verdict = guard.onResult(result(2, "OK", "bash:b"));
  assert.equal(verdict.duplicates, 0);
});

test("reset forgets everything, as after a compaction", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, BODY));
  guard.reset();
  assert.deepEqual(guard.onResult(result(2, BODY)), { duplicates: 0 });
});

test("the memory is bounded by the window", () => {
  const guard = new LoopGuard({ ...DEFAULT_GUARD, window: 3 });
  guard.onResult(result(1, `first ${BODY}`));
  for (let turn = 2; turn <= 5; turn++) guard.onResult(result(turn, `other ${turn} ${BODY}`));
  assert.equal(guard.onResult(result(6, `first ${BODY}`)).duplicates, 0);
});

test("the recorded loop shape: 67 identical results become one full copy and stubs", () => {
  const guard = new LoopGuard();
  let kept = 0;
  let refused = 0;
  for (let turn = 1; turn <= 67; turn++) {
    if (guard.beforeCall("bash", "bash:git diff | grep cursor | head -N")) {
      refused++;
      continue;
    }
    const verdict = guard.onResult(result(turn, BODY, "bash:git diff | grep cursor | head -N"));
    kept += (verdict.replacement ?? BODY).length;
  }
  assert.ok(refused > 0);
  assert.ok(kept < BODY.length * 8);
});

test("edits are exempt: identical short success messages never block the next edit", () => {
  const guard = new LoopGuard();
  for (let turn = 1; turn <= 8; turn++) {
    guard.onResult({ turn, tool: "edit", shape: "edit:a.py", text: "Successfully replaced 1 block(s) in a.py.", isError: false });
    assert.equal(guard.beforeCall("edit", "edit:a.py"), undefined);
  }
});

test("an edit between identical command results ends the run", () => {
  const guard = new LoopGuard();
  for (let turn = 1; turn <= 3; turn++) guard.onResult(result(turn, BODY, "bash:go test"));
  guard.onResult({ turn: 4, tool: "edit", shape: "edit:a.go", text: "ok", isError: false });
  assert.equal(guard.beforeCall("bash", "bash:go test"), undefined);
});

test("two outputs that alternate are never refused as a run", () => {
  const guard = new LoopGuard();
  const other = `different ${BODY}`;
  for (let turn = 1; turn <= 9; turn++) {
    const shape = turn % 2 ? "bash:cat a" : "bash:cat b";
    guard.onResult(result(turn, turn % 2 ? BODY : other, shape));
    assert.equal(guard.beforeCall("bash", turn % 2 ? "bash:cat b" : "bash:cat a"), undefined);
  }
});

test("an error and a success with the same text are different results", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, BODY));
  const verdict = guard.onResult({ ...result(2, BODY), isError: true });
  assert.equal(verdict.duplicates, 0);
});

test("the same text from two tools is not a repeat", () => {
  const guard = new LoopGuard();
  guard.onResult(result(1, BODY));
  assert.equal(guard.onResult({ ...result(2, BODY), tool: "grep" }).duplicates, 0);
});

test("identity decides, not the shortened text the model sees", () => {
  const guard = new LoopGuard();
  const capped = `head\n${"x".repeat(300)}\ntail`;
  guard.onResult({ ...result(1, capped), identity: "full output one ".repeat(40) });
  assert.equal(guard.onResult({ ...result(2, capped), identity: "full output two ".repeat(40) }).duplicates, 0);
  assert.equal(guard.onResult({ ...result(3, capped), identity: "full output two ".repeat(40) }).duplicates, 1);
});
