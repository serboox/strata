import assert from "node:assert/strict";
import { test } from "node:test";
import { buildCompaction, planCompaction, type EntryLike, type FindCutPoint, type PreparationLike } from "../src/compaction.ts";
import type { DigestMessage } from "../src/digest.ts";

function msg(id: string, role: string, extra: Partial<DigestMessage> = {}): EntryLike {
  return { id, type: "message", message: { role, content: [{ type: "text", text: `text of ${id}` }], ...extra } };
}

const entries: EntryLike[] = [
  msg("e0", "user"),
  msg("e1", "assistant"),
  msg("e2", "toolResult", { toolCallId: "x" }),
  msg("e3", "assistant"),
  msg("e4", "toolResult", { toolCallId: "y" }),
  msg("e5", "assistant"),
  msg("e6", "toolResult", { toolCallId: "z" }),
];

const preparation: PreparationLike = {
  firstKeptEntryId: "e2",
  messagesToSummarize: [entries[0].message as DigestMessage, entries[1].message as DigestMessage],
  turnPrefixMessages: [],
  tokensBefore: 50000,
};

test("the cut moves later and the entries in between join the summarized messages", () => {
  const findCut: FindCutPoint = (_e, start, end) => {
    assert.equal(start, 2);
    assert.equal(end, 7);
    return { firstKeptEntryIndex: 5 };
  };
  const plan = planCompaction(entries, preparation, 2500, findCut);
  assert.equal(plan.firstKeptEntryId, "e5");
  assert.equal(plan.movedEntries, 3);
  assert.equal(plan.messages.length, 5);
});

test("a cut that does not move leaves pi's choice alone", () => {
  const plan = planCompaction(entries, preparation, 2500, () => ({ firstKeptEntryIndex: 2 }));
  assert.equal(plan.firstKeptEntryId, "e2");
  assert.equal(plan.movedEntries, 0);
  assert.equal(plan.messages.length, 2);
});

test("an unknown kept entry or a cut past the end falls back to pi's plan", () => {
  const unknown = planCompaction(entries, { ...preparation, firstKeptEntryId: "nope" }, 2500, () => ({ firstKeptEntryIndex: 5 }));
  assert.equal(unknown.firstKeptEntryId, "nope");
  const pastEnd = planCompaction(entries, preparation, 2500, () => ({ firstKeptEntryIndex: 7 }));
  assert.equal(pastEnd.firstKeptEntryId, "e2");
});

test("buildCompaction uses the model text and appends cumulative file lists", async () => {
  const withFiles: EntryLike[] = [
    { id: "c0", type: "compaction", details: { readFiles: ["old.py"], modifiedFiles: ["gone.py"] } },
    ...entries,
  ];
  const plan = { firstKeptEntryId: "e5", messages: [] as DigestMessage[], movedEntries: 0 };
  const prompts: string[] = [];
  const outcome = await buildCompaction(withFiles, { ...preparation, fileOps: { read: new Set(["r.py"]), written: new Set(), edited: new Set(["e.py"]) } }, plan, {
    digestChars: 4000,
    summarize: async (prompt) => {
      prompts.push(prompt);
      return { text: "## Goal\nthe goal", usage: { total: 7 } };
    },
  });
  assert.match(prompts[0], /<digest>/);
  assert.match(outcome.summary, /^## Goal\nthe goal/);
  assert.match(outcome.summary, /<read-files>\nold\.py\nr\.py\n<\/read-files>/);
  assert.match(outcome.summary, /<modified-files>\ne\.py\ngone\.py\n<\/modified-files>/);
  assert.deepEqual(outcome.usage, { total: 7 });
  assert.equal(outcome.details.digestOnly, false);
  assert.equal(outcome.tokensBefore, 50000);
});

test("buildCompaction falls back to the digest when the model gives nothing", async () => {
  const plan = { firstKeptEntryId: "e2", messages: [entries[0].message as DigestMessage], movedEntries: 0 };
  const outcome = await buildCompaction(entries, preparation, plan, { digestChars: 4000, summarize: async () => undefined });
  assert.match(outcome.summary, /^## Handoff \(automatic digest, no model summary\)\n## Original request\ntext of e0/);
  assert.equal(outcome.details.digestOnly, true);
});

test("the cut does not move across entries that carry context other than messages", () => {
  for (const type of ["branch_summary", "custom_message", "compaction"]) {
    const withOther: EntryLike[] = [...entries.slice(0, 3), { id: "x", type }, ...entries.slice(3)];
    const plan = planCompaction(withOther, { ...preparation, firstKeptEntryId: "e2" }, 2500, () => ({ firstKeptEntryIndex: 6 }));
    assert.equal(plan.firstKeptEntryId, "e2", type);
    assert.equal(plan.movedEntries, 0);
  }
});

test("the cut does not move across a message that a context edit changed or hid", () => {
  const edited: EntryLike[] = [...entries, { id: "ed", type: "context_edit", targetId: "e3" }];
  const plan = planCompaction(edited, preparation, 2500, () => ({ firstKeptEntryIndex: 5 }));
  assert.equal(plan.firstKeptEntryId, "e2");
});

test("state-only entries do not stop the cut from moving", () => {
  const withState: EntryLike[] = [...entries.slice(0, 3), { id: "m", type: "model_change" }, { id: "u", type: "usage" }, ...entries.slice(3)];
  const plan = planCompaction(withState, preparation, 2500, () => ({ firstKeptEntryIndex: 7 }));
  assert.equal(plan.firstKeptEntryId, "e5");
});
