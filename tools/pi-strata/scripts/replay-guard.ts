import { readFileSync } from "node:fs";
import { DEFAULT_GUARD, LoopGuard } from "../src/loop-guard.ts";
import { SAFE_NORMALIZE, capOutput, commandShape, estimateTokens, normalizeOutput } from "../src/text.ts";

const path = process.argv[2];
if (path === undefined) {
  console.error("usage: node scripts/replay-guard.ts <session.jsonl>");
  process.exit(2);
}

interface Block {
  type: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  text?: string;
}
interface Message {
  role: string;
  content?: Block[];
  toolCallId?: string;
  isError?: boolean;
}

const guard = new LoopGuard(DEFAULT_GUARD);
const pending = new Map<string, { name: string; input: Record<string, unknown> }>();
let turn = 0;
let results = 0;
let rawTokens = 0;
let keptTokens = 0;
let stubbed = 0;
let refused = 0;
let firstRefusal = 0;
let aborted = 0;

for (const line of readFileSync(path, "utf8").split("\n")) {
  if (line.trim() === "") continue;
  const message = (JSON.parse(line) as { message?: Message }).message;
  if (message === undefined) continue;
  if (message.role === "assistant") {
    turn++;
    for (const block of message.content ?? []) {
      if (block.type === "toolCall") pending.set(block.id as string, { name: block.name as string, input: block.arguments ?? {} });
    }
  } else if (message.role === "toolResult") {
    const call = pending.get(message.toolCallId as string);
    if (call === undefined) continue;
    const shape = commandShape(call.name, call.input);
    const original = (message.content ?? []).map((block) => block.text ?? "").join("\n");
    rawTokens += estimateTokens(original);
    const verdict = guard.beforeCall(call.name, shape);
    if (verdict !== undefined) {
      refused++;
      if (firstRefusal === 0) firstRefusal = turn;
      if (verdict.abort && aborted === 0) aborted = turn;
      keptTokens += estimateTokens(verdict.reason);
      continue;
    }
    let text = capOutput(normalizeOutput(original, SAFE_NORMALIZE), { maxChars: 6000, hint: "narrow the command" });
    const result = guard.onResult({ turn, tool: call.name, shape, text, isError: message.isError === true });
    if (result.replacement !== undefined) {
      stubbed++;
      text = result.replacement;
    }
    keptTokens += estimateTokens(text);
    results++;
  }
}

console.log(
  JSON.stringify({
    turns: turn,
    toolResults: results + refused,
    stubbed,
    refusedCalls: refused,
    firstRefusalTurn: firstRefusal,
    abortTurn: aborted,
    resultTokensBefore: rawTokens,
    resultTokensAfter: keptTokens,
    savedPercent: Math.round((100 * (rawTokens - keptTokens)) / Math.max(rawTokens, 1)),
  }),
);
