import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { findCutPoint } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { NumberedReads, readView, stripLineNumbers } from "./src/code-view.ts";
import { buildCompaction, planCompaction, type EntryLike, type PreparationLike } from "./src/compaction.ts";
import { DEFAULT_GUARD, LoopGuard, type GuardConfig } from "./src/loop-guard.ts";
import { SUMMARY_SYSTEM_PROMPT } from "./src/summary.ts";
import { SAFE_NORMALIZE, capOutput, commandShape, estimateTokens, normalizeOutput } from "./src/text.ts";

interface Config {
  guard: GuardConfig;
  output: { bashChars: number; otherChars: number };
  read: { lineNumbers: boolean; maxChars: number };
  compaction: { enabled: boolean; keepRecentEstimatedTokens: number; digestChars: number; summaryMaxTokens: number };
  promptGuidance: boolean;
}

const DEFAULTS: Config = {
  guard: DEFAULT_GUARD,
  output: { bashChars: 6000, otherChars: 6000 },
  read: { lineNumbers: true, maxChars: 24000 },
  compaction: { enabled: true, keepRecentEstimatedTokens: 2500, digestChars: 14000, summaryMaxTokens: 700 },
  promptGuidance: true,
};

const GUIDANCE = [
  "This model runs locally: every token a tool returns costs reading time (about 100 tokens per second) and fills a 64K window.",
  "- Prefer narrow commands: grep -n -m 20 PATTERN, sed -n 'A,Bp' FILE, head, tail, wc -l. Read large files in ranges, never whole.",
  "- Do not read a file again, or rerun a command, when its output is already above in this conversation.",
  "- Lines of a read result start with their file line number and a tab; the number is not part of the file, so leave it out of edit text. Cite those numbers when you give file:line references.",
  "- If a command returned the same output as before, changing only a number in it (head -N, a range) will not change the result: use the output you have.",
  "- When several independent lookups are needed, issue them together in one response.",
].join("\n");

function loadConfig(): Config {
  try {
    const raw = JSON.parse(readFileSync(new URL("./config.json", import.meta.url), "utf8")) as Partial<Config>;
    return {
      guard: { ...DEFAULTS.guard, ...raw.guard },
      output: { ...DEFAULTS.output, ...raw.output },
      read: { ...DEFAULTS.read, ...raw.read },
      compaction: { ...DEFAULTS.compaction, ...raw.compaction },
      promptGuidance: raw.promptGuidance ?? DEFAULTS.promptGuidance,
    };
  } catch {
    return DEFAULTS;
  }
}

const textOf = (content: { type: string; text?: string }[]): string =>
  content.filter((block) => block.type === "text").map((block) => block.text ?? "").join("\n");

export default function strataAgent(pi: ExtensionAPI) {
  const config = loadConfig();
  const guard = new LoopGuard(config.guard);
  const reads = new NumberedReads();
  let turn = 0;
  const stats = { cappedTokens: 0, stubbedTokens: 0, refusedCalls: 0, compactions: 0, summaryTokensSaved: 0 };

  const forgetContext = () => {
    guard.reset();
    reads.forget();
  };
  pi.on("session_start", () => {
    forgetContext();
    turn = 0;
  });
  pi.on("session_compact", forgetContext);
  pi.on("session_tree", forgetContext);
  pi.on("turn_start", () => {
    turn++;
  });

  pi.on("before_agent_start", (event) => {
    if (config.promptGuidance) event.systemPromptOptions.sections.local_model_efficiency = GUIDANCE;
  });

  pi.on("tool_call", (event, ctx) => {
    if (event.toolName === "edit") {
      const input = event.input as { path?: unknown; edits?: { oldText?: unknown; newText?: unknown }[]; oldText?: unknown; newText?: unknown };
      const key = typeof input.path === "string" ? resolve(ctx.cwd, input.path) : "";
      for (const edit of [input, ...(Array.isArray(input.edits) ? input.edits : [])]) {
        if (typeof edit.oldText !== "string") continue;
        const stripped = reads.strip(key, edit.oldText);
        if (stripped === edit.oldText) continue;
        edit.oldText = stripped;
        if (typeof edit.newText === "string") edit.newText = stripLineNumbers(edit.newText);
      }
    }
    const verdict = guard.beforeCall(event.toolName, commandShape(event.toolName, event.input));
    if (verdict === undefined) return undefined;
    stats.refusedCalls++;
    if (verdict.abort) {
      const message = "strata: a command kept repeating with identical output; the run is stopped";
      if (ctx.hasUI) ctx.ui.notify(message, "warning");
      else console.error(message);
      ctx.abort();
    }
    return { block: true, reason: verdict.reason, terminate: verdict.abort };
  });

  pi.on("tool_result", (event, ctx) => {
    if (event.content.some((block) => block.type !== "text")) return undefined;
    const original = textOf(event.content);
    const hint = "narrow the command, for example grep -n -m 20, sed -n 'A,Bp' or head";

    let text: string;
    if (event.toolName === "read" && !event.isError) {
      const offset = (event.input as { offset?: unknown }).offset;
      text = readView(original, { firstLine: typeof offset === "number" && offset > 0 ? offset : 1, lineNumbers: config.read.lineNumbers, maxChars: config.read.maxChars });
      const path = (event.input as { path?: unknown }).path;
      if (config.read.lineNumbers && typeof path === "string") reads.remember(resolve(ctx.cwd, path), text);
    } else {
      const budget = event.toolName === "bash" ? config.output.bashChars : config.output.otherChars;
      text = capOutput(normalizeOutput(original, SAFE_NORMALIZE), { maxChars: budget, hint });
    }
    stats.cappedTokens += Math.max(0, estimateTokens(original) - estimateTokens(text));

    const verdict = guard.onResult({
      turn,
      tool: event.toolName,
      shape: commandShape(event.toolName, event.input),
      text,
      identity: original,
      isError: event.isError,
    });
    if (verdict.replacement !== undefined) {
      stats.stubbedTokens += Math.max(0, estimateTokens(text) - estimateTokens(verdict.replacement));
      text = verdict.replacement;
    }
    return text === original ? undefined : { content: [{ type: "text" as const, text }] };
  });

  pi.on("session_before_compact", async (event, ctx) => {
    if (!config.compaction.enabled || ctx.model === undefined) return undefined;
    const entries = event.branchEntries as unknown as EntryLike[];
    const preparation = event.preparation as unknown as PreparationLike;
    const plan = planCompaction(entries, preparation, config.compaction.keepRecentEstimatedTokens, findCutPoint as never);

    const outcome = await buildCompaction(entries, preparation, plan, {
      digestChars: config.compaction.digestChars,
      customInstructions: event.customInstructions,
      summarize: async (prompt) => {
        try {
          const response = await ctx.modelRegistry.complete(
            ctx.model as NonNullable<typeof ctx.model>,
            { systemPrompt: SUMMARY_SYSTEM_PROMPT, messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }] },
            { maxTokens: config.compaction.summaryMaxTokens, signal: event.signal, cacheRetention: "none", sessionId: randomUUID() },
          );
          if (response.stopReason !== "stop") return undefined;
          return { text: textOf(response.content as { type: string; text?: string }[]), usage: response.usage };
        } catch {
          return undefined;
        }
      },
    });

    stats.compactions++;
    stats.summaryTokensSaved += Math.max(0, preparation.tokensBefore - estimateTokens(outcome.summary));
    return { compaction: outcome as never };
  });

  pi.on("agent_end", () => {
    if (stats.cappedTokens + stats.stubbedTokens + stats.refusedCalls + stats.compactions > 0) pi.appendEntry("strata-agent-stats", { ...stats });
  });
}
