import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, findCutPoint } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import { NumberedReads, readView, stripLineNumbers } from "./src/code-view.ts";
import { buildCompaction, planCompaction, type EntryLike, type PreparationLike } from "./src/compaction.ts";
import { DEFAULT_DOOR, DoorGuard, adviceFor, refusalReason, sitesOfResult, webCall, type DoorConfig } from "./src/door-guard.ts";
import { installLocalOnly, localOnlyRequested } from "./src/local-only.ts";
import { DEFAULT_GUARD, LoopGuard, type GuardConfig } from "./src/loop-guard.ts";
import { SUMMARY_SYSTEM_PROMPT } from "./src/summary.ts";
import { SAFE_NORMALIZE, capOutput, commandShape, estimateTokens, normalizeOutput } from "./src/text.ts";
import { DEFAULT_CHARS, MAX_CHARS, describeFetchError, fetchPage, pageSlice, pageText } from "./src/web.ts";

interface Config {
  guard: GuardConfig;
  doors: DoorConfig;
  output: { bashChars: number; otherChars: number };
  read: { lineNumbers: boolean; maxChars: number };
  compaction: { enabled: boolean; keepRecentEstimatedTokens: number; digestChars: number; summaryMaxTokens: number };
  promptGuidance: boolean;
  web: { enabled: boolean };
}

const DEFAULTS: Config = {
  guard: DEFAULT_GUARD,
  doors: DEFAULT_DOOR,
  output: { bashChars: 6000, otherChars: 6000 },
  read: { lineNumbers: true, maxChars: 24000 },
  compaction: { enabled: true, keepRecentEstimatedTokens: 2500, digestChars: 14000, summaryMaxTokens: 700 },
  promptGuidance: true,
  web: { enabled: true },
};

const GUIDANCE = [
  "This model runs locally: every token a tool returns costs reading time (about 100 tokens per second) and fills a 64K window.",
  "- Prefer narrow commands: grep -n -m 20 PATTERN, sed -n 'A,Bp' FILE, head, tail, wc -l. Read large files in ranges, never whole.",
  "- Do not read a file again, or rerun a command, when its output is already above in this conversation.",
  "- Lines of a read result start with their file line number and a tab; the number is not part of the file, so leave it out of edit text. Cite those numbers when you give file:line references.",
  "- If a command returned the same output as before, changing only a number in it (head -N, a range) will not change the result: use the output you have.",
  "- When several independent lookups are needed, issue them together in one response.",
  "- The web is reachable: web_fetch opens a URL as text, and curl works in bash. Never answer that you cannot open web pages; try, and report what came back.",
  "- A site that refuses programs (HTTP 403, a browser check, Access Denied) refuses every page of it: do not retry or try other pages, take another way (a web search, another shop or the maker's site, an API or a feed, the web archive) or ask the user for the text. A name that does not exist is not worth guessing at.",
].join("\n");

function loadConfig(): Config {
  try {
    const raw = JSON.parse(readFileSync(new URL("./config.json", import.meta.url), "utf8")) as Partial<Config>;
    return {
      guard: { ...DEFAULTS.guard, ...raw.guard },
      doors: { ...DEFAULTS.doors, ...raw.doors },
      output: { ...DEFAULTS.output, ...raw.output },
      read: { ...DEFAULTS.read, ...raw.read },
      compaction: { ...DEFAULTS.compaction, ...raw.compaction },
      promptGuidance: raw.promptGuidance ?? DEFAULTS.promptGuidance,
      web: { ...DEFAULTS.web, ...raw.web },
    };
  } catch {
    return DEFAULTS;
  }
}

const textOf = (content: { type: string; text?: string }[]): string =>
  content.filter((block) => block.type === "text").map((block) => block.text ?? "").join("\n");

const webFetchTool = defineTool({
  name: "web_fetch",
  label: "Web fetch",
  description:
    "Open an http or https URL and return it as plain text: a web page is reduced to its readable text, JSON and plain text come as they are. A long page comes in parts, and the result names the offset to pass for the next one.",
  promptSnippet: "web_fetch: open a web page or an API address and read it as text",
  promptGuidelines: [
    `web_fetch returns about ${DEFAULT_CHARS} characters at a time. When the page continues, call it again with the offset the result names; do not fetch the same page again from the start.`,
    "If web_fetch says a site blocks programs, do not guess what its page contains: tell the user, or look for the information on another site.",
  ],
  parameters: Type.Object({
    url: Type.String({ description: "The http or https address to open" }),
    offset: Type.Optional(Type.Number({ description: "Character to start from, as named by the previous result. Default 0." })),
    max_chars: Type.Optional(Type.Number({ description: `How many characters to return, at most ${MAX_CHARS}. Default ${DEFAULT_CHARS}.` })),
  }),
  async execute(_toolCallId, params, signal) {
    let page;
    try {
      page = await fetchPage(params.url, { signal });
    } catch (error) {
      throw new Error(`web_fetch: ${describeFetchError(error)}`);
    }
    const type = page.contentType.split(";")[0]?.trim();
    const head = `${page.url.slice(0, 300)} (HTTP ${page.status}${type ? `, ${type}` : ""})`;
    const text = `${head}\n${pageSlice(pageText(page), params.offset ?? 0, params.max_chars ?? DEFAULT_CHARS)}`;
    return { content: [{ type: "text" as const, text }], details: undefined };
  },
});

export default function strataAgent(pi: ExtensionAPI) {
  if (localOnlyRequested(process.env)) installLocalOnly(pi, { isSubmitKey: (data) => matchesKey(data, "enter") });
  const config = loadConfig();
  if (config.web.enabled) pi.registerTool(webFetchTool);
  const guard = new LoopGuard(config.guard);
  const doors = new DoorGuard(config.doors);
  const reads = new NumberedReads();
  let turn = 0;
  const stats = { cappedTokens: 0, stubbedTokens: 0, refusedCalls: 0, compactions: 0, summaryTokensSaved: 0 };

  const forgetContext = () => {
    guard.reset();
    reads.forget();
  };
  pi.on("session_start", () => {
    forgetContext();
    doors.reset();
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
    const webTarget = webCall(event.toolName, event.input);
    const shut = webTarget === undefined || !config.doors.enabled || !config.doors.block ? undefined : doors.check(webTarget.hosts, webTarget.text);
    if (shut !== undefined) {
      stats.refusedCalls++;
      return { block: true, reason: shut };
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
    const webTarget = webCall(event.toolName, event.input);
    const advice: string[] = [];
    if (webTarget !== undefined && config.doors.enabled) {
      const reason = refusalReason(original);
      for (const host of sitesOfResult(event.toolName, original, webTarget)) {
        if (reason !== undefined) advice.push(adviceFor(host, reason, doors.refused(host, reason)));
        else if (original.trim() !== "" && !event.isError) doors.opened(host);
      }
    }
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
    if (advice.length > 0) text = `${text}\n\n${advice.join("\n")}`;
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
