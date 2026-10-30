import { buildDigest, touchedFiles, type DigestMessage } from "./digest.ts";
import { buildSummaryPrompt, composeSummary, splitFileLists } from "./summary.ts";

export interface EntryLike {
  id: string;
  type: string;
  message?: DigestMessage;
  details?: unknown;
  targetId?: string;
}

// Entries that add nothing to the model's context, so a cut can move across them without losing anything.
const INERT_ENTRY_TYPES = new Set(["message", "thinking_level_change", "model_change", "usage", "custom"]);

export interface PreparationLike {
  firstKeptEntryId: string;
  messagesToSummarize: DigestMessage[];
  turnPrefixMessages: DigestMessage[];
  previousSummary?: string;
  tokensBefore: number;
  fileOps?: { read: Set<string>; written: Set<string>; edited: Set<string> };
}

export type FindCutPoint = (entries: EntryLike[], startIndex: number, endIndex: number, keepRecentTokens: number) => { firstKeptEntryIndex: number };

export interface Plan {
  firstKeptEntryId: string;
  messages: DigestMessage[];
  movedEntries: number;
}

export interface SummarizeResult {
  text: string;
  usage?: unknown;
}

export interface CompactionOutcome {
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  usage?: unknown;
  details: { readFiles: string[]; modifiedFiles: string[]; digestOnly: boolean };
}

// Moves pi's cut point later, so fewer tokens are kept verbatim and fewer have to be read again after the compaction.
export function planCompaction(
  entries: EntryLike[],
  preparation: PreparationLike,
  keepRecentTokens: number,
  findCutPoint: FindCutPoint,
): Plan {
  const piCut = entries.findIndex((entry) => entry.id === preparation.firstKeptEntryId);
  const base = [...preparation.messagesToSummarize, ...preparation.turnPrefixMessages];
  if (piCut < 0) return { firstKeptEntryId: preparation.firstKeptEntryId, messages: base, movedEntries: 0 };

  const cut = findCutPoint(entries, piCut, entries.length, keepRecentTokens).firstKeptEntryIndex;
  if (cut <= piCut || cut >= entries.length) return { firstKeptEntryId: preparation.firstKeptEntryId, messages: base, movedEntries: 0 };

  const between = entries.slice(piCut, cut);
  const edited = new Set(entries.filter((entry) => entry.type === "context_edit").map((entry) => entry.targetId));
  const safe = between.every((entry) => INERT_ENTRY_TYPES.has(entry.type) && !edited.has(entry.id));
  if (!safe) return { firstKeptEntryId: preparation.firstKeptEntryId, messages: base, movedEntries: 0 };

  const moved = between.flatMap((entry) => (entry.type === "message" && entry.message !== undefined ? [entry.message] : []));
  return { firstKeptEntryId: entries[cut].id, messages: [...base, ...moved], movedEntries: cut - piCut };
}

function previousFileLists(entries: EntryLike[]): { read: string[]; modified: string[] } {
  const last = [...entries].reverse().find((entry) => entry.type === "compaction");
  const details = last?.details as { readFiles?: string[]; modifiedFiles?: string[] } | undefined;
  return { read: details?.readFiles ?? [], modified: details?.modifiedFiles ?? [] };
}

export interface CompactionDeps {
  summarize: (prompt: string) => Promise<SummarizeResult | undefined>;
  digestChars: number;
  customInstructions?: string;
}

export async function buildCompaction(
  entries: EntryLike[],
  preparation: PreparationLike,
  plan: Plan,
  deps: CompactionDeps,
): Promise<CompactionOutcome> {
  const digest = buildDigest(plan.messages, { maxChars: deps.digestChars, previousSummary: preparation.previousSummary });

  const touched = touchedFiles(plan.messages);
  const earlier = previousFileLists(entries);
  const read = new Set([...earlier.read, ...touched.read, ...(preparation.fileOps?.read ?? [])]);
  const modified = new Set([...earlier.modified, ...touched.modified, ...(preparation.fileOps?.written ?? []), ...(preparation.fileOps?.edited ?? [])]);
  const { readFiles, modifiedFiles } = splitFileLists(read, modified);

  const generated = await deps.summarize(buildSummaryPrompt(digest, deps.customInstructions));
  const body = generated?.text.trim() ? generated.text : `## Handoff (automatic digest, no model summary)\n${digest}`;
  return {
    summary: composeSummary(body, readFiles, modifiedFiles),
    firstKeptEntryId: plan.firstKeptEntryId,
    tokensBefore: preparation.tokensBefore,
    usage: generated?.usage,
    details: { readFiles, modifiedFiles, digestOnly: !generated?.text.trim() },
  };
}
