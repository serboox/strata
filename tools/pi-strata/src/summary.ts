const MAX_LISTED_FILES = 40;

export const SUMMARY_SYSTEM_PROMPT =
  "You write short handoff notes for a coding agent whose older conversation is being removed to save context. " +
  "You never continue the work and never call tools.";

export function buildSummaryPrompt(digest: string, customInstructions?: string): string {
  const extra = customInstructions?.trim() ? `\nExtra focus requested by the user: ${customInstructions.trim()}\n` : "";
  return [
    "Write the handoff note that lets the agent continue the work below without the old messages.",
    "Be terse: at most 300 words, no preamble. Use exactly these sections:",
    "## Goal",
    "## Done",
    "## Findings",
    "## Next steps",
    "Under Findings put the facts the agent still needs: paths, line numbers, names, values, error messages, decisions. " +
      "Use only the digest, keep identifiers exactly as written, invent nothing, and skip what one command can re-derive.",
    extra,
    "<digest>",
    digest,
    "</digest>",
  ].join("\n");
}

function list(tag: string, files: string[]): string {
  if (files.length === 0) return "";
  const shown = files.slice(0, MAX_LISTED_FILES);
  const more = files.length > shown.length ? [`... and ${files.length - shown.length} more`] : [];
  return `\n\n<${tag}>\n${[...shown, ...more].join("\n")}\n</${tag}>`;
}

export function composeSummary(body: string, readFiles: string[], modifiedFiles: string[]): string {
  return `${body.trim()}${list("read-files", readFiles)}${list("modified-files", modifiedFiles)}`;
}

export function splitFileLists(read: Set<string>, modified: Set<string>): { readFiles: string[]; modifiedFiles: string[] } {
  const modifiedFiles = [...modified].sort();
  const readFiles = [...read].filter((path) => !modified.has(path)).sort();
  return { readFiles, modifiedFiles };
}
