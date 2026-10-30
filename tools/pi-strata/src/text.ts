import { createHash } from "node:crypto";

// Measured on real pi tool results with the Qwen3.6 tokenizer: 2.76 characters per token. Kept in tenths to stay exact.
const CHARS_PER_TOKEN_TENTHS = 28;

const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g;

export interface NormalizeOptions {
  tidyWhitespace: boolean;
  minRun: number;
  maxLine: number;
}

// Tool output a model may copy back into an edit: whitespace and blank lines stay exactly as they were.
export const SAFE_NORMALIZE: NormalizeOptions = { tidyWhitespace: false, minRun: 5, maxLine: 4000 };
export const TIDY_NORMALIZE: NormalizeOptions = { tidyWhitespace: true, minRun: 3, maxLine: 1200 };

export function estimateTokens(text: string): number {
  return Math.ceil((text.length * 10) / CHARS_PER_TOKEN_TENTHS);
}

export function fingerprint(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

export function countLines(text: string): number {
  return text === "" ? 0 : text.replace(/\n$/, "").split("\n").length;
}

function afterLastCarriageReturn(line: string): string {
  const at = line.lastIndexOf("\r");
  return at === -1 ? line : line.slice(at + 1);
}

function capLine(line: string, maxLine: number): string {
  if (line.length <= maxLine) return line;
  return `${line.slice(0, maxLine)} [... ${line.length - maxLine} more characters]`;
}

// Drops escape sequences and progress-bar rewrites, caps very long lines and folds runs of identical lines.
export function normalizeOutput(text: string, options: NormalizeOptions = TIDY_NORMALIZE): string {
  const lines = text.replace(ANSI, "").split("\n");
  const out: string[] = [];
  let blankRun = 0;
  let runLine: string | undefined;
  let runLength = 0;

  const flushRun = () => {
    if (runLength >= options.minRun) out.push(`[the line above repeated ${runLength - 1} more times]`);
    else for (let i = 1; i < runLength; i++) out.push(runLine as string);
    runLine = undefined;
    runLength = 0;
  };

  for (const raw of lines) {
    const carriage = raw.endsWith("\r");
    const stripped = afterLastCarriageReturn(carriage ? raw.slice(0, -1) : raw);
    const kept = options.tidyWhitespace ? stripped.trimEnd() : stripped;
    const line = capLine(kept, options.maxLine) + (carriage && !options.tidyWhitespace ? "\r" : "");
    if (line.trim() === "") {
      flushRun();
      blankRun++;
      if (!options.tidyWhitespace) out.push(line);
      else if (blankRun === 1) out.push("");
      continue;
    }
    blankRun = 0;
    if (line === runLine) {
      runLength++;
      continue;
    }
    flushRun();
    runLine = line;
    runLength = 1;
    out.push(line);
  }
  flushRun();
  const joined = out.join("\n");
  return options.tidyWhitespace ? joined.replace(/^\n+/, "").replace(/\n+$/, "") : joined;
}

export interface CapOptions {
  maxChars: number;
  headShare?: number;
  hint: string;
}

// Keeps the start and the end of an oversized output and says what was left out and how to get it.
export function capOutput(text: string, options: CapOptions): string {
  if (text.length <= options.maxChars) return text;
  const headBudget = Math.floor(options.maxChars * (options.headShare ?? 0.6));
  const tailBudget = options.maxChars - headBudget;
  const lines = text.split("\n");

  let head = 0;
  let headChars = 0;
  while (head < lines.length && headChars + lines[head].length + 1 <= headBudget) headChars += lines[head++].length + 1;

  let tail = lines.length;
  let tailChars = 0;
  while (tail > head && tailChars + lines[tail - 1].length + 1 <= tailBudget) tailChars += lines[--tail].length + 1;

  if (tail <= head) return text;
  const omitted = lines.slice(head, tail).join("\n");
  const marker = `[... ${tail - head} lines (~${estimateTokens(omitted)} tokens) omitted; ${options.hint}]`;
  return [...lines.slice(0, head), marker, ...lines.slice(tail)].join("\n");
}

// Only whole numeric arguments and sed-style ranges are masked, so file1 and file2 stay different commands.
function maskNumbers(text: string): string {
  return text
    .replace(/(^|\s)-?\d+(?=\s|$)/g, "$1N")
    .replace(/(?<![\w.])\d+,\d+(?=[pPdq'"\s]|$)/g, "N,N")
    .replace(/\s+/g, " ")
    .trim();
}

// Calls that differ only in numbers (head -50 and head -80, other offsets) share a shape.
export function commandShape(toolName: string, input: Record<string, unknown>): string {
  const pick = (key: string) => (typeof input[key] === "string" ? (input[key] as string) : "");
  switch (toolName) {
    case "bash":
    case "powershell":
      return `${toolName}:${maskNumbers(pick("command")).slice(0, 240)}`;
    case "read":
    case "edit":
    case "write":
      return `${toolName}:${pick("path")}`;
    default:
      return `${toolName}:${maskNumbers(JSON.stringify(input)).slice(0, 240)}`;
  }
}
