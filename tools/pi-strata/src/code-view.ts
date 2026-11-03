import { estimateTokens } from "./text.ts";

const NOTICE = /\n\n(\[(?:Showing lines|\d+ more lines)[^\]]*\])$/;
const NUMBERED = /^\d+\t/;
const DEFINITION = /^(?: {0,4}|\t)(?:async\s+)?(?:def|class|func|function|fn|struct|enum|trait|impl|interface|type|module|package|export\s+(?:default\s+)?(?:async\s+)?(?:function|class))\b/;
const HEADING = /^#{1,4}\s+\S/;
const MAX_OUTLINE = 40;
const OUTLINE_ITEM_CHARS = 80;

export interface ReadParts {
  body: string;
  notice: string;
}

// pi's read tool appends a continuation notice after a blank line when it cuts a file.
export function splitReadNotice(text: string): ReadParts {
  const match = NOTICE.exec(text);
  return match === null ? { body: text, notice: "" } : { body: text.slice(0, match.index), notice: match[1] };
}

function splitLines(body: string): { lines: string[]; endsWithNewline: boolean } {
  const endsWithNewline = body.endsWith("\n");
  return { lines: (endsWithNewline ? body.slice(0, -1) : body).split("\n"), endsWithNewline };
}

export function numberLines(body: string, firstLine: number): string {
  const { lines, endsWithNewline } = splitLines(body);
  return `${lines.map((line, i) => `${firstLine + i}\t${line}`).join("\n")}${endsWithNewline ? "\n" : ""}`;
}

function outlineOf(lines: string[], firstLine: number, from: number, to: number): string {
  const found: string[] = [];
  for (let i = from; i < to && found.length < MAX_OUTLINE; i++) {
    const line = lines[i];
    if (DEFINITION.test(line) || HEADING.test(line)) found.push(`${firstLine + i} ${line.trim().slice(0, OUTLINE_ITEM_CHARS)}`);
  }
  return found.length === 0 ? "" : ` Definitions in the omitted part: ${found.join(" | ")}.`;
}

export interface ReadViewOptions {
  firstLine: number;
  lineNumbers: boolean;
  maxChars: number;
}

// A numbered, size-bounded view of a file read. When the middle is left out, the marker lists the definitions in it.
export function readView(text: string, options: ReadViewOptions): string {
  const { body, notice } = splitReadNotice(text);
  if (body === "") return text;
  const { lines } = splitLines(body);
  const prefix = (i: number) => (options.lineNumbers ? `${options.firstLine + i}\t` : "");
  const total = lines.reduce((sum, line, i) => sum + prefix(i).length + line.length + 1, 0);
  const tailNotice = notice === "" ? "" : `\n\n${notice}`;

  if (total <= options.maxChars) return `${options.lineNumbers ? numberLines(body, options.firstLine) : body}${tailNotice}`;

  const headBudget = Math.floor(options.maxChars * 0.6);
  const tailBudget = options.maxChars - headBudget;
  let head = 0;
  let used = 0;
  while (head < lines.length && used + prefix(head).length + lines[head].length + 1 <= headBudget) used += prefix(head).length + lines[head++].length + 1;
  let tail = lines.length;
  used = 0;
  while (tail > head && used + prefix(tail - 1).length + lines[tail - 1].length + 1 <= tailBudget) used += prefix(tail - 1).length + lines[--tail].length + 1;
  if (tail <= head) return `${options.lineNumbers ? numberLines(body, options.firstLine) : body}${tailNotice}`;

  const omittedText = lines.slice(head, tail).join("\n");
  const first = options.firstLine + head;
  const last = options.firstLine + tail - 1;
  const marker =
    `[... lines ${first}-${last} omitted (~${estimateTokens(omittedText)} tokens).${outlineOf(lines, options.firstLine, head, tail)} ` +
    `Read the part you need with offset=${first} and limit.]`;
  const shown = (from: number, to: number) =>
    lines
      .slice(from, to)
      .map((line, i) => (options.lineNumbers ? `${options.firstLine + from + i}\t${line}` : line))
      .join("\n");
  return `${[shown(0, head), marker, shown(tail, lines.length)].filter((part) => part !== "").join("\n")}${tailNotice}`;
}

const NUMBERED_LINE = /^(\d+)\t(.*)$/;

// The numbered lines of a view, by line number; the omission marker and the notice are not numbered.
export function parseNumbered(view: string): Map<number, string> {
  const lines = new Map<number, string>();
  for (const line of view.split("\n")) {
    const match = NUMBERED_LINE.exec(line);
    if (match !== null) lines.set(Number(match[1]), match[2]);
  }
  return lines;
}

function numberedParts(text: string): { number: number; rest: string }[] | undefined {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  const parts: { number: number; rest: string }[] = [];
  for (const line of lines) {
    const match = NUMBERED_LINE.exec(line);
    if (match === null) return undefined;
    parts.push({ number: Number(match[1]), rest: match[2] });
  }
  return parts.length === 0 ? undefined : parts;
}

function dropPrefixes(text: string): string {
  return text
    .split("\n")
    .map((line) => line.replace(/^\d+\t/, ""))
    .join("\n");
}

// A model may copy numbered lines into an edit; the numbers are not part of the file. Without a record of what a
// read showed, only a run of consecutive numbers on every filled line counts.
export function stripLineNumbers(text: string): string {
  const parts = numberedParts(text);
  if (parts === undefined) return text;
  for (let i = 1; i < parts.length; i++) if (parts[i].number <= parts[i - 1].number) return text;
  return dropPrefixes(text);
}

const MAX_REMEMBERED_FILES = 24;

// What the numbered reads showed, per file, so that a prefix is removed only when it is really one of those numbers.
export class NumberedReads {
  private readonly files = new Map<string, Map<number, string>>();

  remember(path: string, view: string): void {
    const lines = parseNumbered(view);
    if (lines.size === 0) return;
    const known = this.files.get(path) ?? new Map<number, string>();
    this.files.delete(path);
    for (const [number, rest] of lines) known.set(number, rest);
    this.files.set(path, known);
    while (this.files.size > MAX_REMEMBERED_FILES) this.files.delete(this.files.keys().next().value as string);
  }

  forget(): void {
    this.files.clear();
  }

  // Text copied from a numbered read: every filled line has the number and the text a read showed for that file.
  verified(path: string, text: string): boolean {
    const shown = this.files.get(path);
    const parts = numberedParts(text);
    if (shown === undefined || parts === undefined) return false;
    return parts.every((part) => shown.get(part.number) === part.rest);
  }

  strip(path: string, text: string): string {
    return this.verified(path, text) ? dropPrefixes(text) : text;
  }
}
