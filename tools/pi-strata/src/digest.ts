export interface DigestMessage {
  role: string;
  content?: unknown;
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  command?: string;
  output?: string;
  summary?: string;
  excludeFromContext?: boolean;
}

export interface DigestOptions {
  maxChars: number;
  previousSummary?: string;
}

export interface FileTouches {
  read: Set<string>;
  modified: Set<string>;
}

interface Action {
  label: string;
  brief: string;
  count: number;
}

const FIRST_REQUEST_CHARS = 1500;
const LATER_REQUEST_CHARS = 600;
const LATER_REQUESTS_KEPT = 4;
const NOTE_CHARS = 320;
const NOTES_KEPT = 6;
const PREVIOUS_SUMMARY_CHARS = 6000;
const LABEL_CHARS = 160;
const BRIEF_CHARS = 110;

function clip(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1)}…`;
}

function blocks(content: unknown): Record<string, unknown>[] {
  return Array.isArray(content) ? (content as Record<string, unknown>[]) : [];
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  return blocks(content)
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text as string)
    .join("\n");
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function labelOf(tool: string, args: Record<string, unknown>): string {
  switch (tool) {
    case "bash":
    case "powershell":
      return clip(`${tool}: ${str(args.command)}`, LABEL_CHARS);
    case "read": {
      const range = args.offset === undefined ? "" : `:${String(args.offset)}${args.limit === undefined ? "" : `+${String(args.limit)}`}`;
      return `read ${str(args.path)}${range}`;
    }
    case "edit":
    case "write":
      return `${tool} ${str(args.path)}`;
    case "grep":
      return clip(`grep ${str(args.pattern)} ${str(args.path)}`, LABEL_CHARS);
    case "find":
    case "ls":
      return clip(`${tool} ${str(args.pattern)} ${str(args.path)}`, LABEL_CHARS);
    default:
      return clip(`${tool} ${JSON.stringify(args)}`, LABEL_CHARS);
  }
}

function briefOf(text: string, isError: boolean): string {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  const first = lines[0] === undefined ? "(no output)" : clip(lines[0], BRIEF_CHARS);
  const more = lines.length > 1 ? ` [${lines.length} lines]` : "";
  return `${isError ? "ERROR " : ""}${first}${more}`;
}

export function touchedFiles(messages: DigestMessage[]): FileTouches {
  const touched: FileTouches = { read: new Set(), modified: new Set() };
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const block of blocks(message.content)) {
      if (block.type !== "toolCall") continue;
      const path = str((block.arguments as Record<string, unknown> | undefined)?.path);
      if (path === "") continue;
      if (block.name === "read") touched.read.add(path);
      if (block.name === "edit" || block.name === "write") touched.modified.add(path);
    }
  }
  return touched;
}

// The file lists are appended again to every new summary, so they are not carried inside the text.
function withoutFileBlocks(summary: string): string {
  return summary.replace(/\n*<(read|modified)-files>[\s\S]*?<\/\1-files>/g, "").trim();
}

function collapse(actions: Action[]): Action[] {
  const out: Action[] = [];
  for (const action of actions) {
    const last = out[out.length - 1];
    if (last !== undefined && last.label === action.label && last.brief === action.brief) last.count++;
    else out.push({ ...action });
  }
  return out;
}

function renderAction(action: Action, index: number): string {
  const times = action.count > 1 ? ` x${action.count}` : "";
  return `${index + 1}. ${action.label}${times} -> ${action.brief}`;
}

// A compact, deterministic record of what happened in the part of the conversation that is being dropped.
export function buildDigest(messages: DigestMessage[], options: DigestOptions): string {
  const requests: string[] = [];
  const notes: string[] = [];
  const pending = new Map<string, { tool: string; args: Record<string, unknown> }>();
  const raw: Action[] = [];

  for (const message of messages) {
    if (message.excludeFromContext === true) continue;
    if (message.role === "user") {
      const text = textOf(message.content).trim();
      if (text !== "") requests.push(text);
    } else if (message.role === "assistant") {
      for (const block of blocks(message.content)) {
        if (block.type === "text" && str(block.text).trim().length >= 20) notes.push(clip(str(block.text), NOTE_CHARS));
        if (block.type === "toolCall") {
          pending.set(str(block.id), { tool: str(block.name), args: (block.arguments ?? {}) as Record<string, unknown> });
        }
      }
    } else if (message.role === "toolResult") {
      const call = pending.get(message.toolCallId ?? "");
      const label = call === undefined ? clip(message.toolName ?? "tool", LABEL_CHARS) : labelOf(call.tool, call.args);
      raw.push({ label, brief: briefOf(textOf(message.content), message.isError === true), count: 1 });
    } else if (message.role === "bashExecution" && typeof message.command === "string") {
      raw.push({ label: clip(`bash (run by the user): ${message.command}`, LABEL_CHARS), brief: briefOf(message.output ?? "", false), count: 1 });
    }
  }

  const actions = collapse(raw);
  const head: string[] = [];
  if (requests.length > 0) {
    head.push("## Original request", clip(requests[0], FIRST_REQUEST_CHARS));
    const later = requests.slice(1).slice(-LATER_REQUESTS_KEPT);
    if (later.length > 0) head.push("", "## Later user messages", ...later.map((text) => `- ${clip(text, LATER_REQUEST_CHARS)}`));
  }
  if (options.previousSummary !== undefined && options.previousSummary.trim() !== "") {
    head.push("", "## Earlier summary", withoutFileBlocks(options.previousSummary).slice(0, PREVIOUS_SUMMARY_CHARS));
  }
  const tail = notes.length > 0 ? ["", "## Assistant notes", ...notes.slice(-NOTES_KEPT).map((text) => `- ${text}`)] : [];

  let first = 0;
  const render = () => {
    const shown = actions.slice(first);
    const omitted = actions.length - shown.length;
    const title = `## Actions, oldest first${omitted > 0 ? ` (${omitted} earlier actions omitted)` : ""}`;
    return [...head, "", title, ...shown.map((action, i) => renderAction(action, first + i)), ...tail].join("\n");
  };
  let text = render();
  while (text.length > options.maxChars && first < actions.length - 1) {
    first++;
    text = render();
  }
  return text.length > options.maxChars ? text.slice(0, options.maxChars) : text;
}
