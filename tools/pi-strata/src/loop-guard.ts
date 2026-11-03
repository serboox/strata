import { countLines, estimateTokens, fingerprint } from "./text.ts";

export interface GuardConfig {
  minChars: number;
  window: number;
  nudgeAt: number;
  blockAt: number;
  abortAfterBlocks: number;
  exempt: string[];
}

export const DEFAULT_GUARD: GuardConfig = {
  minChars: 160,
  window: 24,
  nudgeAt: 2,
  blockAt: 3,
  abortAfterBlocks: 3,
  exempt: ["edit", "write"],
};

export interface ResultInput {
  turn: number;
  tool: string;
  shape: string;
  text: string;
  identity?: string;
  isError: boolean;
}

export interface ResultVerdict {
  replacement?: string;
  duplicates: number;
}

export interface CallVerdict {
  reason: string;
  abort: boolean;
}

interface Sighting {
  turn: number;
  lines: number;
  tokens: number;
}

// Finds a run of identical tool results (a stuck agent retrying a command that cannot return more), swaps the
// repeats for a short stub, and refuses the same command once the run is long enough.
export class LoopGuard {
  private readonly config: GuardConfig;
  private seen = new Map<string, Sighting>();
  private order: string[] = [];
  private lastKey: string | undefined;
  private duplicates = 0;
  private loopShape: string | undefined;
  private blocked = 0;

  constructor(config: GuardConfig = DEFAULT_GUARD) {
    this.config = config;
  }

  reset(): void {
    this.seen.clear();
    this.order = [];
    this.lastKey = undefined;
    this.duplicates = 0;
    this.loopShape = undefined;
    this.blocked = 0;
  }

  get run(): number {
    return this.duplicates;
  }

  onResult(input: ResultInput): ResultVerdict {
    if (this.config.exempt.includes(input.tool)) {
      this.lastKey = undefined;
      this.duplicates = 0;
      this.loopShape = undefined;
      this.blocked = 0;
      return { duplicates: 0 };
    }
    const identity = input.identity ?? input.text;
    const long = identity.length >= this.config.minChars;
    const base = `${input.tool}:${input.isError ? "e" : "o"}:${fingerprint(identity)}`;
    const key = long ? base : `${base}:${input.shape}`;
    const earlier = long ? this.seen.get(key) : this.lastKey === key ? { turn: input.turn, lines: 0, tokens: 0 } : undefined;
    this.lastKey = key;

    if (earlier === undefined) {
      this.remember(key, input);
      this.duplicates = 0;
      this.loopShape = undefined;
      this.blocked = 0;
      return { duplicates: 0 };
    }

    if (this.loopShape !== input.shape) this.duplicates = 0;
    this.duplicates++;
    this.loopShape = input.shape;
    if (!long) return { duplicates: this.duplicates };
    return { replacement: this.stub(earlier), duplicates: this.duplicates };
  }

  beforeCall(tool: string, shape: string): CallVerdict | undefined {
    if (this.config.exempt.includes(tool)) return undefined;
    if (this.duplicates < this.config.blockAt || shape !== this.loopShape) return undefined;
    this.blocked++;
    const reason =
      `Refused: this ${tool} command was already run ${this.duplicates + 1} times with identical output. ` +
      "Running it again, or changing only a number in it, returns the same result. " +
      "Use the output you already have, or choose a different command.";
    return { reason, abort: this.blocked >= this.config.abortAfterBlocks };
  }

  private remember(key: string, input: ResultInput): void {
    if ((input.identity ?? input.text).length < this.config.minChars) return;
    this.seen.set(key, { turn: input.turn, lines: countLines(input.text), tokens: estimateTokens(input.text) });
    this.order.push(key);
    while (this.order.length > this.config.window) this.seen.delete(this.order.shift() as string);
  }

  private stub(earlier: Sighting): string {
    const base =
      `[output identical to the result of turn ${earlier.turn} (${earlier.lines} lines, about ${earlier.tokens} tokens); ` +
      "not repeated, it is already above in this conversation]";
    if (this.duplicates < this.config.nudgeAt) return base;
    return (
      `${base}\nThis is identical result number ${this.duplicates + 1} in a row. Repeating the command, or making it ` +
      "larger (a bigger head -N, another range), cannot show more than it already did. Use what you have, or take a different approach."
    );
  }
}
