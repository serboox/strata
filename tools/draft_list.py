#!/usr/bin/env python3
import argparse
import collections
import struct
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the token list for STRATA_Q36_DRAFT_LIST (--draft-list): the most frequent token ids of "
                                                 "recorded generations, as little-endian int32. Input files hold token ids separated by whitespace.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=24000, help="how many of the most frequent ids to keep (default 24000)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", action="store_true", help="print the share of all tokens covered by the list")
    args = parser.parse_args()
    counts: collections.Counter = collections.Counter()
    for path in args.inputs:
        counts.update(int(word) for word in path.read_text(encoding="ascii").split() if word.lstrip("-").isdigit())
    if not counts:
        print("draft_list: no token ids in the inputs", file=sys.stderr)
        return 1
    kept = [token for token, _ in counts.most_common(args.top)]
    args.out.write_bytes(struct.pack(f"<{len(kept)}i", *kept))
    if args.report:
        total = sum(counts.values())
        covered = sum(counts[token] for token in kept)
        print(f"{len(kept)} ids cover {covered / total:.2%} of {total} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
