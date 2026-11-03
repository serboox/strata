import argparse
import os
import sys
import time
from pathlib import Path

CHUNK_BYTES = 8 << 20
REPORT_EVERY_BYTES = 4 << 30
GIB = 1 << 30
COUNTED_SHARE = 0.9   # below this share of the bytes read, the scope owns too little of the file to protect it
CGROUP_ROOT = Path("/sys/fs/cgroup")


def warm(path: str, limit: int | None = None, chunk: int = CHUNK_BYTES, progress=None) -> int:
    """Read the start of a file once so its pages sit in the page cache of the control group this process runs in.

    Pages already cached are dropped first, as far as the kernel lets go of them: a page that another process has mapped stays.
    Returns the number of bytes read. Reading is sequential and the data is thrown away."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        total = os.fstat(descriptor).st_size
        wanted = total if limit is None else min(total, limit)
        if wanted <= 0:
            return 0
        # A cached page stays charged to the control group that read it first; dropping it makes this group the owner.
        os.posix_fadvise(descriptor, 0, wanted, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        buffer = bytearray(chunk)
        done = 0
        while done < wanted:
            count = os.preadv(descriptor, [memoryview(buffer)[:min(chunk, wanted - done)]], done)
            if count <= 0:
                break
            done += count
            if progress is not None:
                progress(done, wanted)
        return done
    finally:
        os.close(descriptor)


def scope_cache_bytes(root: Path = CGROUP_ROOT, own: str | None = None) -> int | None:
    """Bytes of page cache counted in the control group this process runs in, or None when they cannot be read."""
    try:
        if own is None:
            own = Path("/proc/self/cgroup").read_text(encoding="ascii").split(":", 2)[2].strip()
        for line in (root / own.lstrip("/") / "memory.stat").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(" ")
            if key == "file":
                return int(value)
    except (OSError, IndexError, ValueError):
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a model file into the page cache, then become the server command after --.")
    parser.add_argument("--limit", type=int, help="read at most this many bytes")
    parser.add_argument("model")
    parser.add_argument("server", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.server[1:] if args.server[:1] == ["--"] else args.server
    if not command:
        parser.error("no server command after --")
    reported = [0]

    def progress(done: int, wanted: int) -> None:
        if done - reported[0] >= REPORT_EVERY_BYTES:
            reported[0] = done
            print(f"strata-prewarm: {done / GIB:.1f} of {wanted / GIB:.1f} GiB", file=sys.stderr, flush=True)

    started = time.monotonic()
    before = scope_cache_bytes()
    try:
        read = warm(args.model, args.limit, progress=progress)
    except OSError as error:
        print(f"strata-prewarm: could not read {args.model}: {error}; starting the server anyway", file=sys.stderr, flush=True)
    else:
        after = scope_cache_bytes()
        counted = None if before is None or after is None else max(0, after - before)
        message = f"strata-prewarm: {read / GIB:.1f} GiB read in {time.monotonic() - started:.0f} s"
        if counted is not None:
            message += f", {counted / GIB:.1f} GiB of it counted in this scope"
        print(message, file=sys.stderr, flush=True)
        if counted is not None and read > 0 and counted < COUNTED_SHARE * read:
            print("strata-prewarm: part of the file stays counted in another control group (a server that has it mapped, for "
                  "example), so the protection of this scope will be partial", file=sys.stderr, flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
