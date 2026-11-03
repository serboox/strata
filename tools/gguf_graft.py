import argparse
import hashlib
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_reader

FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING_TYPE = 8
ARRAY_TYPE = 9
U32_TYPE = 4
COPY_CHUNK = 16 << 20
SYNC_EVERY = 256 << 20
GIB = 1 << 30
ALIGNMENT_KEY = "general.alignment"


def skip_value(buf: bytes, pos: int, value_type: int) -> int:
    if value_type in FIXED_SIZES:
        return pos + FIXED_SIZES[value_type]
    if value_type == STRING_TYPE:
        return pos + 8 + struct.unpack_from("<Q", buf, pos)[0]
    if value_type == ARRAY_TYPE:
        element_type, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if element_type in FIXED_SIZES:
            return pos + FIXED_SIZES[element_type] * count
        for _ in range(count):
            pos = skip_value(buf, pos, element_type)
        return pos
    raise ValueError(f"unknown GGUF value type {value_type}")


def read_string(buf: bytes, pos: int) -> tuple:
    length = struct.unpack_from("<Q", buf, pos)[0]
    return buf[pos + 8:pos + 8 + length].decode("utf-8"), pos + 8 + length


def pack_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def copy_range(source, start: int, size: int, put, chunk: int = COPY_CHUNK) -> None:
    """Pass exactly `size` bytes of `source` from `start` to `put` in chunks; a file that ends early is an error."""
    source.seek(start)
    remaining = size
    while remaining:
        data = source.read(min(chunk, remaining))
        if not data:
            raise OSError(f"{source.name} ended {remaining} bytes early")
        put(data)
        remaining -= len(data)
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


class Layout:
    """Where the parts of a GGUF v3 file sit: the metadata entries, the tensor descriptions and the start of the data."""

    def __init__(self, path: Path):
        self.path = path
        self.size = path.stat().st_size
        parsed = gguf_reader.GGUFFile(path)
        self.alignment = parsed.alignment
        self.data_start = parsed.data_start
        with open(path, "rb") as handle:
            head = handle.read(self.data_start)
        magic, version, tensor_count, kv_count = struct.unpack_from("<4sIQQ", head, 0)
        if magic != b"GGUF" or version != 3:
            raise ValueError(f"{path.name}: not a GGUF v3 file")
        self.tensor_count = tensor_count
        self.kv_count = kv_count
        self.kv_start = 24
        pos = self.kv_start
        self.entries = {}
        for _ in range(kv_count):
            key, pos = read_string(head, pos)
            value_type = struct.unpack_from("<I", head, pos)[0]
            pos += 4
            end = skip_value(head, pos, value_type)
            self.entries[key] = (value_type, pos, end)
            pos = end
        self.kv_end = pos
        for _ in range(tensor_count):
            _, pos = read_string(head, pos)
            dims = struct.unpack_from("<I", head, pos)[0]
            pos += 4 + 8 * dims + 4 + 8
        self.tensor_info_end = pos
        self.head = head[:pos]


def graft(base: Path, donor: Path, out: Path, prefix: str, set_u32: dict, add_u32: dict, progress=None) -> dict:
    """Write `out`: all of `base`, plus the tensors of `donor` whose names start with `prefix`, with some u32 metadata
    values changed (`set_u32`, the key must exist) or added (`add_u32`, the key must not exist).  Returns a summary."""
    layout = Layout(base)
    donor_file = gguf_reader.GGUFFile(donor)
    donor_data_start = donor_file.data_start
    base_names = {tensor.name for tensor in gguf_reader.GGUFFile(base).tensors}
    moved = [tensor for tensor in donor_file.tensors if tensor.name.startswith(prefix)]
    if not moved:
        raise ValueError(f"no tensor of {donor.name} starts with {prefix!r}")
    clash = sorted(base_names & {tensor.name for tensor in moved})
    if clash:
        raise ValueError(f"{base.name} already has {clash[:3]}")
    donor_size = donor.stat().st_size
    for tensor in moved:
        size = tensor.expected_bytes()
        if size is None or donor_data_start + tensor.offset + size > donor_size:
            raise ValueError(f"cannot size the tensor {tensor.name} ({tensor.type_name}) of {donor.name}")
    if ALIGNMENT_KEY in set_u32 or ALIGNMENT_KEY in add_u32:
        raise ValueError(f"{ALIGNMENT_KEY} cannot be changed: it decides where every tensor sits")
    for key in set_u32:
        if key not in layout.entries:
            raise ValueError(f"{base.name} has no metadata key {key}")
        if layout.entries[key][0] != U32_TYPE:
            raise ValueError(f"{key} is not a u32 value")
    for key in add_u32:
        if key in layout.entries:
            raise ValueError(f"{base.name} already has the metadata key {key}")

    alignment = layout.alignment
    base_data_size = layout.size - layout.data_start
    offsets = []
    cursor = align_up(base_data_size, alignment)
    for tensor in moved:
        offsets.append(cursor)
        cursor = align_up(cursor + tensor.expected_bytes(), alignment)
    total_data = cursor

    region = bytearray(layout.head[layout.kv_start:layout.kv_end])
    for key, value in set_u32.items():
        struct.pack_into("<I", region, layout.entries[key][1] - layout.kv_start, value)
    new_entries = b"".join(pack_string(key) + struct.pack("<II", U32_TYPE, value) for key, value in add_u32.items())
    new_infos = b"".join(
        pack_string(tensor.name) + struct.pack("<I", len(tensor.shape)) + b"".join(struct.pack("<Q", dim) for dim in tensor.shape)
        + struct.pack("<IQ", tensor.type_id, offset)
        for tensor, offset in zip(moved, offsets))
    header = (layout.head[:8] + struct.pack("<QQ", layout.tensor_count + len(moved), layout.kv_count + len(add_u32)) + bytes(region)
              + new_entries + layout.head[layout.kv_end:layout.tensor_info_end] + new_infos)
    data_start = align_up(len(header), alignment)

    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    unsynced = 0
    written = 0

    def put(data) -> None:
        nonlocal unsynced, written
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            view = view[count:]
            written += count
            unsynced += count
        if unsynced >= SYNC_EVERY:
            os.fdatasync(descriptor)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            unsynced = 0
            if progress is not None:
                progress(written, data_start + total_data)

    try:
        put(header + bytes(data_start - len(header)))
        with open(base, "rb") as source:
            copy_range(source, layout.data_start, base_data_size, put)
        put(bytes(align_up(base_data_size, alignment) - base_data_size))
        with open(donor, "rb") as source:
            for tensor, offset in zip(moved, offsets):
                size = tensor.expected_bytes()
                copy_range(source, donor_data_start + tensor.offset, size, put)
                put(bytes(align_up(offset + size, alignment) - offset - size))
        os.fdatasync(descriptor)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except BaseException:
        os.close(descriptor)
        os.unlink(out)
        raise
    os.close(descriptor)
    return {"tensors_added": len(moved), "bytes_added": sum(tensor.expected_bytes() for tensor in moved), "size": out.stat().st_size,
            "names": [tensor.name for tensor in moved]}


def digest(path: Path, start: int, size: int) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(start)
        remaining = size
        while remaining:
            chunk = handle.read(min(COPY_CHUNK, remaining))
            if not chunk:
                raise OSError(f"{path.name} ended early")
            sha.update(chunk)
            remaining -= len(chunk)
    return sha.hexdigest()


def verify(base: Path, donor: Path, out: Path, prefix: str, set_u32: dict | None = None, add_u32: dict | None = None) -> list:
    """Problems found in `out` as a graft of `base` and `donor`: an empty list means none.  The metadata must be the base's with
    the given changes, every tensor must have the type and shape of its source, and every grafted tensor and a few base tensors
    (the first, the middle, the last by offset) are compared byte for byte."""
    problems = []
    result = gguf_reader.GGUFFile(out)
    base_file = gguf_reader.GGUFFile(base)
    donor_file = gguf_reader.GGUFFile(donor)
    expected = {tensor.name for tensor in base_file.tensors} | {tensor.name for tensor in donor_file.tensors if tensor.name.startswith(prefix)}
    names = [tensor.name for tensor in result.tensors]
    if set(names) != expected or len(names) != len(expected):
        problems.append("the tensor names differ from base plus the grafted ones")
        return problems
    expected_metadata = dict(base_file.metadata)
    expected_metadata.update(set_u32 or {})
    expected_metadata.update(add_u32 or {})
    if result.metadata != expected_metadata:
        keys = sorted(key for key in set(result.metadata) | set(expected_metadata) if result.metadata.get(key) != expected_metadata.get(key))
        problems.append(f"the metadata differs from the base plus the changes: {keys[:5]}")
    sources = {tensor.name: tensor for tensor in base_file.tensors}
    sources.update({tensor.name: tensor for tensor in donor_file.tensors if tensor.name.startswith(prefix)})
    for tensor in result.tensors:
        source = sources[tensor.name]
        if (tensor.type_id, tensor.shape) != (source.type_id, source.shape):
            problems.append(f"{tensor.name}: the type or shape differs from the source")
    by_name = {tensor.name: tensor for tensor in result.tensors}
    out_size = out.stat().st_size
    for tensor in result.tensors:
        size = tensor.expected_bytes()
        if size is None or tensor.offset % result.alignment or result.data_start + tensor.offset + size > out_size:
            problems.append(f"{tensor.name}: bad offset or size")
    for tensor in donor_file.tensors:
        if tensor.name.startswith(prefix):
            size = tensor.expected_bytes()
            if digest(donor, donor_file.data_start + tensor.offset, size) != digest(out, result.data_start + by_name[tensor.name].offset, size):
                problems.append(f"{tensor.name}: bytes differ from the donor")
    ordered = sorted(base_file.tensors, key=lambda tensor: tensor.offset)
    for tensor in ordered[:1] + ordered[len(ordered) // 2:len(ordered) // 2 + 1] + ordered[-1:]:
        size = tensor.expected_bytes()
        if size is not None and digest(base, base_file.data_start + tensor.offset, size) != digest(out, result.data_start + by_name[tensor.name].offset, size):
            problems.append(f"{tensor.name}: bytes differ from the base")
    return problems


def parse_assignments(items: list, parser: argparse.ArgumentParser) -> dict:
    values = {}
    for item in items:
        match = re.fullmatch(r"([\w.\-]+)=(\d+)", item)
        if match is None:
            parser.error(f"expected KEY=NUMBER, not {item!r}")
        values[match.group(1)] = int(match.group(2))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a GGUF file and graft the tensors of another one onto it (the MTP block "
                                                 "that a re-quantized file lacks), changing or adding u32 metadata.")
    parser.add_argument("base", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prefix", required=True, help="graft the donor tensors whose names start with this, e.g. blk.40.")
    parser.add_argument("--set-u32", action="append", default=[], metavar="KEY=N", help="change an existing u32 metadata value")
    parser.add_argument("--add-u32", action="append", default=[], metavar="KEY=N", help="add a u32 metadata value")
    parser.add_argument("--no-verify", action="store_true", help="skip the byte comparison of the grafted tensors")
    args = parser.parse_args()
    set_u32 = parse_assignments(args.set_u32, parser)
    add_u32 = parse_assignments(args.add_u32, parser)

    def progress(done: int, total: int) -> None:
        print(f"gguf-graft: {done / GIB:.1f} of {total / GIB:.1f} GiB", file=sys.stderr, flush=True)

    summary = graft(args.base, args.donor, args.out, args.prefix, set_u32, add_u32, progress)
    print(f"gguf-graft: wrote {args.out} ({summary['size'] / GIB:.2f} GiB), {summary['tensors_added']} tensors from "
          f"{args.donor.name} ({summary['bytes_added'] / GIB:.2f} GiB)", file=sys.stderr)
    if args.no_verify:
        return 0
    problems = verify(args.base, args.donor, args.out, args.prefix, set_u32, add_u32)
    for problem in problems:
        print(f"gguf-graft: PROBLEM {problem}", file=sys.stderr)
    print("gguf-graft: verified" if not problems else "gguf-graft: verification FAILED", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
