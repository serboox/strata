import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import gguf_graft
import gguf_reader

ALIGNMENT = 32
F32, Q8_0 = 0, 8


def pack_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, entries: list, tensors: list, alignment: int = ALIGNMENT, pad_last: bool = True) -> None:
    """entries: (key, ("u32", n) | ("str", text) | ("strs", [text, ...])); tensors: (name, dims, type id, payload bytes)."""
    kv = b""
    for key, (kind, value) in entries:
        kv += pack_string(key)
        if kind == "u32":
            kv += struct.pack("<II", 4, value)
        elif kind == "str":
            kv += struct.pack("<I", 8) + pack_string(value)
        else:
            kv += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(pack_string(item) for item in value)
    infos, data, offset = b"", b"", 0
    for position, (name, dims, type_id, payload) in enumerate(tensors):
        infos += pack_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", dim) for dim in dims) + struct.pack("<IQ", type_id, offset)
        padded = (len(payload) + alignment - 1) // alignment * alignment
        if position == len(tensors) - 1 and not pad_last:
            padded = len(payload)
        data += payload + bytes(padded - len(payload))
        offset += padded
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(entries)) + kv + infos
    head += bytes((alignment - len(head) % alignment) % alignment)
    path.write_bytes(head + data)


def payload(seed: int, size: int) -> bytes:
    return bytes((seed * 31 + index * 7) % 251 for index in range(size))


class GraftTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.dir)], check=False))
        self.base = self.dir / "base.gguf"
        self.donor = self.dir / "donor.gguf"
        self.out = self.dir / "out.gguf"
        write_gguf(self.base, [("general.architecture", ("str", "toy")), ("toy.block_count", ("u32", 2)),
                               ("tokenizer.ggml.tokens", ("strs", ["a", "bb", "ccc"])), ("toy.note", ("str", "keep"))],
                   [("token_embd.weight", [8, 2], F32, payload(1, 64)), ("blk.0.w", [8], F32, payload(2, 32)),
                    ("blk.1.w", [64], Q8_0, payload(3, 68))])
        write_gguf(self.donor, [("general.architecture", ("str", "toy"))],
                   [("blk.0.w", [8], F32, payload(9, 32)), ("blk.2.a", [16], F32, payload(4, 64)),
                    ("blk.2.q", [64], Q8_0, payload(5, 68)), ("blk.2.n", [4], F32, payload(6, 16))])

    def run_graft(self, **overrides):
        arguments = {"base": self.base, "donor": self.donor, "out": self.out, "prefix": "blk.2.", "set_u32": {"toy.block_count": 3},
                     "add_u32": {"toy.nextn_predict_layers": 1}}
        arguments.update(overrides)
        return gguf_graft.graft(**arguments)

    def test_the_tensors_and_the_metadata_arrive(self):
        summary = self.run_graft()
        self.assertEqual(summary["tensors_added"], 3)
        result = gguf_reader.GGUFFile(self.out)
        self.assertEqual([t.name for t in result.tensors], ["token_embd.weight", "blk.0.w", "blk.1.w", "blk.2.a", "blk.2.q", "blk.2.n"])
        self.assertEqual(result.metadata["toy.block_count"], 3)
        self.assertEqual(result.metadata["toy.nextn_predict_layers"], 1)
        self.assertEqual(result.metadata["tokenizer.ggml.tokens"], ["a", "bb", "ccc"])
        self.assertEqual(result.metadata["toy.note"], "keep")

    def test_every_byte_is_where_the_description_says(self):
        self.run_graft()
        result = gguf_reader.GGUFFile(self.out)
        raw = self.out.read_bytes()
        expected = {"token_embd.weight": payload(1, 64), "blk.0.w": payload(2, 32), "blk.1.w": payload(3, 68),
                    "blk.2.a": payload(4, 64), "blk.2.q": payload(5, 68), "blk.2.n": payload(6, 16)}
        for tensor in result.tensors:
            start = result.data_start + tensor.offset
            self.assertEqual(raw[start:start + tensor.expected_bytes()], expected[tensor.name], tensor.name)

    def test_offsets_and_the_data_start_stay_aligned(self):
        self.run_graft()
        result = gguf_reader.GGUFFile(self.out)
        self.assertEqual(result.data_start % ALIGNMENT, 0)
        self.assertTrue(all(tensor.offset % ALIGNMENT == 0 for tensor in result.tensors))

    def test_the_donor_tensor_of_a_name_the_prefix_excludes_is_left_behind(self):
        self.run_graft()
        names = [t.name for t in gguf_reader.GGUFFile(self.out).tensors]
        self.assertEqual(names.count("blk.0.w"), 1)

    def test_verify_accepts_a_good_graft_and_catches_a_damaged_one(self):
        self.run_graft()
        self.assertEqual(gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3}, {"toy.nextn_predict_layers": 1}), [])
        result = gguf_reader.GGUFFile(self.out)
        target = next(t for t in result.tensors if t.name == "blk.2.q")
        raw = bytearray(self.out.read_bytes())
        raw[result.data_start + target.offset + 5] ^= 0xFF
        self.out.write_bytes(bytes(raw))
        problems = gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3}, {"toy.nextn_predict_layers": 1})
        self.assertEqual(problems, ["blk.2.q: bytes differ from the donor"])

    def test_a_base_with_a_wider_alignment_keeps_it(self):
        write_gguf(self.base, [("general.alignment", ("u32", 64)), ("toy.block_count", ("u32", 2))],
                   [("a", [8], F32, payload(1, 32)), ("b", [64], Q8_0, payload(2, 68))], alignment=64)
        write_gguf(self.donor, [("general.alignment", ("u32", 64))], [("blk.2.a", [16], F32, payload(4, 64))], alignment=64)
        self.run_graft(add_u32={})
        result = gguf_reader.GGUFFile(self.out)
        self.assertEqual(result.alignment, 64)
        self.assertEqual(result.data_start % 64, 0)
        self.assertTrue(all(tensor.offset % 64 == 0 for tensor in result.tensors))
        self.assertEqual(gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3}), [])

    def test_a_base_whose_last_tensor_is_not_padded_still_grafts_aligned(self):
        write_gguf(self.base, [("toy.block_count", ("u32", 2))], [("a", [8], F32, payload(1, 32)), ("b", [64], Q8_0, payload(2, 68))], pad_last=False)
        self.run_graft(add_u32={})
        result = gguf_reader.GGUFFile(self.out)
        self.assertTrue(all(tensor.offset % ALIGNMENT == 0 for tensor in result.tensors))
        self.assertEqual(gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3}), [])

    def test_the_alignment_key_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            self.run_graft(add_u32={"general.alignment": 64})
        self.assertFalse(self.out.exists())

    def test_verify_checks_the_metadata_and_the_tensor_descriptions(self):
        self.run_graft()
        self.assertEqual(gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3}, {"toy.nextn_predict_layers": 1}), [])
        wrong = gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 4}, {"toy.nextn_predict_layers": 1})
        self.assertTrue(any("metadata differs" in problem and "toy.block_count" in problem for problem in wrong))
        missing = gguf_graft.verify(self.base, self.donor, self.out, "blk.2.", {"toy.block_count": 3})
        self.assertTrue(any("toy.nextn_predict_layers" in problem for problem in missing))

    def test_a_donor_that_ends_early_stops_the_copy_and_removes_the_output(self):
        with mock.patch.object(gguf_graft, "copy_range", side_effect=OSError("ended early")), self.assertRaises(OSError):
            self.run_graft()
        self.assertFalse(self.out.exists())

    def test_copy_range_is_exact_chunked_and_strict(self):
        source_path = self.dir / "blob.bin"
        source_path.write_bytes(bytes(range(100)))
        pieces = []
        with open(source_path, "rb") as source:
            gguf_graft.copy_range(source, 10, 50, pieces.append, chunk=16)
        self.assertEqual(b"".join(pieces), bytes(range(10, 60)))
        self.assertEqual([len(piece) for piece in pieces], [16, 16, 16, 2])
        with open(source_path, "rb") as source, self.assertRaisesRegex(OSError, "20 bytes early"):
            gguf_graft.copy_range(source, 90, 30, lambda data: None)

    def test_nothing_to_graft_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no tensor"):
            self.run_graft(prefix="blk.9.")
        self.assertFalse(self.out.exists())

    def test_a_tensor_that_the_base_already_has_is_refused(self):
        with self.assertRaisesRegex(ValueError, "already has"):
            self.run_graft(prefix="blk.0.")
        self.assertFalse(self.out.exists())

    def test_metadata_mistakes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no metadata key"):
            self.run_graft(set_u32={"toy.absent": 1})
        with self.assertRaisesRegex(ValueError, "not a u32"):
            self.run_graft(set_u32={"toy.note": 1})
        with self.assertRaisesRegex(ValueError, "already has the metadata key"):
            self.run_graft(add_u32={"toy.block_count": 1})
        self.assertFalse(self.out.exists())

    def test_an_existing_output_is_never_overwritten(self):
        self.out.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            self.run_graft()
        self.assertEqual(self.out.read_bytes(), b"keep me")

    def test_a_donor_cut_short_is_refused_before_anything_is_written(self):
        cut = self.dir / "cut.gguf"
        cut.write_bytes(self.donor.read_bytes()[:-40])
        with self.assertRaisesRegex(ValueError, "cannot size"):
            self.run_graft(donor=cut)
        self.assertFalse(self.out.exists())

    def test_the_command_line_grafts_and_verifies(self):
        result = subprocess.run([sys.executable, str(TOOLS / "gguf_graft.py"), str(self.base), str(self.donor), str(self.out), "--prefix", "blk.2.",
                                 "--set-u32", "toy.block_count=3", "--add-u32", "toy.nextn_predict_layers=1"],
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gguf-graft: verified", result.stderr)
        self.assertEqual(gguf_reader.GGUFFile(self.out).metadata["toy.block_count"], 3)

    def test_a_malformed_assignment_is_refused_by_the_command_line(self):
        result = subprocess.run([sys.executable, str(TOOLS / "gguf_graft.py"), str(self.base), str(self.donor), str(self.out), "--prefix", "blk.2.",
                                 "--set-u32", "toy.block_count"], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
