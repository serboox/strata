import contextlib
import io
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import gguf_graft
import gguf_reader
import requant_dense as rd

ALIGNMENT = 32
F32, Q8_0, Q6_K, IQ4_XS = 0, 8, 14, 23
IQ4_XS_BLOCK_BYTES = 136
GGML, GGML_UNAVAILABLE_REASON = rd.load_ggml_reference()


def random_blocks(rng: np.random.Generator, count: int) -> np.ndarray:
    scales = 10 ** rng.uniform(-3, 1, (count, 1))
    return (rng.normal(0, 1, (count, rd.Q6_K_BLOCK_WEIGHTS)) * scales).astype(np.float32)


class QuantiserTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_block_sizes_agree_with_the_gguf_reader(self):
        self.assertEqual(gguf_reader.BLOCK_GEOMETRY["Q6_K"], (rd.Q6_K_BLOCK_WEIGHTS, rd.Q6_K_BLOCK_BYTES))
        self.assertEqual(gguf_reader.BLOCK_GEOMETRY["Q8_0"], (rd.Q8_0_BLOCK_WEIGHTS, rd.Q8_0_BLOCK_BYTES))
        self.assertEqual(gguf_reader.GGML_TYPES[rd.Q6_K_TYPE_ID], "Q6_K")
        self.assertEqual(gguf_reader.GGML_TYPES[rd.Q8_0_TYPE_ID], "Q8_0")

    def test_scalar_port_and_vectorised_quantiser_give_the_same_bytes(self):
        sample = np.concatenate([random_blocks(self.rng, 60), np.stack([values for _, values in rd.structured_blocks()])])
        fast = rd.quantize_q6_k(sample)
        slow = np.stack([np.frombuffer(rd.reference_quantize_q6_k_block(row), dtype=np.uint8) for row in sample])
        mismatched = np.flatnonzero((fast != slow).any(axis=1))
        self.assertEqual(len(mismatched), 0, f"blocks differing from the scalar port: {mismatched[:10]}")

    @unittest.skipIf(GGML is None, f"ggml reference unavailable: {GGML_UNAVAILABLE_REASON}")
    def test_quantiser_matches_the_real_ggml_bytes(self):
        heavy_tails = (self.rng.standard_t(3, (500, rd.Q6_K_BLOCK_WEIGHTS)) * 0.02).astype(np.float32)
        cases = [("random", random_blocks(self.rng, 3000)), ("heavy tails", heavy_tails)] + [
            (label, values.reshape(1, -1)) for label, values in rd.structured_blocks()]
        for label, values in cases:
            ours, theirs = rd.quantize_q6_k(values), GGML.quantize_q6_k(values)
            differing = int((ours != theirs).any(axis=1).sum())
            self.assertEqual(differing, 0, f"{label}: {differing} of {len(values)} blocks differ from ggml")

    @unittest.skipIf(GGML is None, f"ggml reference unavailable: {GGML_UNAVAILABLE_REASON}")
    def test_dequantisers_and_the_q8_0_quantiser_match_ggml(self):
        values = random_blocks(self.rng, 400)
        packed = GGML.quantize_q6_k(values)
        np.testing.assert_array_equal(rd.dequantize_q6_k(packed), GGML.dequantize_q6_k(packed))
        q8_values = values.reshape(-1, rd.Q8_0_BLOCK_WEIGHTS)
        np.testing.assert_array_equal(rd.quantize_q8_0(q8_values), GGML.quantize_q8_0(q8_values))
        np.testing.assert_array_equal(rd.dequantize_q8_0(GGML.quantize_q8_0(q8_values)), GGML.dequantize_q8_0(GGML.quantize_q8_0(q8_values)))

    def test_the_result_does_not_depend_on_how_the_blocks_are_sliced(self):
        values = random_blocks(self.rng, rd.SLICE_BLOCKS * 2 + 37)
        whole = rd.quantize_q6_k(values)
        pieces = np.concatenate([rd.quantize_q6_k(values[:1]), rd.quantize_q6_k(values[1:rd.SLICE_BLOCKS + 5]), rd.quantize_q6_k(values[rd.SLICE_BLOCKS + 5:])])
        np.testing.assert_array_equal(whole, pieces)

    def test_zero_blocks_give_zero_bytes_and_no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            zeros = np.zeros((5, rd.Q6_K_BLOCK_WEIGHTS), dtype=np.float32)
            packed = rd.quantize_q6_k(zeros)
            restored = rd.dequantize_q6_k(packed)
        self.assertFalse(packed.any())
        self.assertTrue(np.isfinite(restored).all())
        self.assertFalse(restored.any())

    def test_a_block_with_one_zero_group_keeps_that_group_at_zero(self):
        values = random_blocks(self.rng, 1)
        values[:, 32:48] = 0
        restored = rd.dequantize_q6_k(rd.quantize_q6_k(values))
        self.assertFalse(restored[:, 32:48].any())
        self.assertTrue(restored[:, :32].any())

    def test_q8_0_blocks_dequantise_as_scale_times_quant(self):
        quants = np.arange(-16, 16, dtype=np.int8)
        block = np.frombuffer(np.float16(0.5).tobytes() + quants.tobytes(), dtype=np.uint8).reshape(1, rd.Q8_0_BLOCK_BYTES)
        np.testing.assert_array_equal(rd.dequantize_q8_0(block)[0], quants.astype(np.float32) * np.float32(0.5))

    def test_q6_k_round_trip_stays_within_the_expected_error(self):
        weights = self.rng.normal(0, 0.02, (1500, rd.Q6_K_BLOCK_WEIGHTS)).astype(np.float32)
        q8_values = rd.dequantize_q8_0(rd.quantize_q8_0(weights.reshape(-1, rd.Q8_0_BLOCK_WEIGHTS))).reshape(-1, rd.Q6_K_BLOCK_WEIGHTS)
        q6_values = rd.dequantize_q6_k(rd.quantize_q6_k(q8_values))
        self.assertLess(rd.relative_rms_error(weights, q8_values), 0.01)
        self.assertLess(rd.relative_rms_error(q8_values, q6_values), rd.SELFTEST_MAX_ROUND_TRIP_ERROR)

    def test_a_wrong_block_shape_is_refused(self):
        with self.assertRaises(ValueError):
            rd.quantize_q6_k(np.zeros((4, 100), dtype=np.float32))

    def test_selftest_runs_and_reports(self):
        report = io.StringIO()
        self.assertTrue(rd.selftest(300, report, benchmark_blocks=1000))
        text = report.getvalue()
        for expected in ("scalar port vs numpy", "round trip, gaussian", "blocks/s", "selftest: ok"):
            self.assertIn(expected, text)


def pack_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, entries: list, tensors: list, alignment: int = ALIGNMENT) -> None:
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
    for name, dims, type_id, payload in tensors:
        infos += pack_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", dim) for dim in dims) + struct.pack("<IQ", type_id, offset)
        padded = (len(payload) + alignment - 1) // alignment * alignment
        data += payload + bytes(padded - len(payload))
        offset += padded
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(entries)) + kv + infos
    head += bytes((alignment - len(head) % alignment) % alignment)
    path.write_bytes(head + data)


def q8_0_payload(rng: np.random.Generator, weights: int) -> bytes:
    return rd.quantize_q8_0(rng.normal(0, 0.05, (weights // rd.Q8_0_BLOCK_WEIGHTS, rd.Q8_0_BLOCK_WEIGHTS)).astype(np.float32)).tobytes()


def opaque_payload(rng: np.random.Generator, size: int) -> bytes:
    return rng.integers(0, 256, size, dtype=np.uint8).tobytes()


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.dir)], check=False))
        self.source = self.dir / "src.gguf"
        self.dest = self.dir / "dst.gguf"
        self.part = self.dir / "dst.gguf.part"
        self.entries = [("general.architecture", ("str", "toy")), ("toy.block_count", ("u32", 2)),
                        ("tokenizer.ggml.tokens", ("strs", ["a", "bb", "ccc"])), ("toy.note", ("str", "keep"))]
        self.make_source(self.source)

    def make_source(self, path: Path, alignment: int = ALIGNMENT, entries: list | None = None) -> None:
        rng = np.random.default_rng(11)
        self.payloads = {
            "token_embd.weight": q8_0_payload(rng, 256 * 16),
            "blk.0.attn_qkv.weight": q8_0_payload(rng, 512 * 8),
            "blk.0.attn_norm.weight": opaque_payload(rng, 64 * 4),
            "blk.0.ffn_gate_inp.weight": opaque_payload(rng, 64 * 4 * 4),
            "blk.0.ffn_gate_exps.weight": opaque_payload(rng, IQ4_XS_BLOCK_BYTES * 8),
            "blk.0.ffn_gate_shexp.weight": q8_0_payload(rng, 256 * 4),
            "blk.0.ssm_out.weight": q8_0_payload(rng, 96 * 4),
            "blk.1.nextn.eh_proj.weight": q8_0_payload(rng, 256 * 2),
            "output.weight": opaque_payload(rng, rd.Q6_K_BLOCK_BYTES * 4),
        }
        shapes = {"token_embd.weight": ([256, 16], Q8_0), "blk.0.attn_qkv.weight": ([512, 8], Q8_0), "blk.0.attn_norm.weight": ([64], F32),
                  "blk.0.ffn_gate_inp.weight": ([64, 4], F32), "blk.0.ffn_gate_exps.weight": ([256, 4, 2], IQ4_XS),
                  "blk.0.ffn_gate_shexp.weight": ([256, 4], Q8_0), "blk.0.ssm_out.weight": ([96, 4], Q8_0),
                  "blk.1.nextn.eh_proj.weight": ([256, 2], Q8_0), "output.weight": ([256, 4], Q6_K)}
        tensors = [(name, shapes[name][0], shapes[name][1], payload) for name, payload in self.payloads.items()]
        write_gguf(path, self.entries if entries is None else entries, tensors, alignment)

    def convert(self, groups=rd.DEFAULT_GROUPS, dest: Path | None = None, **options) -> rd.Plan:
        return rd.convert(self.source, dest or self.dest, groups, **options)

    def tensor_bytes(self, path: Path, name: str, parsed: gguf_reader.GGUFFile | None = None) -> bytes:
        parsed = parsed or gguf_reader.GGUFFile(path)
        tensor = next(item for item in parsed.tensors if item.name == name)
        with open(path, "rb") as handle:
            handle.seek(parsed.data_start + tensor.offset)
            return handle.read(tensor.expected_bytes())

    def run_cli(self, *arguments) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rd.main([str(argument) for argument in arguments])
        return code, out.getvalue(), err.getvalue()

    def test_default_groups_change_the_types_of_the_selected_q8_0_tensors_only(self):
        self.convert()
        before, after = gguf_reader.GGUFFile(self.source), gguf_reader.GGUFFile(self.dest)
        self.assertEqual([tensor.name for tensor in after.tensors], [tensor.name for tensor in before.tensors])
        converted = {"blk.0.attn_qkv.weight", "blk.0.ffn_gate_shexp.weight"}
        for old, new in zip(before.tensors, after.tensors):
            self.assertEqual(new.shape, old.shape, old.name)
            self.assertEqual(new.type_id, Q6_K if old.name in converted else old.type_id, old.name)
        self.assertEqual(after.metadata, before.metadata)

    def test_untouched_tensors_are_byte_identical(self):
        self.convert()
        for name in ("token_embd.weight", "blk.0.attn_norm.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
                     "blk.0.ssm_out.weight", "blk.1.nextn.eh_proj.weight", "output.weight"):
            self.assertEqual(self.tensor_bytes(self.dest, name), self.payloads[name], name)

    def test_converted_tensors_are_the_q6_k_quantisation_of_the_q8_0_values(self):
        self.convert()
        for name in ("blk.0.attn_qkv.weight", "blk.0.ffn_gate_shexp.weight"):
            source_values = rd.dequantize_q8_0(np.frombuffer(self.payloads[name], dtype=np.uint8).reshape(-1, rd.Q8_0_BLOCK_BYTES)).reshape(-1, rd.Q6_K_BLOCK_WEIGHTS)
            written = self.tensor_bytes(self.dest, name)
            self.assertEqual(written, rd.quantize_q6_k(source_values).tobytes(), name)
            restored = rd.dequantize_q6_k(np.frombuffer(written, dtype=np.uint8).reshape(-1, rd.Q6_K_BLOCK_BYTES))
            self.assertLess(rd.relative_rms_error(source_values, restored), 0.03, name)

    def test_offsets_and_the_data_start_are_aligned_and_ascending(self):
        self.convert()
        result = gguf_reader.GGUFFile(self.dest)
        self.assertEqual(result.data_start % ALIGNMENT, 0)
        previous_end = 0
        for tensor in result.tensors:
            self.assertEqual(tensor.offset % ALIGNMENT, 0, tensor.name)
            self.assertGreaterEqual(tensor.offset, previous_end, tensor.name)
            previous_end = tensor.offset + tensor.expected_bytes()
        self.assertLessEqual(result.data_start + previous_end, self.dest.stat().st_size)

    def test_the_file_size_is_the_one_the_dry_run_predicts(self):
        code, report, _ = self.run_cli(self.source, self.dest, "--dry-run")
        self.assertEqual(code, 0)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.part.exists())
        predicted = int(re.search(r"new file size: (\d+) bytes", report).group(1))
        self.convert()
        self.assertEqual(self.dest.stat().st_size, predicted)
        self.assertIn("would fit", report)
        self.assertIn("blk.0.ssm_out.weight", report)
        self.assertIn("not a multiple of 256", report)

    def test_the_dry_run_counts_bytes_per_group(self):
        _, report, _ = self.run_cli(self.source, self.dest, "--dry-run")
        qkv_before = len(self.payloads["blk.0.attn_qkv.weight"])
        qkv_after = 512 * 8 // rd.Q6_K_BLOCK_WEIGHTS * rd.Q6_K_BLOCK_BYTES
        row = next(line for line in report.splitlines() if line.startswith("ssm "))
        self.assertEqual(row.split()[1:4], ["1", str(qkv_before), str(qkv_after)])
        self.assertRegex(next(line for line in report.splitlines() if line.startswith("attn ")), r"attn\s+0\s+0\s+0")

    def test_the_metadata_bytes_are_copied_verbatim(self):
        self.convert()
        layout = gguf_graft.Layout(self.source)
        self.assertEqual(self.dest.read_bytes()[:layout.kv_end], layout.head[:layout.kv_end])

    def test_no_part_file_is_left_after_success(self):
        self.convert()
        self.assertTrue(self.dest.exists())
        self.assertFalse(self.part.exists())

    def test_an_existing_destination_is_refused_and_left_alone(self):
        self.dest.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(self.dest.read_bytes(), b"keep me")
        code, _, err = self.run_cli(self.source, self.dest)
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        self.assertEqual(self.run_cli(self.source, self.dest, "--dry-run")[0], 1)

    def test_a_stale_part_file_is_not_overwritten_or_deleted(self):
        self.part.write_bytes(b"stale")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(self.part.read_bytes(), b"stale")
        self.assertFalse(self.dest.exists())

    def test_all_groups_also_convert_the_embedding_and_the_mtp_projection(self):
        code, _, err = self.run_cli(self.source, self.dest, "--groups", "all")
        self.assertEqual(code, 0, err)
        result = {tensor.name: tensor for tensor in gguf_reader.GGUFFile(self.dest).tensors}
        for name in ("token_embd.weight", "blk.1.nextn.eh_proj.weight", "blk.0.attn_qkv.weight", "blk.0.ffn_gate_shexp.weight"):
            self.assertEqual(result[name].type_id, Q6_K, name)
        self.assertEqual(result["blk.0.ssm_out.weight"].type_id, Q8_0)
        self.assertEqual(self.tensor_bytes(self.dest, "output.weight"), self.payloads["output.weight"])
        self.assertEqual(self.tensor_bytes(self.dest, "blk.0.ffn_gate_exps.weight"), self.payloads["blk.0.ffn_gate_exps.weight"])

    def test_a_selected_tensor_that_is_not_q8_0_is_reported_as_kept(self):
        _, report, _ = self.run_cli(self.source, self.dest, "--groups", "head,embd", "--dry-run")
        self.assertIn("output.weight: kept (type Q6_K)", report)
        self.assertNotIn("token_embd.weight: kept", report)

    def test_the_chunked_path_gives_the_same_bytes_as_one_chunk(self):
        self.convert(groups=tuple(rd.GROUPS))
        reference = self.dest.read_bytes()
        for chunk_bytes in (1, 3 * 512 * 4, 1 << 30):
            other = self.dir / f"chunk_{chunk_bytes}.gguf"
            self.convert(groups=tuple(rd.GROUPS), dest=other, chunk_bytes=chunk_bytes)
            self.assertEqual(other.read_bytes(), reference, f"chunk of {chunk_bytes} bytes")

    def test_the_chunk_option_of_the_command_line_is_honoured(self):
        self.assertEqual(self.run_cli(self.source, self.dest, "--chunk-mib", "0.0001")[0], 0)
        other = self.dir / "default.gguf"
        self.convert(dest=other)
        self.assertEqual(self.dest.read_bytes(), other.read_bytes())

    def test_a_failure_in_the_middle_removes_the_part_file(self):
        with mock.patch.object(rd, "quantize_q6_k", side_effect=RuntimeError("boom")), self.assertRaises(RuntimeError):
            self.convert()
        self.assertFalse(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_a_non_finite_q8_0_scale_is_an_error_without_leftovers(self):
        layout = gguf_graft.Layout(self.source)
        parsed = gguf_reader.GGUFFile(self.source)
        target = next(tensor for tensor in parsed.tensors if tensor.name == "blk.0.attn_qkv.weight")
        raw = bytearray(self.source.read_bytes())
        raw[layout.data_start + target.offset:layout.data_start + target.offset + 2] = np.float16(np.inf).tobytes()
        self.source.write_bytes(bytes(raw))
        code, _, err = self.run_cli(self.source, self.dest)
        self.assertEqual(code, 1)
        self.assertIn("non-finite", err)
        self.assertFalse(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_a_selection_that_converts_nothing_is_refused(self):
        code, _, err = self.run_cli(self.source, self.dest, "--groups", "head")
        self.assertEqual(code, 1)
        self.assertIn("would be converted", err)
        self.assertFalse(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_a_truncated_source_is_refused_before_anything_is_written(self):
        self.source.write_bytes(self.source.read_bytes()[:-100])
        code, _, err = self.run_cli(self.source, self.dest)
        self.assertEqual(code, 1)
        self.assertIn("cannot size", err)
        self.assertFalse(self.part.exists())

    def test_too_little_disk_space_is_refused_before_anything_is_written(self):
        tiny = mock.Mock(free=1000)
        with mock.patch.object(rd.shutil, "disk_usage", return_value=tiny):
            with self.assertRaises(OSError):
                self.convert()
            code, report, _ = self.run_cli(self.source, self.dest, "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("would NOT fit", report)
        self.assertFalse(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_a_damaged_output_is_caught_by_the_check(self):
        plan = self.convert()
        self.assertEqual(rd.check_output(plan, self.dest), [])
        damaged = self.dir / "damaged.gguf"
        damaged.write_bytes(self.dest.read_bytes()[:-64])
        self.assertTrue(rd.check_output(plan, damaged))
        flipped = bytearray(self.dest.read_bytes())
        target = next(entry for entry in plan.entries if entry.tensor.name == "blk.0.attn_qkv.weight")
        flipped[plan.data_start + target.new_offset + 208:plan.data_start + target.new_offset + 210] = np.float16(1e4).tobytes()
        damaged.write_bytes(bytes(flipped))
        self.assertTrue(any("does not match" in problem for problem in rd.check_output(plan, damaged)))

    def test_a_larger_alignment_from_the_metadata_is_honoured(self):
        aligned = self.dir / "aligned.gguf"
        self.make_source(aligned, alignment=64, entries=self.entries + [("general.alignment", ("u32", 64))])
        self.source = aligned
        plan = self.convert()
        result = gguf_reader.GGUFFile(self.dest)
        self.assertEqual(result.alignment, 64)
        self.assertEqual(result.data_start % 64, 0)
        self.assertTrue(all(tensor.offset % 64 == 0 for tensor in result.tensors))
        self.assertEqual(self.dest.stat().st_size, plan.total_size)
        self.assertEqual(self.tensor_bytes(self.dest, "blk.0.attn_norm.weight"), self.payloads["blk.0.attn_norm.weight"])

    def test_group_names_are_validated(self):
        self.assertEqual(rd.parse_groups("all"), tuple(rd.GROUPS))
        self.assertEqual(rd.parse_groups("ssm, attn,ssm"), ("ssm", "attn"))
        with self.assertRaises(ValueError) as context:
            rd.parse_groups("attn,bogus")
        self.assertIn("bogus", str(context.exception))
        with self.assertRaises(SystemExit) as exit_context:
            self.run_cli(self.source, self.dest, "--groups", "bogus")
        self.assertEqual(exit_context.exception.code, 2)

    def test_the_group_patterns_select_the_documented_tensor_names(self):
        def selects(group: str, name: str) -> bool:
            return re.match(rd.GROUPS[group].pattern, name) is not None

        for name in ("blk.3.attn_q.weight", "blk.39.attn_k.weight", "blk.3.attn_v.weight", "blk.3.attn_output.weight"):
            self.assertTrue(selects("attn", name), name)
        for name in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight"):
            self.assertTrue(selects("ssm", name), name)
        for name in ("blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.40.ffn_down_shexp.weight"):
            self.assertTrue(selects("shexp", name), name)
        self.assertTrue(selects("head", "output.weight"))
        self.assertTrue(selects("embd", "token_embd.weight"))
        self.assertTrue(selects("mtp", "blk.40.nextn.eh_proj.weight"))
        for name in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight", "blk.0.attn_q_norm.weight",
                     "blk.0.ssm_a", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_conv1d.weight", "blk.0.ssm_norm.weight",
                     "blk.0.attn_norm.weight", "blk.40.nextn.enorm.weight", "output_norm.weight"):
            self.assertFalse(any(selects(group, name) for group in rd.GROUPS), name)

    def test_the_help_text_documents_every_group(self):
        text = rd.build_parser().format_help()
        for name, group in rd.GROUPS.items():
            self.assertIn(group.pattern, text, name)
        self.assertIn("default groups: attn,ssm,shexp", text)


if __name__ == "__main__":
    unittest.main()
