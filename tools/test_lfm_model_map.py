import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lfm_model_map as lfm
from gguf_reader import TensorInfo

REAL_FILE = Path(__file__).resolve().parents[1] / "models" / "lfm2.5-24b-a2b" / "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"


def metadata(**changes):
    values = {"general.architecture": "lfm2moe", "lfm2moe.block_count": 6, "lfm2moe.embedding_length": 256, "lfm2moe.feed_forward_length": 512,
              "lfm2moe.attention.head_count": 4, "lfm2moe.attention.head_count_kv": [0, 0, 2, 0, 0, 2], "lfm2moe.expert_count": 8,
              "lfm2moe.expert_used_count": 2, "lfm2moe.expert_feed_forward_length": 256, "lfm2moe.leading_dense_block_count": 2,
              "lfm2moe.shortconv.l_cache": 3, "lfm2moe.vocab_size": 512}
    values.update(changes)
    return values


def tensors_of(h, type_name="Q4_K"):
    return [TensorInfo(name, list(shape), 0, "F32" if len(shape) == 1 or name.endswith("conv.weight") else type_name, 0)
            for name, shape in lfm.expected_shapes(h).items()]


class HyperTests(unittest.TestCase):
    def test_the_sizes_come_from_the_metadata(self):
        h = lfm.hyper_of(metadata())
        self.assertEqual((h.layers, h.embd, h.head_dim, h.experts, h.experts_used), (6, 256, 64, 8, 2))
        self.assertTrue(h.is_conv(0) and not h.is_conv(2))
        self.assertFalse(h.is_moe(1))
        self.assertTrue(h.is_moe(2))

    def test_another_architecture_a_missing_key_and_a_short_head_list_are_refused(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            lfm.hyper_of(metadata(**{"general.architecture": "qwen35moe"}))
        broken = metadata()
        del broken["lfm2moe.expert_count"]
        with self.assertRaisesRegex(ValueError, "expert_count"):
            lfm.hyper_of(broken)
        with self.assertRaisesRegex(ValueError, "head_count_kv"):
            lfm.hyper_of(metadata(**{"lfm2moe.attention.head_count_kv": [0, 2]}))


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.h = lfm.hyper_of(metadata())

    def test_the_layout_has_the_expected_number_of_tensors(self):
        conv, attention, dense, moe = 4, 2, 2, 4
        self.assertEqual(len(lfm.expected_shapes(self.h)), 2 + 6 * 2 + conv * 3 + attention * 6 + dense * 3 + moe * 5)

    def test_a_complete_list_has_no_problems_with_or_without_the_output_tensor(self):
        tensors = tensors_of(self.h)
        self.assertEqual(lfm.check(self.h, tensors), [])
        tensors.append(TensorInfo("output.weight", [256, 512], 0, "Q8_0", 0))
        self.assertEqual(lfm.check(self.h, tensors), [])

    def test_missing_unknown_duplicate_and_misshapen_tensors_are_reported(self):
        tensors = tensors_of(self.h)
        missing = tensors.pop(0)
        tensors.append(TensorInfo("blk.1.mystery.weight", [4], 0, "F32", 0))
        tensors.append(tensors[3])
        for tensor in tensors:
            if tensor.name == "blk.2.attn_k.weight":
                tensor.shape = [256, 100]
        tensors.append(TensorInfo("output.weight", [1, 1], 0, "Q8_0", 0))
        found = lfm.check(self.h, tensors)
        self.assertIn(f"{missing.name}: missing", found)
        self.assertIn("blk.1.mystery.weight: not part of the lfm2moe layout", found)
        self.assertTrue(any("listed twice" in line for line in found))
        self.assertTrue(any(line.startswith("blk.2.attn_k.weight: shape [256, 100]") for line in found))
        self.assertTrue(any(line.startswith("output.weight: shape") for line in found))

    def test_a_conv_layer_has_no_attention_tensors_and_the_other_way_round(self):
        names = set(lfm.expected_shapes(self.h))
        self.assertIn("blk.0.shortconv.in_proj.weight", names)
        self.assertNotIn("blk.0.attn_q.weight", names)
        self.assertIn("blk.2.attn_q.weight", names)
        self.assertNotIn("blk.2.shortconv.conv.weight", names)
        self.assertIn("blk.1.ffn_down.weight", names)
        self.assertNotIn("blk.1.ffn_gate_exps.weight", names)


class OptionalTensorTests(unittest.TestCase):
    def setUp(self):
        self.h = lfm.hyper_of(metadata())

    def test_a_fused_qkv_matrix_replaces_the_three_separate_ones(self):
        tensors = [tensor for tensor in tensors_of(self.h) if tensor.name not in ("blk.2.attn_q.weight", "blk.2.attn_k.weight", "blk.2.attn_v.weight")]
        tensors.append(TensorInfo("blk.2.attn_qkv.weight", [256, 256 + 128 + 128], 0, "Q8_0", 0))
        self.assertEqual(lfm.check(self.h, tensors), [])
        tensors[-1].shape = [256, 300]
        self.assertTrue(any(line.startswith("blk.2.attn_qkv.weight: shape") for line in lfm.check(self.h, tensors)))

    def test_both_forms_at_once_are_a_problem(self):
        tensors = tensors_of(self.h) + [TensorInfo("blk.2.attn_qkv.weight", [256, 512], 0, "Q8_0", 0)]
        self.assertTrue(any("also separate" in line for line in lfm.check(self.h, tensors)))

    def test_attention_biases_are_accepted_with_their_shapes(self):
        tensors = tensors_of(self.h) + [TensorInfo("blk.2.attn_q.bias", [256], 0, "F32", 0), TensorInfo("blk.2.attn_k.bias", [128], 0, "F32", 0),
                                        TensorInfo("blk.2.attn_v.bias", [128], 0, "F32", 0)]
        self.assertEqual(lfm.check(self.h, tensors), [])
        tensors[-1].shape = [64]
        self.assertTrue(any(line.startswith("blk.2.attn_v.bias: shape") for line in lfm.check(self.h, tensors)))

    def test_a_conv_layer_does_not_take_attention_biases(self):
        tensors = tensors_of(self.h) + [TensorInfo("blk.0.attn_q.bias", [256], 0, "F32", 0)]
        self.assertTrue(any("not part of the lfm2moe layout" in line for line in lfm.check(self.h, tensors)))

    def test_the_extra_output_layer_of_the_embedding_variants_is_accepted(self):
        tensors = tensors_of(self.h) + [TensorInfo("dense_2.weight", [256, 128], 0, "F32", 0), TensorInfo("dense_2.bias", [128], 0, "F32", 0)]
        self.assertEqual(lfm.check(self.h, tensors), [])
        tensors[-2].shape = [100, 128]
        self.assertTrue(any(line.startswith("dense_2.weight: shape") for line in lfm.check(self.h, tensors)))

    def test_the_key_and_value_lengths_of_the_metadata_change_the_shapes_and_the_cache(self):
        h = lfm.hyper_of(metadata(**{"lfm2moe.attention.key_length": 32, "lfm2moe.attention.value_length": 48}))
        shapes = lfm.expected_shapes(h)
        self.assertEqual(shapes["blk.2.attn_q.weight"], [256, 4 * 32])
        self.assertEqual(shapes["blk.2.attn_k.weight"], [256, 2 * 32])
        self.assertEqual(shapes["blk.2.attn_v.weight"], [256, 2 * 48])
        self.assertEqual(shapes["blk.2.attn_output.weight"], [4 * 48, 256])
        self.assertEqual(shapes["blk.2.attn_q_norm.weight"], [32])
        self.assertEqual(lfm.summarize(h, tensors_of(h))["kv_bytes_per_token"], 2 * (2 * (32 + 48) * 2))


class SummaryTests(unittest.TestCase):
    def test_the_bytes_of_experts_one_token_reads_follow_the_expert_count(self):
        h = lfm.hyper_of(metadata())
        summary = lfm.summarize(h, tensors_of(h))
        per_tensor = 256 * 256 * 8 // 256 * 144
        self.assertEqual(summary["expert_bytes"], 4 * 3 * per_tensor)
        self.assertEqual(summary["expert_bytes_per_token"], summary["expert_bytes"] * 2 // 8)
        self.assertEqual(summary["expert_bytes_per_layer"], 3 * per_tensor)
        self.assertEqual(summary["pattern"], "ccAccA")
        self.assertEqual(summary["kv_bytes_per_token"], 2 * (2 * 2 * 64 * 2))
        self.assertEqual(summary["unsupported_types"], [])

    def test_a_type_without_a_kernel_is_named(self):
        h = lfm.hyper_of(metadata())
        summary = lfm.summarize(h, tensors_of(h, "IQ2_XS"))
        self.assertEqual(summary["unsupported_types"], ["IQ2_XS"])

    def test_the_role_drops_the_layer_number_and_the_suffix(self):
        self.assertEqual(lfm.role_of("blk.12.ffn_gate_exps.weight"), "ffn_gate_exps")
        self.assertEqual(lfm.role_of("blk.0.shortconv.conv.weight"), "shortconv.conv")
        self.assertEqual(lfm.role_of("token_embd.weight"), "token_embd")


@unittest.skipUnless(REAL_FILE.is_file(), "the LFM2.5 file is not on this machine")
class RealFileTests(unittest.TestCase):
    def test_the_real_file_matches_the_layout(self):
        lines, problems = lfm.report(REAL_FILE)
        self.assertEqual(problems, [])
        self.assertIn("428 tensors", lines[0])
        self.assertIn("pattern ccAcccAcccAcccAcccAcccAcccAcccAcccAcccAc", lines[0])
        self.assertTrue(any("876 MB of experts" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
