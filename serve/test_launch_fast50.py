import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch


class Fast50Tests(unittest.TestCase):
    def parse(self, *argv):
        return strata_launch.build_parser().parse_args(strata_launch.restore_old_abbreviations(list(argv)))

    def test_the_new_switches_are_off_by_default(self):
        args = self.parse()
        self.assertFalse(any([args.fast50, args.strict, args.strict_cpu, args.interleave, args.mtp_adapt, args.chain_device, args.spec_sampling, args.iq3s_4bit,
                              args.lend_buffers, args.attn_ahead, args.prefetch_lead2, args.head2, args.resident_experts]))
        self.assertEqual((args.kv_lazy, args.warm_experts, args.ngram, args.parallel), (0, 0, "", 1))
        self.assertIsNone(args.platform_profile)
        self.assertIsNone(args.gpu_clocks)

    def test_the_fast50_constants_are_the_measured_set(self):
        self.assertEqual((strata_launch.FAST50_EXPERTS, strata_launch.FAST50_PREFETCH, strata_launch.FAST50_KV_TIER,
                          strata_launch.FAST50_WARM_KIB, strata_launch.FAST50_NGRAM), (6, 3, 8192, 64, "8,4"))

    def test_the_fast50_token_list_is_in_the_repo_and_is_a_list_of_int32_ids(self):
        path = strata_launch.FAST50_DRAFT_LIST
        self.assertTrue(path.exists())
        data = path.read_bytes()
        self.assertEqual(len(data) % 4, 0)
        self.assertGreater(len(data) // 4, 10000)

    def test_the_capped_default_start_uses_the_capped_worker_count(self):
        self.assertEqual(strata_launch.CAPPED_WORKERS, 6)

    def test_the_old_short_forms_still_mean_the_old_options(self):
        restored = strata_launch.restore_old_abbreviations(["--ch=256", "--d", "--dr", "--mt=1", "--r=8G", "--s=card", "--w=2"])
        self.assertEqual(restored, ["--chunk=256", "--dry-run", "--dry-run", "--mtp=1", "--ram=8G", "--sampling=card", "--workers=2"])


if __name__ == "__main__":
    unittest.main()
