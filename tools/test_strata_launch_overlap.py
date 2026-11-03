import argparse
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import strata_launch


class OverlapFlagTest(unittest.TestCase):
    def parse(self, *arguments):
        return strata_launch.build_parser().parse_args(list(arguments))

    def test_off_by_default(self):
        args = self.parse()
        self.assertIsNone(args.overlap)
        self.assertIsNone(strata_launch.overlap_environment(args.overlap))

    def test_bare_flag_means_all(self):
        self.assertEqual(self.parse("--overlap").overlap, "all")
        self.assertEqual(strata_launch.overlap_environment("all"), {"STRATA_Q36_OVERLAP": "all"})

    def test_list_is_normalised(self):
        self.assertEqual(self.parse("--overlap", "quant, glue").overlap, "quant,glue")

    def test_unknown_switch_is_refused(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            strata_launch.overlap_list("quant,rows")
        with self.assertRaises(argparse.ArgumentTypeError):
            strata_launch.overlap_list("")

    def test_environment_is_merged_with_the_others(self):
        env = strata_launch.engine_environment(strata_launch.IQ_KERNELS_ENV, 0, {"STRATA_Q36_OVERLAP": "commit"})
        self.assertEqual(env["STRATA_Q36_OVERLAP"], "commit")
        self.assertTrue(set(strata_launch.IQ_KERNELS_ENV.items()) <= set(env.items()))
        self.assertIsNone(strata_launch.engine_environment(None, 0))


if __name__ == "__main__":
    unittest.main()
