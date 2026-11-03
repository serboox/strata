import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch


class ShareTests(unittest.TestCase):
    def test_a_fraction_between_zero_and_one_is_accepted(self):
        self.assertEqual(strata_launch.igpu_share("0.25"), 0.25)

    def test_zero_means_off(self):
        self.assertEqual(strata_launch.igpu_share("0"), 0.0)

    def test_one_and_above_and_negative_values_are_refused(self):
        for text in ("1", "1.5", "-0.1", "nan", "inf"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                strata_launch.igpu_share(text)

    def test_text_is_refused(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            strata_launch.igpu_share("half")


class EnvironmentTests(unittest.TestCase):
    def test_a_share_reaches_the_engine_environment_next_to_the_other_variables(self):
        self.assertEqual(strata_launch.with_igpu({"STRATA_IQ_VBMI": "1"}, 0.25), {"STRATA_IQ_VBMI": "1", "STRATA_IGPU_EXPERTS": "0.25"})
        self.assertEqual(strata_launch.with_igpu(None, 0.35), {"STRATA_IGPU_EXPERTS": "0.35"})

    def test_off_leaves_the_environment_as_it_was(self):
        self.assertIsNone(strata_launch.with_igpu(None, 0.0))
        env = {"STRATA_IQ_VBMI": "1"}
        self.assertIs(strata_launch.with_igpu(env, 0.0), env)

    def test_the_flag_defaults_to_off_and_parses_a_share(self):
        parser = strata_launch.build_parser()
        self.assertEqual(parser.parse_args([]).igpu_experts, 0.0)
        self.assertEqual(parser.parse_args(["--igpu-experts", "0.15"]).igpu_experts, 0.15)


if __name__ == "__main__":
    unittest.main()
