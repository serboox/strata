import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch


class ExpertsUsedEnvironmentTests(unittest.TestCase):
    def test_nothing_is_set_unless_asked(self):
        self.assertEqual(strata_launch.experts_used_env(None), {})
        self.assertIsNone(strata_launch.engine_environment(None, 0))
        self.assertIsNone(strata_launch.engine_environment(None, 0, None))

    def test_the_count_reaches_the_engine_as_text(self):
        self.assertEqual(strata_launch.experts_used_env(6), {"STRATA_Q36_EXPERTS_USED": "6"})
        self.assertEqual(strata_launch.engine_environment(None, 0, 6), {"STRATA_Q36_EXPERTS_USED": "6"})

    def test_it_joins_the_kernel_and_cap_settings(self):
        env = strata_launch.engine_environment(strata_launch.IQ_KERNELS_ENV, 2200.0, 5)
        self.assertEqual(env, {"STRATA_IQ_VBMI": "1", "STRATA_POOL_SPIN_US": "0", "STRATA_Q36_EXPERTS_USED": "5"})

    def test_the_config_carries_it(self):
        env = strata_launch.engine_environment(None, 0, 7)
        config = strata_launch.build_config(dict(strata_launch.PROFILES["auto"]), 8080, "greedy", 80.0, env)
        self.assertEqual(config["env"], {"STRATA_Q36_EXPERTS_USED": "7"})


class ExpertsUsedArgumentTests(unittest.TestCase):
    def run_main(self, *arguments):
        with mock.patch.object(sys, "argv", ["strata-launch", *arguments]), self.assertRaises(SystemExit) as raised:
            strata_launch.main()
        return raised.exception.code

    def test_it_is_off_by_default(self):
        self.assertIsNone(strata_launch.build_parser().parse_args([]).experts_used)

    def test_a_count_is_read_as_an_integer(self):
        self.assertEqual(strata_launch.build_parser().parse_args(["--experts-used", "6"]).experts_used, 6)

    def test_a_count_outside_the_model_range_is_refused(self):
        for bad in ("0", "-1", "9", "100"):
            self.assertEqual(self.run_main("--experts-used", bad), 2, bad)

    def test_a_count_that_is_not_an_integer_is_refused(self):
        for bad in ("six", "6.5", ""):
            self.assertEqual(self.run_main("--experts-used", bad), 2, repr(bad))

    def test_every_count_from_one_to_the_model_value_is_accepted_by_the_parser(self):
        for count in range(1, strata_launch.MODEL_EXPERTS_USED + 1):
            self.assertEqual(strata_launch.build_parser().parse_args(["--experts-used", str(count)]).experts_used, count)


if __name__ == "__main__":
    unittest.main()
