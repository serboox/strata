import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch

GIB = strata_launch.GIB
MODEL_SIZE = 17 * GIB
PLENTY = 28 * GIB


def resolve(experts_used=None, expert_prefetch=None, profile="auto", available=PLENTY, budget=None, overlap=False, igpu=False):
    return strata_launch.resolve_stack(experts_used, expert_prefetch, profile, MODEL_SIZE, available, budget, overlap, igpu)


class DefaultStackTests(unittest.TestCase):
    def test_nothing_asked_gives_seven_experts_and_a_prefetch_of_three(self):
        self.assertEqual(resolve(), (7, 3, []))

    def test_the_models_own_count_is_passed_on_so_that_an_inherited_environment_cannot_lower_it(self):
        self.assertEqual(resolve(experts_used=strata_launch.MODEL_EXPERTS_USED), (strata_launch.MODEL_EXPERTS_USED, 3, []))
        self.assertEqual(strata_launch.experts_used_env(strata_launch.MODEL_EXPERTS_USED), {"STRATA_Q36_EXPERTS_USED": "8"})

    def test_an_explicit_count_wins(self):
        self.assertEqual(resolve(experts_used=5)[0], 5)
        self.assertEqual(resolve(experts_used=1)[0], 1)

    def test_prefetch_zero_turns_it_off(self):
        self.assertEqual(resolve(expert_prefetch=0), (7, 0, []))

    def test_an_explicit_prefetch_wins_over_every_condition(self):
        self.assertEqual(resolve(expert_prefetch=2, profile="lean", available=GIB, overlap=True, igpu=True), (7, 2, []))

    def test_the_memory_limited_profiles_keep_the_prefetch_off_without_a_message(self):
        for profile in ("lean", "balanced", "fast", "max"):
            self.assertEqual(resolve(profile=profile), (7, 0, []), profile)

    def test_the_experts_count_applies_to_every_profile(self):
        for profile in strata_launch.PROFILES:
            self.assertEqual(resolve(profile=profile)[0], 7, profile)

    def test_overlap_and_the_igpu_share_turn_the_default_prefetch_off_with_a_message(self):
        for overlap, igpu in ((True, False), (False, True)):
            experts, prefetch, notes = resolve(overlap=overlap, igpu=igpu)
            self.assertEqual((experts, prefetch), (7, 0))
            self.assertEqual(len(notes), 1)
            self.assertIn("--overlap", notes[0])

    def test_too_little_free_memory_turns_the_default_prefetch_off_with_a_message(self):
        experts, prefetch, notes = resolve(available=MODEL_SIZE)
        self.assertEqual((experts, prefetch), (7, 0))
        self.assertEqual(len(notes), 1)
        self.assertIn("memory", notes[0])

    def test_the_memory_bound_is_the_model_plus_the_margin(self):
        needed = MODEL_SIZE + strata_launch.PROTECT_MARGIN
        self.assertEqual(resolve(available=needed)[1], 3)
        self.assertEqual(resolve(available=needed - 1)[1], 0)

    def test_an_unreadable_memory_figure_keeps_the_default_prefetch_off(self):
        experts, prefetch, notes = resolve(available=None)
        self.assertEqual((experts, prefetch), (7, 0))
        self.assertEqual(len(notes), 1)

    def test_a_memory_budget_below_the_experts_turns_the_default_prefetch_off_with_a_message(self):
        needed = MODEL_SIZE + strata_launch.PROTECT_MARGIN
        self.assertEqual(resolve(budget=needed)[1], 3)
        experts, prefetch, notes = resolve(budget=8 * GIB)
        self.assertEqual((experts, prefetch), (7, 0))
        self.assertIn("budget", notes[0])

    def test_no_budget_means_no_limit(self):
        self.assertEqual(resolve(budget=None), (7, 3, []))


class DefaultModelTests(unittest.TestCase):
    def test_the_dense_q5_k_file_is_chosen_when_it_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory) / "dense.gguf"
            dense.write_bytes(b"x")
            with mock.patch.object(strata_launch, "DENSE_Q5K_FILE", dense):
                self.assertEqual(strata_launch.default_model_file(), dense)

    def test_the_q4_k_xl_file_is_the_fallback(self):
        with mock.patch.object(strata_launch, "DENSE_Q5K_FILE", Path("/nonexistent/dense.gguf")):
            self.assertEqual(strata_launch.default_model_file(), strata_launch.MODEL_FILE)


class PrefetchArgumentTests(unittest.TestCase):
    def run_main(self, *arguments):
        with mock.patch.object(sys, "argv", ["strata-launch", *arguments]), self.assertRaises(SystemExit) as raised:
            strata_launch.main()
        return raised.exception.code

    def test_the_parser_leaves_both_settings_unset_so_that_the_defaults_apply_in_one_place(self):
        args = strata_launch.build_parser().parse_args([])
        self.assertIsNone(args.experts_used)
        self.assertIsNone(args.expert_prefetch)

    def test_zero_is_accepted_and_the_upper_bound_is_kept(self):
        self.assertEqual(strata_launch.build_parser().parse_args(["--expert-prefetch", "0"]).expert_prefetch, 0)
        self.assertEqual(self.run_main("--expert-prefetch", str(strata_launch.MAX_EXPERT_PREFETCH + 1)), 2)
        self.assertEqual(self.run_main("--expert-prefetch", "-1"), 2)

    def test_an_off_prefetch_is_passed_to_the_engine_so_that_the_environment_cannot_turn_it_on(self):
        profile = dict(strata_launch.PROFILES["auto"], expert_prefetch=0)
        args = strata_launch.engine_args(profile)
        self.assertEqual(args[args.index("--expert-prefetch") + 1], "0")
        self.assertEqual(args[args.index("--experts") + 1], "mapped")

    def test_a_profile_without_a_prefetch_setting_adds_no_flag(self):
        self.assertNotIn("--expert-prefetch", strata_launch.engine_args(dict(strata_launch.PROFILES["auto"])))

    def test_an_active_prefetch_pins_the_experts(self):
        args = strata_launch.engine_args(dict(strata_launch.PROFILES["auto"], expert_prefetch=3))
        self.assertEqual(args[args.index("--experts") + 1], "pinned")
        self.assertEqual(args[args.index("--expert-prefetch") + 1], "3")

    def test_the_memory_plan_does_not_protect_a_page_cache_the_pinned_experts_do_not_use(self):
        profile = dict(strata_launch.PROFILES["auto"])
        self.assertIsNone(strata_launch.protect_bytes(strata_launch.PINNED_EXPERTS_IO, profile["ram"], MODEL_SIZE))
        self.assertIsNotNone(strata_launch.protect_bytes(profile["io"], profile["ram"], MODEL_SIZE))


if __name__ == "__main__":
    unittest.main()
