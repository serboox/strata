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
        self.assertFalse(any([args.no_fast50, args.strict, args.strict_cpu, args.interleave, args.mtp_adapt, args.chain_device, args.spec_sampling, args.iq3s_4bit,
                              args.lend_buffers, args.attn_ahead, args.prefetch_lead2, args.head2, args.resident_experts]))
        self.assertEqual((args.kv_lazy, args.warm_experts, args.ngram, args.parallel), (0, 0, "", 1))
        self.assertIsNone(args.platform_profile)
        self.assertIsNone(args.gpu_clocks)

    def test_the_set_is_the_default_and_no_fast50_or_strict_cpu_turn_it_off(self):
        source = (ROOT / "tools" / "strata_launch.py").read_text(encoding="utf-8")
        self.assertIn("args.fast50 = not args.no_fast50 and not args.strict_cpu", source)

    def test_the_fast50_constants_are_the_measured_set(self):
        self.assertEqual((strata_launch.FAST50_EXPERTS_REPACK, strata_launch.FAST50_EXPERTS_PREFETCH, strata_launch.FAST50_PREFETCH,
                          strata_launch.FAST50_KV_TIER, strata_launch.FAST50_WARM_KIB, strata_launch.FAST50_NGRAM), (8, 7, 3, 8192, 64, "8,4"))

    def choose(self, **changes):
        gib = strata_launch.GIB
        values = {"expert_prefetch": None, "repack": False, "profile_name": "auto", "model_size": 16 * gib, "available": 40 * gib, "budget": None,
                  "overlap": False, "igpu_experts": False}
        return strata_launch.choose_fast_family(**{**values, **changes})

    def test_the_default_start_needs_the_models_size_plus_the_repack_margin(self):
        gib = strata_launch.GIB
        needed = 16 * gib + strata_launch.REPACK_MARGIN
        self.assertEqual(self.choose(available=needed), ("repack", []))
        with self.assertRaises(strata_launch.NotEnoughMemory) as caught:
            self.choose(available=needed - 1)
        self.assertIn("the default start needs 22.0 GiB", str(caught.exception))
        with self.assertRaises(strata_launch.NotEnoughMemory):
            self.choose(available=None)

    def test_a_memory_budget_below_the_need_refuses_too(self):
        gib = strata_launch.GIB
        needed = 16 * gib + strata_launch.REPACK_MARGIN
        with self.assertRaises(strata_launch.NotEnoughMemory) as caught:
            self.choose(budget=needed - 1)
        self.assertIn("memory budget", str(caught.exception))
        self.assertEqual(self.choose(budget=needed)[0], "repack")

    def test_main_returns_one_and_says_why_when_the_memory_is_short(self):
        import contextlib
        import io
        from unittest import mock
        if not strata_launch.DENSE_Q4K_FILE.exists():
            self.skipTest("the dense Q4_K file is not here")
        errors = io.StringIO()
        with mock.patch.object(strata_launch, "memory_available", return_value=strata_launch.GIB), \
                mock.patch.object(sys, "argv", ["strata-launch", "--port", "18999"]), contextlib.redirect_stderr(errors):
            code = strata_launch.main()
        self.assertEqual(code, 1)
        self.assertIn("strata-launch: cannot start: 1.0 GiB of memory is free", errors.getvalue())
        self.assertIn("--no-fast50", errors.getvalue())

    def test_explicit_flags_and_other_profiles_decide_the_family(self):
        low = {"available": 1}
        self.assertEqual(self.choose(repack=True, **low), ("repack", []))
        self.assertEqual(self.choose(expert_prefetch=3, **low), ("prefetch", []))
        self.assertEqual(self.choose(profile_name="lean"), ("mapped", []))
        self.assertEqual(self.choose(overlap=True)[0], "mapped")
        self.assertEqual(self.choose(igpu_experts=True)[0], "mapped")

    def test_the_repack_family_takes_all_experts_and_no_prefetch(self):
        args = self.parse()
        strata_launch.apply_fast_family(args, "repack")
        self.assertEqual((args.experts_used, args.expert_prefetch, args.iq3s_4bit, args.iq3s_4bit_drop, args.warm_experts), (8, 0, True, True, 0))

    def test_the_prefetch_family_takes_seven_experts_and_a_prefetch_of_three(self):
        args = self.parse()
        strata_launch.apply_fast_family(args, "prefetch")
        self.assertEqual((args.experts_used, args.expert_prefetch, args.iq3s_4bit, args.warm_experts), (7, 3, False, 64))

    def test_the_mapped_family_leaves_the_prefetch_to_the_stack_rules(self):
        args = self.parse()
        strata_launch.apply_fast_family(args, "mapped")
        self.assertEqual((args.experts_used, args.expert_prefetch, args.iq3s_4bit, args.warm_experts), (7, None, False, 0))

    def test_values_given_on_the_command_line_stay(self):
        args = self.parse("--experts-used", "6", "--expert-prefetch", "0", "--warm-experts", "16")
        strata_launch.apply_fast_family(args, "repack")
        self.assertEqual((args.experts_used, args.expert_prefetch), (6, 0))
        args = self.parse("--experts-used", "5", "--expert-prefetch", "2", "--warm-experts", "16")
        strata_launch.apply_fast_family(args, "prefetch")
        self.assertEqual((args.experts_used, args.expert_prefetch, args.warm_experts), (5, 2, 16))

    def test_the_temperature_limit_defaults_to_70_everywhere(self):
        from serve import thermal
        self.assertEqual(thermal.DEFAULT_LIMIT_C, 70.0)
        self.assertEqual(self.parse().max_temp, thermal.DEFAULT_LIMIT_C)
        sys.path.insert(0, str(ROOT / "tools"))
        import cooling_kit
        import kld_experts
        import lever_matrix
        import llama_thermal_bench
        import quant_bench
        self.assertEqual((kld_experts.DEFAULT_MAX_TEMP, llama_thermal_bench.DEFAULT_MAX_TEMP_C, quant_bench.DEFAULT_MAX_TEMP_C,
                          lever_matrix.DEFAULT_MAX_TEMP_C, cooling_kit.DEFAULT_LIMIT_C), (thermal.DEFAULT_LIMIT_C,) * 5)

    def test_the_fast50_token_list_is_in_the_repo_and_is_a_list_of_int32_ids(self):
        path = strata_launch.FAST50_DRAFT_LIST
        self.assertTrue(path.exists())
        data = path.read_bytes()
        self.assertEqual(len(data) % 4, 0)
        self.assertGreater(len(data) // 4, 10000)

    def test_a_running_engine_is_found_with_the_port_of_its_server(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            for pid, comm, parent, cmdline in ((100, "python", 1, b"python\0serve/server.py\0--port\08081\0"),
                                               (200, "strata-q36", 100, b"strata-q36\0--serve\0"), (300, "bash", 1, b"bash\0")):
                (proc / str(pid)).mkdir()
                (proc / str(pid) / "comm").write_text(comm + "\n", encoding="utf-8")
                (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) S {parent} 0 0\n", encoding="utf-8")
                (proc / str(pid) / "cmdline").write_bytes(cmdline)
            (proc / "self").mkdir()
            engines = strata_launch.running_engines(proc)
        self.assertEqual(engines, [(200, "8081")])
        message = strata_launch.other_engine_message(engines)
        self.assertIn("process 200 on port 8081", message)
        self.assertIn("--allow-second-engine", message)

    def test_the_capped_default_start_uses_the_capped_worker_count(self):
        self.assertEqual(strata_launch.CAPPED_WORKERS, 6)

    def test_the_old_short_forms_still_mean_the_old_options(self):
        restored = strata_launch.restore_old_abbreviations(["--ch=256", "--d", "--dr", "--mt=1", "--r=8G", "--s=card", "--w=2"])
        self.assertEqual(restored, ["--chunk=256", "--dry-run", "--dry-run", "--mtp=1", "--ram=8G", "--sampling=card", "--workers=2"])


if __name__ == "__main__":
    unittest.main()
