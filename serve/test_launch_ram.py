import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch
import strata_prewarm

GIB = 1 << 30
MODEL = int(20.8 * GIB)
APP_SLICE = "user.slice/user-1000.slice/user@1000.service/app.slice"


class FlagTests(unittest.TestCase):
    def parse(self, *arguments):
        return strata_launch.build_parser().parse_args(list(arguments))

    def test_protection_and_prewarm_are_on_unless_switched_off(self):
        defaults = self.parse()
        self.assertIsNone(defaults.protect_ram)
        self.assertIsNone(defaults.prewarm)
        self.assertIs(self.parse("--protect-ram").protect_ram, True)
        self.assertIs(self.parse("--no-protect-ram").protect_ram, False)
        self.assertIs(self.parse("--prewarm").prewarm, True)
        self.assertIs(self.parse("--no-prewarm").prewarm, False)

    def test_the_help_says_how_to_switch_them_off(self):
        text = strata_launch.build_parser().format_help()
        self.assertIn("--no-protect-ram", text)
        self.assertIn("--no-prewarm", text)

    def test_the_other_defaults_are_untouched(self):
        args = self.parse()
        self.assertEqual((args.profile, args.priority, args.turbo, args.max_temp, args.port), ("auto", "background", "off", 80.0, 8080))


class SizeTests(unittest.TestCase):
    def test_sizes_use_binary_units(self):
        self.assertEqual(strata_launch.parse_size("22G"), 22 * GIB)
        self.assertEqual(strata_launch.parse_size("512m"), 512 << 20)
        self.assertEqual(strata_launch.parse_size("1.5G"), int(1.5 * GIB))
        self.assertEqual(strata_launch.parse_size("4096"), 4096)

    def test_a_size_that_is_not_one_is_refused(self):
        for text in ("abc", "50%", "", "G"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strata_launch.parse_size(text)


class ProtectTests(unittest.TestCase):
    def test_the_model_and_the_engine_around_it_are_protected(self):
        self.assertEqual(strata_launch.protect_bytes("buffered", None, MODEL), MODEL + strata_launch.PROTECT_MARGIN)

    def test_a_memory_budget_caps_what_is_protected(self):
        self.assertEqual(strata_launch.protect_bytes("buffered", "22G", MODEL), 22 * GIB)
        self.assertEqual(strata_launch.protect_bytes("buffered", "64G", MODEL), MODEL + strata_launch.PROTECT_MARGIN)

    def test_a_profile_that_bypasses_the_page_cache_has_nothing_to_protect(self):
        self.assertIsNone(strata_launch.protect_bytes("direct", "16G", MODEL))

    def test_the_value_reaches_the_scope_and_the_default_is_unchanged(self):
        self.assertEqual(strata_launch.scope_properties("22G", False, 5), ["MemoryMax=22G", "MemorySwapMax=0", "MemoryLow=5"])
        self.assertEqual(strata_launch.scope_properties(None, True, 7), ["MemoryLow=7", "CPUWeight=idle", "IOWeight=1"])
        self.assertEqual(strata_launch.scope_properties(None, False), [])
        self.assertEqual(strata_launch.scope_properties("16G", True), ["MemoryMax=16G", "MemorySwapMax=0", "CPUWeight=idle", "IOWeight=1"])

    def test_a_scope_can_be_asked_for_without_any_property(self):
        self.assertEqual(strata_launch.wrap_in_scope(["server"], []), ["server"])
        self.assertEqual(strata_launch.wrap_in_scope(["server"], [], always=True), ["systemd-run", "--user", "--scope", "--quiet", "--", "server"])

    def test_protection_alone_is_enough_to_start_a_scope(self):
        command = strata_launch.wrap_in_scope(["server"], strata_launch.scope_properties(None, False, 5))
        self.assertEqual(command, ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryLow=5", "--", "server"])


class PrewarmPlanTests(unittest.TestCase):
    def test_a_model_that_fits_and_has_free_memory_is_read_whole(self):
        plan = strata_launch.prewarm_plan("buffered", None, MODEL, MODEL + 3 * GIB)
        self.assertEqual(plan, (MODEL, ""))

    def test_a_budget_that_holds_the_model_is_enough(self):
        self.assertEqual(strata_launch.prewarm_plan("buffered", "22G", MODEL, MODEL + 3 * GIB)[0], MODEL)
        self.assertEqual(strata_launch.prewarm_plan("buffered", str(MODEL), MODEL, MODEL + 3 * GIB)[0], MODEL)
        self.assertIsNone(strata_launch.prewarm_plan("buffered", str(MODEL - 1), MODEL, MODEL + 3 * GIB)[0])

    def test_a_budget_smaller_than_the_model_skips_the_read(self):
        limit, reason = strata_launch.prewarm_plan("buffered", "16G", MODEL, 30 * GIB)
        self.assertIsNone(limit)
        self.assertIn("16G", reason)

    def test_not_enough_free_memory_skips_the_read(self):
        limit, reason = strata_launch.prewarm_plan("buffered", None, MODEL, 5 * GIB)
        self.assertIsNone(limit)
        self.assertIn("5.0 GiB", reason)

    def test_direct_reads_never_fill_the_page_cache(self):
        limit, reason = strata_launch.prewarm_plan("direct", "16G", MODEL, 30 * GIB)
        self.assertIsNone(limit)
        self.assertIn("O_DIRECT", reason)

    def test_the_free_memory_comes_from_meminfo(self):
        text = "MemTotal:       32000000 kB\nMemAvailable:       2048 kB\n"
        with mock.patch.object(strata_launch.Path, "read_text", return_value=text):
            self.assertEqual(strata_launch.memory_available(), 2048 * 1024)
        with mock.patch.object(strata_launch.Path, "read_text", return_value="MemTotal: 1 kB\n"):
            self.assertEqual(strata_launch.memory_available(), 0)


class GapTests(unittest.TestCase):
    def tree(self, values: dict) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(root)], check=False))
        for path, value in values.items():
            (root / path).mkdir(parents=True, exist_ok=True)
            if value is not None:
                (root / path / "memory.low").write_text(f"{value}\n", encoding="ascii")
        return root

    def levels(self, **overrides) -> dict:
        names = strata_launch.scope_ancestors(1000)
        return {name: overrides.get(name, "max") for name in names}

    def test_the_levels_above_a_user_scope_are_found(self):
        self.assertEqual(strata_launch.scope_ancestors(1000), [
            "user.slice", "user.slice/user-1000.slice", "user.slice/user-1000.slice/user@1000.service", APP_SLICE])

    def test_the_levels_follow_the_user_id_not_the_calling_group(self):
        self.assertEqual(strata_launch.scope_ancestors(1234)[-1], "user.slice/user-1234.slice/user@1234.service/app.slice")

    def test_a_machine_without_the_levels_reports_every_one_as_unreadable(self):
        gaps = strata_launch.protection_gaps(5 * GIB, 1000, Path(tempfile.mkdtemp()))
        self.assertEqual(gaps, [(path, None) for path in strata_launch.scope_ancestors(1000)])

    def test_levels_that_allow_enough_leave_no_gap(self):
        root = self.tree(self.levels())
        self.assertEqual(strata_launch.protection_gaps(23 * GIB, 1000, root), [])
        root = self.tree(self.levels(**{APP_SLICE: str(23 * GIB)}))
        self.assertEqual(strata_launch.protection_gaps(23 * GIB, 1000, root), [])

    def test_a_level_that_allows_less_is_reported_with_its_value(self):
        root = self.tree(self.levels(**{"user.slice": "262144000"}))
        self.assertEqual(strata_launch.protection_gaps(23 * GIB, 1000, root), [("user.slice", 262144000)])

    def test_a_level_without_the_file_is_reported_as_unreadable(self):
        values = self.levels()
        values[APP_SLICE] = None
        root = self.tree(values)
        self.assertEqual(strata_launch.protection_gaps(23 * GIB, 1000, root), [(APP_SLICE, None)])


class WarmTests(unittest.TestCase):
    def file(self, size: int) -> str:
        handle, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with os.fdopen(handle, "wb") as out:
            out.write(os.urandom(size))
        return path

    def test_the_whole_file_is_read(self):
        path = self.file((3 << 20) + 17)
        self.assertEqual(strata_prewarm.warm(path), (3 << 20) + 17)

    def test_the_limit_stops_the_read(self):
        path = self.file(3 << 20)
        self.assertEqual(strata_prewarm.warm(path, limit=1 << 20), 1 << 20)
        self.assertEqual(strata_prewarm.warm(path, limit=10 << 20), 3 << 20)

    def test_progress_counts_up_to_the_end(self):
        path = self.file(10_000)
        seen = []
        strata_prewarm.warm(path, chunk=3000, progress=lambda done, wanted: seen.append((done, wanted)))
        self.assertEqual(seen, [(3000, 10_000), (6000, 10_000), (9000, 10_000), (10_000, 10_000)])

    def test_pages_already_cached_are_dropped_before_the_read(self):
        path = self.file(3 << 20)
        with mock.patch.object(strata_prewarm.os, "posix_fadvise", wraps=os.posix_fadvise) as fadvise:
            strata_prewarm.warm(path, limit=1 << 20)
        fadvise.assert_any_call(mock.ANY, 0, 1 << 20, os.POSIX_FADV_DONTNEED)

    def test_a_zero_limit_reads_nothing_and_drops_nothing(self):
        path = self.file(1 << 20)
        with mock.patch.object(strata_prewarm.os, "posix_fadvise") as fadvise:
            self.assertEqual(strata_prewarm.warm(path, limit=0), 0)
        fadvise.assert_not_called()

    def test_an_empty_file_reads_nothing(self):
        self.assertEqual(strata_prewarm.warm(self.file(0)), 0)

    def test_a_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            strata_prewarm.warm("/nonexistent/model.gguf")


class PrewarmCommandTests(unittest.TestCase):
    def run_helper(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "tools" / "strata_prewarm.py"), *arguments], capture_output=True, text=True, timeout=60, check=False)

    def test_the_model_is_read_and_the_server_command_takes_over(self):
        handle, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.unlink(path))
        with os.fdopen(handle, "wb") as out:
            out.write(os.urandom(1 << 20))
        result = self.run_helper("--limit", "524288", path, "--", "echo", "started")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "started")
        self.assertIn("strata-prewarm: 0.0 GiB read in", result.stderr)

    def test_a_model_that_cannot_be_read_never_stops_the_server(self):
        result = self.run_helper("/nonexistent/model.gguf", "--", "echo", "started")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "started")
        self.assertIn("could not read", result.stderr)

    def test_a_missing_server_command_is_refused(self):
        self.assertEqual(self.run_helper("/nonexistent/model.gguf", "--").returncode, 2)


class ScopeCacheTests(unittest.TestCase):
    def stat(self, text: str) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(root)], check=False))
        (root / "user.slice" / "x.scope").mkdir(parents=True)
        (root / "user.slice" / "x.scope" / "memory.stat").write_text(text, encoding="ascii")
        return root

    def test_the_page_cache_of_the_group_is_read(self):
        root = self.stat("anon 100\nfile 4096\nkernel 1\n")
        self.assertEqual(strata_prewarm.scope_cache_bytes(root, "/user.slice/x.scope"), 4096)

    def test_an_unreadable_or_odd_file_gives_none(self):
        self.assertIsNone(strata_prewarm.scope_cache_bytes(self.stat("anon 100\n"), "/user.slice/x.scope"))
        self.assertIsNone(strata_prewarm.scope_cache_bytes(self.stat("file lots\n"), "/user.slice/x.scope"))
        self.assertIsNone(strata_prewarm.scope_cache_bytes(self.stat("file 1\n"), "/user.slice/missing.scope"))


class ScopeAvailableTests(unittest.TestCase):
    def test_without_systemd_run_everything_runs_in_the_current_group(self):
        usable, warning = strata_launch.scope_available(False, ["MemoryMax=1G"], False, probe=lambda: self.fail("must not probe"))
        self.assertFalse(usable)
        self.assertIn("systemd-run is not available", warning)

    def test_properties_that_exist_anyway_need_no_probe(self):
        self.assertEqual(strata_launch.scope_available(True, ["CPUWeight=idle"], False, probe=lambda: self.fail("must not probe")), (True, ""))

    def test_a_dry_run_does_not_probe(self):
        self.assertEqual(strata_launch.scope_available(True, [], True, probe=lambda: self.fail("must not probe")), (True, ""))

    def test_a_start_that_would_run_without_a_scope_is_probed(self):
        self.assertEqual(strata_launch.scope_available(True, [], False, probe=lambda: True), (True, ""))
        usable, warning = strata_launch.scope_available(True, [], False, probe=lambda: False)
        self.assertFalse(usable)
        self.assertIn("no user systemd manager", warning)


class PlanMemoryTests(unittest.TestCase):
    def plan(self, protect=None, prewarm=None, io="buffered", budget=None, have_systemd=True, base=("CPUWeight=idle",), dry_run=False,
             available=MODEL + 3 * GIB, gaps=(), probe=lambda: True):
        return strata_launch.plan_memory(protect, prewarm, io, budget, MODEL, have_systemd, list(base), dry_run,
                                         available=lambda: available, gaps=lambda needed: list(gaps), probe=probe)

    def test_both_are_on_by_default(self):
        self.assertEqual(self.plan(), (MODEL + strata_launch.PROTECT_MARGIN, MODEL, True, []))

    def test_each_can_be_switched_off(self):
        self.assertEqual(self.plan(protect=False)[:2], (None, MODEL))
        self.assertEqual(self.plan(prewarm=False)[:2], (MODEL + strata_launch.PROTECT_MARGIN, None))
        self.assertEqual(self.plan(protect=False, prewarm=False), (None, None, False, []))

    def test_the_direct_profiles_stay_quiet_by_default(self):
        self.assertEqual(self.plan(io="direct", budget="16G"), (None, None, False, []))

    def test_the_direct_profiles_explain_when_asked(self):
        messages = self.plan(protect=True, prewarm=True, io="direct", budget="16G")[3]
        self.assertEqual(len(messages), 2)
        self.assertTrue(all("O_DIRECT" in message for message in messages))

    def test_too_little_free_memory_skips_the_prewarm_and_says_so(self):
        low, warm, _, messages = self.plan(available=5 * GIB)
        self.assertIsNotNone(low)
        self.assertIsNone(warm)
        self.assertTrue(any(message.startswith("prewarm skipped") and "5.0 GiB" in message for message in messages))

    def test_every_level_that_allows_less_is_reported(self):
        messages = self.plan(gaps=[("user.slice", 0), ("user.slice/x/app.slice", None)])[3]
        self.assertEqual(len(messages), 2)
        self.assertIn("/user.slice allows 0.0 GiB", messages[0])
        self.assertIn("no memory.low", messages[1])

    def test_without_systemd_run_the_protection_goes_quietly_and_the_prewarm_warns(self):
        low, warm, scope, messages = self.plan(have_systemd=False)
        self.assertEqual((low, warm, scope), (None, MODEL, False))
        self.assertEqual(len(messages), 1)
        self.assertIn("systemd-run is not available", messages[0])

    def test_no_user_manager_turns_the_protection_off_instead_of_failing_the_start(self):
        low, warm, scope, messages = self.plan(base=(), probe=lambda: False)
        self.assertEqual((low, warm, scope), (None, MODEL, False))
        self.assertIn("no user systemd manager", messages[-1])

    def test_a_scope_that_exists_anyway_is_not_probed(self):
        self.assertTrue(self.plan(base=("MemoryMax=22G",), probe=lambda: self.fail("must not probe"))[2])

    def test_a_dry_run_never_probes(self):
        self.assertTrue(self.plan(base=(), dry_run=True, probe=lambda: self.fail("must not probe"))[2])

    def test_a_budget_that_is_not_a_size_skips_both_with_a_reason(self):
        low, warm, _, messages = self.plan(budget="50%")
        self.assertEqual((low, warm), (None, None))
        self.assertEqual(sum("cannot read the size" in message for message in messages), 2)


OLD_FLAG_VALUES = {"--profile": "fast", "--ram": "8G", "--context": "1000", "--chunk": "256", "--sampling": "card", "--mtp": "1",
                   "--workers": "2", "--max-temp": "70", "--turbo": "on", "--priority": "normal", "--port": "9000"}
OLD_FLAGS = list(OLD_FLAG_VALUES) + ["--no-limit", "--dry-run"]


class OldFlagTests(unittest.TestCase):
    def test_every_short_form_the_old_parser_took_still_works(self):
        defaults = strata_launch.build_parser().parse_args([])
        for option in OLD_FLAGS:
            for length in range(3, len(option) + 1):
                prefix = option[:length]
                if [other for other in OLD_FLAGS + ["--help"] if other.startswith(prefix)] != [option]:
                    continue
                with self.subTest(prefix=prefix):
                    argv = [f"{prefix}={OLD_FLAG_VALUES[option]}"] if option in OLD_FLAG_VALUES else [prefix]
                    args = strata_launch.build_parser().parse_args(strata_launch.restore_old_abbreviations(argv))
                    field = option[2:].replace("-", "_")
                    self.assertNotEqual(getattr(args, field), getattr(defaults, field))

    def test_the_forms_that_the_new_flags_made_ambiguous_map_back(self):
        self.assertEqual(strata_launch.restore_old_abbreviations(["--pro=max", "--no", "--n", "--no-", "--ram", "8G"]),
                         ["--profile=max", "--no-limit", "--no-limit", "--no-limit", "--ram", "8G"])

    def test_the_new_flags_are_not_taken_for_old_ones(self):
        args = strata_launch.build_parser().parse_args(strata_launch.restore_old_abbreviations(["--no-prewarm", "--no-protect-ram", "--protect-ram"]))
        self.assertEqual((args.prewarm, args.protect_ram, args.no_limit), (False, True, False))


class UserScopeProbeTests(unittest.TestCase):
    def probe(self, **result):
        with mock.patch.object(strata_launch.subprocess, "run", **result) as run:
            return strata_launch.user_scope_works(), run

    def test_a_scope_that_started_counts(self):
        works, run = self.probe(return_value=subprocess.CompletedProcess([], 0))
        self.assertTrue(works)
        self.assertEqual(run.call_args.args[0], ["systemd-run", "--user", "--scope", "--quiet", "true"])
        self.assertEqual(run.call_args.kwargs["timeout"], strata_launch.PROBE_TIMEOUT_SECONDS)

    def test_a_refused_missing_or_hung_systemd_run_does_not(self):
        self.assertFalse(self.probe(return_value=subprocess.CompletedProcess([], 1))[0])
        self.assertFalse(self.probe(side_effect=FileNotFoundError("systemd-run"))[0])
        self.assertFalse(self.probe(side_effect=subprocess.TimeoutExpired("systemd-run", 10))[0])


if __name__ == "__main__":
    unittest.main()
