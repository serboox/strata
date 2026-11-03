import argparse
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import kld_experts


def llama_summary(mean_kld=0.012345, ppl_ratio=1.004321, same_top=96.5, p99=0.1234) -> str:
    """The summary llama-perplexity prints after `--kl-divergence`, with the format strings of tools/perplexity/perplexity.cpp."""
    return "\n".join([
        "====== Perplexity statistics ======",
        "Mean PPL(Q)                   : %10.6lf ± %10.6lf" % (6.123456, 0.034567),
        "Mean PPL(base)                : %10.6lf ± %10.6lf" % (6.097123, 0.033999),
        "Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%" % 99.12,
        "Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf" % (0.004312, 0.000912),
        "Mean PPL(Q)/PPL(base)         : %10.6lf ± %10.6lf" % (ppl_ratio, 0.000919),
        "Mean PPL(Q)-PPL(base)         : %10.6lf ± %10.6lf" % (0.026333, 0.005551),
        "",
        "====== KL divergence statistics ======",
        "Mean    KLD: %10.6lf ± %10.6lf" % (mean_kld, 0.000321),
        "Maximum KLD: %10.6f" % 7.654321,
        "99.9%%   KLD: %10.6f" % 1.234567,
        "99.0%%   KLD: %10.6f" % p99,
        "95.0%%   KLD: %10.6f" % 0.045678,
        "Median  KLD: %10.6f" % 0.001234,
        "Minimum KLD: %10.6f" % -0.000001,
        "",
        "====== Token probability statistics ======",
        "Mean    Δp: %6.3lf ± %5.3lf %%" % (-0.123, 0.045),
        "RMS Δp    : %6.3lf ± %5.3lf %%" % (3.456, 0.067),
        "Same top p: %6.3lf ± %5.3lf %%" % (same_top, 0.222),
    ]) + "\n"


MODEL_LOG = "print_info: n_expert         = 256\nprint_info: n_expert_used    = %d\nprint_info: n_ctx_train      = 262144\n"


def finished(returncode=0, stopped=None):
    return kld_experts.WatchedRun(returncode, 61.234, 1, 12.0, 74.0, stopped)


class ParsingTests(unittest.TestCase):
    def test_the_statistics_of_a_summary_are_read(self):
        stats = kld_experts.parse_statistics(llama_summary())
        self.assertEqual(stats["mean_kld"], 0.012345)
        self.assertEqual(stats["mean_kld_err"], 0.000321)
        self.assertEqual(stats["p99_kld"], 0.1234)
        self.assertEqual(stats["median_kld"], 0.001234)
        self.assertEqual(stats["max_kld"], 7.654321)
        self.assertEqual(stats["same_top_pct"], 96.5)
        self.assertEqual(stats["ppl"], 6.123456)
        self.assertEqual(stats["ppl_base"], 6.097123)
        self.assertEqual(stats["ppl_ratio"], 1.004321)
        self.assertEqual(stats["rms_dp_pct"], 3.456)

    def test_the_99th_percentile_is_not_taken_for_the_99_9th(self):
        text = "99.9%   KLD:   1.000000\n99.0%   KLD:   0.500000\n"
        self.assertEqual(kld_experts.parse_statistics(text)["p99_kld"], 0.5)

    def test_a_run_without_a_summary_has_no_statistics(self):
        self.assertEqual(kld_experts.parse_statistics("chunk  PPL\n   1   5.1 ± 0.2\n"), {})

    def test_nan_and_exponents_are_numbers(self):
        stats = kld_experts.parse_statistics("Mean    KLD:        nan ±        nan\n99.0%   KLD:   1.5e-05\n")
        self.assertTrue(stats["mean_kld"] != stats["mean_kld"])
        self.assertEqual(stats["p99_kld"], 1.5e-05)

    def test_the_experts_llama_cpp_loaded_are_read_from_its_log(self):
        self.assertEqual(kld_experts.parse_experts_logged(MODEL_LOG % 6), 6)
        self.assertIsNone(kld_experts.parse_experts_logged("nothing here\n"))

    def test_the_expert_counts_are_a_list_of_distinct_positive_integers(self):
        self.assertEqual(kld_experts.parse_ks("8,7, 6"), [8, 7, 6])
        for bad in ("", "8,8", "0,6", "six", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                kld_experts.parse_ks(bad)


class CommandTests(unittest.TestCase):
    binary, model, text, base = Path("/x/llama-perplexity"), Path("/m/model.gguf"), Path("/t/text.txt"), Path("/o/base.kld")

    def command(self, **overrides):
        arguments = dict(binary=self.binary, model=self.model, text=self.text, base_logits=self.base, ctx=512, chunks=24, threads=6,
                         extra=["-ngl", "99"], save_base=False, experts_used=None)
        arguments.update(overrides)
        return kld_experts.perplexity_command(**arguments)

    def test_the_reference_run_saves_the_logits_and_does_not_compare(self):
        command = self.command(save_base=True)
        self.assertIn("--kl-divergence-base", command)
        self.assertNotIn("--kl-divergence", command)
        self.assertNotIn("--override-kv", command)

    def test_a_measured_run_compares_with_the_saved_logits_and_overrides_the_count(self):
        command = self.command(experts_used=6)
        self.assertIn("--kl-divergence", command)
        self.assertEqual(command[command.index("--override-kv") + 1], "qwen35moe.expert_used_count=int:6")
        self.assertEqual(command[command.index("--kl-divergence-base") + 1], str(self.base))
        self.assertEqual(command[command.index("-m") + 1], str(self.model))

    def test_the_text_context_and_chunks_are_the_same_in_every_run(self):
        saved, measured = self.command(save_base=True), self.command(experts_used=5)
        for flag in ("-f", "-c", "--chunks", "-t"):
            self.assertEqual(saved[saved.index(flag) + 1], measured[measured.index(flag) + 1])

    def test_extra_arguments_are_passed_on(self):
        self.assertIn("--cpu-moe", self.command(extra=["--cpu-moe"]))


class FindingTests(unittest.TestCase):
    def test_an_explicit_binary_wins_over_every_search(self):
        self.assertEqual(kld_experts.find_llama_perplexity(Path("/given/bin"), environ={"LLAMA_PERPLEXITY": "/env/bin"}, which=lambda name: "/path/bin"),
                         Path("/given/bin"))

    def test_the_environment_then_the_path_then_a_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            build = home.joinpath(*kld_experts.CONVENTIONAL_LLAMA_CPP, "build", "bin")
            build.mkdir(parents=True)
            (build / "llama-perplexity").write_text("")
            find = lambda environ, which: kld_experts.find_llama_perplexity(None, environ=environ, which=which, home=home)
            self.assertEqual(find({"LLAMA_PERPLEXITY": "/env/bin"}, lambda name: "/path/bin"), Path("/env/bin"))
            self.assertEqual(find({}, lambda name: "/path/bin"), Path("/path/bin"))
            self.assertEqual(find({}, lambda name: None), build / "llama-perplexity")
            self.assertIsNone(kld_experts.find_llama_perplexity(None, environ={}, which=lambda name: None, home=home / "elsewhere"))

    def test_the_build_hint_names_the_source_and_does_not_build(self):
        with tempfile.TemporaryDirectory() as directory:
            hint = kld_experts.build_hint(environ={"LLAMA_CPP_DIR": directory}, home=Path("/nowhere"))
        self.assertIn(f"-S {directory}", hint)
        self.assertIn("--target llama-perplexity", hint)
        self.assertIn("nice -n 19", hint)
        self.assertIn("<llama.cpp checkout>", kld_experts.build_hint(environ={}, home=Path("/nowhere")))


class TextTests(unittest.TestCase):
    def test_the_default_text_is_the_listed_files_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("first")
            (root / "b.md").write_text("second")
            self.assertEqual(kld_experts.default_text(root, ("a.md", "b.md")), "first\n\nsecond")
            self.assertEqual(kld_experts.default_text(root, ("b.md", "a.md")), "second\n\nfirst")

    def test_the_default_files_exist_in_this_checkout(self):
        text = kld_experts.default_text(kld_experts.ROOT)
        self.assertGreater(kld_experts.estimate_tokens(text), kld_experts.DEFAULT_CTX * kld_experts.DEFAULT_CHUNKS)

    def test_the_token_estimate_rounds_up(self):
        self.assertEqual(kld_experts.estimate_tokens("x" * 7), 2)
        self.assertEqual(kld_experts.estimate_tokens(""), 0)

    def test_the_logits_file_size_counts_the_scored_half_of_each_chunk(self):
        size = kld_experts.base_logits_bytes(512, 24)
        self.assertEqual(size, 24 * 255 * kld_experts.LOGITS_VOCAB_ESTIMATE * 2)


class BaseLogitsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ref = self.root / "ref.gguf"
        self.ref.write_bytes(b"0123456789")
        self.base = self.root / "base.kld"

    def tearDown(self):
        self.directory.cleanup()

    def test_a_missing_file_is_not_reusable(self):
        signature = kld_experts.base_signature(self.ref, "abc", 512, 24)
        self.assertEqual(kld_experts.reusable_base(self.base, signature)[0], False)

    def test_a_file_saved_with_the_same_inputs_is_reused(self):
        signature = kld_experts.base_signature(self.ref, "abc", 512, 24)
        self.base.write_bytes(b"x")
        kld_experts.sidecar_path(self.base).write_text(json.dumps(signature))
        self.assertEqual(kld_experts.reusable_base(self.base, signature), (True, ""))

    def test_a_different_text_context_chunks_or_reference_is_refused_and_named(self):
        saved = kld_experts.base_signature(self.ref, "abc", 512, 24)
        self.base.write_bytes(b"x")
        kld_experts.sidecar_path(self.base).write_text(json.dumps(saved))
        usable, reason = kld_experts.reusable_base(self.base, kld_experts.base_signature(self.ref, "other", 1024, 24))
        self.assertFalse(usable)
        self.assertIn("ctx", reason)
        self.assertIn("text_sha256", reason)
        self.ref.write_bytes(b"0123456789 and more")
        usable, reason = kld_experts.reusable_base(self.base, kld_experts.base_signature(self.ref, "abc", 512, 24))
        self.assertFalse(usable)
        self.assertIn("ref_size", reason)

    def test_a_file_without_a_description_is_refused(self):
        self.base.write_bytes(b"x")
        self.assertFalse(kld_experts.reusable_base(self.base, kld_experts.base_signature(self.ref, "abc", 512, 24))[0])


class RowTests(unittest.TestCase):
    def test_a_good_run_is_ok(self):
        row = kld_experts.measurement_row(6, MODEL_LOG % 6 + llama_summary(), finished())
        self.assertEqual((row["k"], row["status"], row["mean_kld"], row["experts_logged"]), (6, "ok", 0.012345, 6))
        self.assertEqual((row["seconds"], row["pauses"], row["peak_c"]), (61.2, 1, 74.0))

    def test_an_override_that_did_not_take_effect_is_an_error(self):
        row = kld_experts.measurement_row(6, MODEL_LOG % 8 + llama_summary(), finished())
        self.assertIn("override was not applied", row["status"])

    def test_a_log_without_the_expert_line_is_not_an_error(self):
        self.assertEqual(kld_experts.measurement_row(6, llama_summary(), finished())["status"], "ok")

    def test_a_failed_run_says_so(self):
        self.assertIn("exited with 1", kld_experts.measurement_row(6, "", finished(returncode=1))["status"])

    def test_a_run_too_short_for_statistics_says_so(self):
        self.assertIn("no KL divergence summary", kld_experts.measurement_row(6, MODEL_LOG % 6, finished())["status"])

    def test_a_stopped_run_says_why(self):
        self.assertEqual(kld_experts.measurement_row(6, "", finished(None, "time budget"))["status"], "stopped: time budget")


class TableTests(unittest.TestCase):
    def rows(self):
        return [kld_experts.measurement_row(8, MODEL_LOG % 8 + llama_summary(0.001234, 1.0004, 98.9, 0.0123), finished()),
                kld_experts.measurement_row(6, MODEL_LOG % 6 + llama_summary(0.034567, 1.0213, 94.2, 0.2345), finished()),
                {"k": 5, "status": "skipped: time budget"}]

    def test_the_table_has_the_asked_columns_and_one_line_per_count(self):
        lines = kld_experts.format_table(self.rows()).splitlines()
        self.assertEqual(lines[0].split(), ["K", "mean", "KLD", "99%", "KLD", "same", "token", "%", "PPL", "ratio", "status"])
        self.assertEqual(len(lines), 5)
        self.assertIn("0.03457 ± 0.00032", lines[3])
        self.assertIn("0.23450", lines[3])
        self.assertIn("94.200 ± 0.222", lines[3])
        self.assertIn("1.02130 ± 0.00092", lines[3])
        self.assertTrue(lines[4].startswith("5 "))
        self.assertIn("skipped: time budget", lines[4])

    def test_the_csv_has_one_row_per_count_and_the_numbers(self):
        lines = kld_experts.format_csv(self.rows()).splitlines()
        self.assertEqual(lines[0].split(",")[:2], ["k", "mean_kld"])
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[2].startswith("6,0.034567,"))
        self.assertTrue(lines[3].startswith("5,,"))


class WatchTests(unittest.TestCase):
    class Process:
        pid = 4242

        def __init__(self, polls):
            self.polls = polls
            self.waited = False

        def poll(self):
            if self.polls:
                self.polls -= 1
                return None
            return 0

        def wait(self, timeout=None):
            self.waited = True

    def run_watch(self, temperatures, polls=3, deadline=None, max_pause=1200.0):
        signals, now = [], [0.0]
        readings = iter(temperatures)
        process = self.Process(polls)

        def sleep(seconds):
            now[0] += seconds

        with tempfile.TemporaryDirectory() as directory:
            run = kld_experts.run_watched(
                ["llama-perplexity"], Path(directory) / "log", 80.0, 70.0, deadline, lambda: next(readings, 50.0),
                popen=lambda *a, **k: process, send_signal=lambda pid, number: signals.append((pid, number)), sleep=sleep,
                clock=lambda: now[0], max_pause=max_pause)
        return run, signals, process

    def test_a_cool_machine_is_not_interrupted(self):
        run, signals, _ = self.run_watch([60.0, 65.0, 70.0])
        self.assertEqual((run.returncode, run.pauses, run.peak_c, signals, run.stopped), (0, 0, 70.0, [], None))

    def test_the_run_is_stopped_at_the_limit_and_continued_when_cool(self):
        run, signals, _ = self.run_watch([60.0, 81.0, 78.0, 69.0, 60.0])
        self.assertEqual(signals, [(4242, signal.SIGSTOP), (4242, signal.SIGCONT)])
        self.assertEqual((run.pauses, run.peak_c, run.stopped), (1, 81.0, None))
        self.assertGreater(run.paused_seconds, 0)

    def test_a_machine_that_does_not_cool_ends_the_run(self):
        run, signals, process = self.run_watch([81.0] + [79.0] * 100, polls=1000, max_pause=30.0)
        self.assertTrue(run.stopped.startswith("too hot"))
        self.assertEqual(signals[0], (4242, signal.SIGSTOP))
        self.assertEqual(signals[-1], (4242, signal.SIGKILL))
        self.assertTrue(process.waited)

    def test_the_deadline_ends_the_run(self):
        run, signals, _ = self.run_watch([60.0] * 100, polls=1000, deadline=10.0)
        self.assertEqual(run.stopped, "time budget")
        self.assertEqual(signals[-1], (4242, signal.SIGKILL))

    def test_an_unreadable_sensor_does_not_stop_the_run(self):
        run, signals, _ = self.run_watch([None, None, None])
        self.assertEqual((run.returncode, signals, run.peak_c), (0, [], None))


class MainTests(unittest.TestCase):
    def run_main(self, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run([sys.executable, "-I", str(TOOLS / "kld_experts.py"), *arguments, "--out", str(Path(directory) / "result")],
                                  capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False, timeout=60,
                                  env={"PATH": "/usr/bin:/bin", "HOME": directory})
            files = sorted(path.name for path in Path(directory).iterdir())
        return done.returncode, done.stdout, done.stderr, files

    def test_a_dry_run_prints_every_command_and_writes_nothing(self):
        code, out, err, files = self.run_main("--dry-run", "--model", "/m/model.gguf", "--ref-model", "/m/ref.gguf", "--ks", "8,6",
                                              "--llama-perplexity", "/b/llama-perplexity", "--ctx", "256", "--chunks", "10")
        self.assertEqual(code, 0, err)
        self.assertEqual(files, [])
        commands = [line for line in out.splitlines() if not line.startswith("#")]
        self.assertEqual(len(commands), 3)
        self.assertIn("--kl-divergence-base", commands[0])
        self.assertNotIn("--override-kv", commands[0])
        self.assertIn("qwen35moe.expert_used_count=int:8", commands[1])
        self.assertIn("qwen35moe.expert_used_count=int:6", commands[2])
        self.assertIn("tokens", err)

    def test_a_missing_binary_prints_the_build_command_and_builds_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            model, ref = Path(directory) / "m.gguf", Path(directory) / "r.gguf"
            model.write_bytes(b"x")
            ref.write_bytes(b"x")
            code, out, err, files = self.run_main("--model", str(model), "--ref-model", str(ref))
        self.assertEqual(code, 2)
        self.assertIn("llama-perplexity was not found", err)
        self.assertIn("cmake -S", err)
        self.assertEqual(files, [])

    def test_a_bad_argument_is_refused_before_anything_starts(self):
        code, out, err, files = self.run_main("--dry-run", "--model", "m", "--ref-model", "r", "--cpu-max-mhz", "100")
        self.assertEqual(code, 2)
        self.assertIn("--cpu-max-mhz", err)


if __name__ == "__main__":
    unittest.main()
