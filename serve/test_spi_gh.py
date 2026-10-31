import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import spi_gh_broker as broker_module
from spi_gh_broker import Broker, BrokerServer, Denied, HostTokens, Policy

BROKER = ROOT / "tools" / "spi_gh_broker.py"
SHIM = ROOT / "tools" / "spi_gh_shim.py"
TOKEN = "gho_" + "a1B2c3D4" * 4 + "xyz0"

FAKE_GH = f"""#!{sys.executable}
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "behaviour")
behaviour = open(path).read().strip() if os.path.exists(path) else ""
open(os.path.join(here, "ran"), "a").write("x")
if behaviour == "flood":
    sys.stdout.write("x" * 100000)
elif behaviour == "sleep":
    time.sleep(30)
elif behaviour == "token":
    sys.stdout.write("token is " + os.environ["GH_TOKEN"] + " and ghp_" + "b" * 30)
elif behaviour == "fail":
    sys.stderr.write("boom\\n")
    sys.exit(3)
else:
    json.dump({{"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()}}, sys.stdout)
"""


def policy(*repos):
    return Policy(list(repos or ("acme/api", "Acme/lib")))


def token_run(command, **kwargs):
    return subprocess.CompletedProcess(command, 0, stdout=TOKEN + "\n")


class RemoteParsingTests(unittest.TestCase):
    def test_remote_urls(self):
        cases = {
            "git@github.com:acme/api.git": "acme/api",
            "git@github.com-work:acme/tool.git": "acme/tool",
            "https://github.com/Third/Party.git": "Third/Party",
            "https://user:secret@github.com/acme/api": "acme/api",
            "ssh://git@github.com/acme/api.git": "acme/api",
            "https://github.com/acme/api/": "acme/api",
            "git@gitlab.com:acme/api.git": None,
            "https://github.com.evil.com/acme/api": None,
            "git@github.com-x.evil.com:acme/api.git": None,
            "https://github.com-attacker/acme/private.git": None,
            "ftp://github.com/acme/private.git": None,
            "file:///srv/git/acme/api.git": None,
            "ssh://git@github.com-work/acme/tool.git": "acme/tool",
            "git://github.com/acme/api.git": "acme/api",
            "https://github.com/acme": None,
            "/srv/git/acme/api.git": None,
            "": None,
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(broker_module.repo_from_remote_url(url), expected)

    def test_git_remotes_put_origin_first_and_skip_other_hosts(self):
        with tempfile.TemporaryDirectory() as folder:
            for args in (
                ("init", "-q"),
                ("remote", "add", "upstream", "https://github.com/Other/Fork.git"),
                ("remote", "add", "mirror", "git@gitlab.com:acme/api.git"),
                ("remote", "add", "origin", "git@github.com-work:acme/tool.git"),
            ):
                subprocess.run(["git", "-C", folder, *args], check=True, capture_output=True)
            self.assertEqual(broker_module.repos_from_git_remotes(folder), ["acme/tool", "Other/Fork"])

    def test_a_folder_without_git_has_no_repositories(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(broker_module.repos_from_git_remotes(folder), [])

    def test_repositories_are_unique_ignoring_case(self):
        self.assertEqual(Policy(["acme/api", "ACME/API", "acme/lib"]).repos, ["acme/api", "acme/lib"])


class AllowedRequestTests(unittest.TestCase):
    def check(self, argv, expected_args, expected_repo):
        plan = policy().translate(argv)
        self.assertEqual(plan.args, expected_args)
        self.assertEqual(plan.repo, expected_repo)

    def test_pull_request_view_with_json_fields(self):
        self.check(["pr", "view", "636", "--json", "title,state"], ["pr", "view", "636", "--json=title,state", "--repo=acme/api"], "acme/api")

    def test_comments_switch_and_short_form(self):
        self.check(["pr", "view", "7", "-c"], ["pr", "view", "7", "--comments", "--repo=acme/api"], "acme/api")

    def test_a_pull_request_url_picks_its_repository_in_the_allowed_spelling(self):
        self.check(["pr", "view", "https://github.com/ACME/lib/pull/7"], ["pr", "view", "7", "--repo=Acme/lib"], "Acme/lib")

    def test_a_branch_name_is_accepted_for_pull_requests(self):
        self.check(["pr", "diff", "feature/x-1"], ["pr", "diff", "feature/x-1", "--repo=acme/api"], "acme/api")

    def test_repo_flag_may_name_any_allowed_repository_in_any_case(self):
        self.check(["pr", "view", "-R", "ACME/LIB", "5"], ["pr", "view", "5", "--repo=Acme/lib"], "Acme/lib")
        self.check(["pr", "view", "5", "--repo=acme/lib"], ["pr", "view", "5", "--repo=Acme/lib"], "Acme/lib")

    def test_list_filters(self):
        self.check(
            ["pr", "list", "-s", "open", "-L", "5", "--search", "is:open label:bug", "--draft"],
            ["pr", "list", "--state=open", "--limit=5", "--search=is:open label:bug", "--draft", "--repo=acme/api"],
            "acme/api",
        )

    def test_flags_are_rebuilt_in_long_form_with_equals(self):
        plan = policy().translate(["issue", "list", "--author", "@me", "-l", "-leading-dash"])
        self.assertEqual(plan.args, ["issue", "list", "--author=@me", "--label=-leading-dash", "--repo=acme/api"])

    def test_run_view_takes_long_run_ids(self):
        self.check(["run", "view", "12345678901234", "--log-failed"], ["run", "view", "12345678901234", "--log-failed", "--repo=acme/api"], "acme/api")

    def test_repo_view_gets_the_repository_as_an_argument_and_no_repo_flag(self):
        self.check(["repo", "view"], ["repo", "view", "acme/api"], "acme/api")
        self.check(["repo", "view", "https://github.com/acme/lib"], ["repo", "view", "Acme/lib"], "Acme/lib")

    def test_api_get_under_the_repository(self):
        self.check(
            ["api", "repos/acme/api/pulls/1/comments", "--paginate"],
            ["api", "--method=GET", "--paginate", "repos/acme/api/pulls/1/comments"],
            "acme/api",
        )

    def test_api_placeholders_and_query(self):
        self.check(
            ["api", "-X", "get", "/repos/{owner}/{repo}/issues?state=all&per_page=100"],
            ["api", "--method=GET", "repos/acme/api/issues?state=all&per_page=100"],
            "acme/api",
        )

    def test_api_other_read_topics(self):
        for path in (
            "repos/ACME/api/commits/abc123/check-runs",
            "repos/acme/api/compare/main...feature",
            "repos/acme/api/actions/runs/123/jobs",
            "repos/acme/api/actions/jobs/5",
            "repos/acme/api/git/trees/main",
            "repos/acme/api/contents/src/main.go?ref=dev",
            "repos/acme/api",
        ):
            with self.subTest(path=path):
                plan = policy().translate(["api", path])
                self.assertEqual(plan.repo, "acme/api")
                self.assertTrue(plan.args[-1].startswith("repos/acme/api"))

    def test_api_header_and_fields(self):
        plan = policy().translate(["api", "repos/acme/api/pulls", "-H", "Accept: application/vnd.github.diff", "-f", "state=open", "-F", "per_page=50"])
        self.assertEqual(
            plan.args,
            ["api", "--method=GET", "--header=Accept: application/vnd.github.diff", "--raw-field=state=open", "--field=per_page=50", "repos/acme/api/pulls"],
        )

    def test_help_and_auth_status_are_answered_without_gh(self):
        with tempfile.TemporaryDirectory() as home:
            broker = Broker(policy(), "/nonexistent/gh", Path(home))
            for argv in ([], ["help"], ["--help"], ["auth", "status"]):
                with self.subTest(argv=argv):
                    reply = broker.serve(json.dumps({"argv": argv}).encode())
                    self.assertEqual(reply.rc, 0)
                    self.assertIn("read-only", reply.stdout)
                    self.assertIn("repositories: acme/api, Acme/lib (acme/api is the default)", reply.stdout)


class DeniedRequestTests(unittest.TestCase):
    def assertDenied(self, argv, policy_under_test=None):
        with self.assertRaises(Denied, msg=f"{argv} was accepted"):
            (policy_under_test or policy()).translate(argv)

    def test_writing_and_account_commands(self):
        for argv in (
            ["pr", "merge", "1"],
            ["pr", "comment", "1", "-b", "x"],
            ["pr", "create"],
            ["pr", "review", "1", "--approve"],
            ["pr", "close", "1"],
            ["issue", "create"],
            ["issue", "comment", "1"],
            ["repo", "delete"],
            ["repo", "clone", "acme/api"],
            ["release", "create"],
            ["gist", "create"],
            ["workflow", "run"],
            ["run", "rerun", "1"],
            ["run", "cancel", "1"],
            ["secret", "list"],
            ["ssh-key", "list"],
            ["auth", "token"],
            ["auth", "login"],
            ["config", "get", "x"],
            ["alias", "set", "x", "y"],
            ["extension", "install", "x"],
            ["codespace", "ssh"],
            ["search", "code", "x"],
            ["pr"],
            ["--version"],
            ["-R", "acme/api", "pr", "view", "1"],
        ):
            with self.subTest(argv=argv):
                self.assertDenied(argv)

    def test_flags_that_evaluate_open_or_widen(self):
        for argv in (
            ["pr", "view", "1", "--jq", ".title"],
            ["pr", "view", "1", "-q", ".title"],
            ["pr", "view", "1", "--template", "{{.}}"],
            ["pr", "view", "1", "--web"],
            ["pr", "view", "1", "-R", "other/repo"],
            ["pr", "view", "1", "--repo=other/repo"],
            ["pr", "view", "--", "1"],
            ["pr", "view", "1", "2"],
            ["pr", "view", "../../x"],
            ["pr", "view", "x..y"],
            ["pr", "view", "-1"],
            ["pr", "view", "1", "--json"],
            ["pr", "view", "1", "--json=a b"],
            ["pr", "view", "1", "--comments=yes"],
            ["pr", "view", "https://github.com/other/repo/pull/1"],
            ["pr", "view", "https://evil.example/acme/api/pull/1"],
            ["pr", "view", "https://github.com/acme/api/pull/1", "-R", "acme/lib"],
            ["pr", "list", "--limit", "abc"],
            ["pr", "list", "--limit", "99999"],
            ["pr", "list", "extra"],
            ["pr", "list", "-L5"],
            ["pr", "list", "--search", "repo:other/private is:pr"],
            ["pr", "list", "--search", "x org:other"],
            ["pr", "list", "--search", "(user:someone)"],
            ["pr", "list", "--search", "x\ny"],
            ["run", "view", "abc"],
            ["repo", "view", "other/repo"],
            ["repo", "view", "-R", "acme/api"],
        ):
            with self.subTest(argv=argv):
                self.assertDenied(argv)

    def test_api_that_writes_leaks_or_leaves_the_repository(self):
        for argv in (
            ["api", "-X", "POST", "repos/acme/api/issues"],
            ["api", "--method=DELETE", "repos/acme/api"],
            ["api", "graphql"],
            ["api", "user"],
            ["api", "repos/other/x/pulls"],
            ["api", "repos/acme/api/../other/x"],
            ["api", "repos/acme/api/pulls/%2e%2e/x"],
            ["api", "repos/acme/api/pulls//1"],
            ["api", "repos/acme/api/actions/secrets"],
            ["api", "repos/acme/api/actions/variables"],
            ["api", "repos/acme/api/hooks"],
            ["api", "repos/acme/api/pulls", "--input", "x"],
            ["api", "repos/acme/api/pulls", "-F", "body=@/home/user/.ssh/id_rsa"],
            ["api", "repos/acme/api/pulls", "-f", "a=@x"],
            ["api", "repos/acme/api/pulls", "-H", "X-HTTP-Method-Override: DELETE"],
            ["api", "repos/acme/api/pulls", "--jq", ".[]"],
            ["api", "repos/acme/api/pulls", "-i"],
            ["api", "repos/acme/api/pulls", "--verbose"],
            ["api", "--hostname", "evil.example", "repos/acme/api"],
            ["api", "repos/acme/api/pulls", "repos/acme/api/issues"],
            ["api", "repos/acme/api/pulls?x=1#frag"],
            ["api", "repos/acme/api/pulls?x=a b"],
            ["api", "https://api.github.com/repos/acme/api"],
            ["api"],
        ):
            with self.subTest(argv=argv):
                self.assertDenied(argv)

    def test_nothing_is_available_without_repositories(self):
        for argv in (["pr", "view", "1"], ["api", "repos/acme/api"]):
            with self.subTest(argv=argv):
                self.assertDenied(argv, Policy([]))


class HostTokensTests(unittest.TestCase):
    def fake_run(self, answers):
        calls = []

        def run(command, **kwargs):
            arguments = command[4:]
            calls.append(arguments)
            stdout, rc = answers.get(tuple(arguments), ("", 1))
            return subprocess.CompletedProcess(command, rc, stdout=stdout)

        return run, calls

    def test_an_account_named_like_the_owner_wins(self):
        run, calls = self.fake_run({("auth", "token", "--user", "jdoe"): ("personal\n", 0), ("auth", "token"): ("active\n", 0)})
        self.assertEqual(HostTokens("gh", run).for_owner("jdoe"), "personal")
        self.assertEqual(calls, [["auth", "token", "--user", "jdoe"]])

    def test_any_other_owner_uses_the_active_account(self):
        run, calls = self.fake_run({("auth", "token"): ("active\n", 0)})
        self.assertEqual(HostTokens("gh", run).for_owner("acme"), "active")
        self.assertEqual(calls, [["auth", "token", "--user", "acme"], ["auth", "token"]])

    def test_no_login_at_all_is_a_denial(self):
        run, _ = self.fake_run({})
        with self.assertRaises(Denied):
            HostTokens("gh", run).for_owner("acme")

    def test_a_missing_gh_binary_is_a_denial(self):
        with self.assertRaises(Denied):
            HostTokens("/nonexistent/gh").for_owner("acme")


class BrokerRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.gh = self.dir / "gh"
        self.gh.write_text(FAKE_GH, encoding="utf-8")
        self.gh.chmod(0o755)
        self.home = self.dir / "home"
        self.home.mkdir()
        self.log = self.dir / "log"
        self.broker = Broker(policy(), str(self.gh), self.home, self.log, HostTokens(str(self.gh), token_run))

    def behave(self, text):
        (self.dir / "behaviour").write_text(text, encoding="utf-8")

    def ask(self, *argv):
        return self.broker.serve(json.dumps({"argv": list(argv)}).encode())

    def test_gh_runs_with_a_clean_environment_and_only_the_token(self):
        reply = self.ask("pr", "view", "1", "--json", "title")
        self.assertEqual(reply.rc, 0, reply.stderr)
        seen = json.loads(reply.stdout)
        self.assertEqual(seen["argv"], ["pr", "view", "1", "--json=title", "--repo=acme/api"])
        self.assertEqual(seen["cwd"], "/")
        passthrough = {"HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE"}
        self.assertEqual(
            set(seen["env"]) - passthrough,
            {"PATH", "HOME", "GH_CONFIG_DIR", "GH_HOST", "GH_TOKEN", "GH_PROMPT_DISABLED", "GH_NO_UPDATE_NOTIFIER", "GH_NO_EXTENSION_UPDATE_NOTIFIER", "NO_COLOR", "LANG"},
        )
        self.assertEqual(seen["env"]["HOME"], str(self.home))
        self.assertEqual(seen["env"]["GH_HOST"], "github.com")

    def test_a_denied_request_never_starts_gh(self):
        reply = self.ask("pr", "merge", "1")
        self.assertEqual(reply.rc, 1)
        self.assertEqual(reply.stdout, "")
        self.assertIn("not available", reply.stderr)
        self.assertIn("gh help", reply.stderr)
        self.assertFalse((self.dir / "ran").exists())
        self.ask("pr", "view", "1")
        self.assertTrue((self.dir / "ran").exists())

    def test_gh_runs_under_coreutils_timeout(self):
        command = broker_module.guarded(["gh", "x"], 7)
        self.assertEqual(command[0], broker_module.TIMEOUT_BINARY)
        self.assertEqual(command[2:], ["7", "gh", "x"])

    def test_without_the_timeout_binary_gh_is_not_started(self):
        with mock.patch.object(broker_module, "TIMEOUT_BINARY", None):
            reply = self.ask("pr", "view", "1")
        self.assertEqual(reply.rc, 2)
        self.assertIn("could not run gh", reply.stderr)
        self.assertFalse((self.dir / "ran").exists())

    def test_the_exit_code_and_stderr_of_gh_are_passed_on(self):
        self.behave("fail")
        reply = self.ask("pr", "list")
        self.assertEqual((reply.rc, reply.stderr), (3, "boom\n"))

    def test_token_like_strings_are_removed_from_the_output(self):
        self.behave("token")
        reply = self.ask("pr", "list")
        self.assertNotIn(TOKEN, reply.stdout)
        self.assertNotIn("ghp_", reply.stdout)
        self.assertEqual(reply.stdout.count("[token removed]"), 2)

    def test_a_flood_is_cut_at_the_limit(self):
        self.behave("flood")
        with mock.patch.object(broker_module, "OUTPUT_LIMIT_BYTES", 1000):
            reply = self.ask("pr", "list")
        self.assertEqual(len(reply.stdout), 1000)
        self.assertIn("output stopped", reply.stderr)

    def test_a_slow_gh_is_stopped_at_the_timeout(self):
        self.behave("sleep")
        started = time.monotonic()
        with mock.patch.object(broker_module, "RUN_TIMEOUT_SECONDS", 1):
            reply = self.ask("pr", "list")
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("output stopped", reply.stderr)
        self.assertNotEqual(reply.rc, 0)

    def test_malformed_requests_are_refused(self):
        for raw in (b"", b"not json", b'{"argv": "pr"}', b'{"argv": [1]}', b'{"argv": ["a\\u0000b"]}', json.dumps({"argv": ["x"] * 60}).encode(), b"x" * 70000):
            with self.subTest(raw=raw[:30]):
                self.assertEqual(self.broker.serve(raw).rc, 2)

    def test_every_request_is_logged_on_one_line(self):
        self.ask("pr", "view", "1")
        self.ask("pr", "merge", "1\nforged ALLOW line")
        lines = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(" ALLOW rc=0 ", lines[0])
        self.assertIn(" DENY rc=1 ", lines[1])
        self.assertNotIn(TOKEN, "".join(lines))

    def test_an_unwritable_log_does_not_break_the_answer(self):
        self.broker.log_path = self.dir / "missing" / "log"
        self.assertEqual(self.ask("pr", "view", "1").rc, 0)

    def test_the_log_stays_bounded_during_a_session(self):
        with mock.patch.object(broker_module, "LOG_TRIM_ABOVE_BYTES", 2000), mock.patch.object(broker_module, "LOG_KEEP_BYTES", 500):
            for _ in range(200):
                self.ask("help")
        self.assertLess(self.log.stat().st_size, 2200)
        self.assertTrue(self.log.read_text(encoding="utf-8").endswith("\n"))

    def test_stopping_the_broker_kills_a_running_gh(self):
        self.behave("sleep")
        results = []
        thread = threading.Thread(target=lambda: results.append(self.ask("pr", "list")))
        thread.start()
        for _ in range(50):
            if (self.dir / "ran").exists():
                break
            time.sleep(0.1)
        self.assertTrue((self.dir / "ran").exists())
        self.broker.tracker.kill_all()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertNotEqual(results[0].rc, 0)
        self.assertEqual(self.broker.tracker.groups, set())

    def test_a_request_that_arrives_after_the_shutdown_sweep_is_killed(self):
        self.behave("sleep")
        self.broker.tracker.kill_all()
        started = time.monotonic()
        reply = self.ask("pr", "list")
        self.assertLess(time.monotonic() - started, 10)
        self.assertNotEqual(reply.rc, 0)
        self.assertEqual(self.broker.tracker.groups, set())

    def test_log_trimming_keeps_whole_lines_of_the_tail(self):
        self.log.write_text("".join(f"line {number}\n" for number in range(100000)), encoding="utf-8")
        broker_module.trim_log(self.log)
        text = self.log.read_text(encoding="utf-8")
        self.assertLessEqual(len(text.encode()), broker_module.LOG_KEEP_BYTES)
        self.assertTrue(text.startswith("line "))
        self.assertTrue(text.endswith("line 99999\n"))


class SocketRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        gh = self.dir / "gh"
        gh.write_text(FAKE_GH, encoding="utf-8")
        gh.chmod(0o755)
        home = self.dir / "home"
        home.mkdir()
        self.socket = str(self.dir / "gh.sock")
        self.server = BrokerServer(self.socket, Broker(policy(), str(gh), home, None, HostTokens(str(gh), token_run)))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def shim(self, *argv):
        env = {"SPI_GH_SOCKET": self.socket, "PATH": os.environ["PATH"]}
        return subprocess.run([sys.executable, str(SHIM), *argv], capture_output=True, text=True, env=env, timeout=30, check=False)

    def test_the_socket_is_private_to_the_user(self):
        self.assertEqual(stat.S_IMODE(os.stat(self.socket).st_mode), 0o600)

    def test_the_shim_returns_stdout_and_the_exit_code(self):
        result = self.shim("pr", "view", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["argv"][:3], ["pr", "view", "5"])

    def test_the_shim_shows_a_denial_on_stderr_with_a_failing_code(self):
        result = self.shim("pr", "merge", "5")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("read-only", result.stderr)

    def test_several_requests_in_parallel(self):
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.shim("pr", "list").returncode)) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(results, [0] * 6)

    def test_the_shim_explains_a_missing_broker(self):
        env = {"SPI_GH_SOCKET": str(self.dir / "nothing.sock"), "PATH": os.environ["PATH"]}
        result = subprocess.run([sys.executable, str(SHIM), "pr", "list"], capture_output=True, text=True, env=env, timeout=30, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not reachable", result.stderr)


def read_all(conn):
    chunks = []
    while chunk := conn.recv(65536):
        chunks.append(chunk)
    return b"".join(chunks)


class ConnectionLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.socket = str(self.dir / "gh.sock")
        broker = Broker(policy(), "/nonexistent/gh", self.dir)
        with mock.patch.object(broker_module, "MAX_CONNECTIONS", 2):
            self.server = BrokerServer(self.socket, broker)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def connect(self):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(5)
        conn.connect(self.socket)
        self.addCleanup(conn.close)
        return conn

    def test_idle_connections_cannot_hold_the_broker(self):
        with mock.patch.object(broker_module, "CONNECTION_TIMEOUT_SECONDS", 1):
            idle = [self.connect(), self.connect()]
            time.sleep(0.3)
            extra = self.connect()
            self.assertEqual(extra.recv(10), b"")
            time.sleep(1.5)
            self.assertEqual(idle[0].recv(10), b"")
            fresh = self.connect()
            fresh.sendall(b'{"argv": ["help"]}\n')
            fresh.shutdown(socket.SHUT_WR)
            self.assertIn(b"read-only", read_all(fresh))

    def test_a_slow_trickle_is_cut_at_the_total_deadline(self):
        with mock.patch.object(broker_module, "CONNECTION_TIMEOUT_SECONDS", 1):
            conn = self.connect()
            started = time.monotonic()
            try:
                for _ in range(20):
                    conn.sendall(b" ")
                    time.sleep(0.4)
            except OSError:
                pass
            self.assertLess(time.monotonic() - started, 4)

    def test_a_request_without_a_newline_is_dropped_after_the_timeout(self):
        with mock.patch.object(broker_module, "CONNECTION_TIMEOUT_SECONDS", 1):
            conn = self.connect()
            conn.sendall(b'{"argv": ["help"]')
            started = time.monotonic()
            self.assertEqual(conn.recv(10), b"")
            self.assertLess(time.monotonic() - started, 4)


class BrokerProcessTests(unittest.TestCase):
    def test_the_broker_exits_and_removes_its_socket_when_the_parent_is_gone(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            socket_path = Path(folder) / "gh.sock"
            parent = subprocess.Popen(["sleep", "2"])
            self.addCleanup(parent.kill)
            process = subprocess.Popen(
                [sys.executable, str(BROKER), "--socket", str(socket_path), "--cwd", folder, "--repos", "acme/api", "--parent-pid", str(parent.pid), "--gh", sys.executable],
                stderr=subprocess.PIPE,
            )
            self.addCleanup(process.kill)
            for _ in range(50):
                if socket_path.exists():
                    break
                time.sleep(0.1)
            self.assertTrue(socket_path.exists())
            parent.wait()
            self.assertEqual(process.wait(timeout=15), 0)
            self.assertFalse(socket_path.exists())

    def test_the_broker_refuses_to_start_without_the_timeout_binary(self):
        with mock.patch.object(broker_module, "TIMEOUT_BINARY", None), mock.patch("sys.stderr") as stderr:
            self.assertEqual(broker_module.main(["--socket", "/tmp/never-created.sock", "--gh", "/usr/bin/true"]), 1)
        self.assertIn("timeout", "".join(call.args[0] for call in stderr.write.call_args_list))

    def test_the_broker_refuses_to_start_without_gh(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            result = subprocess.run([sys.executable, str(BROKER), "--socket", f"{folder}/s", "--gh", ""], capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
