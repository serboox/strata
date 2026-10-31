#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

GITHUB_HOST = "github.com"
REQUEST_LIMIT_BYTES = 64 * 1024
MAX_ARGUMENTS = 48
MAX_ARGUMENT_CHARS = 2048
OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
ERROR_LIMIT_BYTES = 64 * 1024
RUN_TIMEOUT_SECONDS = 60
TOKEN_TIMEOUT_SECONDS = 15
MAX_PARALLEL_RUNS = 4
MAX_CONNECTIONS = 16
CONNECTION_TIMEOUT_SECONDS = 10
REPLY_TIMEOUT_SECONDS = 30
KILL_GRACE_SECONDS = 5
TIMEOUT_EXIT_CODE = 124
TIMEOUT_BINARY = shutil.which("timeout")
READ_CHUNK_BYTES = 4096
REMOTE_SCHEMES = frozenset({"https", "http", "ssh", "git"})
BUSY_WAIT_SECONDS = 30
PARENT_POLL_SECONDS = 2
LOG_TRIM_ABOVE_BYTES = 1024 * 1024
LOG_KEEP_BYTES = 256 * 1024
LOG_ARGUMENTS_CHARS = 400
DENIED_EXIT_CODE = 1
USAGE_EXIT_CODE = 2

REPO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}")
SSH_ALIAS_HOST = re.compile(r"github\.com-[a-z0-9_-]+")
SCP_REMOTE_PATTERN =re.compile(r"(?:[^@/\s]+@)?(?P<host>[^:/\s]+):(?P<path>[^/\s].*)")
TOKEN_PATTERN = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
TOKEN_PLACEHOLDER = "[token removed]"

NUMBER = re.compile(r"[1-9][0-9]{0,9}")
RUN_ID = re.compile(r"[1-9][0-9]{0,19}")
BRANCH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}")
JSON_FIELDS = r"[A-Za-z0-9,]{1,300}"
STATE = r"open|closed|merged|all"
LIMIT = r"[1-9][0-9]{0,3}"
LOGIN = r"[A-Za-z0-9@._-]{1,100}"
REF = r"[A-Za-z0-9._/@-]{1,200}"
TEXT = r"[^\x00-\x1f]{1,200}"
# an extra repo:, org: or user: qualifier would widen the search past the repository this session may read
SEARCH_SCOPE = re.compile(r"(?i)(?:^|[\s(+-])(?:repo|org|user|owner|enterprise):")
PULL_URL = re.compile(r"https://github\.com/(?P<repo>[^/\s]+/[^/\s]+)/pull/(?P<number>[1-9][0-9]{0,9})/?")
ISSUE_URL = re.compile(r"https://github\.com/(?P<repo>[^/\s]+/[^/\s]+)/issues/(?P<number>[1-9][0-9]{0,9})/?")
REPO_URL = re.compile(r"https://github\.com/(?P<repo>[^/\s]+/[^/\s]+?)(?:\.git)?/?")

API_HEADER = r"(?i)(?:accept|x-github-api-version): ?[A-Za-z0-9/.+;= -]{1,100}"
# a value that starts with @ makes gh read that file on the host and send it to GitHub
API_FIELD = r"[A-Za-z_][A-Za-z0-9_.\[\]-]{0,100}=(?!@)[^\x00]{0,1000}"
API_PATH_CHARS = re.compile(r"[A-Za-z0-9._~/@+:=,-]+")
API_QUERY_CHARS = re.compile(r"[A-Za-z0-9._~=&,:+%/@*-]*")
API_REPO_TOPICS = frozenset({"pulls", "issues", "commits", "compare", "branches", "check-runs", "contents"})
API_REPO_PAIRS = frozenset({("git", "trees"), ("actions", "runs"), ("actions", "jobs")})


class Denied(Exception):
    pass


@dataclass(frozen=True)
class Reply:
    rc: int
    stdout: str = ""
    stderr: str = ""

    def as_json(self) -> bytes:
        return json.dumps({"rc": self.rc, "stdout": self.stdout, "stderr": self.stderr}).encode() + b"\n"


@dataclass(frozen=True)
class Option:
    names: tuple[str, ...]
    pattern: re.Pattern[str] | None = None
    forbidden: re.Pattern[str] | None = None

    @property
    def long_name(self) -> str:
        return self.names[0]


def switch(*names: str) -> Option:
    return Option(names)


def valued(*names: str, pattern: str, forbidden: re.Pattern[str] | None = None) -> Option:
    return Option(names, re.compile(pattern), forbidden)


@dataclass(frozen=True)
class Command:
    words: tuple[str, ...]
    options: tuple[Option, ...] = ()
    positional: str | None = None
    repo_flag: bool = True

    def lookup(self) -> dict[str, Option]:
        return {name: option for option in self.options for name in option.names}


JSON_OPTION = valued("--json", pattern=JSON_FIELDS)
COMMENTS = switch("--comments", "-c")
LIST_FILTERS = (
    valued("--state", "-s", pattern=STATE),
    valued("--limit", "-L", pattern=LIMIT),
    valued("--author", "-A", pattern=LOGIN),
    valued("--assignee", "-a", pattern=LOGIN),
    valued("--label", "-l", pattern=TEXT),
    valued("--search", "-S", pattern=TEXT, forbidden=SEARCH_SCOPE),
    JSON_OPTION,
)

COMMANDS: dict[tuple[str, ...], Command] = {
    command.words: command
    for command in (
        Command(("pr", "view"), (JSON_OPTION, COMMENTS), "pull"),
        Command(("pr", "list"), LIST_FILTERS + (valued("--base", "-B", pattern=REF), valued("--head", "-H", pattern=REF), switch("--draft", "-d"))),
        Command(("pr", "diff"), (switch("--name-only"), switch("--patch")), "pull"),
        Command(("pr", "checks"), (JSON_OPTION, switch("--required")), "pull"),
        Command(("pr", "status"), (JSON_OPTION,)),
        Command(("issue", "view"), (JSON_OPTION, COMMENTS), "issue"),
        Command(("issue", "list"), LIST_FILTERS),
        Command(
            ("run", "list"),
            (
                valued("--limit", "-L", pattern=LIMIT),
                valued("--workflow", "-w", pattern=REF),
                valued("--branch", "-b", pattern=REF),
                valued("--status", "-s", pattern=r"[a-z_]{1,30}"),
                valued("--event", "-e", pattern=r"[a-z_]{1,40}"),
                valued("--user", "-u", pattern=LOGIN),
                valued("--commit", "-c", pattern=r"[0-9a-fA-F]{7,40}"),
                JSON_OPTION,
            ),
        ),
        Command(
            ("run", "view"),
            (switch("--log"), switch("--log-failed"), valued("--job", "-j", pattern=r"[1-9][0-9]{0,18}"), JSON_OPTION, switch("--exit-status")),
            "run",
        ),
        Command(("repo", "view"), (JSON_OPTION,), "repo", repo_flag=False),
    )
}
REPO_OPTION = valued("--repo", "-R", pattern=REPO_PATTERN.pattern)
API_OPTIONS = (
    switch("--paginate"),
    switch("--slurp"),
    valued("--header", "-H", pattern=API_HEADER),
    valued("--raw-field", "-f", pattern=API_FIELD),
    valued("--field", "-F", pattern=API_FIELD),
    valued("--method", "-X", pattern=r"(?i)GET"),
)
API_COMMAND = Command(("api",), API_OPTIONS)


@dataclass(frozen=True)
class Plan:
    args: list[str]
    repo: str


def normalize_repo(text: str) -> str | None:
    path = text.strip().strip("/").removesuffix(".git")
    return path if REPO_PATTERN.fullmatch(path) else None


def is_branch(text: str) -> bool:
    return BRANCH.fullmatch(text) is not None and ".." not in text and not text.endswith("/")


def repo_from_remote_url(url: str) -> str | None:
    url = url.strip()
    if "://" in url:
        parts = urlsplit(url)
        scheme, host, path = parts.scheme.lower(), parts.hostname or "", parts.path
        if scheme not in REMOTE_SCHEMES:
            return None
    else:
        match = SCP_REMOTE_PATTERN.fullmatch(url)
        if match is None:
            return None
        scheme, host, path = "ssh", match["host"], match["path"]
    host = host.lower()
    # github.com-NAME is an ssh config alias of github.com, so only the ssh forms can use it
    if host != GITHUB_HOST and not (scheme == "ssh" and SSH_ALIAS_HOST.fullmatch(host)):
        return None
    return normalize_repo(path)


def repos_from_git_remotes(folder: str) -> list[str]:
    try:
        result = subprocess.run(["git", "-C", folder, "remote", "-v"], capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    remotes: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == "(fetch)":
            remotes.append((fields[0], fields[1]))
    remotes.sort(key=lambda remote: remote[0] != "origin")
    return [repo for _, url in remotes if (repo := repo_from_remote_url(url))]


def unique_repos(repos: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for repo in repos:
        if repo.lower() not in seen:
            seen.add(repo.lower())
            unique.append(repo)
    return unique


class Policy:
    def __init__(self, repos: list[str]):
        self.repos = unique_repos(repos)

    @property
    def default_repo(self) -> str | None:
        return self.repos[0] if self.repos else None

    def allowed_repo(self, text: str) -> str:
        candidate = normalize_repo(text)
        for repo in self.repos:
            if candidate is not None and repo.lower() == candidate.lower():
                return repo
        raise Denied(f"repository {text} is not available in this session; available: {', '.join(self.repos) or 'none'}")

    def translate(self, argv: list[str]) -> Plan:
        if not self.repos:
            raise Denied("no GitHub repository is available here: the folder has no github.com remote and SPI_GH_REPOS is empty")
        if argv[:1] == ["api"]:
            return self._translate_api(argv[1:])
        command = COMMANDS.get(tuple(argv[:2]))
        if command is None:
            raise Denied(f"'gh {' '.join(argv[:2])}' is not available: this gh is read-only and covers a fixed set of commands")
        return self._translate_command(command, argv[2:])

    def _parse(self, command: Command, args: list[str], extra: tuple[Option, ...]) -> tuple[dict[str, list[str]], list[str]]:
        options = command.lookup() | {name: option for option in extra for name in option.names}
        values: dict[str, list[str]] = {}
        positionals: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            index += 1
            if token == "--":
                raise Denied("the -- separator is not accepted")
            if not token.startswith("-"):
                positionals.append(token)
                continue
            name, has_inline, inline = token.partition("=") if token.startswith("--") else (token, "", "")
            option = options.get(name)
            if option is None:
                raise Denied(f"flag {name} is not available for 'gh {' '.join(command.words)}'")
            if option.pattern is None:
                if has_inline:
                    raise Denied(f"flag {name} takes no value")
                values.setdefault(option.long_name, [])
                continue
            if has_inline:
                value = inline
            elif index < len(args):
                value = args[index]
                index += 1
            else:
                raise Denied(f"flag {name} needs a value")
            if option.pattern.fullmatch(value) is None or (option.forbidden and option.forbidden.search(value)):
                raise Denied(f"the value of {name} is not accepted")
            values.setdefault(option.long_name, []).append(value)
        return values, positionals

    def _translate_command(self, command: Command, args: list[str]) -> Plan:
        values, positionals = self._parse(command, args, (REPO_OPTION,) if command.repo_flag else ())
        repo = self.allowed_repo(values["--repo"][-1]) if "--repo" in values else None
        rebuilt = list(command.words)
        if command.positional is None:
            if positionals:
                raise Denied(f"'gh {' '.join(command.words)}' takes no argument here")
        elif len(positionals) > 1:
            raise Denied("only one argument is accepted")
        elif positionals:
            reference, url_repo = self._resolve_positional(command.positional, positionals[0])
            if url_repo is not None and repo is not None and url_repo != repo:
                raise Denied("the URL and --repo name different repositories")
            repo = repo or url_repo
            rebuilt.append(reference)
        repo = repo or self.default_repo
        assert repo is not None
        if command.positional == "repo" and not positionals:
            rebuilt.append(repo)
        for name, items in values.items():
            if name == "--repo":
                continue
            if items:
                rebuilt.extend(f"{name}={item}" for item in items)
            else:
                rebuilt.append(name)
        if command.repo_flag:
            rebuilt.append(f"--repo={repo}")
        return Plan(rebuilt, repo)

    def _resolve_positional(self, kind: str, text: str) -> tuple[str, str | None]:
        if kind == "repo":
            match = REPO_URL.fullmatch(text)
            repo = self.allowed_repo(match["repo"] if match else text)
            return repo, repo
        if kind == "run":
            if RUN_ID.fullmatch(text):
                return text, None
        else:
            match = (PULL_URL if kind == "pull" else ISSUE_URL).fullmatch(text)
            if match:
                return match["number"], self.allowed_repo(match["repo"])
            if NUMBER.fullmatch(text) or (kind == "pull" and is_branch(text)):
                return text, None
        raise Denied(f"the argument {text!r} is not accepted")

    def _translate_api(self, args: list[str]) -> Plan:
        values, positionals = self._parse(API_COMMAND, args, ())
        if len(positionals) != 1:
            raise Denied("gh api takes exactly one endpoint")
        repo, endpoint = self._check_endpoint(positionals[0])
        rebuilt = ["api", "--method=GET"]
        for name in ("--paginate", "--slurp"):
            if name in values:
                rebuilt.append(name)
        for name in ("--header", "--raw-field", "--field"):
            rebuilt.extend(f"{name}={item}" for item in values.get(name, []))
        return Plan(rebuilt + [endpoint], repo)

    def _check_endpoint(self, endpoint: str) -> tuple[str, str]:
        default = self.default_repo
        if default is not None:
            owner, name = default.split("/")
            endpoint = endpoint.replace("{owner}", owner).replace("{repo}", name)
        path, has_query, query = endpoint.lstrip("/").partition("?")
        segments = path.split("/")
        if not API_PATH_CHARS.fullmatch(path) or any(segment in ("", ".", "..") for segment in segments):
            raise Denied("the endpoint path is not accepted")
        if API_QUERY_CHARS.fullmatch(query) is None:
            raise Denied("the endpoint query is not accepted")
        if len(segments) < 3 or segments[0] != "repos":
            raise Denied("gh api only reads endpoints under repos/OWNER/REPO/")
        repo = self.allowed_repo("/".join(segments[1:3]))
        topic = segments[3:]
        if topic and topic[0] not in API_REPO_TOPICS and tuple(topic[:2]) not in API_REPO_PAIRS:
            raise Denied(f"the endpoint repos/OWNER/REPO/{topic[0]} is not available")
        return repo, "/".join(["repos", repo, *topic]) + (has_query + query if has_query else "")

    def help_text(self) -> str:
        commands = ", ".join(" ".join(words) for words in COMMANDS)
        return (
            "gh here is read-only and goes through a broker on the host: no token and no network inside the sandbox.\n"
            f"commands: {commands}, api (GET only, under repos/OWNER/REPO/)\n"
            f"repositories: {', '.join(self.repos) or 'none'}" + (f" ({self.default_repo} is the default)" if self.repos else "") + "\n"
            "--repo/-R may name any of them. --jq and --template are not available: pipe the output through jq instead.\n"
            "Anything that writes (create, comment, review, merge, close, edit, delete) is refused.\n"
        )


def scrub(text: str, token: str | None) -> str:
    text = TOKEN_PATTERN.sub(TOKEN_PLACEHOLDER, text)
    return text.replace(token, TOKEN_PLACEHOLDER) if token else text


def kill_group(group: int) -> None:
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ProcessTracker:
    def __init__(self) -> None:
        self.groups: set[int] = set()
        self.lock = threading.Lock()
        self.closed = False

    def add(self, group: int) -> bool:
        with self.lock:
            if self.closed:
                return False
            self.groups.add(group)
            return True

    def discard(self, group: int) -> None:
        with self.lock:
            self.groups.discard(group)

    def kill_all(self) -> None:
        with self.lock:
            self.closed = True
            for group in self.groups:
                kill_group(group)


def guarded(command: list[str], seconds: int) -> list[str]:
    # coreutils timeout ends gh even when the broker itself is killed
    if TIMEOUT_BINARY is None:
        raise OSError("coreutils timeout is not installed")
    return [TIMEOUT_BINARY, f"--kill-after={KILL_GRACE_SECONDS}", str(seconds), *command]


def run_capped(command: list[str], env: dict[str, str], tracker: ProcessTracker) -> tuple[int, bytes, bytes, bool]:
    process = subprocess.Popen(guarded(command, RUN_TIMEOUT_SECONDS), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd="/", start_new_session=True)
    truncated = threading.Event()
    if not tracker.add(process.pid):
        kill_group(process.pid)

    def stop() -> None:
        truncated.set()
        kill_group(process.pid)

    def pump(stream, limit: int, sink: bytearray) -> None:
        while chunk := stream.read(65536):
            room = limit - len(sink)
            sink += chunk[:room]
            if len(chunk) > room:
                stop()
                return

    out, err = bytearray(), bytearray()
    readers = [threading.Thread(target=pump, args=(process.stdout, OUTPUT_LIMIT_BYTES, out)), threading.Thread(target=pump, args=(process.stderr, ERROR_LIMIT_BYTES, err))]
    timer = threading.Timer(RUN_TIMEOUT_SECONDS + KILL_GRACE_SECONDS, stop)
    timer.start()
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    process.wait()
    timer.cancel()
    tracker.discard(process.pid)
    return process.returncode, bytes(out), bytes(err), truncated.is_set() or process.returncode == TIMEOUT_EXIT_CODE


class HostTokens:
    def __init__(self, gh: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.gh = gh
        self.run = run

    def for_owner(self, owner: str) -> str:
        # an account named like the owner reads its own repositories; any other owner uses the active account
        for arguments in (["auth", "token", "--user", owner], ["auth", "token"]):
            command = guarded([self.gh, *arguments], TOKEN_TIMEOUT_SECONDS)
            try:
                result = self.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=TOKEN_TIMEOUT_SECONDS + 2 * KILL_GRACE_SECONDS,
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            token = result.stdout.strip()
            if result.returncode == 0 and token:
                return token
        raise Denied("the host has no gh login to use")


class Broker:
    def __init__(self, policy: Policy, gh: str, home: Path, log_path: Path | None = None, tokens: HostTokens | None = None):
        self.policy = policy
        self.gh = gh
        self.home = home
        self.log_path = log_path
        self.tokens = tokens or HostTokens(gh)
        self.slots = threading.BoundedSemaphore(MAX_PARALLEL_RUNS)
        self.tracker = ProcessTracker()
        self.log_lock = threading.Lock()

    def serve(self, raw: bytes) -> Reply:
        started = time.monotonic()
        argv = self._parse_request(raw)
        if argv is None:
            return Reply(USAGE_EXIT_CODE, stderr="gh: the request was not understood\n")
        verdict = "ALLOW"
        try:
            reply = self._answer(argv)
        except Denied as denied:
            verdict = "DENY"
            reply = Reply(DENIED_EXIT_CODE, stderr=f"gh: {denied}. Run 'gh help' to see what is available.\n")
        except OSError as error:
            verdict = "ERROR"
            reply = Reply(USAGE_EXIT_CODE, stderr=f"gh: the broker could not run gh: {error}\n")
        self._log(verdict, reply, argv, started)
        return reply

    @staticmethod
    def _parse_request(raw: bytes) -> list[str] | None:
        if len(raw) > REQUEST_LIMIT_BYTES:
            return None
        try:
            argv = json.loads(raw)["argv"]
        except (ValueError, KeyError, TypeError):
            return None
        if not isinstance(argv, list) or len(argv) > MAX_ARGUMENTS:
            return None
        if not all(isinstance(item, str) and len(item) <= MAX_ARGUMENT_CHARS and "\x00" not in item for item in argv):
            return None
        return argv

    def _answer(self, argv: list[str]) -> Reply:
        if not argv or argv[0] in ("help", "--help", "-h") or argv[:2] == ["auth", "status"]:
            return Reply(0, self.policy.help_text())
        plan = self.policy.translate(argv)
        if not self.slots.acquire(timeout=BUSY_WAIT_SECONDS):
            raise Denied("the broker is busy")
        try:
            return self._run_gh(plan)
        finally:
            self.slots.release()

    def _run_gh(self, plan: Plan) -> Reply:
        token = self.tokens.for_owner(plan.repo.split("/")[0])
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.home),
            "GH_CONFIG_DIR": str(self.home / "gh"),
            "GH_HOST": GITHUB_HOST,
            "GH_TOKEN": token,
            "GH_PROMPT_DISABLED": "1",
            "GH_NO_UPDATE_NOTIFIER": "1",
            "GH_NO_EXTENSION_UPDATE_NOTIFIER": "1",
            "NO_COLOR": "1",
            "LANG": "C.UTF-8",
        }
        env.update({name: os.environ[name] for name in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE") if name in os.environ})
        rc, out, err, stopped = run_capped([self.gh, *plan.args], env, self.tracker)
        stderr = scrub(err.decode("utf-8", "replace"), token)
        if stopped:
            stderr += f"gh: output stopped at {OUTPUT_LIMIT_BYTES // 1024} KiB or {RUN_TIMEOUT_SECONDS} s; narrow the request\n"
        return Reply(rc, scrub(out.decode("utf-8", "replace"), token), stderr)

    def _log(self, verdict: str, reply: Reply, argv: list[str], started: float) -> None:
        if self.log_path is None:
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {verdict} rc={reply.rc} bytes={len(reply.stdout)} {elapsed_ms}ms {json.dumps(argv)[:LOG_ARGUMENTS_CHARS]}\n"
        try:
            with self.log_lock:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(line)
                trim_log(self.log_path)
        except OSError:
            pass


class BrokerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, broker: Broker):
        self.broker = broker
        self.connections = threading.BoundedSemaphore(MAX_CONNECTIONS)
        previous = os.umask(0o177)
        try:
            super().__init__(path, BrokerHandler)
        finally:
            os.umask(previous)

    def process_request(self, request, client_address) -> None:
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def read_request(connection: socket.socket) -> bytes:
    deadline = time.monotonic() + CONNECTION_TIMEOUT_SECONDS
    data = bytearray()
    while len(data) <= REQUEST_LIMIT_BYTES and b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the request did not arrive in time")
        connection.settimeout(remaining)
        chunk = connection.recv(READ_CHUNK_BYTES)
        if not chunk:
            break
        data += chunk
    return bytes(data)


class BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            reply = self.server.broker.serve(read_request(self.connection))
            # sendall treats the timeout as the total time for the whole reply
            self.connection.settimeout(REPLY_TIMEOUT_SECONDS)
            self.wfile.write(reply.as_json())
        except OSError:
            pass


def trim_log(path: Path) -> None:
    try:
        if path.stat().st_size > LOG_TRIM_ABOVE_BYTES:
            tail = path.read_bytes()[-LOG_KEEP_BYTES:]
            path.write_bytes(tail[tail.find(b"\n") + 1 :])
    except OSError:
        pass


def watch_parent(pid: int, server: BrokerServer) -> None:
    while True:
        time.sleep(PARENT_POLL_SECONDS)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            server.shutdown()
            return


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only gh for the spi sandbox: answers requests on a unix socket.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cwd", default=".", help="folder whose github.com remotes name the repositories")
    parser.add_argument("--repos", default="", help="more OWNER/REPO names, separated by commas or colons")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process is gone")
    parser.add_argument("--log")
    parser.add_argument("--gh", default=shutil.which("gh"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.gh:
        print("spi-gh-broker: gh is not installed", file=sys.stderr)
        return 1
    if TIMEOUT_BINARY is None:
        print("spi-gh-broker: coreutils timeout is not installed", file=sys.stderr)
        return 1
    extra = [repo for name in re.split(r"[,:\s]+", args.repos) if name and (repo := normalize_repo(name))]
    policy = Policy(repos_from_git_remotes(args.cwd) + extra)
    log_path = Path(args.log) if args.log else None
    if log_path:
        trim_log(log_path)
    with tempfile.TemporaryDirectory(prefix="spi-gh-") as home:
        broker = Broker(policy, args.gh, Path(home), log_path)
        Path(args.socket).unlink(missing_ok=True)
        server = BrokerServer(args.socket, broker)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
        if args.parent_pid:
            threading.Thread(target=watch_parent, args=(args.parent_pid, server), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            broker.tracker.kill_all()
            server.server_close()
            Path(args.socket).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
