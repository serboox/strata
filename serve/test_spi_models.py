import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import spi_models
from spi_models import Config, ConfigError, LocalModel, ModelProxy, ModelSupervisor, load_config

PROGRAM = ROOT / "tools" / "spi_models.py"

FAKE_SERVER = '''#!{python}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
stem = args[args.index("--model") + 1].rsplit("/", 1)[-1].split(".")[0]
port = int(args[args.index("--port") + 1])
if os.environ.get("FAKE_START_DELAY"):
    time.sleep(float(os.environ["FAKE_START_DELAY"]))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {{"status": "ok", "model": "fake-" + stem}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{{}}")
        if body.get("slow"):
            time.sleep(0.6)
        if body.get("hang"):
            time.sleep(30)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for number in range(3):
                self.wfile.write(("data: " + json.dumps({{"n": number, "by": stem}}) + "\\n\\n").encode())
                self.wfile.flush()
                time.sleep(0.05)
            return
        self.reply(200, {{"served_by": stem, "model": body.get("model"), "auth": self.headers.get("Authorization")}})

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def parse_response(raw: bytes) -> tuple[int, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split(b" ", 2)[1]), body


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "models.json"

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_relative_files_are_under_the_root_and_absolute_ones_stay(self):
        self.write({"default": "b", "models": {"a": {"file": "models/a.gguf"}, "b": {"file": "/data/b.gguf", "args": ["--profile", "lean"]}}})
        config = load_config(self.path, self.root)
        self.assertEqual(config.default, "b")
        self.assertEqual(config.models["a"].file, self.root / "models" / "a.gguf")
        self.assertEqual(config.models["b"], LocalModel("b", Path("/data/b.gguf"), ("--profile", "lean")))

    def test_the_root_comes_from_the_config_when_it_is_not_given_and_a_given_one_wins(self):
        self.write({"root": "/srv/strata", "models": {"a": {"file": "models/a.gguf"}}})
        self.assertEqual(load_config(self.path, None).models["a"].file, Path("/srv/strata/models/a.gguf"))
        self.assertEqual(load_config(self.path, self.root).models["a"].file, self.root / "models" / "a.gguf")

    def test_a_home_relative_root_and_file_are_expanded(self):
        with mock.patch.dict(os.environ, {"HOME": "/home/someone"}):
            self.write({"root": "~/strata", "models": {"a": {"file": "a.gguf"}, "b": {"file": "~/b.gguf"}}})
            config = load_config(self.path, None)
        self.assertEqual(config.models["a"].file, Path("/home/someone/strata/a.gguf"))
        self.assertEqual(config.models["b"].file, Path("/home/someone/b.gguf"))

    def test_the_fallback_root_comes_last(self):
        self.write({"models": {"a": {"file": "a.gguf"}}})
        self.assertEqual(load_config(self.path, None, Path("/fallback")).models["a"].file, Path("/fallback/a.gguf"))
        self.write({"root": "/from-config", "models": {"a": {"file": "a.gguf"}}})
        self.assertEqual(load_config(self.path, None, Path("/fallback")).models["a"].file, Path("/from-config/a.gguf"))
        self.assertEqual(load_config(self.path, Path("/given"), Path("/fallback")).models["a"].file, Path("/given/a.gguf"))

    def test_a_relative_file_without_any_root_is_refused(self):
        self.write({"models": {"a": {"file": "a.gguf"}}})
        with self.assertRaises(ConfigError):
            load_config(self.path, None)

    def test_the_default_is_the_first_model_when_none_is_named(self):
        self.write({"models": {"a": {"file": "a.gguf"}, "b": {"file": "b.gguf"}}})
        self.assertEqual(load_config(self.path, self.root).default, "a")

    def test_bad_configs_are_refused(self):
        bad = [
            {"models": {}},
            {"models": []},
            {"models": {"a": {}}},
            {"models": {"a": {"file": ""}}},
            {"models": {"a": {"file": "a.gguf", "args": "--x"}}},
            {"models": {"a": {"file": "a.gguf", "args": [1]}}},
            {"default": "z", "models": {"a": {"file": "a.gguf"}}},
        ]
        for data in bad:
            with self.subTest(data=data):
                self.write(data)
                with self.assertRaises(ConfigError):
                    load_config(self.path, self.root)

    def test_a_missing_or_broken_file_is_refused(self):
        with self.assertRaises(ConfigError):
            load_config(self.root / "nothing.json", self.root)
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.path, self.root)


class ProxyTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.script = root / "fake-serve"
        self.script.write_text(FAKE_SERVER.format(python=sys.executable), encoding="utf-8")
        self.script.chmod(0o755)
        for stem in ("alpha", "beta"):
            (root / f"{stem}.gguf").write_text("", encoding="utf-8")
        config = Config("alpha", {name: LocalModel(name, root / f"{name}.gguf") for name in ("alpha", "beta")})
        self.port = free_port()
        self.messages = []
        self.supervisor = ModelSupervisor(config, root, self.port, self.script, root / "server.log", self.messages.append)
        self.proxy = ModelProxy(self.supervisor)
        self.socket_path = str(root / "spi.sock")
        self.server = await asyncio.start_unix_server(self.proxy.handle, path=self.socket_path, limit=spi_models.HEAD_LIMIT_BYTES)
        patcher = mock.patch.object(spi_models, "READY_POLL_SECONDS", 0.05)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        self.server.close()
        await self.supervisor.stop_child()

    async def raw(self, payload: bytes) -> bytes:
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        data = await asyncio.wait_for(reader.read(), 20)
        writer.close()
        return data

    async def ask(self, method: str, path: str, body: dict | None = None, headers: str = "") -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else b""
        head = f"{method} {path} HTTP/1.1\r\nHost: spi\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n{headers}\r\n"
        return parse_response(await self.raw(head.encode() + data))

    async def chat(self, model: str | None, **extra) -> tuple[int, dict]:
        body = {"messages": [], **extra}
        if model is not None:
            body["model"] = model
        status, payload = await self.ask("POST", "/v1/chat/completions", body)
        return status, json.loads(payload)


class SwitchTests(ProxyTestCase):
    async def test_a_request_starts_its_model_and_the_server_gets_its_own_name_for_it(self):
        status, body = await self.chat("alpha")
        self.assertEqual(status, 200)
        self.assertEqual(body["served_by"], "alpha")
        self.assertEqual(body["model"], "fake-alpha")

    async def test_headers_of_the_client_reach_the_server(self):
        status, payload = await self.ask("POST", "/v1/chat/completions", {"model": "alpha"}, headers="Authorization: Bearer secret\r\n")
        self.assertEqual(json.loads(payload)["auth"], "Bearer secret")

    async def test_asking_for_another_model_stops_the_first_server_and_starts_the_second(self):
        await self.chat("alpha")
        first = self.supervisor.child
        status, body = await self.chat("beta")
        self.assertEqual((status, body["served_by"]), (200, "beta"))
        self.assertIsNotNone(first.returncode)
        self.assertFalse(alive(first.pid))
        self.assertEqual(self.supervisor.current, "beta")

    async def test_the_same_model_keeps_its_server(self):
        await self.chat("alpha")
        pid = self.supervisor.child.pid
        await self.chat("alpha")
        self.assertEqual(self.supervisor.child.pid, pid)

    async def test_an_unknown_model_is_refused_and_nothing_starts(self):
        status, body = await self.chat("gpt-4o")
        self.assertEqual(status, 404)
        self.assertIn("alpha, beta", body["error"]["message"])
        self.assertIsNone(self.supervisor.child)

    async def test_a_request_without_a_model_goes_to_the_current_one_or_the_default(self):
        self.assertEqual((await self.chat(None))[1]["served_by"], "alpha")
        await self.chat("beta")
        self.assertEqual((await self.chat(None))[1]["served_by"], "beta")

    async def test_a_switch_waits_for_the_request_that_is_running(self):
        await self.chat("alpha")
        slow = asyncio.create_task(self.chat("alpha", slow=True))
        for _ in range(100):
            if self.supervisor.active:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(self.supervisor.active, 1)
        other = asyncio.create_task(self.chat("beta"))
        await asyncio.sleep(0.2)
        self.assertFalse(other.done())
        slow_status, slow_body = await slow
        other_status, other_body = await other
        self.assertEqual((slow_status, slow_body["served_by"]), (200, "alpha"))
        self.assertEqual((other_status, other_body["served_by"]), (200, "beta"))

    async def test_a_stream_is_relayed_to_the_end(self):
        status, payload = await self.ask("POST", "/v1/chat/completions", {"model": "alpha", "stream": True})
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in payload.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual([event["n"] for event in events], [0, 1, 2])

    async def test_a_server_that_died_is_started_again_by_the_next_request(self):
        await self.chat("alpha")
        first = self.supervisor.child
        os.kill(first.pid, signal.SIGKILL)
        await first.wait()
        status, body = await self.chat("alpha")
        self.assertEqual((status, body["served_by"]), (200, "alpha"))
        self.assertNotEqual(self.supervisor.child.pid, first.pid)


class FailureTests(ProxyTestCase):
    async def test_a_server_that_ends_while_it_starts_gives_a_503_and_the_next_request_tries_again(self):
        self.supervisor.serve_command = Path("/bin/false")
        status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("ended with code", body["error"]["message"])
        self.supervisor.serve_command = self.script
        self.assertEqual((await self.chat("alpha"))[0], 200)

    async def test_a_program_that_cannot_be_run_gives_a_503(self):
        self.supervisor.serve_command = Path("/nonexistent/strata-serve")
        status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("could not be started", body["error"]["message"])

    async def test_a_missing_model_file_gives_a_503(self):
        self.supervisor.config.models["alpha"].file.unlink()
        status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("missing", body["error"]["message"])

    async def test_a_taken_port_gives_a_503_and_nothing_is_started(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", self.port))
            taken.listen()
            status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("already taken", body["error"]["message"])
        self.assertIsNone(self.supervisor.child)

    async def test_a_server_that_never_gets_ready_is_stopped(self):
        with mock.patch.object(spi_models, "READY_TIMEOUT_SECONDS", 0.5), mock.patch.object(spi_models, "probe_health", mock.AsyncMock(return_value=None)):
            status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("did not become ready", body["error"]["message"])
        self.assertIsNone(self.supervisor.child)

    async def test_a_server_that_ignores_the_stop_signal_is_killed(self):
        ignoring = Path(self.tmp.name) / "stubborn"
        ignoring.write_text(FAKE_SERVER.format(python=sys.executable).replace("ThreadingHTTPServer(", "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)\nThreadingHTTPServer("), encoding="utf-8")
        ignoring.chmod(0o755)
        self.supervisor.serve_command = ignoring
        await self.chat("alpha")
        pid = self.supervisor.child.pid
        with mock.patch.object(spi_models, "STOP_TIMEOUT_SECONDS", 0.5):
            await self.supervisor.stop_child()
        self.assertFalse(alive(pid))


class ShutdownAndLimitTests(ProxyTestCase):
    async def test_a_stop_that_was_cancelled_keeps_the_server_in_view_and_the_next_stop_ends_it(self):
        stubborn = Path(self.tmp.name) / "stubborn"
        stubborn.write_text(FAKE_SERVER.format(python=sys.executable).replace("ThreadingHTTPServer(", "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)\nThreadingHTTPServer("), encoding="utf-8")
        stubborn.chmod(0o755)
        self.supervisor.serve_command = stubborn
        await self.chat("alpha")
        child = self.supervisor.child
        stopping = asyncio.create_task(self.supervisor.stop_child())
        await asyncio.sleep(0.2)
        stopping.cancel()
        await asyncio.gather(stopping, return_exceptions=True)
        self.assertIs(self.supervisor.child, child)
        self.assertTrue(alive(child.pid))
        with mock.patch.object(spi_models, "STOP_TIMEOUT_SECONDS", 0.3):
            await self.supervisor.stop_child()
        self.assertFalse(alive(child.pid))
        self.assertIsNone(self.supervisor.child)

    async def test_nothing_starts_once_the_supervisor_is_closing(self):
        self.supervisor.closing = True
        status, body = await self.chat("alpha")
        self.assertEqual(status, 503)
        self.assertIn("closing", body["error"]["message"])
        self.assertIsNone(self.supervisor.child)

    async def test_connections_over_the_limit_are_refused_at_once(self):
        with mock.patch.object(spi_models, "MAX_CONNECTIONS", 2):
            holders = [await asyncio.open_unix_connection(self.socket_path) for _ in range(2)]
            await asyncio.sleep(0.2)
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            reply = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            self.assertEqual(parse_response(reply)[0], 503)
            self.assertIn(b"too many connections", reply)
            for _, held in holders:
                held.close()
            await asyncio.sleep(0.3)
            self.assertEqual((await self.ask("GET", "/health"))[0], 200)

    async def test_a_server_that_does_not_take_the_request_ends_it_with_a_504_and_frees_the_slot(self):
        await self.chat("alpha")
        real_open = asyncio.open_connection

        async def connection_that_never_drains(*args, **kwargs):
            reader, writer = await real_open(*args, **kwargs)

            async def drain():
                await asyncio.sleep(30)

            writer.drain = drain
            return reader, writer

        with mock.patch.object(spi_models, "RELAY_IDLE_TIMEOUT_SECONDS", 0.3), mock.patch("asyncio.open_connection", connection_that_never_drains):
            status, body = await self.chat("alpha")
        self.assertEqual(status, 504)
        self.assertIn("did not take the request", body["error"]["message"])
        self.assertEqual(self.supervisor.active, 0)

    async def test_a_server_that_stays_silent_ends_the_request_with_a_504_and_frees_the_slot(self):
        await self.chat("alpha")
        with mock.patch.object(spi_models, "RELAY_IDLE_TIMEOUT_SECONDS", 0.4):
            status, body = await self.chat("alpha", hang=True)
        self.assertEqual(status, 504)
        self.assertIn("did not answer in time", body["error"]["message"])
        self.assertEqual(self.supervisor.active, 0)


class RequestTests(ProxyTestCase):
    async def test_the_model_list_names_every_local_model_and_marks_the_running_one(self):
        status, payload = await self.ask("GET", "/v1/models")
        listing = json.loads(payload)
        self.assertEqual([item["id"] for item in listing["data"]], ["alpha", "beta"])
        self.assertEqual({item["status"]["value"] for item in listing["data"]}, {"unloaded"})
        await self.chat("beta")
        states = {item["id"]: item["status"]["value"] for item in json.loads((await self.ask("GET", "/models"))[1])["data"]}
        self.assertEqual(states, {"alpha": "unloaded", "beta": "loaded"})

    async def test_asking_for_the_list_or_the_health_starts_nothing(self):
        await self.ask("GET", "/v1/models")
        status, payload = await self.ask("GET", "/health")
        self.assertEqual((status, json.loads(payload)["service"]), (200, "spi-models"))
        self.assertIsNone(self.supervisor.child)

    async def test_a_chunked_body_is_refused(self):
        raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: spi\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
        self.assertEqual(parse_response(await self.raw(raw))[0], 411)

    async def test_a_body_over_the_limit_is_refused(self):
        with mock.patch.object(spi_models, "BODY_LIMIT_BYTES", 10):
            status, _ = await self.ask("POST", "/v1/chat/completions", {"model": "alpha", "messages": []})
        self.assertEqual(status, 413)
        self.assertIsNone(self.supervisor.child)

    async def test_a_body_that_is_not_json_is_refused(self):
        raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: spi\r\nContent-Type: application/json\r\nContent-Length: 5\r\n\r\n{nope"
        self.assertEqual(parse_response(await self.raw(raw))[0], 400)

    async def test_malformed_requests_are_refused(self):
        for raw in (b"NONSENSE\r\n\r\n", b"GET / HTTP/1.1\r\nbad header\r\n\r\n", b"GET / HTTP/1.1\r\nContent-Length: x\r\n\r\n",
                    b"GET / HTTP/1.1\r\n Host: spi\r\n\r\n"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_response(await self.raw(raw))[0], 400)

    async def test_a_client_that_leaves_without_a_word_is_ignored(self):
        self.assertEqual(await self.raw(b""), b"")


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.script = root / "fake-serve"
        self.script.write_text(FAKE_SERVER.format(python=sys.executable), encoding="utf-8")
        self.script.chmod(0o755)
        (root / "alpha.gguf").write_text("", encoding="utf-8")
        self.config = root / "models.json"
        self.config.write_text(json.dumps({"models": {"alpha": {"file": "alpha.gguf"}}}), encoding="utf-8")
        self.socket_path = root / "spi.sock"
        self.port = free_port()
        self.processes = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stderr:
                process.stderr.close()

    def start(self, parent_pid: int = 0) -> subprocess.Popen:
        command = [sys.executable, str(PROGRAM), "--socket", str(self.socket_path), "--config", str(self.config), "--root", self.tmp.name,
                   "--port", str(self.port), "--serve-command", str(self.script), "--server-log", str(Path(self.tmp.name) / "server.log")]
        if parent_pid:
            command += ["--parent-pid", str(parent_pid)]
        process = subprocess.Popen(command, stderr=subprocess.PIPE)
        self.processes.append(process)
        return process

    def wait_for(self, condition, seconds: float = 20) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.05)
        return False

    def server_answers(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                return True
        except OSError:
            return False

    def test_the_default_model_starts_at_once_and_stops_with_the_program(self):
        process = self.start()
        self.assertTrue(self.wait_for(self.server_answers))
        self.assertTrue(self.socket_path.exists())
        self.assertEqual(oct(self.socket_path.stat().st_mode & 0o777), "0o600")
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0)
        self.assertFalse(self.server_answers())
        self.assertFalse(self.socket_path.exists())

    def test_the_program_ends_and_stops_the_server_when_its_parent_is_gone(self):
        parent = subprocess.Popen(["sleep", "60"])
        self.processes.append(parent)
        process = self.start(parent.pid)
        self.assertTrue(self.wait_for(self.server_answers))
        parent.kill()
        parent.wait()
        self.assertEqual(process.wait(timeout=30), 0)
        self.assertFalse(self.server_answers())

    def test_a_broken_config_ends_the_program_with_code_2_and_a_message(self):
        self.config.write_text("{}", encoding="utf-8")
        process = self.start()
        self.assertEqual(process.wait(timeout=20), 2)
        self.assertIn("models", process.stderr.read().decode())

    def test_a_config_without_any_known_root_ends_the_program_with_code_2(self):
        self.config.write_text(json.dumps({"models": {"alpha": {"file": "alpha.gguf"}}}), encoding="utf-8")
        command = [sys.executable, str(PROGRAM), "--socket", str(self.socket_path), "--config", str(self.config),
                   "--server-log", str(Path(self.tmp.name) / "server.log")]
        done = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(done.returncode, 2)
        self.assertIn("relative file", done.stderr)

    def test_a_stop_while_the_server_is_still_starting_ends_both(self):
        self.processes.append(process := subprocess.Popen(
            [sys.executable, str(PROGRAM), "--socket", str(self.socket_path), "--config", str(self.config), "--root", self.tmp.name,
             "--port", str(self.port), "--serve-command", str(self.script), "--server-log", str(Path(self.tmp.name) / "server.log"),
             "--log", str(Path(self.tmp.name) / "models.log")], env={**os.environ, "FAKE_START_DELAY": "30"}, stderr=subprocess.PIPE))
        log = Path(self.tmp.name) / "models.log"
        self.assertTrue(self.wait_for(lambda: log.exists() and "starting alpha" in log.read_text(encoding="utf-8")))
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0)
        self.assertFalse(self.server_answers())
        listing = subprocess.run(["pgrep", "-f", str(self.script)], capture_output=True, text=True)
        self.assertEqual(listing.stdout.strip(), "")

    def test_a_default_that_is_not_in_the_list_ends_the_program_with_code_2(self):
        command = [sys.executable, str(PROGRAM), "--socket", str(self.socket_path), "--config", str(self.config), "--root", self.tmp.name,
                   "--server-log", str(Path(self.tmp.name) / "server.log"), "--default", "nope"]
        done = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(done.returncode, 2)
        self.assertIn("nope", done.stderr)


if __name__ == "__main__":
    unittest.main()
