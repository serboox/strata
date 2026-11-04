#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HEAD_LIMIT_BYTES = 64 * 1024
HEAD_TIMEOUT_SECONDS = 10
BODY_LIMIT_BYTES = 8 * 1024 * 1024
MAX_CONNECTIONS = 16
RELAY_IDLE_TIMEOUT_SECONDS = 1200
BODY_TIMEOUT_SECONDS = 120
READY_TIMEOUT_SECONDS = 300
READY_POLL_SECONDS = 1.0
STOP_TIMEOUT_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 3
PARENT_POLL_SECONDS = 2
RELAY_CHUNK_BYTES = 64 * 1024
SERVE_COMMAND = Path("bin") / "strata-serve"
MODEL_LIST_PATHS = frozenset({"/v1/models", "/models"})
HEALTH_PATHS = frozenset({"/health", "/api/health"})
RELAYED_HEADERS_DROPPED = frozenset({"connection", "proxy-connection", "keep-alive", "transfer-encoding", "content-length", "host"})


class ConfigError(Exception):
    pass


class Refused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class LocalModel:
    name: str
    file: Path
    args: tuple[str, ...] = ()


@dataclass
class Config:
    default: str
    models: dict[str, LocalModel] = field(default_factory=dict)
    root: Path | None = None


def load_config(path: Path, root: Path | None = None, fallback_root: Path | None = None) -> Config:
    """The local models spi may start: {"root": folder, "default": id, "models": {id: {"file": path, "args": [...]}}}.

    The Strata folder is the root argument, else the "root" of the file, else the fallback; a relative model file is under it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if isinstance(data, dict) and root is None and isinstance(data.get("root"), str) and data["root"]:
        root = Path(data["root"]).expanduser()
    root = root or fallback_root
    raw_models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError(f'{path}: "models" must be an object with at least one model')
    models = {}
    for name, entry in raw_models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or not entry["file"]:
            raise ConfigError(f'{path}: model "{name}" needs a "file"')
        args = entry.get("args", [])
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ConfigError(f'{path}: "args" of model "{name}" must be a list of strings')
        file = Path(entry["file"]).expanduser()
        if not file.is_absolute() and root is None:
            raise ConfigError(f'{path}: model "{name}" has a relative file, so the Strata folder must be given ("root" or --root)')
        models[name] = LocalModel(name, file if file.is_absolute() else root / file, tuple(args))
    default = data.get("default", next(iter(models)))
    if default not in models:
        raise ConfigError(f'{path}: "default" names no model of the list ({default})')
    return Config(default, models, root)


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: list[tuple[str, str]]
    body: bytes

    @property
    def path(self) -> str:
        return self.target.split("?", 1)[0]

    def header(self, name: str) -> str | None:
        for key, value in self.headers:
            if key.lower() == name:
                return value
        return None


async def read_request(reader: asyncio.StreamReader) -> Request | None:
    """The next request of the connection, None when the client closed it quietly; Refused for a request that cannot be served."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_SECONDS)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise Refused(400, "the request head is incomplete") from error
    except asyncio.LimitOverrunError as error:
        raise Refused(431, "the request head is too long") from error
    except TimeoutError as error:
        raise Refused(408, "the request head did not arrive in time") from error
    lines = head[:-4].decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise Refused(400, "the request line is malformed")
    headers = []
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if not separator or not name or name != name.strip():
            raise Refused(400, "a header line is malformed")
        headers.append((name, value.strip()))
    request = Request(parts[0], parts[1], parts[2], headers, b"")
    if (request.header("transfer-encoding") or "").lower() not in ("", "identity"):
        raise Refused(411, "send the body with a Content-Length")
    length_text = request.header("content-length") or "0"
    if not length_text.isdigit():
        raise Refused(400, "Content-Length is not a number")
    length = int(length_text)
    if length > BODY_LIMIT_BYTES:
        raise Refused(413, "the request body is too large")
    try:
        request.body = await asyncio.wait_for(reader.readexactly(length), BODY_TIMEOUT_SECONDS) if length else b""
    except (asyncio.IncompleteReadError, TimeoutError) as error:
        raise Refused(400, "the request body is incomplete") from error
    return request


def response_bytes(status: int, body: dict) -> bytes:
    reasons = {200: "OK", 400: "Bad Request", 404: "Not Found", 408: "Request Timeout", 411: "Length Required", 413: "Payload Too Large",
               431: "Request Header Fields Too Large", 502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}
    payload = json.dumps(body).encode("utf-8")
    head = (f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n")
    return head.encode("latin-1") + payload


def error_body(message: str) -> dict:
    return {"error": {"message": message, "type": "spi_models"}}


async def probe_health(port: int) -> dict | None:
    """The /health answer of the server on the port, None while it does not answer with a 200."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), PROBE_TIMEOUT_SECONDS)
    except (OSError, TimeoutError):
        return None
    try:
        writer.write(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        await writer.drain()
        head = (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), PROBE_TIMEOUT_SECONDS)).decode("latin-1")
        lines = head.split("\r\n")
        if lines[0].split(" ")[1:2] != ["200"]:
            return None
        lengths = [line.partition(":")[2].strip() for line in lines[1:] if line.lower().startswith("content-length:")]
        if len(lengths) != 1 or not lengths[0].isdigit():
            return None
        body = await asyncio.wait_for(reader.readexactly(int(lengths[0])), PROBE_TIMEOUT_SECONDS)
        data = json.loads(body.decode("utf-8"))
    except (OSError, TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
        return None
    finally:
        writer.close()
    return data if isinstance(data, dict) else None


def port_is_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(PROBE_TIMEOUT_SECONDS)
        return probe.connect_ex(("127.0.0.1", port)) == 0


class ModelSupervisor:
    """Keeps one Strata server alive for the model that was asked for last, and replaces it when another one is asked for."""

    def __init__(self, config: Config, root: Path, port: int, serve_command: Path, server_log: Path, log):
        self.config = config
        self.root = root
        self.port = port
        self.serve_command = serve_command
        self.server_log = server_log
        self.log = log
        self.current: str | None = None
        self.backend_name = ""
        self.child: asyncio.subprocess.Process | None = None
        self.active = 0
        self.switching = False
        self.closing = False
        self.condition = asyncio.Condition()

    def running(self) -> bool:
        return self.child is not None and self.child.returncode is None and self.current is not None

    async def acquire(self, name: str) -> str:
        """Waits until the server of the model runs and is idle of other models' requests; returns the model id the server knows itself by."""
        async with self.condition:
            while True:
                if self.closing:
                    raise Refused(503, "spi is closing")
                if self.switching:
                    await self.condition.wait()
                elif self.running() and self.current == name:
                    self.active += 1
                    return self.backend_name
                elif self.active > 0 and self.running():
                    await self.condition.wait()
                else:
                    self.switching = True
                    break
        try:
            await self.switch_to(name)
        finally:
            async with self.condition:
                self.switching = False
                self.condition.notify_all()
        return await self.acquire(name)

    async def release(self) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def switch_to(self, name: str) -> None:
        await self.stop_child()
        model = self.config.models[name]
        if not model.file.is_file():
            raise Refused(503, f'the file of model "{name}" is missing: {model.file}')
        if port_is_taken(self.port):
            raise Refused(503, f"port {self.port} is already taken by another program, so model \"{name}\" cannot be started")
        command = [str(self.serve_command), "--model", str(model.file), "--port", str(self.port), *model.args]
        self.log(f"starting {name}: {' '.join(command)}")
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.server_log, "ab") as output:
            output.write(f"\n=== {time.strftime('%F %T')} starting {name}\n".encode("utf-8"))
            try:
                self.child = await asyncio.create_subprocess_exec(*command, stdout=output, stderr=output, stdin=asyncio.subprocess.DEVNULL,
                                                                  start_new_session=True)
            except OSError as error:
                raise Refused(503, f'model "{name}" could not be started: {error}') from error
        started = time.monotonic()
        while time.monotonic() - started < READY_TIMEOUT_SECONDS:
            if self.child.returncode is not None:
                code = self.child.returncode
                self.child = None
                raise Refused(503, f'the server of model "{name}" ended with code {code} while it started (log: {self.server_log})')
            health = await probe_health(self.port)
            if health is not None:
                self.current = name
                self.backend_name = str(health.get("model") or name)
                self.log(f"{name} is ready after {time.monotonic() - started:.0f} s as {self.backend_name}")
                return
            await asyncio.sleep(READY_POLL_SECONDS)
        await self.stop_child()
        raise Refused(503, f'model "{name}" did not become ready within {READY_TIMEOUT_SECONDS} s (log: {self.server_log})')

    async def stop_child(self) -> None:
        """Stops the server and keeps its record until it is gone, so a stop that was cancelled can be finished by the next call."""
        child = self.child
        if child is None:
            return
        self.current = None
        if child.returncode is None:
            self.log(f"stopping process {child.pid}")
            try:
                child.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(child.wait()), STOP_TIMEOUT_SECONDS)
            except TimeoutError:
                self.log(f"process {child.pid} did not stop in {STOP_TIMEOUT_SECONDS} s, killing its group")
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await child.wait()
        self.child = None


class ModelProxy:
    def __init__(self, supervisor: ModelSupervisor):
        self.supervisor = supervisor
        self.handlers: set[asyncio.Task] = set()

    def model_list(self) -> dict:
        data = [{"id": name, "object": "model", "owned_by": "local",
                 "status": {"value": "loaded" if self.supervisor.running() and self.supervisor.current == name else "unloaded"}}
                for name in self.supervisor.config.models]
        return {"object": "list", "data": data}

    def requested_model(self, request: Request) -> str:
        """The model a request names, the current one (or the default) when it names none; Refused for a model that is not in the list."""
        wanted = None
        if request.body and (request.header("content-type") or "").lower().startswith("application/json"):
            try:
                body = json.loads(request.body)
            except ValueError as error:
                raise Refused(400, "the request body is not valid JSON") from error
            if isinstance(body, dict) and isinstance(body.get("model"), str):
                wanted = body["model"]
        if wanted is None:
            return self.supervisor.current or self.supervisor.config.default
        if wanted not in self.supervisor.config.models:
            known = ", ".join(self.supervisor.config.models)
            raise Refused(404, f'model "{wanted}" is not one of the local models of spi ({known})')
        return wanted

    def forwarded(self, request: Request, backend_name: str) -> bytes:
        body = request.body
        if body and (request.header("content-type") or "").lower().startswith("application/json"):
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "model" in parsed:
                parsed["model"] = backend_name
                body = json.dumps(parsed).encode("utf-8")
        head = [f"{request.method} {request.target} HTTP/1.1", "Host: 127.0.0.1", "Connection: close", f"Content-Length: {len(body)}"]
        head += [f"{name}: {value}" for name, value in request.headers if name.lower() not in RELAYED_HEADERS_DROPPED]
        return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self.handlers) >= MAX_CONNECTIONS:
            writer.write(response_bytes(503, error_body("too many connections")))
            writer.close()
            return
        # a client that half-closes its side leaves the task with no owner but the event loop's weak set, and a collection would end it midway
        task = asyncio.current_task()
        self.handlers.add(task)
        task.add_done_callback(self.handlers.discard)
        try:
            try:
                request = await read_request(reader)
                if request is None:
                    return
                if request.method == "GET" and request.path in MODEL_LIST_PATHS:
                    writer.write(response_bytes(200, self.model_list()))
                elif request.method == "GET" and request.path in HEALTH_PATHS:
                    writer.write(response_bytes(200, {"status": "ok", "service": "spi-models", "current": self.supervisor.current}))
                else:
                    await self.relay(request, writer)
            except Refused as refusal:
                writer.write(response_bytes(refusal.status, error_body(refusal.message)))
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def relay(self, request: Request, writer: asyncio.StreamWriter) -> None:
        name = self.requested_model(request)
        backend_name = await self.supervisor.acquire(name)
        try:
            try:
                back_reader, back_writer = await asyncio.open_connection("127.0.0.1", self.supervisor.port)
            except OSError as error:
                raise Refused(502, f'the server of model "{name}" does not answer: {error}') from error
            try:
                sent = False
                try:
                    back_writer.write(self.forwarded(request, backend_name))
                    await asyncio.wait_for(back_writer.drain(), RELAY_IDLE_TIMEOUT_SECONDS)
                except TimeoutError as error:
                    self.supervisor.log(f"the server of {name} did not take the request in {RELAY_IDLE_TIMEOUT_SECONDS} s, it is dropped")
                    raise Refused(504, f'the server of model "{name}" did not take the request in time') from error
                while True:
                    try:
                        chunk = await asyncio.wait_for(back_reader.read(RELAY_CHUNK_BYTES), RELAY_IDLE_TIMEOUT_SECONDS)
                    except TimeoutError as error:
                        self.supervisor.log(f"the server of {name} sent nothing for {RELAY_IDLE_TIMEOUT_SECONDS} s, the request is dropped")
                        if sent:
                            return
                        raise Refused(504, f'the server of model "{name}" did not answer in time') from error
                    if not chunk:
                        return
                    sent = True
                    writer.write(chunk)
                    await writer.drain()
            finally:
                back_writer.close()
        finally:
            await self.supervisor.release()


class Logger:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, message: str) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")


async def watch_parent(parent_pid: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            stop.set()
            return
        except PermissionError:
            pass
        await asyncio.sleep(PARENT_POLL_SECONDS)


async def serve(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.config), Path(args.root) if args.root else None, Path(args.fallback_root) if args.fallback_root else None)
    except ConfigError as error:
        print(f"spi_models: {error}", file=sys.stderr)
        return 2
    if config.root is None and not args.serve_command:
        print("spi_models: the Strata folder is unknown: give --root, a \"root\" in the config or --serve-command", file=sys.stderr)
        return 2
    root = config.root or Path()
    default = args.default or config.default
    if default not in config.models:
        print(f'spi_models: the default model "{default}" is not in {args.config}', file=sys.stderr)
        return 2
    log_path = Path(args.log) if args.log else None
    if log_path is not None:
        log_path.write_text("", encoding="utf-8")
    supervisor = ModelSupervisor(config, root, args.port, Path(args.serve_command) if args.serve_command else root / SERVE_COMMAND,
                                 Path(args.server_log), Logger(log_path))
    proxy = ModelProxy(supervisor)
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(proxy.handle, path=str(socket_path), limit=HEAD_LIMIT_BYTES)
    os.chmod(socket_path, 0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, stop.set)
    tasks = [asyncio.create_task(prepare(supervisor, default))]
    if args.parent_pid:
        tasks.append(asyncio.create_task(watch_parent(args.parent_pid, stop)))
    await stop.wait()
    supervisor.closing = True
    server.close()
    handlers = list(proxy.handlers)
    for task in (*tasks, *handlers):
        task.cancel()
    await asyncio.gather(*tasks, *handlers, return_exceptions=True)
    await supervisor.stop_child()
    socket_path.unlink(missing_ok=True)
    return 0


async def prepare(supervisor: ModelSupervisor, name: str) -> None:
    """Starts the default model in the background so that the first request does not wait for it."""
    try:
        await supervisor.acquire(name)
    except Refused as refusal:
        supervisor.log(f"the default model did not start: {refusal.message}")
        return
    await supervisor.release()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Starts the Strata server of the local model that spi asks for and replaces it when another one is asked for.")
    parser.add_argument("--socket", required=True, help="the unix socket that the sandbox reaches the server through")
    parser.add_argument("--config", required=True, help="JSON file with the local models")
    parser.add_argument("--root", help="the Strata folder (relative model files and bin/strata-serve are under it); default: the \"root\" of the config")
    parser.add_argument("--port", type=int, default=8081, help="the port of the server that is started (default 8081)")
    parser.add_argument("--fallback-root", help="the Strata folder to use when neither --root nor the config names one")
    parser.add_argument("--default", help="the model that is started first (default: the one of the config)")
    parser.add_argument("--serve-command", help="the program that starts the server (default: bin/strata-serve)")
    parser.add_argument("--server-log", required=True, help="the file for the output of the servers")
    parser.add_argument("--log", help="the file for the log of this program")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit, and stop the server, when this process is gone")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return asyncio.run(serve(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
