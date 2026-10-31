#!/usr/bin/env python3
import argparse
import asyncio
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

# every port except the ones used to send mail, which would make the proxy a spam relay
DENIED_PORTS = frozenset({25, 465, 587})
VPN_MODES = ("hosts", "block", "allow")
# a route this narrow leads to a particular host or small network (a company endpoint), a wider one to a provider or a CDN
NARROW_ROUTE_PREFIX = {4: 23, 6: 48}
HEAD_LIMIT_BYTES = 64 * 1024
HEAD_TIMEOUT_SECONDS = 10
RESOLVE_TIMEOUT_SECONDS = 10
CONNECT_TIMEOUT_SECONDS = 15
IDLE_TIMEOUT_SECONDS = 120
MAX_TUNNEL_SECONDS = 3600
LINGER_SECONDS = 1
MAX_CONNECTIONS = 64
MAX_REFUSALS = 64
RESOLVER_SLOTS = 16
ROUTE_LOOKUP_TIMEOUT_SECONDS = 5
ROUTE_CACHE_SECONDS = 5
ROUTE_SLOTS = 8
MAX_ADDRESSES = 16
RELAY_CHUNK_BYTES = 64 * 1024
PARENT_POLL_SECONDS = 2
LOG_TRIM_ABOVE_BYTES = 1024 * 1024
LOG_KEEP_BYTES = 256 * 1024
REASON_MAX_CHARS = 120

HOSTNAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,252}")
IPV6_LITERAL_PATTERN = re.compile(r"\[([0-9A-Fa-f:.]{2,45})\]")
METHOD_PATTERN = re.compile(r"[A-Z]{3,7}")
HEADER_NAME_PATTERN = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]{1,100}")
# IPv6 forms that carry an IPv4 address inside, so a private one could hide in them, and ranges that ipaddress still calls global
BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(text) for text in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32", "fec0::/10", "192.88.99.0/24")
)
# documentation addresses are never routed on purpose, so the kernel sends them along the default route
CANARY_ADDRESSES = {4: "192.0.2.1", 6: "2001:db8::1"}
IP_BINARY = shutil.which("ip")
HOP_BY_HOP_HEADERS = frozenset({"host", "connection", "proxy-connection", "proxy-authorization", "proxy-authenticate", "keep-alive", "te", "upgrade"})
HTTP_PORT = 80


class Refused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def normalized_address(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_public_address(text: str) -> bool:
    address = normalized_address(text)
    if address is None:
        return False
    # is_global alone lets multicast addresses through
    if not address.is_global or address.is_multicast or address.is_reserved or address.is_unspecified:
        return False
    return not any(address in network for network in BLOCKED_NETWORKS)


@dataclass(frozen=True)
class Route:
    local: bool
    path: tuple[str | None, str | None, str | None, str]


async def run_ip(*arguments: str) -> bytes | None:
    if IP_BINARY is None:
        return None
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            IP_BINARY, "-j", *arguments, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        async with asyncio.timeout(ROUTE_LOOKUP_TIMEOUT_SECONDS):
            output, _ = await process.communicate()
        return output if process.returncode == 0 else None
    except (OSError, TimeoutError):
        return None
    finally:
        if process is not None and process.returncode is None:
            process.kill()


async def route_of(address: str, port: int) -> Route | None:
    """The kernel's own routing decision for a TCP connection to an address, or None when it cannot be had."""
    if normalized_address(address) is None:
        return None
    output = await run_ip("-6" if ":" in address else "-4", "route", "get", address, "ipproto", "tcp", "dport", str(port))
    if output is None:
        return None
    try:
        entry = json.loads(output)[0]
        path = (entry.get("gateway"), entry.get("dev"), entry.get("prefsrc"), json.dumps(entry.get("nexthops"), sort_keys=True))
        return Route(local=entry.get("type", "unicast") != "unicast", path=path)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None


async def matching_prefix(address: str) -> int | None:
    """Prefix length of the most specific route other than the default one that covers the address, or None.

    Every routing table is looked at: policy rules decide which one the kernel uses, so the narrowest match anywhere is taken.
    """
    if normalized_address(address) is None:
        return None
    output = await run_ip("-6" if ":" in address else "-4", "route", "show", "table", "all", "match", address)
    if output is None:
        return None
    prefixes = []
    try:
        for entry in json.loads(output):
            destination = entry.get("dst")
            if destination not in (None, "default"):
                prefixes.append(ipaddress.ip_network(destination, strict=False).prefixlen)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return max(prefixes) if prefixes else None


class RouteChecker:
    """Decides whether the kernel would reach an address the way it reaches the internet.

    The address passes when the kernel sends it along the default route. A split-tunnel VPN adds routes for some public
    ranges, and the lookup honours policy rules and every routing table. What to do with such an address depends on `vpn`:
    "hosts" lets a wide route through (a provider or CDN block is the internet) and refuses a narrow one (a company
    endpoint), "block" refuses every such address, "allow" refuses none. An address of this host, an unreachable one and a
    failed lookup are always refused. The route can change between the lookup and the connection, so the connected socket is
    checked as well: the kernel picks its source address from the route it really used.
    """

    def __init__(
        self,
        vpn: str = "hosts",
        lookup: Callable[[str, int], Awaitable[Route | None]] = route_of,
        prefix_lookup: Callable[[str], Awaitable[int | None]] = matching_prefix,
    ):
        self.vpn = vpn
        self.lookup = lookup
        self.prefix_lookup = prefix_lookup
        self.slots = asyncio.Semaphore(ROUTE_SLOTS)
        self.defaults: dict[tuple[int, int], tuple[float, Route | None]] = {}

    async def limited_lookup(self, address: str, port: int) -> Route | None:
        async with self.slots:
            return await self.lookup(address, port)

    async def default_route(self, family: int, port: int) -> Route | None:
        cached = self.defaults.get((family, port))
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        route = await self.limited_lookup(CANARY_ADDRESSES[family], port)
        self.defaults[(family, port)] = (time.monotonic() + ROUTE_CACHE_SECONDS, route)
        return route

    async def is_narrow(self, address: str) -> bool:
        async with self.slots:
            prefix = await self.prefix_lookup(address)
        return prefix is None or prefix >= NARROW_ROUTE_PREFIX[6 if ":" in address else 4]

    async def accepted_route(self, address: str, port: int) -> Route | None:
        route = await self.limited_lookup(address, port)
        if route is None or route.local:
            return None
        if self.vpn == "allow":
            return route
        default = await self.default_route(6 if ":" in address else 4, port)
        if default is not None and not default.local and route.path == default.path:
            return route
        if self.vpn == "hosts" and default is not None and not await self.is_narrow(address):
            return route
        return None

    async def __call__(self, address: str, port: int) -> bool:
        return await self.accepted_route(address, port) is not None

    async def source_ok(self, local_address: str, address: str, port: int) -> bool:
        route = await self.accepted_route(address, port)
        if route is None:
            return False
        expected = route.path[2]
        if expected is None:
            return True
        actual, wanted = normalized_address(local_address), normalized_address(expected)
        return actual is not None and actual == wanted


@dataclass(frozen=True)
class Target:
    host: str
    port: int

    def authority(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == HTTP_PORT else f"{host}:{self.port}"


def parse_authority(text: str, default_port: int | None) -> Target:
    literal = IPV6_LITERAL_PATTERN.match(text)
    if literal:
        host, rest = literal.group(1), text[literal.end() :]
        try:
            ipaddress.IPv6Address(host)
        except ValueError as error:
            raise Refused(400, "the IPv6 address is not valid") from error
    else:
        host, colon, port_text = text.partition(":")
        rest = colon + port_text
        if HOSTNAME_PATTERN.fullmatch(host) is None:
            raise Refused(400, "the host name is not valid")
    if not rest:
        if default_port is None:
            raise Refused(400, "the port is missing")
        return Target(host, default_port)
    port_text = rest[1:]
    if not rest.startswith(":") or not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise Refused(400, "the port is not valid")
    return Target(host, int(port_text))


@dataclass(frozen=True)
class Request:
    method: str
    target: Target
    path: str
    headers: tuple[tuple[str, str], ...]

    def forward_head(self) -> bytes:
        lines = [f"{self.method} {self.path} HTTP/1.1", f"Host: {self.target.authority()}"]
        lines += [f"{name}: {value}" for name, value in self.headers if name.lower() not in HOP_BY_HOP_HEADERS]
        lines.append("Connection: close")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def parse_request(head: bytes) -> Request:
    lines = head.decode("latin-1")[: -len("\r\n\r\n")].split("\r\n")
    if any(character in line for line in lines for character in "\r\n\x00"):
        raise Refused(400, "the request has a stray control character")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise Refused(400, "the request line is not valid")
    method, uri, _ = parts
    if METHOD_PATTERN.fullmatch(method) is None:
        raise Refused(400, "the method is not valid")
    headers = []
    for line in lines[1:]:
        name, colon, value = line.partition(":")
        if not colon or HEADER_NAME_PATTERN.fullmatch(name) is None:
            raise Refused(400, "a header line is not valid")
        headers.append((name, value.strip()))
    if method == "CONNECT":
        return Request(method, parse_authority(uri, None), "", tuple(headers))
    if not uri.lower().startswith("http://"):
        raise Refused(400, "this is a proxy: send CONNECT, or a request with a full http:// address")
    try:
        parts_of_uri = urlsplit(uri)
    except ValueError as error:
        raise Refused(400, "the address is not valid") from error
    if "@" in parts_of_uri.netloc or any(character.isspace() for character in uri):
        raise Refused(400, "the address is not valid")
    path = (parts_of_uri.path or "/") + (f"?{parts_of_uri.query}" if parts_of_uri.query else "")
    return Request(method, parse_authority(parts_of_uri.netloc, HTTP_PORT), path, tuple(headers))


async def read_head(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        if len(data) > HEAD_LIMIT_BYTES:
            raise Refused(431, "the request head is too large")
        chunk = await reader.read(RELAY_CHUNK_BYTES)
        if not chunk:
            raise Refused(400, "the connection closed before a full request")
        data += chunk
    end = data.index(b"\r\n\r\n") + 4
    if end > HEAD_LIMIT_BYTES:
        raise Refused(431, "the request head is too large")
    return bytes(data[:end]), bytes(data[end:])


class ThreadResolver:
    """Name lookups on daemon threads, at most `slots` alive at once.

    A lookup that times out keeps its slot until the resolver call returns, so slow DNS cannot pile up threads, and a hung
    call cannot hold the process at exit.
    """

    def __init__(self, slots: int = RESOLVER_SLOTS):
        self.slots = asyncio.Semaphore(slots)

    async def __call__(self, host: str, port: int) -> list[str]:
        await self.slots.acquire()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[str]] = loop.create_future()

        def deliver(addresses: list[str], error: OSError | None) -> None:
            self.slots.release()
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(addresses)

        def lookup() -> None:
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                outcome: tuple[list[str], OSError | None] = (list(dict.fromkeys(str(info[4][0]) for info in infos)), None)
            except OSError as error:
                outcome = ([], error)
            try:
                loop.call_soon_threadsafe(deliver, *outcome)
            except RuntimeError:
                pass

        try:
            threading.Thread(target=lookup, daemon=True).start()
        except RuntimeError as error:
            self.slots.release()
            raise OSError("cannot start a lookup thread") from error
        return await future


@dataclass
class Transfer:
    up: int = 0
    down: int = 0
    started: float = field(default_factory=time.monotonic)


def trim_log(path: Path) -> None:
    try:
        if path.stat().st_size > LOG_TRIM_ABOVE_BYTES:
            tail = path.read_bytes()[-LOG_KEEP_BYTES:]
            path.write_bytes(tail[tail.find(b"\n") + 1 :])
    except OSError:
        pass


def parse_ports(text: str) -> tuple[int, ...] | None:
    if text.strip().lower() == "any":
        return None
    ports = tuple(int(item) for item in re.split(r"[,\s]+", text) if item.isdigit() and 0 < int(item) < 65536)
    if not ports:
        raise ValueError(f"no usable port in {text!r}")
    return ports


class WebProxy:
    def __init__(
        self,
        ports: tuple[int, ...] | None = None,
        resolve: Callable[[str, int], Awaitable[list[str]]] | None = None,
        address_ok: Callable[[str], bool] = is_public_address,
        route_ok: Callable[[str, int], Awaitable[bool]] | None = None,
        log_path: Path | None = None,
    ):
        self.ports = ports
        self.resolve = resolve or ThreadResolver()
        self.route_ok = route_ok or RouteChecker()
        self.source_ok = getattr(self.route_ok, "source_ok", None)
        self.address_ok = address_ok
        self.log_path = log_path
        self.active = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        transfer = Transfer()
        label, verdict, detail = "-", "DENY", ""
        upstream_writer = None
        dropped = False
        self.active += 1
        try:
            if self.active > MAX_CONNECTIONS + MAX_REFUSALS:
                dropped = True
                return
            if self.active > MAX_CONNECTIONS:
                raise Refused(503, "too many connections")
            async with asyncio.timeout(HEAD_TIMEOUT_SECONDS):
                head, leftover = await read_head(reader)
            request = parse_request(head)
            label = f"{request.method} {request.target.host}:{request.target.port}"
            if not self.port_allowed(request.target.port):
                raise Refused(403, f"port {request.target.port} is not allowed")
            addresses = await self.public_addresses(request.target)
            upstream_reader, upstream_writer = await self.connect(addresses, request.target.port)
            if request.method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                first = leftover
            else:
                first = request.forward_head() + leftover
            upstream_writer.write(first)
            transfer.up += len(first)
            verdict = "ALLOW"
            await self.relay(reader, writer, upstream_reader, upstream_writer, transfer)
        except Refused as refused:
            detail = refused.reason
            await self.reply_error(writer, refused)
            await self.linger(reader)
        except (TimeoutError, OSError) as error:
            detail = f"{type(error).__name__}"
        finally:
            self.active -= 1
            if not dropped:
                self.log(verdict, label, detail, transfer)
            await self.close(writer, upstream_writer)

    def port_allowed(self, port: int) -> bool:
        return port not in DENIED_PORTS if self.ports is None else port in self.ports

    async def public_addresses(self, target: Target) -> list[str]:
        try:
            async with asyncio.timeout(RESOLVE_TIMEOUT_SECONDS):
                addresses = await self.resolve(target.host, target.port)
        except (TimeoutError, OSError):
            addresses = []
        # the same answer for a name that does not resolve and one that resolves inside, so the proxy cannot be used to probe
        # internal names; one private answer among public ones is how a rebinding trick looks, so the whole name is refused
        refusal = Refused(403, f"{target.host} does not resolve to a public address")
        if not addresses or not all(self.address_ok(address) for address in addresses):
            raise refusal
        # an address the kernel would not send along the default route (an IPv6 one on a host without IPv6, or one behind a
        # VPN) is left out, and the name is refused only when none is left
        candidates = addresses[:MAX_ADDRESSES]
        verdicts = await asyncio.gather(*(self.route_ok(address, target.port) for address in candidates))
        reachable = [address for address, verdict in zip(candidates, verdicts, strict=True) if verdict]
        if not reachable:
            raise refusal
        return reachable

    async def connect(self, addresses: list[str], port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        for address in addresses:
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                    reader, writer = await asyncio.open_connection(address, port)
            except (TimeoutError, OSError):
                continue
            if self.source_ok is not None and not await self.source_ok(writer.get_extra_info("sockname")[0], address, port):
                writer.close()
                continue
            return reader, writer
        raise Refused(502, f"cannot connect to port {port}")

    @staticmethod
    async def relay(client_reader, client_writer, upstream_reader, upstream_writer, transfer: Transfer) -> None:
        async def pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter, upstream: bool) -> None:
            try:
                while chunk := await asyncio.wait_for(source.read(RELAY_CHUNK_BYTES), IDLE_TIMEOUT_SECONDS):
                    sink.write(chunk)
                    await sink.drain()
                    if upstream:
                        transfer.up += len(chunk)
                    else:
                        transfer.down += len(chunk)
                if sink.can_write_eof():
                    sink.write_eof()
            except (TimeoutError, OSError):
                client_writer.close()
                upstream_writer.close()

        uplink = asyncio.create_task(pipe(client_reader, upstream_writer, True))
        try:
            async with asyncio.timeout(MAX_TUNNEL_SECONDS):
                await pipe(upstream_reader, client_writer, False)
        except TimeoutError:
            pass
        finally:
            # the server is done, so a client that keeps its side open must not hold the connection
            uplink.cancel()
            await asyncio.gather(uplink, return_exceptions=True)

    @staticmethod
    async def reply_error(writer: asyncio.StreamWriter, refused: Refused) -> None:
        reason = "".join(character for character in refused.reason if character.isprintable() and character.isascii())[:REASON_MAX_CHARS]
        body = f"spi web proxy: {reason}\n".encode("ascii")
        head = f"HTTP/1.1 {refused.status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        try:
            writer.write(head.encode("ascii") + body)
            await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
        except OSError:
            pass

    @staticmethod
    async def linger(reader: asyncio.StreamReader) -> None:
        # closing with unread request bytes would reset the connection and the client would lose the error reply
        try:
            async with asyncio.timeout(LINGER_SECONDS):
                while await reader.read(RELAY_CHUNK_BYTES):
                    pass
        except (TimeoutError, OSError):
            pass

    @staticmethod
    async def close(*writers: asyncio.StreamWriter | None) -> None:
        for writer in writers:
            if writer is None:
                continue
            try:
                writer.close()
                async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                    await writer.wait_closed()
            except (TimeoutError, OSError):
                pass

    def log(self, verdict: str, label: str, detail: str, transfer: Transfer) -> None:
        if self.log_path is None:
            return
        elapsed = time.monotonic() - transfer.started
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {verdict} {label} up={transfer.up} down={transfer.down} {elapsed:.1f}s"
        try:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(line + (f" {detail}" if detail else "") + "\n")
            trim_log(self.log_path)
        except OSError:
            pass


async def watch_parent(pid: int, stop: asyncio.Event) -> None:
    while True:
        await asyncio.sleep(PARENT_POLL_SECONDS)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stop.set()
            return


async def serve(args: argparse.Namespace) -> None:
    proxy = WebProxy(parse_ports(args.ports), route_ok=RouteChecker(vpn=args.vpn), log_path=Path(args.log) if args.log else None)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_number, stop.set)
    Path(args.socket).unlink(missing_ok=True)
    previous = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(proxy.handle, path=args.socket)
    finally:
        os.umask(previous)
    watcher = asyncio.create_task(watch_parent(args.parent_pid, stop)) if args.parent_pid else None
    try:
        await stop.wait()
    finally:
        server.close()
        if watcher:
            watcher.cancel()
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        Path(args.socket).unlink(missing_ok=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtering web proxy for the spi sandbox: public addresses only, on a unix socket.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--ports", default="any", help="ports that may be reached, separated by commas, or any (all but the mail ports)")
    parser.add_argument("--vpn", choices=VPN_MODES, default="hosts", help="public addresses that a VPN routes: hosts = only wide provider ranges, block = none, allow = all")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process is gone")
    parser.add_argument("--log")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        parse_ports(args.ports)
    except ValueError as error:
        print(f"spi-web-proxy: {error}", file=sys.stderr)
        return 1
    if IP_BINARY is None:
        print("spi-web-proxy: ip (iproute2) is not installed", file=sys.stderr)
        return 1
    if args.log:
        trim_log(Path(args.log))
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
