import asyncio
import os
import shutil
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

import spi_web_proxy as proxy_module
from spi_web_proxy import (
    Refused,
    WebProxy,
    is_provider_host,
    is_public_address,
    parse_authority,
    parse_request,
)

PROXY = ROOT / "tools" / "spi_web_proxy.py"
CRLF2 = b"\r\n\r\n"


def allow_everything(address):
    return True


async def allow_route(address, port):
    return True


class AddressTests(unittest.TestCase):
    def test_public_addresses_pass(self):
        for text in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "::ffff:8.8.8.8"):
            with self.subTest(text=text):
                self.assertTrue(is_public_address(text))

    def test_everything_local_private_or_special_is_refused(self):
        for text in (
            "127.0.0.1",
            "127.1.2.3",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "198.18.0.1",
            "192.88.99.1",
            "fec0::1",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::a00:1",
            "2002:a00:1::",
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",
            "not an address",
            "",
            "1.2.3",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_public_address(text))


FAKE_IP = f"""#!{sys.executable}
import os, sys, time
mode = os.environ.get("FAKE_IP_MODE", "ok")
if mode == "fail":
    sys.stderr.write("RTNETLINK answers: Network is unreachable\\n")
    sys.exit(2)
if mode == "hang":
    open(os.environ["FAKE_IP_PIDFILE"], "w").write(str(os.getpid()))
    time.sleep(30)
if os.environ.get("FAKE_IP_ARGS"):
    open(os.environ["FAKE_IP_ARGS"], "w").write(" ".join(sys.argv[1:]))
answer = os.environ["FAKE_IP_SHOW_JSON"] if "show" in sys.argv else os.environ["FAKE_IP_JSON"]
sys.stdout.write("not json" if mode == "garbage" else answer)
"""
DEFAULT_ROUTE = '[{"dst":"93.184.216.34","gateway":"192.168.1.1","dev":"wlan0","prefsrc":"192.168.1.5","flags":[],"uid":1000,"cache":[]}]'
VPN_ROUTE = '[{"dst":"104.16.0.34","dev":"wlan0","prefsrc":"10.9.8.7","flags":[],"uid":1000,"cache":[]}]'
LOCAL_ROUTE = '[{"type":"local","dst":"192.168.1.5","dev":"lo","table":"local","prefsrc":"192.168.1.5","flags":[],"uid":1000,"cache":["local"]}]'
MATCHING_ROUTES = '[{"dst":"default","gateway":"192.168.1.1","dev":"wlan0"},{"dst":"104.16.0.0/13","dev":"wlan0"},{"dst":"104.26.0.0/16","dev":"wlan0"}]'


class RouteLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fake = Path(self.tmp.name) / "ip"
        fake.write_text(FAKE_IP, encoding="utf-8")
        fake.chmod(0o755)
        patcher = mock.patch.object(proxy_module, "IP_BINARY", str(fake))
        patcher.start()
        self.addCleanup(patcher.stop)

    def environment(self, mode, answer="", shown="[]"):
        return mock.patch.dict(os.environ, {"FAKE_IP_MODE": mode, "FAKE_IP_JSON": answer, "FAKE_IP_SHOW_JSON": shown, "FAKE_IP_PIDFILE": str(Path(self.tmp.name) / "pid"), "FAKE_IP_ARGS": str(Path(self.tmp.name) / "args")})

    async def lookup(self, mode, answer="", address="93.184.216.34"):
        with self.environment(mode, answer):
            return await proxy_module.route_of(address, 443)

    async def prefix(self, mode, shown="[]", address="104.26.5.4"):
        with self.environment(mode, shown=shown):
            return await proxy_module.matching_prefix(address)

    async def test_default_vpn_and_local_routes_are_told_apart(self):
        default = await self.lookup("ok", DEFAULT_ROUTE)
        vpn = await self.lookup("ok", VPN_ROUTE)
        local = await self.lookup("ok", LOCAL_ROUTE)
        self.assertFalse(default.local)
        self.assertFalse(vpn.local)
        self.assertTrue(local.local)
        self.assertNotEqual(default.path, vpn.path)
        self.assertEqual(default.path, (await self.lookup("ok", DEFAULT_ROUTE, "93.184.216.99")).path)

    async def test_the_most_specific_route_gives_the_prefix(self):
        self.assertEqual(await self.prefix("ok", MATCHING_ROUTES), 16)
        self.assertEqual(await self.prefix("ok", '[{"dst":"8.8.8.8"}]'), 32)
        self.assertEqual(await self.prefix("ok", '[{"dst":"2a00:1450::/32"}]', "2a00:1450::1"), 32)

    async def test_the_prefix_looks_at_every_routing_table(self):
        await self.prefix("ok", MATCHING_ROUTES)
        self.assertIn("route show table all match 104.26.5.4", (Path(self.tmp.name) / "args").read_text())

    async def test_a_prefix_that_cannot_be_had_gives_none(self):
        self.assertIsNone(await self.prefix("ok", '[{"dst":"default","gateway":"192.168.1.1"}]'))
        self.assertIsNone(await self.prefix("ok", "[]"))
        self.assertIsNone(await self.prefix("ok", '[{"dst":"not a network"}]'))
        self.assertIsNone(await self.prefix("fail"))
        self.assertIsNone(await self.prefix("garbage"))
        self.assertIsNone(await self.prefix("ok", MATCHING_ROUTES, "--help"))

    async def test_a_lookup_that_cannot_be_had_gives_none(self):
        self.assertIsNone(await self.lookup("fail"))
        self.assertIsNone(await self.lookup("garbage"))
        self.assertIsNone(await self.lookup("ok", "[]"))
        self.assertIsNone(await self.lookup("ok", '["x"]'))
        with mock.patch.object(proxy_module, "ROUTE_LOOKUP_TIMEOUT_SECONDS", 1):
            started = time.monotonic()
            self.assertIsNone(await self.lookup("hang"))
            self.assertLess(time.monotonic() - started, 5)
        with mock.patch.object(proxy_module, "IP_BINARY", None):
            self.assertIsNone(await proxy_module.route_of("93.184.216.34", 443))
        with mock.patch.object(proxy_module, "IP_BINARY", "/nonexistent/ip"):
            self.assertIsNone(await proxy_module.route_of("93.184.216.34", 443))

    async def test_only_an_ip_literal_reaches_the_command(self):
        for odd in ("--help", "-4", "1.2.3.4; ls", "example.com", "", "93.184.216.34 dev lo"):
            with self.subTest(odd=odd):
                self.assertIsNone(await self.lookup("ok", DEFAULT_ROUTE, odd))

    async def test_a_cancelled_lookup_does_not_leave_the_command_running(self):
        pid_file = Path(self.tmp.name) / "pid"
        with self.environment("hang"):
            task = asyncio.create_task(proxy_module.route_of("93.184.216.34", 443))
            for _ in range(50):
                if pid_file.exists() and pid_file.read_text():
                    break
                await asyncio.sleep(0.1)
            pid = int(pid_file.read_text())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        for _ in range(30):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.1)
        self.fail("the lookup command is still running after the cancellation")

    async def test_the_real_ip_command_calls_loopback_local(self):
        if shutil.which("ip") is None:
            self.skipTest("no ip command")
        with mock.patch.object(proxy_module, "IP_BINARY", shutil.which("ip")):
            route = await proxy_module.route_of("127.0.0.1", 443)
        self.assertTrue(route.local)


def route(local=False, via="192.168.1.1", src="192.168.1.5"):
    return proxy_module.Route(local=local, path=(via, "wlan0", src, "null"))


VPN_PATH = {"via": None, "src": "10.9.8.7"}
CANARY = {"192.0.2.1": route(), "2001:db8::1": route(src="2001:db8:1::5")}


class RouteCheckerTests(unittest.IsolatedAsyncioTestCase):
    def checker(self, answers, prefixes=None, **kwargs):
        calls = []

        async def lookup(address, port):
            calls.append((address, port))
            return answers.get(address)

        async def prefix_lookup(address):
            return (prefixes or {}).get(address)

        return proxy_module.RouteChecker(lookup=lookup, prefix_lookup=prefix_lookup, **kwargs), calls

    async def test_block_mode_accepts_only_the_default_route(self):
        checker, _ = self.checker(
            {**CANARY, "93.184.216.34": route(), "34.7.131.105": route(**VPN_PATH), "104.26.5.4": route(**VPN_PATH), "192.168.1.5": route(local=True)},
            {"34.7.131.105": 32, "104.26.5.4": 14},
            vpn="block",
        )
        self.assertTrue(await checker("93.184.216.34", 443))
        self.assertFalse(await checker("34.7.131.105", 443))
        self.assertFalse(await checker("104.26.5.4", 443))
        self.assertFalse(await checker("192.168.1.5", 443))
        self.assertFalse(await checker("203.0.113.9", 443))

    async def test_hosts_mode_lets_wide_ranges_through_and_stops_narrow_routes(self):
        answers = {**CANARY, "93.184.216.34": route()}
        prefixes = {}
        for address, prefix in (("104.26.5.4", 14), ("185.15.59.224", 8), ("103.21.244.9", 22), ("18.239.69.7", 24), ("34.7.131.105", 32), ("185.94.84.9", 23)):
            answers[address] = route(**VPN_PATH)
            prefixes[address] = prefix
        answers["198.51.100.7"] = route(**VPN_PATH)
        checker, _ = self.checker(answers, prefixes)
        self.assertTrue(await checker("93.184.216.34", 443))
        for address in ("104.26.5.4", "185.15.59.224", "103.21.244.9"):
            with self.subTest(address=address):
                self.assertTrue(await checker(address, 443))
        for address in ("18.239.69.7", "34.7.131.105", "185.94.84.9", "198.51.100.7"):
            with self.subTest(address=address):
                self.assertFalse(await checker(address, 443))

    async def test_the_narrow_limit_differs_for_ipv6(self):
        answers = {**CANARY, "2a00:1450::1": route(**VPN_PATH), "2a00:1451::1": route(**VPN_PATH)}
        checker, _ = self.checker(answers, {"2a00:1450::1": 47, "2a00:1451::1": 48})
        self.assertTrue(await checker("2a00:1450::1", 443))
        self.assertFalse(await checker("2a00:1451::1", 443))

    async def test_allow_mode_admits_every_route_but_never_this_host(self):
        checker, _ = self.checker({"34.7.131.105": route(**VPN_PATH), "192.168.1.5": route(local=True)}, {"34.7.131.105": 32}, vpn="allow")
        self.assertTrue(await checker("34.7.131.105", 443))
        self.assertFalse(await checker("192.168.1.5", 443))
        self.assertFalse(await checker("203.0.113.9", 443))

    async def test_a_missing_default_route_refuses_everything_except_in_allow_mode(self):
        for mode in ("hosts", "block"):
            checker, _ = self.checker({"93.184.216.34": route(), "104.26.5.4": route(**VPN_PATH)}, {"104.26.5.4": 14}, vpn=mode)
            self.assertFalse(await checker("93.184.216.34", 443))
            self.assertFalse(await checker("104.26.5.4", 443))

    async def test_the_default_route_is_looked_up_once_per_cache_period(self):
        checker, calls = self.checker({**CANARY, "93.184.216.34": route(), "93.184.216.35": route()})
        for address in ("93.184.216.34", "93.184.216.35", "93.184.216.34"):
            self.assertTrue(await checker(address, 443))
        self.assertEqual(calls.count(("192.0.2.1", 443)), 1)
        await checker("93.184.216.34", 80)
        self.assertEqual(calls.count(("192.0.2.1", 80)), 1)
        with mock.patch.object(proxy_module, "ROUTE_CACHE_SECONDS", 0):
            checker, calls = self.checker({**CANARY, "93.184.216.34": route()})
            await checker("93.184.216.34", 443)
            await checker("93.184.216.34", 443)
        self.assertEqual(calls.count(("192.0.2.1", 443)), 2)

    async def test_the_source_address_of_the_connection_must_be_the_one_the_route_gave(self):
        checker, _ = self.checker(
            {**CANARY, "93.184.216.34": route(), "104.26.5.4": route(**VPN_PATH), "2606:4700::1": route(src="2001:db8:1::5")}, {"104.26.5.4": 14}
        )
        self.assertTrue(await checker.source_ok("192.168.1.5", "93.184.216.34", 443))
        self.assertFalse(await checker.source_ok("10.9.8.7", "93.184.216.34", 443))
        self.assertTrue(await checker.source_ok("10.9.8.7", "104.26.5.4", 443))
        self.assertFalse(await checker.source_ok("192.168.1.5", "104.26.5.4", 443))
        self.assertTrue(await checker.source_ok("2001:0db8:1:0::5", "2606:4700::1", 443))
        self.assertFalse(await checker.source_ok("not an address", "93.184.216.34", 443))
        self.assertFalse(await checker.source_ok("192.168.1.5", "203.0.113.9", 443))

    async def test_a_route_without_a_source_address_has_nothing_to_compare_the_socket_with(self):
        checker, _ = self.checker({"192.0.2.1": route(src=None), "93.184.216.34": route(src=None)})
        self.assertTrue(await checker("93.184.216.34", 443))
        self.assertTrue(await checker.source_ok("192.168.1.5", "93.184.216.34", 443))

    async def test_a_route_that_differs_from_the_default_one_is_refused_in_block_mode_even_if_only_the_source_differs(self):
        checker, _ = self.checker({**CANARY, "93.184.216.34": route(src="10.9.8.7")}, vpn="block")
        self.assertFalse(await checker("93.184.216.34", 443))

    async def test_lookups_run_at_most_route_slots_at_a_time(self):
        running, peak = 0, 0

        async def lookup(address, port):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.05)
            running -= 1
            return route()

        with mock.patch.object(proxy_module, "ROUTE_SLOTS", 2):
            checker = proxy_module.RouteChecker(lookup=lookup)
        await asyncio.gather(*(checker(f"93.184.216.{number}", 443) for number in range(12)))
        self.assertEqual(peak, 2)


class ThreadResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookups_run_on_daemon_threads_and_at_most_slots_at_once(self):
        running, peak, lock = 0, 0, threading.Lock()

        def slow_lookup(host, port, type=0):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.3)
            with lock:
                running -= 1
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        resolver = proxy_module.ThreadResolver(slots=2)
        with mock.patch.object(socket, "getaddrinfo", slow_lookup):
            results = await asyncio.gather(*(resolver(f"h{number}.test", 443) for number in range(6)))
        self.assertEqual(results, [["93.184.216.34"]] * 6)
        self.assertEqual(peak, 2)

    async def test_a_timed_out_lookup_keeps_its_slot_until_the_thread_returns(self):
        starts = []

        def lookup(host, port, type=0):
            starts.append(time.monotonic())
            time.sleep(0.8)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        resolver = proxy_module.ThreadResolver(slots=1)
        with mock.patch.object(socket, "getaddrinfo", lookup):
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await resolver("slow.test", 443)
            self.assertEqual(await resolver("next.test", 443), ["93.184.216.34"])
        self.assertGreaterEqual(starts[1] - starts[0], 0.7)

    async def test_a_thread_that_cannot_start_gives_its_slot_back(self):
        def lookup(host, port, type=0):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        resolver = proxy_module.ThreadResolver(slots=1)
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("can't start new thread")), self.assertRaises(OSError):
            await resolver("a.test", 443)
        with mock.patch.object(socket, "getaddrinfo", lookup):
            self.assertEqual(await asyncio.wait_for(resolver("b.test", 443), 5), ["93.184.216.34"])

    async def test_a_failed_lookup_raises_oserror(self):
        def fail(host, port, type=0):
            raise socket.gaierror("no such name")

        with mock.patch.object(socket, "getaddrinfo", fail), self.assertRaises(OSError):
            await proxy_module.ThreadResolver()("nowhere.test", 443)


class ParseTests(unittest.TestCase):
    def test_authorities(self):
        self.assertEqual(parse_authority("example.com:443", None).port, 443)
        self.assertEqual(parse_authority("example.com", 80).port, 80)
        self.assertEqual(parse_authority("[2001:db8::1]:443", None).host, "2001:db8::1")
        self.assertEqual(parse_authority("[::1]", 80).host, "::1")

    def test_bad_authorities(self):
        for text, default in (
            ("example.com", None),
            ("example.com:0", None),
            ("example.com:99999", None),
            ("example.com:http", None),
            ("a:b:80", None),
            ("[zz]:80", None),
            ("[::1", 80),
            ("exa mple.com:80", None),
            ("exa/mple.com:80", None),
            ("", 80),
            (":80", None),
        ):
            with self.subTest(text=text), self.assertRaises(Refused) as caught:
                parse_authority(text, default)
            self.assertEqual(caught.exception.status, 400)

    def test_connect_request(self):
        request = parse_request(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
        self.assertEqual((request.method, request.target.host, request.target.port), ("CONNECT", "example.com", 443))

    def test_a_plain_request_is_rewritten_for_the_origin(self):
        request = parse_request(
            b"GET http://example.com:8080/a?b=1 HTTP/1.1\r\nHost: evil\r\nProxy-Connection: keep-alive\r\n"
            b"Proxy-Authorization: x\r\nConnection: keep-alive\r\nUser-Agent: t\r\n\r\n"
        )
        self.assertEqual(
            request.forward_head(),
            b"GET /a?b=1 HTTP/1.1\r\nHost: example.com:8080\r\nUser-Agent: t\r\nConnection: close\r\n\r\n",
        )

    def test_the_default_port_stays_out_of_the_host_header(self):
        request = parse_request(b"GET http://example.com HTTP/1.1\r\n\r\n")
        self.assertEqual(request.forward_head(), b"GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n")

    def test_bad_requests(self):
        for head in (
            b"GET / HTTP/1.1\r\n\r\n",
            b"GET https://example.com/ HTTP/1.1\r\n\r\n",
            b"GET http://example.com/ HTTP/2\r\n\r\n",
            b"GET http://example.com/\r\n\r\n",
            b"get http://example.com/ HTTP/1.1\r\n\r\n",
            b"GET http://user@example.com/ HTTP/1.1\r\n\r\n",
            b"GET http://exa mple.com/ HTTP/1.1\r\n\r\n",
            b"GET http://example.com/ HTTP/1.1\r\nBad Header\r\n\r\n",
            b"GET http://example.com/ HTTP/1.1\r\n folded: x\r\n\r\n",
            b"GET http://example.com/ HTTP/1.1\r\nX: a\x00b\r\n\r\n",
            b"GET http://example.com/ HTTP/1.1\r\nX: a\nb\r\n\r\n",
            b"CONNECT example.com HTTP/1.1\r\n\r\n",
            b"CONNECT http://example.com:80/ HTTP/1.1\r\n\r\n",
            b"GET http://[::1 HTTP/1.1\r\n\r\n",
            b"\r\n\r\n",
        ):
            with self.subTest(head=head), self.assertRaises(Refused):
                parse_request(head)


class ProxyTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.socket = str(self.dir / "web.sock")
        self.log = self.dir / "log"

    async def start_proxy(self, **kwargs):
        kwargs.setdefault("route_ok", allow_route)
        proxy = WebProxy(log_path=self.log, **kwargs)
        server = await asyncio.start_unix_server(proxy.handle, path=self.socket)
        self.addCleanup(server.close)
        return proxy

    async def start_upstream(self, handler):
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        self.addCleanup(server.close)
        return server.sockets[0].getsockname()[1]

    async def connect(self):
        reader, writer = await asyncio.open_unix_connection(self.socket)
        self.addCleanup(writer.close)
        return reader, writer

    async def exchange(self, raw):
        reader, writer = await self.connect()
        writer.write(raw)
        await writer.drain()
        async with asyncio.timeout(10):
            return await reader.read()

    @staticmethod
    def status_of(reply):
        return int(reply.split(b" ", 2)[1])

    async def eventually_logged(self, text):
        for _ in range(50):
            if self.log.exists() and text in self.log.read_text(encoding="utf-8"):
                return
            await asyncio.sleep(0.1)
        self.fail(f"{text!r} never reached the log: {self.log.read_text(encoding='utf-8') if self.log.exists() else 'no log'}")


class TunnelTests(ProxyTestCase):
    async def test_connect_relays_both_ways_and_counts_the_bytes(self):
        async def echo(reader, writer):
            writer.write((await reader.read(100)).upper())
            await writer.drain()
            writer.close()

        port = await self.start_upstream(echo)
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        reader, writer = await self.connect()
        writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        self.assertEqual(await asyncio.wait_for(reader.readuntil(CRLF2), 5), b"HTTP/1.1 200 Connection established\r\n\r\n")
        writer.write(b"hello")
        self.assertEqual(await asyncio.wait_for(reader.read(100), 5), b"HELLO")
        writer.close()
        await self.eventually_logged(f"ALLOW CONNECT 127.0.0.1:{port} up=5 down=5")

    async def test_bytes_sent_with_the_first_request_are_counted(self):
        async def echo(reader, writer):
            writer.write((await reader.read(100)).upper())
            await writer.drain()
            writer.close()

        port = await self.start_upstream(echo)
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        reader, writer = await self.connect()
        writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\nhello".encode())
        await asyncio.wait_for(reader.readuntil(CRLF2), 5)
        self.assertEqual(await asyncio.wait_for(reader.read(100), 5), b"HELLO")
        writer.close()
        await self.eventually_logged(f"ALLOW CONNECT 127.0.0.1:{port} up=5 down=5")

    async def test_half_close_still_delivers_the_reply(self):
        async def reply_after_eof(reader, writer):
            await reader.read()
            writer.write(b"done")
            await writer.drain()
            writer.close()

        port = await self.start_upstream(reply_after_eof)
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        reader, writer = await self.connect()
        writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
        await asyncio.wait_for(reader.readuntil(CRLF2), 5)
        writer.write(b"x")
        writer.write_eof()
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"done")

    async def test_the_name_is_resolved_once_and_the_connection_goes_to_that_address(self):
        async def hello(reader, writer):
            writer.write(b"hello")
            await writer.drain()
            writer.close()

        port = await self.start_upstream(hello)
        calls = []

        async def resolve(host, port_number):
            calls.append((host, port_number))
            return ["127.0.0.1"]

        await self.start_proxy(ports=(port,), resolve=resolve, address_ok=allow_everything)
        reader, writer = await self.connect()
        writer.write(f"CONNECT pinned.test:{port} HTTP/1.1\r\n\r\n".encode())
        await asyncio.wait_for(reader.readuntil(CRLF2), 5)
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"hello")
        self.assertEqual(calls, [("pinned.test", port)])

    async def test_an_idle_tunnel_is_closed(self):
        async def silent(reader, writer):
            await reader.read()

        port = await self.start_upstream(silent)
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        with mock.patch.object(proxy_module, "IDLE_TIMEOUT_SECONDS", 1):
            reader, writer = await self.connect()
            writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
            await asyncio.wait_for(reader.readuntil(CRLF2), 5)
            started = time.monotonic()
            self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"")
            self.assertLess(time.monotonic() - started, 4)

    async def test_a_connection_whose_source_address_shows_another_route_is_dropped(self):
        received = []

        async def upstream(reader, writer):
            received.append(await reader.read(10))
            writer.close()

        class MovedRoute:
            async def __call__(self, address, port):
                return True

            async def source_ok(self, local_address, address, port):
                return False

        port = await self.start_upstream(upstream)
        await self.start_proxy(ports=(port,), address_ok=allow_everything, route_ok=MovedRoute())
        reply = await self.exchange(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\nhello".encode())
        self.assertEqual(self.status_of(reply), 502)
        self.assertNotIn(b"hello", b"".join(received))

    async def test_a_connection_whose_source_address_is_expected_goes_through(self):
        async def hello(reader, writer):
            writer.write(b"hello")
            await writer.drain()
            writer.close()

        class SameRoute:
            def __init__(self):
                self.seen = []

            async def __call__(self, address, port):
                return True

            async def source_ok(self, local_address, address, port):
                self.seen.append(local_address)
                return True

        checker = SameRoute()
        port = await self.start_upstream(hello)
        await self.start_proxy(ports=(port,), address_ok=allow_everything, route_ok=checker)
        reader, writer = await self.connect()
        writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
        await asyncio.wait_for(reader.readuntil(CRLF2), 5)
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"hello")
        self.assertEqual(checker.seen, ["127.0.0.1"])

    async def test_an_unreachable_port_is_a_502(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        reply = await self.exchange(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
        self.assertEqual(self.status_of(reply), 502)


class PlainHttpTests(ProxyTestCase):
    async def test_the_request_is_rewritten_and_the_body_passes(self):
        seen = []

        async def server(reader, writer):
            head = await reader.readuntil(CRLF2)
            length = int(next(line.split(b":")[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")))
            body = await reader.readexactly(length)
            seen.append((head, body))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        port = await self.start_upstream(server)
        await self.start_proxy(ports=(port,), address_ok=allow_everything)
        raw = (
            f"POST http://127.0.0.1:{port}/x?y=1 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nProxy-Connection: keep-alive\r\n"
            "Content-Length: 4\r\n\r\nbody"
        ).encode()
        reply = await self.exchange(raw)
        self.assertTrue(reply.endswith(b"ok"))
        head, body = seen[0]
        self.assertTrue(head.startswith(b"POST /x?y=1 HTTP/1.1\r\n"))
        self.assertIn(b"Connection: close", head)
        self.assertNotIn(b"Proxy-Connection", head)
        self.assertEqual(body, b"body")
        await self.eventually_logged(f"ALLOW POST 127.0.0.1:{port}")


class ProviderHostTests(unittest.TestCase):
    def test_hosts_of_model_providers_are_recognised_with_their_subdomains(self):
        for host in ("api.openai.com", "API.OpenAI.com.", "eu.api.openai.com", "my-resource.openai.azure.com", "api.anthropic.com", "claude.ai",
                     "us-central1-aiplatform.googleapis.com", "bedrock-runtime.eu-west-1.amazonaws.com", "openrouter.ai", "api.groq.com"):
            with self.subTest(host=host):
                self.assertTrue(is_provider_host(host))

    def test_other_hosts_stay_reachable(self):
        for host in ("example.com", "docs.python.org", "github.com", "api.github.com", "platform.openai.com", "notopenai.com", "evilapi.openai.com.example.com",
                     "storage.googleapis.com", "s3.eu-west-1.amazonaws.com", "huggingface.co", "127.0.0.1", "[::1]"):
            with self.subTest(host=host):
                self.assertFalse(is_provider_host(host))


class RefusalTests(ProxyTestCase):
    async def test_a_model_provider_is_refused_before_resolving_or_connecting(self):
        calls = []

        async def resolve(host, port):
            calls.append(host)
            return ["8.8.8.8"]

        await self.start_proxy(resolve=resolve)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock()) as connect:
            for raw in (b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n", b"GET http://api.anthropic.com/v1/messages HTTP/1.1\r\n\r\n"):
                with self.subTest(raw=raw):
                    reply = await self.exchange(raw)
                    self.assertEqual(self.status_of(reply), 403)
                    self.assertIn(b"local models only", reply)
            connect.assert_not_called()
        self.assertEqual(calls, [])

    async def test_local_private_and_odd_targets_never_connect(self):
        proxy = await self.start_proxy()
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock()) as connect:
            for raw, accepted in (
                (b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT [::1]:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT [::ffff:127.0.0.1]:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT [fec0::1]:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 192.88.99.1:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 2130706433:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 0x7f.1:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 169.254.169.254:80 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 10.0.0.1:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 192.168.0.1:80 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 100.64.0.1:443 HTTP/1.1\r\n\r\n", (403,)),
                (b"GET http://127.0.0.1:80/ HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 127.0.0.1:3306 HTTP/1.1\r\n\r\n", (403,)),
                (b"CONNECT 127.0.0.1:25 HTTP/1.1\r\n\r\n", (403,)),
            ):
                with self.subTest(raw=raw):
                    reply = await self.exchange(raw)
                    self.assertIn(self.status_of(reply), accepted)
                    self.assertIn(b"spi web proxy", reply)
            connect.assert_not_called()
        for _ in range(30):
            if proxy.active == 0:
                break
            await asyncio.sleep(0.1)
        self.assertEqual(proxy.active, 0)

    async def test_malformed_requests_are_400(self):
        await self.start_proxy()
        for raw in (
            b"GET / HTTP/1.1\r\n\r\n",
            b"CONNECT example.com HTTP/1.1\r\n\r\n",
            b"GET http://user@example.com/ HTTP/1.1\r\n\r\n",
            b"GET http://[::1 HTTP/1.1\r\n\r\n",
            b"\r\n\r\n",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.status_of(await self.exchange(raw)), 400)

    async def test_any_port_is_allowed_except_the_mail_ports(self):
        proxy = WebProxy()
        self.assertTrue(all(proxy.port_allowed(port) for port in (80, 443, 8080, 8443, 3000, 22, 5432, 65535)))
        self.assertFalse(any(proxy.port_allowed(port) for port in (25, 465, 587)))
        listed = WebProxy(ports=(80, 443))
        self.assertTrue(listed.port_allowed(443))
        self.assertFalse(listed.port_allowed(8080))
        self.assertIsNone(proxy_module.parse_ports("any"))
        self.assertEqual(proxy_module.parse_ports("80, 8080"), (80, 8080))

    async def test_a_public_name_on_an_unusual_port_is_connected(self):
        async def resolve(host, port):
            return ["93.184.216.34"]

        await self.start_proxy(resolve=resolve)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock(side_effect=Refused(502, "stop here"))) as connect:
            reply = await self.exchange(b"CONNECT shop.test:8443 HTTP/1.1\r\n\r\n")
        self.assertEqual(self.status_of(reply), 502)
        connect.assert_awaited_once_with(["93.184.216.34"], 8443)

    async def test_a_disallowed_port_is_refused_before_resolving(self):
        calls = []

        async def resolve(host, port):
            calls.append(host)
            return ["8.8.8.8"]

        await self.start_proxy(resolve=resolve)
        reply = await self.exchange(b"CONNECT example.com:25 HTTP/1.1\r\n\r\n")
        self.assertEqual(self.status_of(reply), 403)
        self.assertEqual(calls, [])

    async def test_one_private_answer_among_public_ones_refuses_the_name(self):
        async def resolve(host, port):
            return ["8.8.8.8", "10.0.0.1"]

        await self.start_proxy(resolve=resolve)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock()) as connect:
            reply = await self.exchange(b"CONNECT mixed.test:443 HTTP/1.1\r\n\r\n")
            connect.assert_not_called()
        self.assertEqual(self.status_of(reply), 403)

    async def test_only_addresses_that_pass_the_route_check_are_used(self):
        async def resolve(host, port):
            return ["93.184.216.34", "93.184.216.35", "2606:4700::1"]

        async def route_ok(address, port):
            return address == "93.184.216.35"

        await self.start_proxy(resolve=resolve, route_ok=route_ok)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock(side_effect=Refused(502, "stop here"))) as connect:
            reply = await self.exchange(b"CONNECT mixed.test:443 HTTP/1.1\r\n\r\n")
        self.assertEqual(self.status_of(reply), 502)
        connect.assert_awaited_once_with(["93.184.216.35"], 443)

    async def test_a_name_with_many_answers_is_checked_for_the_first_few_only(self):
        checked = []

        async def resolve(host, port):
            return [f"93.184.216.{number}" for number in range(1, 41)]

        async def route_ok(address, port):
            checked.append(address)
            return True

        await self.start_proxy(resolve=resolve, route_ok=route_ok)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock(side_effect=Refused(502, "stop here"))) as connect:
            await self.exchange(b"CONNECT many.test:443 HTTP/1.1\r\n\r\n")
        self.assertEqual(len(checked), proxy_module.MAX_ADDRESSES)
        self.assertEqual(len(connect.await_args.args[0]), proxy_module.MAX_ADDRESSES)

    async def test_a_name_with_no_address_on_the_default_route_is_refused_like_an_inside_name(self):
        async def resolve(host, port):
            return ["93.184.216.34"]

        async def route_ok(address, port):
            return False

        await self.start_proxy(resolve=resolve, route_ok=route_ok)
        with mock.patch.object(WebProxy, "connect", new=mock.AsyncMock()) as connect:
            reply = await self.exchange(b"CONNECT vpn.test:443 HTTP/1.1\r\n\r\n")
            connect.assert_not_called()
        self.assertEqual(self.status_of(reply), 403)
        self.assertIn(b"does not resolve to a public address", reply)

    async def test_a_name_that_does_not_resolve_looks_like_one_that_resolves_inside(self):
        async def resolve(host, port):
            if host == "nowhere.test":
                raise OSError("no such name")
            return ["10.0.0.5"]

        await self.start_proxy(resolve=resolve)
        replies = [await self.exchange(f"CONNECT {host}:443 HTTP/1.1\r\n\r\n".encode()) for host in ("nowhere.test", "inside.test")]
        self.assertEqual([self.status_of(reply) for reply in replies], [403, 403])
        self.assertEqual([reply.split(b"\r\n", 1)[0].split(b" ", 2)[2].split(b" ", 1)[1] for reply in replies], [b"does not resolve to a public address"] * 2)

    async def test_an_oversized_head_is_431_with_or_without_its_end(self):
        await self.start_proxy()
        with mock.patch.object(proxy_module, "HEAD_LIMIT_BYTES", 100):
            self.assertEqual(self.status_of(await self.exchange(b"GET http://example.com/ HTTP/1.1\r\n" + b"X: " + b"a" * 300)), 431)
            self.assertEqual(self.status_of(await self.exchange(b"GET http://example.com/ HTTP/1.1\r\nX: " + b"a" * 150 + b"\r\n\r\n")), 431)

    async def test_a_slow_head_is_cut_at_the_total_deadline(self):
        await self.start_proxy()
        with mock.patch.object(proxy_module, "HEAD_TIMEOUT_SECONDS", 1):
            _, writer = await self.connect()
            started = time.monotonic()
            try:
                for _ in range(20):
                    writer.write(b"G")
                    await writer.drain()
                    await asyncio.sleep(0.4)
            except OSError:
                pass
            self.assertLess(time.monotonic() - started, 4)

    async def test_too_many_connections_get_a_503(self):
        await self.start_proxy()
        with mock.patch.object(proxy_module, "MAX_CONNECTIONS", 2):
            await self.connect()
            await self.connect()
            await asyncio.sleep(0.3)
            self.assertEqual(self.status_of(await self.exchange(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")), 503)

    async def test_refused_connections_are_counted_and_a_flood_is_dropped(self):
        proxy = await self.start_proxy()
        with mock.patch.object(proxy_module, "MAX_CONNECTIONS", 1), mock.patch.object(proxy_module, "MAX_REFUSALS", 1):
            await self.connect()
            second_reader, second_writer = await self.connect()
            second_writer.write(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
            third_reader, third_writer = await self.connect()
            third_writer.write(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
            await asyncio.sleep(0.3)
            self.assertEqual(proxy.active, 2)
            second = await asyncio.wait_for(second_reader.read(), 5)
            try:
                third = await asyncio.wait_for(third_reader.read(), 5)
            except ConnectionResetError:
                third = b""
        self.assertEqual(self.status_of(second), 503)
        self.assertEqual(third, b"")

    async def test_the_log_has_one_line_per_connection_and_stays_bounded(self):
        await self.start_proxy()
        for _ in range(3):
            await self.exchange(b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n")
        await self.eventually_logged("DENY CONNECT 127.0.0.1:443")
        await asyncio.sleep(0.3)
        lines = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("does not resolve to a public address", lines[0])
        with mock.patch.object(proxy_module, "LOG_TRIM_ABOVE_BYTES", 2000), mock.patch.object(proxy_module, "LOG_KEEP_BYTES", 500):
            for _ in range(60):
                await self.exchange(b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n")
        self.assertLess(self.log.stat().st_size, 2300)


class ProcessTests(unittest.TestCase):
    def test_the_real_process_is_private_filters_and_follows_its_parent(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            socket_path = Path(folder) / "web.sock"
            parent = subprocess.Popen(["sleep", "3"])
            self.addCleanup(parent.kill)
            process = subprocess.Popen([sys.executable, str(PROXY), "--socket", str(socket_path), "--parent-pid", str(parent.pid), "--log", str(Path(folder) / "log")])
            self.addCleanup(process.kill)
            for _ in range(50):
                if socket_path.exists():
                    break
                time.sleep(0.1)
            self.assertTrue(socket_path.exists())
            self.assertEqual(stat.S_IMODE(os.stat(socket_path).st_mode), 0o600)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(10)
                conn.connect(str(socket_path))
                conn.sendall(b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n")
                self.assertIn(b" 403 ", conn.recv(1000))
            parent.wait()
            self.assertEqual(process.wait(timeout=15), 0)
            self.assertFalse(socket_path.exists())

    def test_a_bad_port_list_stops_the_start(self):
        result = subprocess.run([sys.executable, str(PROXY), "--socket", "/tmp/never-created.sock", "--ports", "abc"], capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no usable port", result.stderr)

    def test_log_trimming_keeps_whole_lines_of_the_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "log"
            log.write_text("".join(f"line {number}\n" for number in range(100000)), encoding="utf-8")
            proxy_module.trim_log(log)
            text = log.read_text(encoding="utf-8")
            self.assertLessEqual(len(text.encode()), proxy_module.LOG_KEEP_BYTES)
            self.assertTrue(text.startswith("line ") and text.endswith("line 99999\n"))


if __name__ == "__main__":
    unittest.main()
