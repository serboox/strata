#!/usr/bin/python3
import json
import os
import socket
import sys

SOCKET_PATH = os.environ.get("SPI_GH_SOCKET", "/tmp/gh.sock")
REPLY_TIMEOUT_SECONDS = 120
CHUNK_BYTES = 65536
UNREACHABLE_EXIT_CODE = 1


def ask(argv: list[str]) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(REPLY_TIMEOUT_SECONDS)
        conn.connect(SOCKET_PATH)
        conn.sendall(json.dumps({"argv": argv}).encode() + b"\n")
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := conn.recv(CHUNK_BYTES):
            chunks.append(chunk)
    return json.loads(b"".join(chunks))


def main() -> int:
    try:
        reply = ask(sys.argv[1:])
        streams = ((sys.stdout, str(reply["stdout"])), (sys.stderr, str(reply["stderr"])))
        rc = int(reply["rc"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"gh: the GitHub broker is not reachable: {error}", file=sys.stderr)
        return UNREACHABLE_EXIT_CODE
    try:
        for stream, text in streams:
            stream.buffer.write(text.encode("utf-8", "replace"))
            stream.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return rc


if __name__ == "__main__":
    sys.exit(main())
