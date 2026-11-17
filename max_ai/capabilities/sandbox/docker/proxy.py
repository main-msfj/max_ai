"""The only way out of a DockerSandbox with ``networks``: a proxy that lets
through the listed domains (and their subdomains) and nothing else.

Standard library only; runs in its own container:
``python3 -c <this file> pypi.org files.pythonhosted.org``.
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlsplit

ALLOWED: list[str] = [domain.lower().strip(".") for domain in sys.argv[1:]]
PORT: int = 3128


def allowed(host: str) -> bool:
    host = host.lower().strip(".")
    return any(host == domain or host.endswith("." + domain) for domain in ALLOWED)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        method, target, version = (await reader.readline()).decode("latin-1").split()
        headers = b""
        while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
            headers += line
        if method == "CONNECT":  # https: a tunnel to host:port
            host, _, number = target.rpartition(":")
            port = int(number)
            first = b""
        else:  # plain http: the request is sent on, with its path only
            url = urlsplit(target)
            host, port = url.hostname or "", url.port or 80
            path = (url.path or "/") + (f"?{url.query}" if url.query else "")
            first = f"{method} {path} {version}\r\n".encode("latin-1") + headers + b"\r\n"
        if not allowed(host):
            writer.write(f"HTTP/1.1 403 Forbidden\r\n\r\n{host} is not in the allowed networks\n".encode())
            return
        there, back = await asyncio.open_connection(host, port)
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        back.write(first)
        await asyncio.gather(pipe(reader, back), pipe(there, writer))
    except (OSError, ValueError):
        pass
    finally:
        writer.close()


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    async with server:
        await server.serve_forever()


asyncio.run(main())
