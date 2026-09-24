"""Expose infrastructure UIs on the devcontainer loopback for VS Code tunnels."""

import select
import socket
import socketserver
import threading

PORTS = {
    8081: ("mongo-express", 8081),
    3000: ("langfuse-web", 3000),
    9001: ("minio", 9001),  # MinIO Console (the agents' files)
}


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        destination = self.server.destination
        try:
            upstream = socket.create_connection(destination, timeout=10)
        except OSError:
            return

        with upstream:
            sockets = (self.request, upstream)
            while True:
                try:
                    readable, _, _ = select.select(sockets, [], [], 60)
                    if not readable:
                        continue
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        target = upstream if source is self.request else self.request
                        target.sendall(data)
                except OSError:
                    return


def main() -> None:
    servers = []
    for local_port, destination in PORTS.items():
        try:
            server = ProxyServer(("127.0.0.1", local_port), ProxyHandler)
        except OSError:
            continue  # An existing proxy or another service already owns the port.
        server.destination = destination
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    if servers:
        threading.Event().wait()


if __name__ == "__main__":
    main()
