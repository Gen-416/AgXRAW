# SPDX-License-Identifier: GPL-3.0-or-later
"""Uvicorn launcher for the dngscan localhost browser GUI."""
from __future__ import annotations

import socket
import threading
import webbrowser
from uuid import uuid4

import dngscan as dg


def _bind_localhost_socket() -> socket.socket:
    """Bind once so another process cannot claim the selected random port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    return sock


def _open_browser_when_ready(
    server: object,
    url: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(0.05):
        if bool(getattr(server, "started", False)):
            webbrowser.open(url)
            return
        if bool(getattr(server, "should_exit", False)):
            return


def main() -> int:
    if dg.IMPORT_ERRORS:
        print(
            "警告：dngscan 依赖未就绪，导出会失败。"
            "请先安装 rawpy/numpy/matplotlib/pillow："
        )
        print("  " + "\n  ".join(dg.IMPORT_ERRORS))

    import uvicorn

    from .fastapi_app import create_app
    from . import service

    with _bind_localhost_socket() as sock:
        port = int(sock.getsockname()[1])
        url = f"http://127.0.0.1:{port}/"
        session_token = uuid4().hex + uuid4().hex
        app = create_app(
            service_module=service,
            session_token=session_token,
            expected_port=port,
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            workers=1,
            reload=False,
            loop="asyncio",
            http="h11",
            ws="none",
            proxy_headers=False,
            lifespan="on",
            timeout_graceful_shutdown=2,
            access_log=False,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        stop_browser_thread = threading.Event()
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(server, url, stop_browser_thread),
            name="dngscan-browser-opener",
            daemon=True,
        )

        print(f"dngscan GUI: {url}  (Ctrl+C 退出)")
        opener.start()
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            print("\n已退出")
        finally:
            stop_browser_thread.set()
            opener.join(timeout=1.0)
    return 0
