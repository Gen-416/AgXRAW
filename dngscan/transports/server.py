# SPDX-License-Identifier: GPL-3.0-or-later
"""Uvicorn launcher for the dngscan localhost browser GUI."""
from __future__ import annotations

import socket
import json
import sys
from pathlib import Path
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


def run(*, frontend_dir: Path | None = None, open_browser: bool = True,
        handshake: bool = False) -> int:
    if dg.IMPORT_ERRORS:
        print(
            "警告：dngscan 依赖未就绪，导出会失败。"
            "请先安装 rawpy/numpy/matplotlib/pillow：", file=sys.stderr,
        )
        print("  " + "\n  ".join(dg.IMPORT_ERRORS), file=sys.stderr)

    import uvicorn

    from .http import create_app
    from dngscan.application import service

    with _bind_localhost_socket() as sock:
        port = int(sock.getsockname()[1])
        url = f"http://127.0.0.1:{port}/"
        session_token = uuid4().hex + uuid4().hex
        app = create_app(
            service_module=service,
            session_token=session_token,
            expected_port=port,
            frontend_dir=frontend_dir,
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

        if frontend_dir is not None:
            url += 'workspace/'
        # stdout belongs solely to the parent handshake in desktop mode.
        # Pipeline diagnostics continue on stderr, never corrupting readiness.
        if handshake:
            import os
            import signal
            channel = sys.stdout
            sys.stdout = sys.stderr
            def watch_parent():
                # A piped stdin is the ownership lease. Closing the desktop
                # or dev launcher requests normal Uvicorn/Python cleanup.
                sys.stdin.buffer.read()
                os.kill(os.getpid(), signal.SIGTERM)
            threading.Thread(target=watch_parent, daemon=True, name='agxraw-parent').start()
            def announce_ready():
                while not stop_browser_thread.wait(.05):
                    if server.started:
                        channel.write(json.dumps({'version': 1, 'url': url,
                                                  'session_token': session_token}) + '\n')
                        channel.flush()
                        return
                    if server.should_exit:
                        return
            opener = threading.Thread(target=announce_ready, daemon=True, name='agxraw-ready')
        elif open_browser:
            opener = threading.Thread(target=_open_browser_when_ready,
                                      args=(server, url, stop_browser_thread), daemon=True)
        print(f"dngscan GUI: {url}  (Ctrl+C 退出)", file=sys.stderr)
        if handshake or open_browser:
            opener.start()
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            print("\n已退出")
        finally:
            stop_browser_thread.set()
            if handshake or open_browser:
                opener.join(timeout=1.0)
    return 0


def main() -> int:
    return run()
