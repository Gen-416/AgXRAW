# SPDX-License-Identifier: GPL-3.0-or-later
"""Localhost HTTP server for the dngscan web GUI."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import dngscan as dg
from dngscan.debug_util import maybe_print_exc

from .constants import RAW_EXTS
from .page import render_page
from .service import list_dir, prepare_preview, raw9_support, run_export_isolated, run_preview


_UPLOADS = tempfile.TemporaryDirectory(prefix="dngscan-gui-uploads-")
_UPLOAD_ROOT = Path(_UPLOADS.name)


def store_upload(
    filename: str,
    source: BinaryIO,
    content_length: int,
    upload_root: Path = _UPLOAD_ROOT,
) -> Path:
    """Store one browser-selected RAW in a process-scoped temporary directory."""
    clean_name = Path(filename.replace("\\", "/")).name
    suffix = Path(clean_name).suffix.lower()
    if suffix not in RAW_EXTS:
        allowed = "、".join(sorted(RAW_EXTS))
        raise ValueError(f"不支持的 RAW 文件类型；请选择：{allowed}")
    if content_length <= 0:
        raise ValueError("选择的 RAW 文件为空")

    stem = Path(clean_name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    safe_stem = safe_stem.strip("._")[:64] or "photo"
    slot = upload_root / uuid4().hex
    slot.mkdir(parents=True)
    target = slot / f"{safe_stem}{suffix}"
    partial = slot / f".{safe_stem}.uploading"

    remaining = content_length
    try:
        with partial.open("wb") as handle:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("RAW 文件传输未完成")
                handle.write(chunk)
                remaining -= len(chunk)
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        try:
            slot.rmdir()
        except OSError:
            pass
        raise
    return target


def reveal_path(params: dict) -> dict:
    path = Path(str(params.get("path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    result = subprocess.run(["open", "-R", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "open -R failed")
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = render_page(str(_default_dir()))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/list":
            q = parse_qs(parsed.query)
            self._json(list_dir(q.get("dir", [""])[0]))
        elif parsed.path == "/hdr-status":
            available, reason = dg.apple_gainmap_backend_status()
            self._json({"ok": True, "available": available, "reason": reason})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/upload":
            try:
                length = int(self.headers.get("Content-Length", 0))
                name = parse_qs(parsed.query).get("name", [""])[0]
                saved = store_upload(name, self.rfile, length)
                self._json({"ok": True, "path": str(saved), "name": saved.name})
            except Exception as exc:
                maybe_print_exc()
                self._json({"ok": False, "error": str(exc)}, code=200)
            return
        if path not in ("/export", "/preview", "/prepare", "/raw9-support", "/reveal"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
            if path == "/preview":
                result = run_preview(params)
            elif path == "/export":
                result = run_export_isolated(params)
            elif path == "/prepare":
                result = prepare_preview(params)
            elif path == "/raw9-support":
                result = raw9_support(params)
            else:
                result = reveal_path(params)
            self._json(result)
        except Exception as exc:  # surface any pipeline error to the UI
            maybe_print_exc()
            self._json({"ok": False, "error": str(exc)}, code=200)

    def log_message(self, fmt: str, *args: object) -> None:  # keep the console quiet
        return


def _default_dir() -> Path:
    pics = Path.home() / "Pictures"
    return pics if pics.is_dir() else Path.home()


def main() -> int:
    if dg.IMPORT_ERRORS:
        print("警告：dngscan 依赖未就绪，导出会失败。请先安装 rawpy/numpy/matplotlib/pillow：")
        print("  " + "\n  ".join(dg.IMPORT_ERRORS))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dngscan GUI: {url}  (Ctrl+C 退出)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0
