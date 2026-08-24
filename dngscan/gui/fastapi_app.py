# SPDX-License-Identifier: GPL-3.0-or-later
"""FastAPI transport adapter for the existing browser GUI."""
from __future__ import annotations

import asyncio
import hmac
import json
import subprocess
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import dngscan as dg
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from dngscan.debug_util import maybe_print_exc

from .page import render_page
from .uploads import UploadStore


# TODO(stage-2): Remove the localhost auth exception, dependency, and handler
# once Tauri commands fully replace the browser HTTP transport.
class _UnauthorizedRequest(Exception):
    pass


def _complete_future(
    future: asyncio.Future[Any],
    result: Any,
    error: BaseException | None,
) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


async def _run_service_call(function: Callable[..., Any], *args: Any) -> Any:
    """Run legacy synchronous work with the old server's exit semantics.

    ``ThreadingHTTPServer`` made request workers daemon threads. AnyIO's shared
    worker threads are non-daemon, which can keep this local app alive after a
    forced Uvicorn shutdown if a decoder wedges. A per-request daemon worker
    preserves the previous lifecycle and also avoids putting the latest
    preview behind an unrelated shared thread-pool queue.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def invoke() -> None:
        try:
            result = function(*args)
        except BaseException as exc:
            if isinstance(exc, Exception):
                error = exc
            else:
                error = RuntimeError(
                    f"service worker aborted with {type(exc).__name__}"
                )
                error.__cause__ = exc
            callback = partial(_complete_future, future, None, error)
        else:
            callback = partial(_complete_future, future, result, None)
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError:
            # The process is already closing; this daemon worker must not
            # resurrect or delay the transport event loop.
            pass

    threading.Thread(
        target=invoke,
        name="dngscan-http-worker",
        daemon=True,
    ).start()
    return await future


def _default_dir() -> Path:
    pictures = Path.home() / "Pictures"
    return pictures if pictures.is_dir() else Path.home()


def _origin_is_allowed(origin: str, expected_port: int | None) -> bool:
    try:
        parsed = urlparse(origin)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            return False
        return expected_port is None or parsed.port == expected_port
    except ValueError:
        return False


def _reveal_path(params: dict[str, Any]) -> dict[str, bool]:
    path = Path(str(params.get("path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    result = subprocess.run(
        ["open", "-R", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "open -R failed")
    return {"ok": True}


def create_app(
    *,
    service_module: Any,
    session_token: str,
    upload_root: Path | None = None,
    expected_port: int | None = None,
) -> FastAPI:
    """Build one localhost HTTP adapter around the existing service module."""
    upload_store = UploadStore(upload_root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            upload_store.close()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.service_module = service_module
    app.state.upload_store = upload_store

    @app.exception_handler(_UnauthorizedRequest)
    async def unauthorized_handler(
        _request: Request, _exc: _UnauthorizedRequest
    ) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "error": "unauthorized"}, status_code=403
        )

    async def require_session(request: Request) -> None:
        origin = request.headers.get("Origin") or request.headers.get("Referer")
        if origin and not _origin_is_allowed(origin, expected_port):
            raise _UnauthorizedRequest()

        token = request.headers.get("X-DngScan-Token", "")
        if not token:
            token = request.query_params.get("token", "")
        if not hmac.compare_digest(token, session_token):
            raise _UnauthorizedRequest()

    async def legacy_call(
        request: Request,
        function: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> JSONResponse:
        try:
            params = json.loads(await request.body() or b"{}")
            if not isinstance(params, dict):
                raise ValueError("JSON 请求必须是对象")
            result = await _run_service_call(function, params)
            return JSONResponse(result)
        except Exception as exc:
            maybe_print_exc()
            return JSONResponse({"ok": False, "error": str(exc)})

    @app.get("/")
    async def page() -> Response:
        body = await _run_service_call(
            render_page,
            str(_default_dir()),
            session_token,
        )
        return Response(content=body, media_type="text/html")

    @app.get("/hdr-status")
    async def hdr_status() -> JSONResponse:
        available, reason = await _run_service_call(
            dg.apple_gainmap_backend_status
        )
        return JSONResponse(
            {"ok": True, "available": available, "reason": reason}
        )

    protected = APIRouter(dependencies=[Depends(require_session)])

    @protected.get("/list")
    async def list_directory(request: Request) -> JSONResponse:
        try:
            result = await _run_service_call(
                service_module.list_dir, request.query_params.get("dir", "")
            )
            return JSONResponse(result)
        except Exception as exc:
            maybe_print_exc()
            return JSONResponse({"ok": False, "error": str(exc)})

    @protected.post("/upload")
    async def upload(request: Request) -> JSONResponse:
        try:
            content_length = int(request.headers.get("Content-Length", "0"))
            saved = await upload_store.store_stream(
                request.query_params.get("name", ""),
                request.stream(),
                content_length,
            )
            return JSONResponse(
                {"ok": True, "path": str(saved), "name": saved.name}
            )
        except Exception as exc:
            maybe_print_exc()
            return JSONResponse({"ok": False, "error": str(exc)})

    @protected.post("/raw9-support")
    async def raw9_support(request: Request) -> JSONResponse:
        return await legacy_call(request, service_module.raw9_support)

    @protected.post("/prepare")
    async def prepare(request: Request) -> JSONResponse:
        return await legacy_call(request, service_module.prepare_preview)

    @protected.post("/preview")
    async def preview(request: Request) -> JSONResponse:
        return await legacy_call(request, service_module.run_preview)

    @protected.post("/export")
    async def export(request: Request) -> JSONResponse:
        return await legacy_call(request, service_module.run_export_isolated)

    @protected.post("/reveal")
    async def reveal(request: Request) -> JSONResponse:
        return await legacy_call(request, _reveal_path)

    app.include_router(protected)
    return app
