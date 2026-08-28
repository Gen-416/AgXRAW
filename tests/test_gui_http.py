# SPDX-License-Identifier: GPL-3.0-or-later
"""Black-box contract for the temporary localhost HTTP adapter.

The FastAPI layer is deliberately disposable: Tauri will eventually replace
HTTP with native commands.  These tests therefore pin only the contract the
current PAGE consumes and keep the computation service as plain Python calls.
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

# R7: the GUI transport is an OPTIONAL extra (pyproject [gui]); a CLI-only
# environment must SKIP these tests, not error unittest discovery.
# Audit R11: only the THIRD-PARTY import may skip this module — a broken
# first-party fastapi_app used to be swallowed as "gui extra unavailable"
# and CI showed a skip instead of a failure.
try:
    from fastapi.testclient import TestClient
except Exception as _exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"gui extra unavailable: {_exc}")

from dngscan.gui.fastapi_app import _run_service_call, create_app  # noqa: E402


TOKEN = "test-session-token"
PORT = 48765
BASE_URL = f"http://127.0.0.1:{PORT}"


class RecordingService:
    """Small injected service that proves routes do not reshape commands."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {
            "raw9_support": {"ok": True, "kind": "raw9"},
            "prepare_preview": {"ok": True, "kind": "prepare"},
            "run_preview": {"ok": True, "kind": "preview"},
            "run_export_isolated": {"ok": True, "kind": "export"},
        }
        self.raise_from: str | None = None

    def list_dir(self, raw: str) -> dict[str, Any]:
        self.calls.append(("list_dir", raw))
        return {"cwd": raw, "parent": "/", "dirs": ["A"], "files": []}

    def _job(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if self.raise_from == method:
            raise ValueError("service exploded")
        return self.results[method]

    def raw9_support(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._job("raw9_support", params)

    def prepare_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._job("prepare_preview", params)

    def run_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._job("run_preview", params)

    def run_export_isolated(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._job("run_export_isolated", params)


class FastApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.upload_root = Path(self.temp.name)
        self.service = RecordingService()
        self.app = create_app(
            service_module=self.service,
            session_token=TOKEN,
            upload_root=self.upload_root,
            expected_port=PORT,
        )
        self.client_context = TestClient(self.app, base_url=BASE_URL)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temp.cleanup()

    @staticmethod
    def _token_headers(**extra: str) -> dict[str, str]:
        return {"X-DngScan-Token": TOKEN, **extra}

    def test_root_serves_the_existing_page_with_its_session_token(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn(TOKEN, response.text)
        self.assertIn("X-DngScan-Token", response.text)

    def test_hdr_status_remains_public_and_keeps_its_response_shape(self) -> None:
        with patch(
            "dngscan.gui.fastapi_app.dg.apple_gainmap_backend_status",
            return_value=(False, "test reason"),
        ):
            response = self.client.get("/hdr-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"ok": True, "available": False, "reason": "test reason"},
        )

    def test_protected_route_accepts_header_and_query_token(self) -> None:
        by_header = self.client.get(
            "/list?dir=%2Fheader", headers=self._token_headers()
        )
        by_query = self.client.get(f"/list?dir=%2Fquery&token={TOKEN}")

        self.assertEqual(by_header.status_code, 200)
        self.assertEqual(by_header.json()["cwd"], "/header")
        self.assertEqual(by_query.status_code, 200)
        self.assertEqual(by_query.json()["cwd"], "/query")
        self.assertEqual(
            self.service.calls,
            [("list_dir", "/header"), ("list_dir", "/query")],
        )

    def test_missing_or_wrong_token_is_the_legacy_403_envelope(self) -> None:
        for headers in ({}, {"X-DngScan-Token": "wrong"}):
            with self.subTest(headers=headers):
                response = self.client.get("/list?dir=%2Ftmp", headers=headers)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(
                    response.json(), {"ok": False, "error": "unauthorized"}
                )
        self.assertEqual(self.service.calls, [])

    def test_every_page_post_route_is_protected_before_body_dispatch(self) -> None:
        for route in (
            "/upload?name=photo.dng",
            "/raw9-support",
            "/prepare",
            "/preview",
            "/clip-overlay",
            "/export",
            "/reveal",
        ):
            with self.subTest(route=route):
                response = self.client.post(route, content=b"{}")
                self.assertEqual(response.status_code, 403)
                self.assertEqual(
                    response.json(), {"ok": False, "error": "unauthorized"}
                )
        self.assertEqual(self.service.calls, [])

    def test_host_must_be_the_expected_loopback_host(self) -> None:
        """R7 hardening: a DNS-rebinding page suppresses Origin/Referer
        (Referrer-Policy: no-referrer) and its rebound origin makes the
        tokenless "/" same-origin — but it cannot suppress or forge the
        Host header back to loopback. Every route, including "/", must
        refuse a non-loopback or wrong-port Host outright."""
        for path, headers in (
            ("/", {}),
            ("/hdr-status", {}),
            ("/list?dir=%2Fx", self._token_headers()),
        ):
            with self.subTest(path=path):
                rejected = self.client.get(
                    path, headers={**headers, "Host": "evil.example:48765"}
                )
                self.assertEqual(rejected.status_code, 403)
                wrong_port = self.client.get(
                    path, headers={**headers, "Host": "127.0.0.1:1"}
                )
                self.assertEqual(wrong_port.status_code, 403)
        ok = self.client.get(
            "/list?dir=%2Fgood2", headers=self._token_headers()
        )
        self.assertEqual(ok.status_code, 200)

    def test_origin_must_be_the_expected_loopback_origin(self) -> None:
        accepted = self.client.get(
            "/list?dir=%2Fgood",
            headers=self._token_headers(Origin=BASE_URL),
        )
        rejected = self.client.get(
            "/list?dir=%2Fevil",
            headers=self._token_headers(Origin="https://evil.example"),
        )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(
            rejected.json(), {"ok": False, "error": "unauthorized"}
        )
        self.assertEqual(self.service.calls, [("list_dir", "/good")])

    def test_job_routes_forward_exact_params_and_results(self) -> None:
        body = {
            "input": "/photos/frame.dng",
            "generation": 7,
            "nested": {"values": [1, "two", False]},
        }
        routes = (
            ("/raw9-support", "raw9_support"),
            ("/prepare", "prepare_preview"),
            ("/preview", "run_preview"),
            ("/export", "run_export_isolated"),
        )

        for route, method in routes:
            with self.subTest(route=route):
                response = self.client.post(
                    route, json=body, headers=self._token_headers()
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), self.service.results[method])

        self.assertEqual(
            self.service.calls,
            [(method, body) for _route, method in routes],
        )

    def test_service_work_uses_daemon_workers_for_app_shutdown(self) -> None:
        daemon_flags: list[bool] = []

        def record_worker(params: dict[str, Any]) -> dict[str, Any]:
            daemon_flags.append(threading.current_thread().daemon)
            return {"ok": True, "params": params}

        self.service.run_preview = record_worker  # type: ignore[method-assign]
        response = self.client.post(
            "/preview", json={"generation": 9}, headers=self._token_headers()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["params"], {"generation": 9})
        self.assertEqual(daemon_flags, [True])

    def test_worker_base_exception_becomes_a_request_error(self) -> None:
        def abort_worker(_params: dict[str, Any]) -> dict[str, Any]:
            raise KeyboardInterrupt()

        self.service.run_preview = abort_worker  # type: ignore[method-assign]
        response = self.client.post(
            "/preview", json={"generation": 10}, headers=self._token_headers()
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertIn("KeyboardInterrupt", response.json()["error"])
        self.assertEqual(
            self.client.get(
                "/list?dir=/still-alive", headers=self._token_headers()
            ).status_code,
            200,
        )

    def test_service_error_keeps_the_legacy_http_200_envelope(self) -> None:
        self.service.raise_from = "run_preview"

        response = self.client.post(
            "/preview", json={"generation": 1}, headers=self._token_headers()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"ok": False, "error": "service exploded"}
        )

    def test_empty_job_body_keeps_the_legacy_empty_object_behavior(self) -> None:
        response = self.client.post(
            "/prepare", content=b"", headers=self._token_headers()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.service.results["prepare_preview"])
        self.assertEqual(self.service.calls, [("prepare_preview", {})])

    def test_app_owned_upload_directory_is_removed_at_shutdown(self) -> None:
        owned_app = create_app(
            service_module=RecordingService(),
            session_token=TOKEN,
            expected_port=PORT,
        )
        owned_root = owned_app.state.upload_store.root

        with TestClient(owned_app, base_url=BASE_URL):
            self.assertTrue(owned_root.is_dir())

        self.assertFalse(owned_root.exists())

    def test_framework_documentation_routes_are_not_exposed(self) -> None:
        for route in ("/docs", "/redoc", "/openapi.json"):
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 404)

    def test_upload_streams_exact_bytes_and_sanitizes_the_name(self) -> None:
        payload = b"example raw bytes"

        response = self.client.post(
            "/upload?name=..%2F..%2Ffolder%2FMy%20Photo.DNG",
            content=payload,
            headers=self._token_headers(**{"Content-Type": "application/octet-stream"}),
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["ok"], result)
        saved = Path(result["path"])
        self.assertEqual(result["name"], "My_Photo.dng")
        self.assertEqual(saved.name, "My_Photo.dng")
        self.assertEqual(saved.read_bytes(), payload)
        self.assertEqual(saved.parent.parent, self.upload_root)

    def test_upload_rejects_non_raw_and_incomplete_bodies_in_json(self) -> None:
        invalid = self.client.post(
            "/upload?name=photo.jpg",
            content=b"jpeg",
            headers=self._token_headers(**{"Content-Type": "application/octet-stream"}),
        )
        incomplete = self.client.post(
            "/upload?name=photo.dng",
            content=b"short",
            headers=self._token_headers(
                **{
                    "Content-Type": "application/octet-stream",
                    "Content-Length": "10",
                }
            ),
        )

        self.assertEqual(invalid.status_code, 200)
        self.assertFalse(invalid.json()["ok"])
        self.assertIn("不支持的 RAW", invalid.json()["error"])
        self.assertEqual(incomplete.status_code, 200)
        self.assertFalse(incomplete.json()["ok"])
        self.assertIn("传输未完成", incomplete.json()["error"])
        self.assertEqual(list(self.upload_root.rglob("*.uploading")), [])

    def test_upload_rejects_the_declared_single_file_limit_before_writing(self) -> None:
        response = self.client.post(
            "/upload?name=huge.dng",
            content=b"x",
            headers=self._token_headers(
                **{
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(512 * 1024 * 1024 + 1),
                }
            ),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertIn("512 MB", response.json()["error"])
        self.assertEqual(list(self.upload_root.rglob("*")), [])

    def test_deleting_a_staged_copy_restores_upload_quota(self) -> None:
        with patch("dngscan.gui.uploads.UPLOAD_TOTAL_MAX_BYTES", 6):
            first = self.client.post(
                "/upload?name=first.dng",
                content=b"1234",
                headers=self._token_headers(),
            )
            Path(first.json()["path"]).unlink()
            second = self.client.post(
                "/upload?name=second.dng",
                content=b"5678",
                headers=self._token_headers(),
            )

        self.assertTrue(first.json()["ok"], first.json())
        self.assertTrue(second.json()["ok"], second.json())


class ServiceWorkerLifecycleTests(unittest.TestCase):
    def test_cancelling_a_request_does_not_wait_for_its_daemon_worker(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocked_call() -> None:
            started.set()
            release.wait(timeout=5.0)

        async def exercise() -> None:
            task = asyncio.create_task(_run_service_call(blocked_call))
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.5)

        try:
            asyncio.run(exercise())
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
