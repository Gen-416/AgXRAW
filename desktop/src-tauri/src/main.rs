#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// SPDX-License-Identifier: GPL-3.0-or-later
use serde::Deserialize;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Manager, State};
use tauri_plugin_opener::OpenerExt;

#[derive(Deserialize)]
struct Ready {
    version: u32,
    url: String,
    session_token: String,
}

struct Backend {
    child: Child,
    ready: Ready,
}
impl Drop for Backend {
    fn drop(&mut self) {
        // Closing the pipe requests normal Python shutdown. Force termination
        // only if it fails to exit within the bounded grace period.
        drop(self.child.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
#[derive(Clone, Default)]
struct EngineState(Arc<Mutex<Option<Backend>>>);

impl EngineState {
    fn connection(&self) -> Result<(String, String), String> {
        let mut guard = self.0.lock().map_err(|_| "处理服务状态不可用")?;
        if let Some(backend) = guard.as_mut() {
            if backend
                .child
                .try_wait()
                .map_err(|e| e.to_string())?
                .is_some()
            {
                *guard = None;
            }
        }
        if guard.is_none() {
            *guard = Some(start_backend()?);
        }
        let backend = guard.as_ref().unwrap();
        Ok((
            backend.ready.url.clone(),
            backend.ready.session_token.clone(),
        ))
    }
}

fn start_backend() -> Result<Backend, String> {
    // Development entry, not a self-contained release package. A packaged
    // Python engine will replace this resolver without changing the API.
    let root = std::env::var_os("AGXRAW_BACKEND_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
    let python = std::env::var_os("AGXRAW_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let local = root.join(".venv/bin/python");
            if local.is_file() {
                local
            } else {
                PathBuf::from("python3")
            }
        });
    let mut child = Command::new(python)
        .args(["-u", "-m", "dngscan.application", "--serve", "--handshake"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("无法启动处理服务，请配置 AGXRAW_PYTHON：{e}"))?;
    let output = child.stdout.take().ok_or("无法读取处理服务启动信息")?;
    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(output).read_line(&mut line).map(|_| line);
        let _ = tx.send(result);
    });
    let parsed = rx
        .recv_timeout(Duration::from_secs(30))
        .ok()
        .and_then(Result::ok)
        .and_then(|line| serde_json::from_str::<Ready>(&line).ok());
    if let Some(ready) = parsed {
        // The app-owned bootstrap can only direct requests to its loopback server.
        let valid_url = reqwest::Url::parse(&ready.url).is_ok_and(|url| {
            url.scheme() == "http"
                && url.host_str() == Some("127.0.0.1")
                && url.port().is_some()
                && url.path() == "/"
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        });
        if ready.version == 1 && valid_url && !ready.session_token.is_empty() {
            return Ok(Backend { child, ready });
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    Err("处理服务未能就绪；请检查 Python 环境和终端日志，然后重试".into())
}

async fn connection(state: &State<'_, EngineState>) -> Result<(String, String), String> {
    // Startup waits belong off the UI/event-loop thread.
    // The mutex is held only for startup/connection lookup, never for rendering.
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || state.connection())
        .await
        .map_err(|e| e.to_string())?
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(1200))
        .build()
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn engine_bootstrap(state: State<'_, EngineState>) -> Result<Value, String> {
    let (url, _) = connection(&state).await?;
    let mut result: Value = http_client()?
        .get(format!("{url}api/v1/bootstrap"))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    // Useful for developer automation and opening a source from a launcher.
    // The frontend still sends the same ordinary prepare/render commands.
    let mut args = std::env::args();
    if let Some(path) = args.find(|arg| arg == "--open").and_then(|_| args.next()) {
        if let Some(object) = result.as_object_mut() {
            object.insert("initial_source".into(), Value::String(path));
        }
    }
    Ok(result)
}

#[tauri::command]
async fn engine_request(state: State<'_, EngineState>, payload: Value) -> Result<Value, String> {
    let (url, token) = connection(&state).await?;
    http_client()?
        .post(format!("{url}api/v1/commands"))
        .header("X-DngScan-Token", token)
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn open_legacy_workspace(
    app: tauri::AppHandle,
    state: State<'_, EngineState>,
) -> Result<(), String> {
    let (url, _) = connection(&state).await?;
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

fn main() {
    let app = tauri::Builder::default()
        .manage(EngineState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            engine_bootstrap,
            engine_request,
            open_legacy_workspace
        ])
        .build(tauri::generate_context!())
        .expect("cannot initialize AgXRAW desktop");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            // Do not block window shutdown behind the startup handshake.
            // If startup currently owns the mutex, OS pipe closure still
            // releases the Python parent's ownership lease when this exits.
            if let Ok(mut backend) = app.state::<EngineState>().0.try_lock() {
                *backend = None;
            }
        }
    });
}
