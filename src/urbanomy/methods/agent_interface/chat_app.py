"""Local browser chat UI for interacting with Urbanomy agents."""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HTML_PAGE = """<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Urbanomy Agent Chat</title>
    <style>
      :root {
        --bg: #f2efe8;
        --panel: rgba(255, 255, 255, 0.82);
        --ink: #1f2a2e;
        --muted: #6a7478;
        --accent: #0b6e4f;
        --accent-2: #c96f3b;
        --line: rgba(31, 42, 46, 0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(201,111,59,0.16), transparent 28%),
          radial-gradient(circle at bottom right, rgba(11,110,79,0.18), transparent 30%),
          linear-gradient(160deg, #f6f3ed 0%, #ece6dc 100%);
      }
      .shell {
        max-width: 980px;
        margin: 0 auto;
        padding: 24px 18px 36px;
      }
      .hero {
        display: grid;
        gap: 10px;
        margin-bottom: 18px;
      }
      .eyebrow {
        font-size: 12px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
      }
      h1 {
        margin: 0;
        font-size: clamp(28px, 4vw, 44px);
        line-height: 0.95;
      }
      .subhead {
        margin: 0;
        max-width: 720px;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.5;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 18px 48px rgba(31, 42, 46, 0.08);
      }
      .toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 14px;
        align-items: center;
        justify-content: space-between;
        padding: 16px 18px;
        margin-bottom: 14px;
      }
      .toolbar-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
      }
      .badge {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(11, 110, 79, 0.08);
        font-size: 13px;
      }
      .dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
      }
      button {
        appearance: none;
        border: 0;
        border-radius: 999px;
        padding: 10px 16px;
        font: inherit;
        cursor: pointer;
        transition: transform 0.12s ease, opacity 0.12s ease, background 0.12s ease;
      }
      button:hover { transform: translateY(-1px); }
      button:disabled { opacity: 0.55; cursor: default; transform: none; }
      .ghost {
        background: rgba(31, 42, 46, 0.07);
        color: var(--ink);
      }
      .primary {
        background: var(--accent);
        color: #fff;
      }
      .chat {
        min-height: 56vh;
        max-height: 68vh;
        overflow: auto;
        padding: 18px;
        display: grid;
        gap: 14px;
      }
      .message {
        display: grid;
        gap: 8px;
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.72);
      }
      .message.user {
        background: rgba(11, 110, 79, 0.09);
      }
      .message.assistant {
        background: rgba(201, 111, 59, 0.08);
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px 14px;
        font-size: 12px;
        color: var(--muted);
      }
      .content {
        white-space: pre-wrap;
        line-height: 1.55;
        font-size: 15px;
      }
      .payload {
        white-space: pre-wrap;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 12px;
        color: var(--muted);
        background: rgba(31, 42, 46, 0.04);
        border-radius: 14px;
        padding: 12px;
      }
      .image-wrap {
        display: grid;
        gap: 8px;
      }
      .image-label {
        font-size: 12px;
        color: var(--muted);
      }
      .chat-image {
        width: 100%;
        max-width: 100%;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.92);
      }
      .composer {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        padding: 16px 18px 18px;
      }
      textarea {
        width: 100%;
        min-height: 90px;
        resize: vertical;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        font: inherit;
        background: rgba(255,255,255,0.88);
        color: var(--ink);
      }
      .composer-actions {
        display: grid;
        gap: 10px;
        align-content: end;
      }
      .hint {
        margin-top: 12px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
      }
      @media (max-width: 720px) {
        .composer { grid-template-columns: 1fr; }
        .composer-actions { grid-auto-flow: column; justify-content: flex-end; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="hero">
        <div class="eyebrow">Local Runtime</div>
        <h1 id="title">Urbanomy Agent Chat</h1>
        <p id="welcome" class="subhead">Подключение к локальному orchestrator...</p>
      </header>

      <section class="panel toolbar">
        <div class="toolbar-meta">
          <div class="badge"><span class="dot"></span><span id="thread-label">thread: ...</span></div>
          <div class="badge"><span>route</span><strong id="last-route">-</strong></div>
        </div>
        <button id="reset" class="ghost" type="button">Сбросить диалог</button>
      </section>

      <section class="panel">
        <div id="chat" class="chat"></div>
        <form id="composer" class="composer">
          <textarea id="message" placeholder="Напиши запрос к Urbanomy orchestrator..."></textarea>
          <div class="composer-actions">
            <button id="send" class="primary" type="submit">Отправить</button>
          </div>
        </form>
      </section>
      <p class="hint">Для визуализаций chat показывает ответ и inline-preview картинки. Matplotlib-окна при этом тоже могут открываться локально в процессе сервера.</p>
    </div>

    <script>
      const chatEl = document.getElementById("chat");
      const formEl = document.getElementById("composer");
      const msgEl = document.getElementById("message");
      const sendEl = document.getElementById("send");
      const resetEl = document.getElementById("reset");
      const titleEl = document.getElementById("title");
      const welcomeEl = document.getElementById("welcome");
      const routeEl = document.getElementById("last-route");
      const threadEl = document.getElementById("thread-label");

      const threadIdKey = "urbanomy-chat-thread-id";
      let threadId = localStorage.getItem(threadIdKey);
      if (!threadId) {
        threadId = "thread-" + Math.random().toString(36).slice(2, 10);
        localStorage.setItem(threadIdKey, threadId);
      }
      threadEl.textContent = "thread: " + threadId;

      function setBusy(isBusy) {
        sendEl.disabled = isBusy;
        resetEl.disabled = isBusy;
        msgEl.disabled = isBusy;
      }

      function appendMessage(kind, text, meta = {}, payload = null, images = []) {
        const box = document.createElement("article");
        box.className = "message " + kind;

        const metaParts = [];
        if (kind === "user") metaParts.push("user");
        if (metaParts.length > 0) {
          const metaEl = document.createElement("div");
          metaEl.className = "meta";
          metaEl.textContent = metaParts.join(" • ");
          box.appendChild(metaEl);
        }

        const contentEl = document.createElement("div");
        contentEl.className = "content";
        contentEl.textContent = text;
        box.appendChild(contentEl);

        if (Array.isArray(images)) {
          for (const item of images) {
            if (!item || !item.url) continue;
            const wrapEl = document.createElement("div");
            wrapEl.className = "image-wrap";
            if (item.label) {
              const labelEl = document.createElement("div");
              labelEl.className = "image-label";
              labelEl.textContent = item.label;
              wrapEl.appendChild(labelEl);
            }
            const imgEl = document.createElement("img");
            imgEl.className = "chat-image";
            imgEl.src = item.url;
            imgEl.alt = item.label || "visualization";
            wrapEl.appendChild(imgEl);
            box.appendChild(wrapEl);
          }
        }

        chatEl.appendChild(box);
        chatEl.scrollTop = chatEl.scrollHeight;
      }

      async function loadConfig() {
        const response = await fetch("/api/config");
        const data = await response.json();
        titleEl.textContent = data.title || "Urbanomy Agent Chat";
        welcomeEl.textContent = data.welcome || "";
      }

      async function sendMessage(message) {
        setBusy(true);
        appendMessage("user", message);
        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, thread_id: threadId }),
          });
          const data = await response.json();
          if (!response.ok || !data.ok) {
            appendMessage("assistant", data.error || "Ошибка сервера.");
            return;
          }
          routeEl.textContent = data.result.route || "-";
          appendMessage(
            "assistant",
            data.result.response || "(empty response)",
            {
              route: data.result.route,
              reasoning: data.result.reasoning,
              tool: data.result.tool_name || null,
            },
            data.result.details || null,
            data.result.images || [],
          );
        } catch (error) {
          appendMessage("assistant", "Ошибка сети: " + error.message);
        } finally {
          setBusy(false);
          msgEl.focus();
        }
      }

      formEl.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = msgEl.value.trim();
        if (!text) return;
        msgEl.value = "";
        await sendMessage(text);
      });

      msgEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          formEl.requestSubmit();
        }
      });

      resetEl.addEventListener("click", async () => {
        setBusy(true);
        try {
          await fetch("/api/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ thread_id: threadId }),
          });
          chatEl.innerHTML = "";
          routeEl.textContent = "-";
          appendMessage("assistant", "Память текущего thread очищена.");
        } finally {
          setBusy(false);
        }
      });

      loadConfig().catch((error) => {
        welcomeEl.textContent = "Не удалось загрузить конфиг: " + error.message;
      });
      appendMessage("assistant", "Чат готов. Можешь писать запросы к локальному orchestrator.");
    </script>
  </body>
</html>
"""


@dataclass
class ChatAppState:
    """Runtime objects exposed by the local chat server."""

    orchestrator: Any
    title: str
    welcome: str


def _load_runtime_module(specifier: str):
    if not specifier:
        raise ValueError(
            "Runtime module is required. Pass --runtime or set URBANOMY_CHAT_RUNTIME."
        )
    candidate = Path(specifier)
    if candidate.exists():
        module_name = f"urbanomy_chat_runtime_{candidate.stem}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load runtime from file: {specifier}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(specifier)


def _build_state(runtime_specifier: str) -> ChatAppState:
    module = _load_runtime_module(runtime_specifier)
    runtime_factory = getattr(module, "create_chat_runtime", None)
    if callable(runtime_factory):
        runtime = runtime_factory()
    else:
        orchestrator_factory = getattr(module, "create_orchestrator", None)
        if not callable(orchestrator_factory):
            raise AttributeError(
                "Runtime module must expose create_chat_runtime() or create_orchestrator()."
            )
        runtime = orchestrator_factory()

    if isinstance(runtime, dict):
        orchestrator = runtime.get("orchestrator")
        title = str(runtime.get("title", "Urbanomy Agent Chat")).strip() or "Urbanomy Agent Chat"
        welcome = (
            str(runtime.get("welcome", "Локальный чат подключён к Urbanomy orchestrator.")).strip()
            or "Локальный чат подключён к Urbanomy orchestrator."
        )
    else:
        orchestrator = runtime
        title = "Urbanomy Agent Chat"
        welcome = "Локальный чат подключён к Urbanomy orchestrator."

    if orchestrator is None or not hasattr(orchestrator, "invoke"):
        raise TypeError("Runtime did not provide a valid orchestrator with invoke(...).")
    return ChatAppState(orchestrator=orchestrator, title=title, welcome=welcome)


def _serialize_visualization_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "route": getattr(result, "route", None),
        "metric_kind": getattr(result, "metric_kind", None),
        "title": getattr(result, "title", None),
        "target_id": getattr(result, "target_id", None),
        "reasoning": getattr(result, "reasoning", None),
        "tool_payload": getattr(result, "tool_payload", None),
    }


def _figure_to_data_url(figure: Any) -> str | None:
    if figure is None or not hasattr(figure, "savefig"):
        return None
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight")
    except Exception:
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _serialize_images(result: Any) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []

    visualization_result = getattr(result, "visualization_result", None)
    artifact = getattr(visualization_result, "artifact", None)
    figure_url = _figure_to_data_url(getattr(artifact, "figure", None))
    if figure_url is not None:
        label = str(getattr(artifact, "title", "") or getattr(visualization_result, "title", "")).strip()
        images.append(
            {
                "label": label or "Визуализация",
                "url": figure_url,
            }
        )

    district = getattr(result, "district_optimization_result", None)
    tool_output = district.get("tool_output") if isinstance(district, dict) else None
    if isinstance(tool_output, dict):
        district_figure_url = _figure_to_data_url(tool_output.get("figure"))
        if district_figure_url is not None:
            images.append(
                {
                    "label": str(tool_output.get("summary_text", "")).strip() or "Результат оптимизации",
                    "url": district_figure_url,
                }
            )

    return images


def _sanitize_json_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json_payload(item)
            for key, item in value.items()
            if key != "figure"
        }
    if isinstance(value, list):
        return [_sanitize_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_details_payload(
    *,
    visualization: dict[str, Any] | None,
    block_parameters: dict[str, Any] | None,
    district: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if visualization is None and block_parameters is None and district is None:
        return None
    payload: dict[str, Any] = {}
    if visualization is not None:
        payload["visualization"] = visualization
    if block_parameters is not None:
        payload["block_parameters"] = block_parameters
    if district is not None:
        payload["district_optimization"] = district
    return payload or None


def _compact_block_parameters_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "target_id": result.get("target_id"),
        "parameters": _sanitize_json_payload(result.get("parameters")),
    }


def _compact_district_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": result.get("status"),
        "tool_name": result.get("tool_name"),
        "tool_output": _sanitize_json_payload(result.get("tool_output")),
    }
    if result.get("used_tool_fallback"):
        payload["used_tool_fallback"] = True
    return payload


def _serialize_orchestrator_result(result: Any) -> dict[str, Any]:
    visualization = _serialize_visualization_result(getattr(result, "visualization_result", None))
    block_parameters = getattr(result, "block_parameters_result", None)
    district = getattr(result, "district_optimization_result", None)
    tool_name = district.get("tool_name") if isinstance(district, dict) else None
    details = _compact_details_payload(
        visualization=visualization,
        block_parameters=(
            _compact_block_parameters_payload(block_parameters)
            if isinstance(block_parameters, dict)
            else None
        ),
        district=_compact_district_payload(district) if isinstance(district, dict) else None,
    )
    return {
        "response": str(getattr(result, "response", "")).strip(),
        "route": getattr(result, "route", None),
        "reasoning": getattr(result, "reasoning", None),
        "tool_name": tool_name,
        "details": details,
        "images": _serialize_images(result),
    }


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, text: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _make_handler(state: ChatAppState):
    class ChatHandler(BaseHTTPRequestHandler):
        """Request handler bound to one chat runtime."""

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                _text_response(self, HTTPStatus.OK, HTML_PAGE)
                return
            if path == "/api/config":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "title": state.title,
                        "welcome": state.welcome,
                    },
                )
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            if path == "/api/chat":
                message = str(payload.get("message", "")).strip()
                thread_id = str(payload.get("thread_id", "chat")).strip() or "chat"
                if not message:
                    _json_response(
                        self,
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": "message cannot be empty"},
                    )
                    return
                try:
                    result = state.orchestrator.invoke(message, thread_id=thread_id)
                except Exception as exc:
                    _json_response(
                        self,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": f"orchestrator error: {exc}"},
                    )
                    return
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "thread_id": thread_id,
                        "result": _serialize_orchestrator_result(result),
                    },
                )
                return

            if path == "/api/reset":
                thread_id = str(payload.get("thread_id", "chat")).strip() or "chat"
                try:
                    if hasattr(state.orchestrator, "clear_thread_memory"):
                        state.orchestrator.clear_thread_memory(thread_id)
                except Exception as exc:
                    _json_response(
                        self,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": f"reset error: {exc}"},
                    )
                    return
                _json_response(self, HTTPStatus.OK, {"ok": True, "thread_id": thread_id})
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            return data

    return ChatHandler


def serve_chat_app(
    *,
    runtime: str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the local browser chat server for one runtime module."""
    state = _build_state(runtime)
    server = ThreadingHTTPServer((host, int(port)), _make_handler(state))
    print(f"Urbanomy chat is running on http://{host}:{port}")
    print(f"Runtime: {runtime}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the local Urbanomy chat server."""
    parser = argparse.ArgumentParser(description="Run a local browser chat for Urbanomy agents.")
    parser.add_argument(
        "--runtime",
        default=os.environ.get(
            "URBANOMY_CHAT_RUNTIME",
            "urbanomy.methods.agent_interface.chat_runtime",
        ),
        help="Python module path or file path with create_chat_runtime()/create_orchestrator().",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    args = parser.parse_args(argv)
    serve_chat_app(runtime=args.runtime, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
