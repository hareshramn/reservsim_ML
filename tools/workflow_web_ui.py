#!/usr/bin/env python3
"""Browser-based local UI for workflow command execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflow"


MODE_SPECS = {
    "doctor": [],
    "run": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "10"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "gpu_init_retries", "label": "GPU Init Retries", "kind": "text", "default": "0"},
        {"key": "tag", "label": "Tag", "kind": "text"},
        {"key": "case_file", "label": "Case File", "kind": "text"},
        {"key": "out", "label": "Out", "kind": "text", "default": "auto"},
    ],
    "ml-data-gen": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "plan", "label": "Plan CSV", "kind": "text"},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "200"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "gpu_init_retries", "label": "GPU Init Retries", "kind": "text", "default": "0"},
        {"key": "keep_temp", "label": "Keep Temp", "kind": "bool"},
    ],
    "ml-check": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "plan", "label": "Plan CSV", "kind": "text"},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "200"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "gpu_init_retries", "label": "GPU Init Retries", "kind": "text", "default": "2"},
        {"key": "bench_repeats", "label": "Bench Repeats", "kind": "text", "default": "3"},
        {"key": "bench_steps", "label": "Bench Steps", "kind": "text", "default": "50"},
        {"key": "bench_output_every", "label": "Bench Output Every", "kind": "text", "default": "10"},
        {"key": "skip_parity", "label": "Skip Parity", "kind": "bool"},
        {"key": "skip_bench", "label": "Skip Bench", "kind": "bool"},
    ],
    "validate": [
        {"key": "run", "label": "Run Path/ID", "kind": "text", "required": True},
    ],
    "parity": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "cpu_run", "label": "CPU Run", "kind": "text"},
        {"key": "gpu_run", "label": "GPU Run", "kind": "text"},
        {"key": "out", "label": "Out JSON Path", "kind": "text"},
    ],
    "bench": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "repeats", "label": "Repeats", "kind": "text", "default": "3"},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "50"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "10"},
        {"key": "gpu_init_retries", "label": "GPU Init Retries", "kind": "text", "default": "2"},
        {"key": "out", "label": "Out CSV", "kind": "text"},
    ],
    "plot": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "run", "label": "Run Path/ID", "kind": "text"},
        {"key": "out", "label": "Out Dir", "kind": "text", "default": "figs"},
        {"key": "check_only", "label": "Check Only", "kind": "bool"},
    ],
    "clean": [
        {"key": "model", "label": "Model", "kind": "enum"},
        {"key": "all", "label": "All Models", "kind": "bool"},
        {"key": "bucket", "label": "Bucket(s)", "kind": "text"},
        {"key": "keep", "label": "Keep Newest N", "kind": "text", "default": "5"},
        {"key": "apply", "label": "Apply", "kind": "bool"},
    ],
}


ARG_FLAG = {
    "model": "--model",
    "mode": "--mode",
    "backend": "--backend",
    "steps": "--steps",
    "output_every": "--output-every",
    "gpu_init_retries": "--gpu-init-retries",
    "tag": "--tag",
    "case_file": "--case-file",
    "out": "--out",
    "plan": "--plan",
    "keep_temp": "--keep-temp",
    "bench_repeats": "--bench-repeats",
    "bench_steps": "--bench-steps",
    "bench_output_every": "--bench-output-every",
    "skip_parity": "--skip-parity",
    "skip_bench": "--skip-bench",
    "run": "--run",
    "cpu_run": "--cpu-run",
    "gpu_run": "--gpu-run",
    "repeats": "--repeats",
    "check_only": "--check-only",
    "all": "--all",
    "bucket": "--bucket",
    "keep": "--keep",
    "apply": "--apply",
}


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reserv ML Workflow UI</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d0d7de;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-2: #134e4a;
      --danger: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #e6f4f1 0%, #f4f6f8 45%, #eef2ff 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1100px;
      margin: 20px auto;
      padding: 0 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    h1 {
      margin: 0 0 12px 0;
      font-size: 22px;
    }
    .muted { color: var(--muted); font-size: 13px; }
    .grid {
      display: grid;
      gap: 10px;
      grid-template-columns: 190px 1fr;
      margin-top: 12px;
    }
    label { font-weight: 600; align-self: center; }
    input, select {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 14px;
      background: #fff;
    }
    .bool-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 9px 12px;
      cursor: pointer;
      font-weight: 600;
    }
    .run { background: var(--accent); color: #fff; }
    .run:hover { background: var(--accent-2); }
    .copy { background: #e5e7eb; color: #111827; }
    .stop { background: #fee2e2; color: var(--danger); }
    .preview {
      margin-top: 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      overflow-x: auto;
      white-space: pre;
    }
    pre#log {
      margin: 14px 0 0 0;
      background: #0b1020;
      color: #cbe4ff;
      border-radius: 10px;
      padding: 12px;
      min-height: 340px;
      max-height: 520px;
      overflow: auto;
      font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.35;
    }
    @media (max-width: 780px) {
      .grid { grid-template-columns: 1fr; }
      label { margin-top: 6px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Reserv ML Workflow UI</h1>
      <div class="muted">Choose mode, fill only relevant arguments, run from browser.</div>
      <div class="grid" id="form-grid"></div>
      <div class="actions">
        <button class="run" id="run-btn">Run</button>
        <button class="copy" id="copy-btn">Copy Command</button>
        <button class="copy" id="clear-btn">Clear Log</button>
      </div>
      <div class="preview" id="preview"></div>
      <pre id="log"></pre>
    </div>
  </div>
  <script>
    const MODE_SPECS = __MODE_SPECS__;
    const ARG_FLAG = __ARG_FLAG__;
    const MODE_KEYS = Object.keys(MODE_SPECS).sort();

    let models = [];
    const state = {};

    function q(id) { return document.getElementById(id); }
    function esc(s) { return String(s).replaceAll('"', '\\"'); }
    function shquote(s) {
      const t = String(s);
      if (t.includes(" ") || t.includes("\\t")) return `"${esc(t)}"`;
      return t;
    }

    async function loadModels() {
      const r = await fetch("/api/models");
      const j = await r.json();
      models = j.models || [];
    }

    function defaultFor(spec) {
      if (spec.kind === "bool") return false;
      if (spec.kind === "enum") {
        if (spec.key === "model") return models[0] || "";
        if (spec.default) return spec.default;
        return (spec.choices && spec.choices.length) ? spec.choices[0] : "";
      }
      return spec.default || "";
    }

    function buildForm() {
      const grid = q("form-grid");
      grid.innerHTML = "";
      const modeSpec = { key: "mode", label: "Mode", kind: "enum", choices: MODE_KEYS };

      if (state.mode === undefined) state.mode = "run";
      const specs = [modeSpec, ...MODE_SPECS[state.mode]];
      for (const spec of specs) {
        if (state[spec.key] === undefined) state[spec.key] = defaultFor(spec);

        const lab = document.createElement("label");
        lab.textContent = spec.label;
        grid.appendChild(lab);

        if (spec.kind === "bool") {
          const box = document.createElement("div");
          box.className = "bool-row";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.checked = !!state[spec.key];
          input.onchange = () => { state[spec.key] = input.checked; updatePreview(); };
          box.appendChild(input);
          const t = document.createElement("span");
          t.textContent = "Enable";
          box.appendChild(t);
          grid.appendChild(box);
          continue;
        }

        const sel = document.createElement("select");
        const inp = document.createElement("input");

        if (spec.kind === "enum") {
          let choices = spec.choices || [];
          if (spec.key === "mode") choices = MODE_KEYS;
          if (spec.key === "model") choices = models;
          for (const c of choices) {
            const o = document.createElement("option");
            o.value = c;
            o.textContent = c;
            sel.appendChild(o);
          }
          sel.value = state[spec.key] || defaultFor(spec);
          sel.onchange = () => {
            state[spec.key] = sel.value;
            if (spec.key === "mode") {
              for (const k of Object.keys(state)) {
                if (k !== "mode") delete state[k];
              }
              buildForm();
            } else {
              updatePreview();
            }
          };
          grid.appendChild(sel);
        } else {
          inp.type = "text";
          inp.value = state[spec.key] || "";
          inp.oninput = () => { state[spec.key] = inp.value; updatePreview(); };
          grid.appendChild(inp);
        }
      }
      updatePreview();
    }

    function buildCommandPayload() {
      const mode = state.mode;
      const specs = MODE_SPECS[mode] || [];
      const args = {};
      for (const spec of specs) {
        const v = state[spec.key];
        if (spec.kind === "bool") {
          if (v) args[spec.key] = true;
          continue;
        }
        const s = String(v || "").trim();
        if (spec.required && !s) throw new Error(`Missing required field: ${spec.label}`);
        if (!s) continue;
        args[spec.key] = s;
      }
      if (mode === "clean") {
        const all = !!args.all;
        const model = (args.model || "").trim();
        if (!all && !model) throw new Error("Clean requires either Model or All Models.");
        if (all && model) throw new Error("Choose either Model or All Models for clean.");
      }
      return { mode, args };
    }

    function cmdText(payload) {
      const parts = ["./workflow", payload.mode];
      for (const [k, v] of Object.entries(payload.args)) {
        const f = ARG_FLAG[k];
        if (!f) continue;
        if (typeof v === "boolean") {
          if (v) parts.push(f);
        } else {
          parts.push(f, String(v));
        }
      }
      return parts.map(shquote).join(" ");
    }

    function updatePreview() {
      try {
        const payload = buildCommandPayload();
        q("preview").textContent = cmdText(payload);
      } catch (e) {
        q("preview").textContent = `Invalid: ${e.message}`;
      }
    }

    async function runCmd() {
      let payload;
      try {
        payload = buildCommandPayload();
      } catch (e) {
        alert(e.message);
        return;
      }
      const log = q("log");
      log.textContent += `$ ${cmdText(payload)}\\n`;
      const r = await fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      log.textContent += (j.stdout || "");
      if (!log.textContent.endsWith("\\n")) log.textContent += "\\n";
      log.textContent += `[exit] code=${j.returncode}\\n`;
      log.scrollTop = log.scrollHeight;
    }

    async function copyCmd() {
      try {
        const payload = buildCommandPayload();
        await navigator.clipboard.writeText(cmdText(payload));
      } catch (e) {
        alert(e.message);
      }
    }

    async function init() {
      await loadModels();
      buildForm();
      q("run-btn").onclick = runCmd;
      q("copy-btn").onclick = copyCmd;
      q("clear-btn").onclick = () => { q("log").textContent = ""; };
    }
    init();
  </script>
</body>
</html>
"""


def list_models() -> list[str]:
    cases = ROOT / "cases"
    if not cases.exists():
        return []
    return sorted([p.name for p in cases.iterdir() if p.is_dir() and (p / "model.yaml").exists()])


def build_cli(mode: str, args: dict[str, object]) -> list[str]:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    cmd = [str(WORKFLOW), mode]
    for spec in MODE_SPECS[mode]:
        key = spec["key"]
        flag = ARG_FLAG.get(key)
        if flag is None:
            continue
        value = args.get(key)
        if spec["kind"] == "bool":
            if value:
                cmd.append(flag)
            continue
        text = str(value or "").strip()
        if spec.get("required") and not text:
            raise ValueError(f"Missing required field: {spec['label']}")
        if not text:
            continue
        cmd.extend([flag, text])
    return cmd


class Handler(BaseHTTPRequestHandler):
    server_version = "WorkflowWebUI/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _json(self, data: dict, code: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            page = HTML.replace("__MODE_SPECS__", json.dumps(MODE_SPECS)).replace("__ARG_FLAG__", json.dumps(ARG_FLAG))
            self._html(page)
            return
        if self.path == "/api/models":
            self._json({"models": list_models()})
            return
        self._json({"error": "not found"}, code=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._json({"error": "not found"}, code=HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
            mode = str(req.get("mode", "")).strip()
            args = req.get("args", {}) or {}
            if not isinstance(args, dict):
                raise ValueError("args must be an object")
            cmd = build_cli(mode, args)
        except Exception as exc:
            self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._json({"command": cmd, "stdout": proc.stdout, "returncode": proc.returncode})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local web UI for workflow commands.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8765, type=int, help="Bind port (default: 8765)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not WORKFLOW.exists():
        print(f"workflow script not found: {WORKFLOW}")
        return 2
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Workflow web UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
