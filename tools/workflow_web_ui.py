#!/usr/bin/env python3
"""Browser-based local UI for workflow command execution."""

from __future__ import annotations

import argparse
import csv
import errno
import html
import json
import mimetypes
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflow"
PROGRESS_LOCK = threading.Lock()
RUN_PROGRESS: dict[str, object] = {
    "running": False,
    "step_current": 0,
    "step_total": 0,
    "last_line": "",
}
RUN_JOB_LOCK = threading.Lock()
RUN_JOBS: dict[str, dict[str, object]] = {}


MODE_SPECS = {
    "doctor": [],
    "history-run": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "10"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "tag", "label": "Tag", "kind": "text"},
        {"key": "out", "label": "Out", "kind": "text", "default": "auto"},
        {"key": "case_file", "label": "Case YAML (optional)", "kind": "text"},
    ],
    "ml-data-gen": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "plan", "label": "Plan CSV", "kind": "text"},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "200"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "keep_temp", "label": "Keep Temp", "kind": "bool"},
    ],
    "ml-train": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "data", "label": "Data Dir (optional)", "kind": "text"},
        {"key": "config", "label": "Config YAML (optional)", "kind": "text"},
        {"key": "seed", "label": "Seed", "kind": "text", "default": "42"},
        {"key": "out", "label": "Out Dir (optional)", "kind": "text"},
    ],
    "ml-eval": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "checkpoint", "label": "Checkpoint (.npz)", "kind": "text", "required": True},
        {"key": "data", "label": "Data Dir (optional)", "kind": "text"},
        {"key": "out", "label": "Out Dir", "kind": "text"},
    ],
    "ml-score": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "checkpoint", "label": "Checkpoint (.npz)", "kind": "text", "required": True},
        {"key": "candidates", "label": "Candidate CSV (optional)", "kind": "text"},
        {"key": "out", "label": "Out Dir", "kind": "text"},
    ],
    "ml-check": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "plan", "label": "Plan CSV", "kind": "text"},
        {"key": "mode", "label": "Mode", "kind": "enum", "default": "release", "choices": ["debug", "release"]},
        {"key": "backend", "label": "Backend", "kind": "enum", "default": "cpu", "choices": ["cpu", "gpu"]},
        {"key": "steps", "label": "Steps", "kind": "text", "default": "200"},
        {"key": "output_every", "label": "Output Every", "kind": "text", "default": "1"},
        {"key": "config", "label": "Config YAML (optional)", "kind": "text"},
        {"key": "seed", "label": "Seed", "kind": "text", "default": "42"},
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
        {"key": "out", "label": "Out CSV", "kind": "text"},
    ],
    "plot": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "run", "label": "Run Path/ID", "kind": "text"},
        {"key": "out", "label": "Out Dir", "kind": "text", "default": "figs"},
        {"key": "check_only", "label": "Check Only", "kind": "bool"},
    ],
    "clean": [
        {"key": "model", "label": "Model", "kind": "enum", "required": True},
        {"key": "bucket", "label": "Type of run", "kind": "enum", "default": "all", "choices": ["all", "history", "benchmark", "ml-data", "legacy"]},
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
    "data": "--data",
    "config": "--config",
    "seed": "--seed",
    "checkpoint": "--checkpoint",
    "candidates": "--candidates",
    "keep_temp": "--keep-temp",
    "run": "--run",
    "cpu_run": "--cpu-run",
    "gpu_run": "--gpu-run",
    "repeats": "--repeats",
    "check_only": "--check-only",
    "plot_after_run": None,
    "animate_after_run": None,
    "schedule_end_step": None,
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
    .status {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
      background: #f8fafc;
      white-space: pre-wrap;
    }
    .status.ok {
      border-color: #a7f3d0;
      background: #ecfdf5;
      color: #065f46;
    }
    .status.warn {
      border-color: #fecaca;
      background: #fef2f2;
      color: #991b1b;
    }
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
      gap: 0;
    }
    .input-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: center;
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
    .result-panel {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f8fafc;
      padding: 10px;
    }
    .result-panel h3 {
      margin: 0 0 8px 0;
      font-size: 15px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .panel-head h3 { margin: 0; }
    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-bottom: 8px;
    }
    .kv {
      border: 1px solid #dbe3ea;
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .kv .k {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .kv .v {
      font-weight: 700;
      font-size: 14px;
    }
    .mono {
      font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
    }
    .picker-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.45);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    .picker-backdrop[hidden] { display: none; }
    .picker {
      width: min(760px, 100%);
      max-height: 80vh;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .picker-path {
      font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      word-break: break-all;
    }
    .picker-list {
      border: 1px solid var(--line);
      border-radius: 8px;
      max-height: 42vh;
      overflow: auto;
    }
    .picker-item {
      width: 100%;
      text-align: left;
      background: transparent;
      border: 0;
      border-bottom: 1px solid #e5e7eb;
      border-radius: 0;
      padding: 8px 10px;
      font-weight: 500;
    }
    .picker-item:last-child { border-bottom: 0; }
    .picker-item:hover { background: #f1f5f9; }
    .picker-actions {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .editor-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.45);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    .editor-backdrop[hidden] { display: none; }
    .editor {
      width: min(980px, 100%);
      max-height: 86vh;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .editor textarea {
      width: 100%;
      min-height: 360px;
      font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      resize: vertical;
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
      <h1>Reserv ML Web UI</h1>
      <div class="muted">Primary user entrypoint for browser-based runs. Use the CLI only for advanced manual workflows and debugging.</div>
      <div class="status" id="doctor-status">Checking environment...</div>
      <div class="grid" id="form-grid"></div>
      <div class="actions">
        <button class="run" id="run-btn">Run</button>
        <button class="copy" id="copy-btn">Copy Command</button>
        <button class="copy" id="clear-btn">Clear Log</button>
      </div>
      <div class="status" id="run-status" hidden></div>
      <div class="preview" id="preview"></div>
      <div class="result-panel" id="summary-panel" hidden>
        <div class="panel-head">
          <h3>Run Summary</h3>
          <button class="copy" id="summary-visualize-btn" type="button">Visualize</button>
        </div>
        <div id="summary-content"></div>
      </div>
      <pre id="log"></pre>
    </div>
  </div>
  <div class="picker-backdrop" id="dir-picker" hidden>
    <div class="picker">
      <div class="picker-path" id="picker-path"></div>
      <div class="picker-list" id="picker-list"></div>
      <div class="picker-actions">
        <button class="copy" id="picker-up-btn">Up</button>
        <button class="copy" id="picker-cancel-btn">Cancel</button>
        <button class="run" id="picker-select-btn">Use This Folder</button>
      </div>
    </div>
  </div>
  <div class="editor-backdrop" id="csv-editor" hidden>
    <div class="editor">
      <div class="picker-path" id="csv-editor-path"></div>
      <textarea id="csv-editor-text"></textarea>
      <div class="picker-actions">
        <button class="copy" id="csv-editor-cancel-btn">Cancel</button>
        <button class="run" id="csv-editor-save-btn">Save CSV</button>
      </div>
    </div>
  </div>
  <script>
    const MODE_SPECS = __MODE_SPECS__;
    const ARG_FLAG = __ARG_FLAG__;
    const COMMAND_KEY = "__command";
    const ML_KIND_KEY = "__ml_kind";
    const COMMAND_CHOICES = ["history-run", "machine-learning", "validate", "clean", "parity"];
    const ML_KIND_TO_MODE = {
      "data-generation": "ml-data-gen",
      "training": "ml-train",
      "evaluate": "ml-eval",
      "score": "ml-score",
      "full-check": "ml-check",
    };
    const ML_KIND_CHOICES = Object.keys(ML_KIND_TO_MODE);
    const COMMAND_LABELS = {
      "history-run": "History Run",
      "machine-learning": "Machine Learning",
      "validate": "Validate",
      "clean": "Clean",
      "parity": "CPU-GPU values check",
    };
    const ML_KIND_LABELS = {
      "data-generation": "Candidate Data",
      "training": "Train Ranker",
      "evaluate": "Evaluate Ranker",
      "score": "Score Candidates",
      "full-check": "Full ML Check",
    };

    let models = [];
    const state = {};
    const pickerState = { targetKey: "", path: "", parent: null };
    const csvEditorState = { path: "" };
    let lastRunDir = "";
    let progressTimer = null;
    let ensuredMlPlanModel = "";

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

    async function loadDirs(path) {
      const p = path ? `?path=${encodeURIComponent(path)}` : "";
      const r = await fetch(`/api/dirs${p}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Failed to load directories.");
      return j;
    }

    async function ensureMlPlanDefault(model) {
      const r = await fetch(`/api/ml-plan-default?model=${encodeURIComponent(model)}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Failed to resolve default ML plan.");
      return j;
    }

    async function maybeEnsureMlPlan() {
      if (selectedMode() !== "ml-data-gen") return;
      const model = String(state.model || "").trim();
      if (!model) return;
      const desired = `cases/${model}/ml_scenarios.csv`;
      if (!String(state.plan || "").trim()) state.plan = desired;
      if (ensuredMlPlanModel === model) return;
      const info = await ensureMlPlanDefault(model);
      ensuredMlPlanModel = model;
      const path = String(info.path || desired);
      const changed = state.plan !== path;
      state.plan = path;
      if (changed) {
        buildForm();
        return;
      }
      if (info.created) {
        setRunStatus(`Created default Plan CSV: ${path}`, "ok");
      }
    }

    async function openPlanCsvEditor() {
      try {
        if (selectedMode() !== "ml-data-gen") return;
        const model = String(state.model || "").trim();
        if (!model) throw new Error("Select a model first.");
        const info = await ensureMlPlanDefault(model);
        const planPath = String(info.path || "").trim();
        if (!planPath) throw new Error("Default plan path is empty.");
        state.plan = planPath;
        const r = await fetch(`/api/ml-plan?path=${encodeURIComponent(planPath)}`);
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "Failed to load plan CSV.");
        csvEditorState.path = String(j.path || planPath);
        q("csv-editor-path").textContent = csvEditorState.path;
        q("csv-editor-text").value = String(j.content || "");
        q("csv-editor").hidden = false;
        updatePreview();
      } catch (e) {
        alert(e.message || "Failed to open plan CSV editor.");
      }
    }

    async function savePlanCsvEditor() {
      try {
        const path = String(csvEditorState.path || "").trim();
        if (!path) throw new Error("No CSV path selected.");
        const content = q("csv-editor-text").value;
        const r = await fetch("/api/ml-plan-save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, content }),
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "Failed to save plan CSV.");
        q("csv-editor").hidden = true;
        state.plan = String(j.path || path);
        setRunStatus(`Saved Plan CSV: ${state.plan}`, "ok");
        buildForm();
      } catch (e) {
        alert(e.message || "Failed to save plan CSV.");
      }
    }

    async function runDoctorCheck() {
      const status = q("doctor-status");
      try {
        const r = await fetch("/api/doctor");
        const j = await r.json();
        const out = String(j.stdout || "");
        const rc = Number(j.returncode ?? -1);
        if (!r.ok || j.returncode !== 0) {
          status.className = "status warn";
          status.textContent = `Environment check: FAIL (exit code ${rc})\n${out || "(no output)"}`;
          return;
        }
        status.className = "status ok";
        status.textContent = "Environment check: PASS";
      } catch (e) {
        status.className = "status warn";
        status.textContent = `Environment check: FAIL (request error)\n${e.message || e}`;
      }
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

    function selectedMode() {
      const command = state[COMMAND_KEY];
      if (!command) return "";
      if (command === "history-run") return "history-run";
      if (command === "machine-learning") {
        const mlKind = state[ML_KIND_KEY] || "data-generation";
        return ML_KIND_TO_MODE[mlKind] || "ml-data-gen";
      }
      return command;
    }

    function buildForm() {
      const grid = q("form-grid");
      grid.innerHTML = "";
      const commandSpec = { key: COMMAND_KEY, label: "Command", kind: "enum", choices: COMMAND_CHOICES };
      const mlKindSpec = { key: ML_KIND_KEY, label: "ML Workflow", kind: "enum", choices: ML_KIND_CHOICES, default: "data-generation" };

      if (state[COMMAND_KEY] === undefined) state[COMMAND_KEY] = "";
      if (state[COMMAND_KEY] && !COMMAND_CHOICES.includes(state[COMMAND_KEY])) {
        state[COMMAND_KEY] = "";
      }
      const commandSpecWithChoices = { ...commandSpec, choices: COMMAND_CHOICES };
      const mode = selectedMode();
      const specs = [commandSpecWithChoices];
      if (state[COMMAND_KEY] === "machine-learning") {
        if (state[ML_KIND_KEY] === undefined) state[ML_KIND_KEY] = "data-generation";
        specs.push(mlKindSpec);
      }
      if (mode) specs.push(...(MODE_SPECS[mode] || []));
      for (const spec of specs) {
        if (spec.key !== COMMAND_KEY && state[spec.key] === undefined) state[spec.key] = defaultFor(spec);

        const lab = document.createElement("label");
        lab.textContent = spec.label;
        grid.appendChild(lab);

        if (spec.kind === "bool") {
          const box = document.createElement("div");
          box.className = "bool-row";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.checked = !!state[spec.key];
          input.onchange = () => {
            state[spec.key] = input.checked;
            updatePreview();
          };
          box.appendChild(input);
          grid.appendChild(box);
          continue;
        }

        const sel = document.createElement("select");
        const inp = document.createElement("input");

        if (spec.kind === "enum") {
          let choices = spec.choices || [];
          if (spec.key === COMMAND_KEY) choices = COMMAND_CHOICES;
          if (spec.key === "model") choices = models;
          if (spec.key === COMMAND_KEY) {
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "-- Select command --";
            sel.appendChild(placeholder);
          }
          for (const c of choices) {
            const o = document.createElement("option");
            o.value = c;
            if (spec.key === COMMAND_KEY) {
              o.textContent = COMMAND_LABELS[c] || c;
            } else if (spec.key === ML_KIND_KEY) {
              o.textContent = ML_KIND_LABELS[c] || c;
            } else {
              o.textContent = c;
            }
            sel.appendChild(o);
          }
          sel.value = spec.key === COMMAND_KEY ? (state[spec.key] || "") : (state[spec.key] || defaultFor(spec));
          sel.onchange = () => {
            state[spec.key] = sel.value;
            if (spec.key === COMMAND_KEY || spec.key === ML_KIND_KEY) {
              if (spec.key === COMMAND_KEY || spec.key === ML_KIND_KEY) ensuredMlPlanModel = "";
              for (const k of Object.keys(state)) {
                if (k !== COMMAND_KEY && k !== ML_KIND_KEY) delete state[k];
              }
              buildForm();
            } else {
              if (spec.key === "model" && selectedMode() === "ml-data-gen") {
                ensuredMlPlanModel = "";
                state.plan = "";
                buildForm();
                return;
              }
              updatePreview();
              if (spec.key === "model" || spec.key === "plan") {
                void maybeEnsureMlPlan().catch((e) => setRunStatus(`Plan setup failed: ${e.message || e}`, "warn"));
              }
            }
          };
          grid.appendChild(sel);
        } else {
          inp.type = "text";
          inp.value = state[spec.key] || "";
          inp.oninput = () => { state[spec.key] = inp.value; updatePreview(); };
          const PICKER_KEYS = new Set(["out", "run", "case_file", "data", "plan", "config", "checkpoint", "candidates", "cpu_run", "gpu_run"]);
          const needsFolderPicker = PICKER_KEYS.has(spec.key);
          if (needsFolderPicker) {
            const row = document.createElement("div");
            row.className = "input-row";
            row.appendChild(inp);
            const browse = document.createElement("button");
            browse.type = "button";
            browse.className = "copy";
            browse.textContent = "Browse";
            browse.onclick = async () => { await openDirPicker(spec.key); };
            row.appendChild(browse);
            if (mode === "ml-data-gen" && spec.key === "plan") {
              const edit = document.createElement("button");
              edit.type = "button";
              edit.className = "copy";
              edit.textContent = "Edit CSV";
              edit.onclick = async () => { await openPlanCsvEditor(); };
              row.appendChild(edit);
            }
            grid.appendChild(row);
          } else {
            grid.appendChild(inp);
          }
        }
      }
      updatePreview();
      void maybeEnsureMlPlan().catch((e) => setRunStatus(`Plan setup failed: ${e.message || e}`, "warn"));
    }

    async function openDirPicker(targetKey) {
      pickerState.targetKey = targetKey;
      pickerState.path = "/";
      const raw = String(state[targetKey] || "").trim();
      if (raw && raw !== "auto") {
        // For file fields (for example plan.csv), open the parent directory.
        if (raw.endsWith(".csv") || raw.endsWith(".yaml") || raw.endsWith(".yml") || raw.endsWith(".json")) {
          const slash = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\\\"));
          pickerState.path = slash > 0 ? raw.slice(0, slash) : "/";
        } else {
          pickerState.path = raw;
        }
      }
      try {
        const data = await loadDirs(pickerState.path);
        renderDirPicker(data);
        q("dir-picker").hidden = false;
      } catch (e) {
        alert(e.message || "Failed to open folder picker.");
      }
    }

    function renderDirPicker(data) {
      pickerState.path = data.path;
      pickerState.parent = data.parent;
      q("picker-path").textContent = data.path;
      const list = q("picker-list");
      list.innerHTML = "";
      for (const d of data.dirs || []) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "picker-item";
        b.textContent = d.name;
        b.onclick = async () => {
          try {
            renderDirPicker(await loadDirs(d.path));
          } catch (e) {
            alert(e.message || "Failed to open folder.");
          }
        };
        list.appendChild(b);
      }
      q("picker-up-btn").disabled = !data.parent;
    }

    function buildCommandPayload() {
      const mode = selectedMode();
      if (!mode || !MODE_SPECS[mode]) throw new Error("Please select a command.");
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
        if (mode === "clean" && spec.key === "bucket" && s === "all") continue;
        args[spec.key] = s;
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

    function fmtNum(v, digits = 4) {
      const n = Number(v);
      if (!Number.isFinite(n)) return String(v ?? "");
      return n.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function setRunStatus(msg, kind = "") {
      const el = q("run-status");
      if (!msg) {
        el.hidden = true;
        el.className = "status";
        el.textContent = "";
        return;
      }
      el.hidden = false;
      el.className = kind ? `status ${kind}` : "status";
      el.textContent = msg;
    }

    function setRunBusy(isBusy) {
      q("run-btn").disabled = !!isBusy;
      q("copy-btn").disabled = !!isBusy;
      q("summary-visualize-btn").disabled = !!isBusy || !lastRunDir;
    }

    async function pollRunProgress() {
      try {
        const r = await fetch("/api/progress");
        const j = await r.json();
        if (!r.ok || !j.running) return;
        const cur = Number(j.step_current || 0);
        const total = Number(j.step_total || 0);
        if (cur > 0) {
          const right = total > 0 ? total : "?";
          setRunStatus(`Run in progress... step ${cur}/${right}`, "");
        } else if (j.last_line) {
          setRunStatus(`Run in progress... ${j.last_line}`, "");
        } else {
          setRunStatus("Run in progress...", "");
        }
      } catch (_e) {
        // Keep prior status if polling fails transiently.
      }
    }

    function simplifyRunId(raw) {
      const s = String(raw || "").trim();
      if (!s) return "";
      const parts = s.split("__");
      if (parts.length >= 3) return `${parts[0]} (${parts[2]})`;
      return s;
    }

    function setSummary(summary) {
      const panel = q("summary-panel");
      const content = q("summary-content");
      const vizBtn = q("summary-visualize-btn");
      if (!summary) {
        panel.hidden = true;
        content.innerHTML = "";
        vizBtn.disabled = true;
        return;
      }
      const timing = summary.timing || {};
      const last = summary.step_last || {};
      const history = summary.history || null;
      const historyGroups = Array.isArray(summary.history_groups) ? summary.history_groups : [];
      const historyRows = Array.isArray(summary.history_rows) ? summary.history_rows : [];
      const cards = [
        ["Run ID", simplifyRunId(summary.run_id || "")],
        ["Backend", summary.backend || ""],
        ["Steps", `${summary.steps_completed || 0}`],
        ["Requested Steps", `${summary.steps_requested || 0}`],
        ["Checkpoints", summary.checkpoints_written || 0],
        ["Total Time (s)", fmtNum(timing.total_time_s, 6)],
        ["Avg Step Time (s)", fmtNum(timing.avg_step_time_s, 6)],
        ["Sim Day", fmtNum(last.simulation_day, 4)],
        ["Last dt (days)", fmtNum(last.dt_days, 6)],
        ["Pressure Avg", fmtNum(last.pressure_avg, 4)],
        ["Sw Avg", fmtNum(last.sw_avg, 6)],
        ["Mass Balance Rel", fmtNum(last.mass_balance_rel, 8)],
      ];
      const historyOverview = history
        ? `<div class="kv-grid">
             <div class="kv"><div class="k">History Objective</div><div class="v">${fmtNum(history.objective_value, 6)}</div></div>
             <div class="kv"><div class="k">Compare Count</div><div class="v">${history.compare_count || 0}</div></div>
             <div class="kv"><div class="k">Objective Name</div><div class="v mono">${history.objective_name || ""}</div></div>
           </div>`
        : "";
      const historyGroupsHtml = historyGroups.length
        ? [
            '<div style="margin-top:10px"><h3 style="margin:0 0 8px 0;font-size:15px">Top Mismatch Groups</h3>',
            '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid #dbe3ea;border-radius:8px;overflow:hidden">',
            '<thead><tr style="background:#f1f5f9"><th style="text-align:left;padding:8px">Well</th><th style="text-align:left;padding:8px">Observable</th><th style="text-align:right;padding:8px">RMSE</th><th style="text-align:right;padding:8px">Weighted Misfit</th></tr></thead><tbody>',
            ...historyGroups.map((row) => `<tr><td style="padding:8px;border-top:1px solid #e5e7eb">${row.well || ""}</td><td style="padding:8px;border-top:1px solid #e5e7eb">${row.observable || ""}</td><td style="padding:8px;border-top:1px solid #e5e7eb;text-align:right">${fmtNum(row.rmse, 6)}</td><td style="padding:8px;border-top:1px solid #e5e7eb;text-align:right">${fmtNum(row.weighted_misfit, 6)}</td></tr>`),
            '</tbody></table></div></div>',
          ].join("")
        : "";
      const historyRowsHtml = historyRows.length
        ? [
            '<div style="margin-top:10px"><h3 style="margin:0 0 8px 0;font-size:15px">Largest Point Errors</h3>',
            '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid #dbe3ea;border-radius:8px;overflow:hidden">',
            '<thead><tr style="background:#f1f5f9"><th style="text-align:left;padding:8px">Day</th><th style="text-align:left;padding:8px">Well</th><th style="text-align:left;padding:8px">Observable</th><th style="text-align:right;padding:8px">Observed</th><th style="text-align:right;padding:8px">Simulated</th><th style="text-align:right;padding:8px">Weighted Error</th></tr></thead><tbody>',
            ...historyRows.map((row) => `<tr><td style="padding:8px;border-top:1px solid #e5e7eb">${fmtNum(row.day, 4)}</td><td style="padding:8px;border-top:1px solid #e5e7eb">${row.well || ""}</td><td style="padding:8px;border-top:1px solid #e5e7eb">${row.observable || ""}</td><td style="padding:8px;border-top:1px solid #e5e7eb;text-align:right">${fmtNum(row.observed_value, 6)}</td><td style="padding:8px;border-top:1px solid #e5e7eb;text-align:right">${fmtNum(row.simulated_value, 6)}</td><td style="padding:8px;border-top:1px solid #e5e7eb;text-align:right">${fmtNum(row.weighted_error, 6)}</td></tr>`),
            '</tbody></table></div></div>',
          ].join("")
        : "";
      const html = [
        '<div class="kv-grid">',
        ...cards.map(([k, v]) => `<div class="kv"><div class="k">${k}</div><div class="v">${v}</div></div>`),
        "</div>",
        historyOverview,
        historyGroupsHtml,
        historyRowsHtml,
        `<div class="mono">Run directory: ${summary.run_dir || ""}</div>`,
      ].join("");
      content.innerHTML = html;
      panel.hidden = false;
      lastRunDir = String(summary.run_dir || "").trim();
      vizBtn.disabled = !lastRunDir;
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
      setSummary(null);
      setRunBusy(true);
      setRunStatus("Run in progress...", "");
      progressTimer = setInterval(pollRunProgress, 500);
      log.textContent += `$ ${cmdText(payload)}\\n`;
      try {
        const start = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const sj = await start.json();
        if (!start.ok) throw new Error(sj.error || "Run failed to start.");
        const jobId = String(sj.job_id || "");
        if (!jobId) throw new Error("Missing run job id.");

        let j = null;
        while (true) {
          const rs = await fetch(`/api/run-status?job_id=${encodeURIComponent(jobId)}`);
          const jr = await rs.json();
          if (!rs.ok) throw new Error(jr.error || "Run status failed.");
          if (!jr.running) {
            j = jr;
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 500));
        }

        if (!j) throw new Error("Run status unavailable.");
        if (j.error) throw new Error(String(j.error));
        log.textContent += (j.stdout || "");
        if (!log.textContent.endsWith("\\n")) log.textContent += "\\n";
        log.textContent += `[exit] code=${j.returncode ?? "?"}\\n`;
        log.scrollTop = log.scrollHeight;
        if (payload.mode === "history-run") {
          setSummary(j.summary || null);
          if (j.run_dir) lastRunDir = String(j.run_dir);
        }
        if (Number(j.returncode || 0) === 0) {
          setRunStatus("Run completed successfully.", "ok");
        } else {
          setRunStatus(`Run finished with exit code ${j.returncode}.`, "warn");
        }
      } catch (e) {
        setRunStatus(`Run failed: ${e.message || e}`, "warn");
      } finally {
        if (progressTimer) {
          clearInterval(progressTimer);
          progressTimer = null;
        }
        setRunBusy(false);
      }
    }

    async function visualizeLastRun() {
      if (!lastRunDir) {
        alert("No run directory available to visualize.");
        return;
      }
      const btn = q("summary-visualize-btn");
      const old = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Visualizing...";
      const tab = window.open("about:blank", "_blank");
      if (tab) {
        tab.document.write("<!doctype html><html><body style='font-family:Segoe UI,Noto Sans,sans-serif;padding:20px'>Generating visuals, please wait...</body></html>");
        tab.document.close();
      }
      try {
        const r = await fetch("/api/visualize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ run_dir: lastRunDir }),
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "Visualization failed.");
        const log = q("log");
        log.textContent += `[visualize] run=${lastRunDir}\\n`;
        log.textContent += (j.stdout || "");
        if (!log.textContent.endsWith("\\n")) log.textContent += "\\n";
        log.textContent += `[visualize-exit] code=${j.returncode}\\n`;
        log.scrollTop = log.scrollHeight;
        const target = `/visuals?run_dir=${encodeURIComponent(lastRunDir)}`;
        if (tab) {
          tab.location.href = target;
        } else {
          window.open(target, "_blank");
        }
      } catch (e) {
        if (tab) {
          const msg = String(e.message || e).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
          tab.document.open();
          tab.document.write(`<!doctype html><html><body style='font-family:Segoe UI,Noto Sans,sans-serif;padding:20px'><h3>Visualization failed</h3><pre>${msg}</pre></body></html>`);
          tab.document.close();
        } else {
          alert(e.message || "Visualization failed.");
        }
      } finally {
        btn.textContent = old;
        btn.disabled = !lastRunDir;
      }
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
      await runDoctorCheck();
      await loadModels();
      buildForm();
      q("run-btn").onclick = runCmd;
      q("copy-btn").onclick = copyCmd;
      q("clear-btn").onclick = () => { q("log").textContent = ""; };
      q("summary-visualize-btn").onclick = visualizeLastRun;
      q("summary-visualize-btn").disabled = true;
      q("picker-cancel-btn").onclick = () => { q("dir-picker").hidden = true; };
      q("csv-editor-cancel-btn").onclick = () => { q("csv-editor").hidden = true; };
      q("csv-editor-save-btn").onclick = savePlanCsvEditor;
      q("picker-up-btn").onclick = async () => {
        if (!pickerState.parent) return;
        try {
          renderDirPicker(await loadDirs(pickerState.parent));
        } catch (e) {
          alert(e.message || "Failed to open parent folder.");
        }
      };
      q("picker-select-btn").onclick = () => {
        if (!pickerState.targetKey) return;
        state[pickerState.targetKey] = pickerState.path;
        q("dir-picker").hidden = true;
        buildForm();
      };
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


def list_dirs(path_arg: str) -> dict[str, object]:
    if path_arg.strip():
        requested = Path(path_arg).expanduser()
        base = requested if requested.is_absolute() else (ROOT / requested)
    else:
        base = ROOT
    current = base.resolve()
    if not current.exists() or not current.is_dir():
        raise ValueError(f"Directory not found: {current}")
    dirs: list[dict[str, str]] = []
    try:
        for item in current.iterdir():
            if item.is_dir():
                dirs.append({"name": item.name, "path": str(item.resolve())})
    except PermissionError as exc:
        raise ValueError(f"Permission denied: {current}") from exc
    dirs.sort(key=lambda x: x["name"].lower())
    parent = None if current.parent == current else str(current.parent)
    return {"path": str(current), "parent": parent, "dirs": dirs}


def resolve_repo_path(path_arg: str) -> Path:
    if not path_arg.strip():
        raise ValueError("Path cannot be empty.")
    raw = Path(path_arg).expanduser()
    target = raw if raw.is_absolute() else (ROOT / raw)
    resolved = target.resolve()
    root_resolved = ROOT.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("Invalid path outside repository.")
    return resolved


def repo_rel(path_obj: Path) -> str:
    try:
        return path_obj.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path_obj.resolve())


def default_ml_plan_content() -> str:
    return (
        "tag,rock.permeability_md,rock.porosity,fluid.nw,fluid.no,producer_pi\n"
        "candidate-a,120.0,0.18,1.8,2.1,1.05\n"
        "candidate-b,140.0,0.19,2.0,2.0,1.20\n"
        "candidate-c,165.0,0.17,2.3,1.9,1.35\n"
    )


def ensure_default_ml_plan(model: str) -> tuple[Path, bool]:
    model_name = model.strip()
    if not model_name:
        raise ValueError("Model is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise ValueError(f"Invalid model name: {model_name}")
    model_dir = ROOT / "cases" / model_name
    model_yaml = model_dir / "model.yaml"
    if not model_yaml.exists():
        raise ValueError(f"Model not found: {model_name}")
    plan_path = model_dir / "ml_scenarios.csv"
    created = False
    if not plan_path.exists():
        plan_path.write_text(default_ml_plan_content(), encoding="utf-8")
        created = True
    return plan_path.resolve(), created


def python_exec() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return "python3"


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
    if mode == "history-run":
        case_file = str(args.get("case_file", "")).strip()
        if case_file:
            cmd.extend(["--case-file", case_file])
    return cmd


def run_command_logged(cmd: list[str], label: str) -> tuple[str, int]:
    pretty = " ".join(shlex.quote(p) for p in cmd)
    print(f"[web-ui] {label} start: {pretty}", flush=True)
    if label == "history-run":
        with PROGRESS_LOCK:
            RUN_PROGRESS["running"] = True
            RUN_PROGRESS["step_current"] = 0
            RUN_PROGRESS["step_total"] = 0
            RUN_PROGRESS["last_line"] = "Run started"
    requested_steps: int | None = None
    if label == "history-run":
        for i, token in enumerate(cmd):
            if token == "--steps" and i + 1 < len(cmd):
                try:
                    requested_steps = int(cmd[i + 1])
                except ValueError:
                    requested_steps = None
                break
    if label == "history-run" and requested_steps is not None:
        with PROGRESS_LOCK:
            RUN_PROGRESS["step_total"] = requested_steps

    stop_event = threading.Event()
    watcher_thread: threading.Thread | None = None

    def start_step_watcher(run_dir: str) -> None:
        nonlocal watcher_thread
        if watcher_thread is not None:
            return
        run_path = Path(run_dir).expanduser()
        if not run_path.is_absolute():
            run_path = (ROOT / run_path).resolve()
        csv_path = run_path / "step_stats.csv"

        def _watch() -> None:
            last_seen = -1
            while not stop_event.is_set():
                if csv_path.exists():
                    try:
                        with csv_path.open("r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        if rows:
                            step_idx = int(float(rows[-1].get("step_idx", "-1")))
                            if step_idx > last_seen:
                                last_seen = step_idx
                                current = step_idx + 1
                                total = requested_steps if requested_steps is not None else "?"
                                print(f"[web-ui:history-run] progress step {current}/{total}", flush=True)
                                with PROGRESS_LOCK:
                                    RUN_PROGRESS["step_current"] = current
                                    RUN_PROGRESS["step_total"] = requested_steps or 0
                                    RUN_PROGRESS["last_line"] = f"step {current}/{total}"
                    except Exception:
                        pass
                time.sleep(0.5)

        watcher_thread = threading.Thread(target=_watch, name="history-run-step-watcher", daemon=True)
        watcher_thread.start()

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output.append(line)
        print(f"[web-ui:{label}] {line.rstrip()}", flush=True)
        if label == "history-run":
            progress_match = re.search(r"\[progress\]\s+step\s+(\d+)/(\d+)", line)
            with PROGRESS_LOCK:
                if progress_match:
                    RUN_PROGRESS["step_current"] = int(progress_match.group(1))
                    RUN_PROGRESS["step_total"] = int(progress_match.group(2))
                    RUN_PROGRESS["last_line"] = f"step {progress_match.group(1)}/{progress_match.group(2)}"
                else:
                    RUN_PROGRESS["last_line"] = line.rstrip()
        if label == "history-run":
            stripped = line.strip()
            if stripped.startswith("out="):
                run_dir = stripped.split("=", 1)[1].strip()
                if run_dir:
                    start_step_watcher(run_dir)
    proc.stdout.close()
    rc = proc.wait()
    stop_event.set()
    if watcher_thread is not None:
        watcher_thread.join(timeout=1.0)
    if label == "history-run":
        with PROGRESS_LOCK:
            RUN_PROGRESS["running"] = False
            RUN_PROGRESS["last_line"] = f"Run finished (exit {rc})"
    print(f"[web-ui] {label} exit={rc}", flush=True)
    return "".join(output), rc


def infer_run_dir(stdout_text: str, out_arg: str) -> str | None:
    matches = re.findall(r"Output directory:\s*(.+)", stdout_text or "")
    if matches:
        return matches[-1].strip()
    out_clean = out_arg.strip()
    if out_clean and out_clean != "auto":
        p = Path(out_clean).expanduser()
        return str((ROOT / p).resolve()) if not p.is_absolute() else str(p.resolve())
    return None


def maybe_build_case_override(model: str, args: dict[str, object]) -> str | None:
    if not model:
        return None
    raw_schedule = str(args.get("schedule_end_step", "")).strip()
    if not raw_schedule:
        return None

    case_path = ROOT / "cases" / model / "model.yaml"
    if not case_path.exists():
        raise ValueError(f"Model case file not found: {case_path}")
    text = case_path.read_text(encoding="utf-8")

    replacements: dict[str, str] = {}
    if raw_schedule:
        schedule_val = int(raw_schedule)
        if schedule_val <= 0:
            raise ValueError("Maximum Steps must be a positive integer.")
        replacements["schedule_end_step"] = str(schedule_val)

    for key, val in replacements.items():
        pattern = rf"(?m)^({re.escape(key)}\s*:\s*).*$"
        if re.search(pattern, text):
            text = re.sub(pattern, lambda m: f"{m.group(1)}{val}", text)
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += f"{key}: {val}\n"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"case_override_{model}_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(text)
        return tmp.name


def _float(v: str | int | float | None, default: float = 0.0) -> float:
    try:
        return float(v if v is not None else default)
    except (TypeError, ValueError):
        return default


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_run_summary(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Run summary unavailable; missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    step_rows = _load_csv_rows(run_dir / "step_stats.csv")
    timing_rows = _load_csv_rows(run_dir / "timing.csv")

    step_last: dict[str, object] = {}
    if step_rows:
        last = step_rows[-1]
        step_last = {
            "step_idx": int(_float(last.get("step_idx"), 0.0)),
            "dt_days": _float(last.get("dt_days")),
            "simulation_day": _float(last.get("simulation_day")),
            "pressure_avg": _float(last.get("pressure_avg")),
            "sw_avg": _float(last.get("sw_avg")),
            "mass_balance_rel": _float(last.get("mass_balance_rel")),
            "total_time_s": _float(last.get("total_time_s")),
        }

    step_timing = [r for r in timing_rows if (r.get("row_type") or "").strip() == "step"]
    if not step_timing:
        step_timing = timing_rows
    total_pressure = sum(_float(r.get("pressure_time_s")) for r in step_timing)
    total_transport = sum(_float(r.get("transport_time_s")) for r in step_timing)
    total_io = sum(_float(r.get("io_time_s")) for r in step_timing)
    total_time = sum(_float(r.get("total_time_s")) for r in step_timing)
    count = len(step_timing)

    history_summary_rows = _load_csv_rows(run_dir / "well_observed_vs_simulated.csv")
    history_detail_rows = _load_csv_rows(run_dir / "history_match.csv")
    top_history_groups: list[dict[str, object]] = []
    for row in history_summary_rows:
        top_history_groups.append(
            {
                "well": str(row.get("well", "")),
                "observable": str(row.get("observable", "")),
                "rmse": _float(row.get("rmse")),
                "mae": _float(row.get("mae")),
                "weighted_misfit": _float(row.get("weighted_misfit")),
            }
        )
    top_history_groups.sort(key=lambda r: float(r.get("weighted_misfit", 0.0)), reverse=True)

    top_history_rows: list[dict[str, object]] = []
    for row in history_detail_rows:
        top_history_rows.append(
            {
                "day": _float(row.get("day")),
                "well": str(row.get("well", "")),
                "observable": str(row.get("observable", "")),
                "observed_value": _float(row.get("observed_value")),
                "simulated_value": _float(row.get("simulated_value")),
                "weighted_error": _float(row.get("weighted_error")),
            }
        )
    top_history_rows.sort(key=lambda r: float(r.get("weighted_error", 0.0)), reverse=True)

    return {
        "run_dir": str(run_dir),
        "run_id": str(meta.get("run_id", run_dir.name)),
        "backend": str(meta.get("backend", "")),
        "steps_completed": int(_float(meta.get("steps_completed"), 0.0)),
        "steps_requested": int(_float(meta.get("steps_requested"), 0.0)),
        "checkpoints_written": int(_float(meta.get("checkpoints_written"), 0.0)),
        "timing": {
            "total_time_s": total_time,
            "pressure_time_s": total_pressure,
            "transport_time_s": total_transport,
            "io_time_s": total_io,
            "avg_step_time_s": (total_time / count) if count else 0.0,
        },
        "step_last": step_last,
        "history": (
            json.loads((run_dir / "history_mismatch.json").read_text(encoding="utf-8"))
            if (run_dir / "history_mismatch.json").exists()
            else None
        ),
        "history_groups": top_history_groups[:4],
        "history_rows": top_history_rows[:5],
    }


def list_visual_artifacts(run_dir: Path) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    candidates: list[Path] = []
    figs = run_dir / "figs"
    anim = run_dir / "animations"
    if figs.is_dir():
        candidates.extend(sorted(figs.glob("*.png")))
    if anim.is_dir():
        for ext in ("*.gif", "*.mp4", "*.png"):
            candidates.extend(sorted(anim.glob(ext)))
    for p in candidates:
        rel = str(p.resolve().relative_to(ROOT))
        ext = p.suffix.lower()
        kind = "image" if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"} else "video"
        results.append({"name": p.name, "path": rel, "kind": kind})
    return results


def build_visuals_page(run_dir: Path) -> str:
    meta: dict[str, object] = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    nx = int(_float(meta.get("nx"), 0.0))
    ny = int(_float(meta.get("ny"), 0.0))
    nz = int(_float(meta.get("nz"), 0.0))
    if nz > 1:
        dim_msg = f"Detected volumetric run: nx={nx}, ny={ny}, nz={nz}. 3D slice visuals are expected."
        dim_style = "background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46"
    else:
        dim_msg = (
            "Detected legacy 2D run metadata (missing nz or nz<=1). "
            "This run will only show 2D maps. Rebuild and re-run the case with nz>1 to generate 3D outputs."
        )
        dim_style = "background:#fff7ed;border:1px solid #fed7aa;color:#9a3412"

    visuals = list_visual_artifacts(run_dir)
    cards: list[str] = []
    for v in visuals:
        # Hide performance figures for now.
        if any(tok in v["name"] for tok in ("runtime_bar", "kernel_breakdown", "speedup_bar")):
            continue
        src = f"/api/file?path={quote(v['path'])}"
        safe_name = html.escape(v["name"])
        safe_src = html.escape(src, quote=True)
        safe_click_src = safe_src.replace("'", "&#39;")
        if v["kind"] == "video":
            media = (
                f'<video controls preload="metadata" src="{safe_src}" '
                f'onclick="openLightbox(\'video\', \'{safe_click_src}\')" '
                'style="width:100%;border-radius:8px;background:#000;cursor:zoom-in"></video>'
            )
        else:
            media = (
                f'<img src="{safe_src}" alt="{safe_name}" '
                f'onclick="openLightbox(\'image\', \'{safe_click_src}\')" '
                'style="width:100%;border-radius:8px;border:1px solid #ddd;cursor:zoom-in" />'
            )
        cards.append(
            "<div style='background:#fff;border:1px solid #d7dee6;border-radius:10px;padding:10px'>"
            f"<div style='font-family:ui-monospace,Consolas,monospace;font-size:12px;margin-bottom:8px'>{safe_name}</div>"
            f"{media}"
            "</div>"
        )
    if not cards:
        cards_html = "<div style='color:#6b7280'>No visuals found. Click Visualize again after a run.</div>"
    else:
        cards_html = "".join(cards)
    safe_run = html.escape(str(run_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visual Outputs</title>
</head>
<body style="margin:0;font-family:Segoe UI,Noto Sans,sans-serif;background:#f3f6f9;color:#0f172a">
  <div style="max-width:1200px;margin:20px auto;padding:0 12px">
    <h1 style="margin:0 0 6px 0">Visual Outputs</h1>
    <div style="font-family:ui-monospace,Consolas,monospace;font-size:12px;margin-bottom:12px;word-break:break-all">{safe_run}</div>
    <div style="padding:10px 12px;border-radius:10px;margin:0 0 12px 0;{dim_style}">{html.escape(dim_msg)}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">{cards_html}</div>
  </div>
  <div id="lightbox" onclick="closeLightbox()" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;align-items:center;justify-content:center;padding:20px">
    <div onclick="event.stopPropagation()" style="max-width:95vw;max-height:95vh">
      <img id="lightbox-image" alt="" style="display:none;max-width:95vw;max-height:95vh;border-radius:10px" />
      <video id="lightbox-video" controls style="display:none;max-width:95vw;max-height:95vh;border-radius:10px;background:#000"></video>
    </div>
  </div>
  <script>
    function openLightbox(kind, src) {{
      const box = document.getElementById("lightbox");
      const img = document.getElementById("lightbox-image");
      const vid = document.getElementById("lightbox-video");
      if (kind === "video") {{
        img.style.display = "none";
        vid.style.display = "block";
        vid.src = src;
      }} else {{
        vid.pause();
        vid.removeAttribute("src");
        vid.load();
        vid.style.display = "none";
        img.style.display = "block";
        img.src = src;
      }}
      box.style.display = "flex";
    }}
    function closeLightbox() {{
      const box = document.getElementById("lightbox");
      const vid = document.getElementById("lightbox-video");
      vid.pause();
      box.style.display = "none";
    }}
  </script>
</body>
</html>"""


def execute_run_request(mode: str, args_in: dict[str, object]) -> dict[str, object]:
    temp_case_file: str | None = None
    args = dict(args_in)
    try:
        if mode == "history-run":
            model = str(args.get("model", "")).strip()
            temp_case_file = maybe_build_case_override(model, args)
            if temp_case_file:
                args["case_file"] = temp_case_file
        cmd = build_cli(mode, args)
        full_stdout, final_rc = run_command_logged(cmd, "history-run")
        run_dir_for_response: str | None = None
        summary: dict[str, object] | None = None

        if mode == "history-run" and final_rc == 0:
            run_dir = infer_run_dir(full_stdout, str(args.get("out", "")))
            model = str(args.get("model", "")).strip()
            run_dir_for_response = run_dir

            if run_dir:
                run_path = Path(run_dir).expanduser().resolve()
                try:
                    summary = build_run_summary(run_path)
                except Exception as exc:
                    full_stdout += f"\n[post-run] summary unavailable: {exc}\n"

        return {
            "command": cmd,
            "stdout": full_stdout,
            "returncode": final_rc,
            "run_dir": run_dir_for_response,
            "summary": summary,
        }
    finally:
        if temp_case_file:
            try:
                os.unlink(temp_case_file)
            except OSError:
                pass


def start_run_job(mode: str, args: dict[str, object]) -> str:
    job_id = uuid.uuid4().hex
    with RUN_JOB_LOCK:
        RUN_JOBS[job_id] = {"running": True, "stdout": "", "returncode": None, "summary": None, "run_dir": None, "error": ""}

    def _worker() -> None:
        try:
            result = execute_run_request(mode, args)
            with RUN_JOB_LOCK:
                RUN_JOBS[job_id].update(
                    {
                        "running": False,
                        "stdout": result.get("stdout", ""),
                        "returncode": result.get("returncode"),
                        "summary": result.get("summary"),
                        "run_dir": result.get("run_dir"),
                        "command": result.get("command"),
                        "error": "",
                    }
                )
        except Exception as exc:
            with RUN_JOB_LOCK:
                RUN_JOBS[job_id].update({"running": False, "error": str(exc), "returncode": 2})

    threading.Thread(target=_worker, name=f"run-job-{job_id[:8]}", daemon=True).start()
    return job_id


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
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/":
            page = HTML.replace("__MODE_SPECS__", json.dumps(MODE_SPECS)).replace("__ARG_FLAG__", json.dumps(ARG_FLAG))
            self._html(page)
            return
        if path == "/api/models":
            self._json({"models": list_models()})
            return
        if path == "/api/progress":
            with PROGRESS_LOCK:
                snap = dict(RUN_PROGRESS)
            self._json(snap)
            return
        if path == "/api/run-status":
            job_id = qs.get("job_id", [""])[0].strip()
            if not job_id:
                self._json({"error": "Missing required query param: job_id"}, code=HTTPStatus.BAD_REQUEST)
                return
            with RUN_JOB_LOCK:
                job = dict(RUN_JOBS.get(job_id, {}))
            if not job:
                self._json({"error": "job not found"}, code=HTTPStatus.NOT_FOUND)
                return
            self._json(job)
            return
        if path == "/api/doctor":
            out, rc = run_command_logged([str(WORKFLOW), "doctor"], "doctor")
            self._json({"stdout": out, "returncode": rc})
            return
        if path == "/api/dirs":
            try:
                path_arg = qs.get("path", [""])[0]
                self._json(list_dirs(path_arg))
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/ml-plan-default":
            try:
                model = qs.get("model", [""])[0]
                plan_path, created = ensure_default_ml_plan(model)
                self._json({"path": repo_rel(plan_path), "created": created})
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/ml-plan":
            try:
                path_arg = qs.get("path", [""])[0]
                if not path_arg:
                    raise ValueError("Missing required query param: path")
                file_path = resolve_repo_path(path_arg)
                if not file_path.exists():
                    raise ValueError(f"File not found: {file_path}")
                if not file_path.is_file():
                    raise ValueError(f"Not a file: {file_path}")
                content = file_path.read_text(encoding="utf-8")
                self._json({"path": repo_rel(file_path), "content": content})
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        if path == "/visuals":
            try:
                run_dir = qs.get("run_dir", [""])[0].strip()
                if not run_dir:
                    raise ValueError("Missing required query param: run_dir")
                run_path = Path(run_dir).expanduser().resolve()
                if not run_path.is_dir() or not (run_path / "meta.json").exists():
                    raise ValueError(f"Invalid run directory: {run_path}")
                page = build_visuals_page(run_path)
                self._html(page)
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/file":
            try:
                rel = qs.get("path", [""])[0]
                if not rel:
                    raise ValueError("Missing required query param: path")
                file_path = resolve_repo_path(rel)
                if not file_path.is_file():
                    raise ValueError(f"File not found: {file_path}")
                data = file_path.read_bytes()
                ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        self._json({"error": "not found"}, code=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/run", "/api/visualize", "/api/ml-plan-save"}:
            self._json({"error": "not found"}, code=HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/api/ml-plan-save":
            try:
                req = json.loads(raw.decode("utf-8"))
                path_arg = str(req.get("path", "")).strip()
                content = str(req.get("content", ""))
                if not path_arg:
                    raise ValueError("Missing required field: path")
                file_path = resolve_repo_path(path_arg)
                if file_path.suffix.lower() != ".csv":
                    raise ValueError("Plan editor only supports CSV files.")
                parent = file_path.parent
                parent.mkdir(parents=True, exist_ok=True)
                header_line = ""
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    header_line = stripped
                    break
                if not header_line:
                    raise ValueError("CSV must include a header line.")
                cols = [c.strip() for c in header_line.split(",")]
                if "tag" not in cols:
                    raise ValueError("CSV header must include a 'tag' column.")
                normalized = content if content.endswith("\n") else (content + "\n")
                file_path.write_text(normalized, encoding="utf-8")
                self._json({"path": repo_rel(file_path), "saved": True})
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/api/visualize":
            try:
                req = json.loads(raw.decode("utf-8"))
                run_dir = str(req.get("run_dir", "")).strip()
                if not run_dir:
                    raise ValueError("Missing required field: run_dir")
                run_path = Path(run_dir).expanduser().resolve()
                if not run_path.is_dir():
                    raise ValueError(f"Run directory not found: {run_path}")
                if not (run_path / "meta.json").exists():
                    raise ValueError(f"{run_path} does not contain meta.json")
                anim_script = ROOT / "python" / "viz" / "make_animation.py"
                py_bin = ROOT / ".venv" / "bin" / "python"
                steps: list[tuple[list[str], str]] = [
                    ([str(ROOT / "tools" / "plot_run.sh"), "--run", str(run_path)], "plot"),
                    ([str(py_bin), str(anim_script), "--run", str(run_path), "--field", "sw", "--out", "animations"], "animate"),
                    ([str(py_bin), str(anim_script), "--run", str(run_path), "--field", "pressure", "--out", "animations"], "animate-pressure"),
                ]
                full_stdout = ""
                final_rc = 0
                for cmd, label in steps:
                    step_out, step_rc = run_command_logged(cmd, f"visualize-{label}")
                    full_stdout += f"[{label}] " + " ".join(cmd) + "\n"
                    full_stdout += step_out
                    if step_rc != 0 and final_rc == 0:
                        final_rc = step_rc
                self._json({"stdout": full_stdout, "returncode": final_rc})
            except Exception as exc:
                self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return

        try:
            req = json.loads(raw.decode("utf-8"))
            mode = str(req.get("mode", "")).strip()
            args = req.get("args", {}) or {}
            if not isinstance(args, dict):
                raise ValueError("args must be an object")
            args = dict(args)
        except Exception as exc:
            self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)
            return
        try:
            job_id = start_run_job(mode, args)
            self._json({"job_id": job_id})
        except Exception as exc:
            self._json({"error": str(exc)}, code=HTTPStatus.BAD_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local web UI for workflow commands.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8770, type=int, help="Bind port (default: 8770)")
    return parser.parse_args()


def create_server(host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    # If the default port is busy, scan a short range so users can still start the UI.
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), Handler), candidate
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
    raise OSError(errno.EADDRINUSE, f"No free port found in range {port}-{port + 19}")


def main() -> int:
    args = parse_args()
    if not WORKFLOW.exists():
        print(f"workflow script not found: {WORKFLOW}")
        return 2
    server, bound_port = create_server(args.host, args.port)
    if bound_port != args.port:
        print(f"Requested port {args.port} is busy; using {bound_port} instead.")
    print(f"Workflow web UI running at http://{args.host}:{bound_port}")
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
