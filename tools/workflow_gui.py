#!/usr/bin/env python3
"""Simple Tk GUI for workflow command execution."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, ttk, messagebox
from tkinter.scrolledtext import ScrolledText


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str  # text|enum|bool
    default: str = ""
    choices: tuple[str, ...] = ()
    required: bool = False


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflow"


def list_models() -> list[str]:
    cases = ROOT / "cases"
    if not cases.exists():
        return []
    return sorted([p.name for p in cases.iterdir() if p.is_dir() and (p / "model.yaml").exists()])


MODE_SPECS: dict[str, list[FieldSpec]] = {
    "doctor": [],
    "history-run": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("mode", "Mode", "enum", default="release", choices=("debug", "release")),
        FieldSpec("backend", "Backend", "enum", default="cpu", choices=("cpu", "gpu")),
        FieldSpec("steps", "Steps", "text", default="10"),
        FieldSpec("output_every", "Output Every", "text", default="1"),
        FieldSpec("tag", "Tag", "text"),
        FieldSpec("out", "Out", "text", default="auto"),
    ],
    "ml-data-gen": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("plan", "Plan CSV", "text"),
        FieldSpec("mode", "Mode", "enum", default="release", choices=("debug", "release")),
        FieldSpec("backend", "Backend", "enum", default="cpu", choices=("cpu", "gpu")),
        FieldSpec("steps", "Steps", "text", default="200"),
        FieldSpec("output_every", "Output Every", "text", default="1"),
        FieldSpec("keep_temp", "Keep Temp", "bool"),
    ],
    "ml-train": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("data", "Data Dir", "text"),
        FieldSpec("config", "Config YAML", "text"),
        FieldSpec("seed", "Seed", "text", default="42"),
        FieldSpec("out", "Out Dir", "text"),
    ],
    "ml-eval": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("checkpoint", "Checkpoint", "text", required=True),
        FieldSpec("data", "Data Dir", "text"),
        FieldSpec("out", "Out Dir", "text"),
    ],
    "ml-score": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("checkpoint", "Checkpoint", "text", required=True),
        FieldSpec("candidates", "Candidate CSV", "text"),
        FieldSpec("out", "Out Dir", "text"),
    ],
    "ml-check": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("plan", "Plan CSV", "text"),
        FieldSpec("config", "Config YAML", "text"),
        FieldSpec("mode", "Mode", "enum", default="release", choices=("debug", "release")),
        FieldSpec("backend", "Backend", "enum", default="cpu", choices=("cpu", "gpu")),
        FieldSpec("steps", "Steps", "text", default="200"),
        FieldSpec("output_every", "Output Every", "text", default="1"),
        FieldSpec("seed", "Seed", "text", default="42"),
    ],
    "validate": [
        FieldSpec("run", "Run Path/ID", "text", required=True),
    ],
    "parity": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("cpu_run", "CPU Run", "text"),
        FieldSpec("gpu_run", "GPU Run", "text"),
        FieldSpec("out", "Out JSON Path", "text"),
    ],
    "bench": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("repeats", "Repeats", "text", default="3"),
        FieldSpec("steps", "Steps", "text", default="50"),
        FieldSpec("output_every", "Output Every", "text", default="10"),
        FieldSpec("out", "Out CSV", "text"),
    ],
    "plot": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("run", "Run Path/ID", "text"),
        FieldSpec("out", "Out Dir", "text", default="figs"),
        FieldSpec("check_only", "Check Only", "bool"),
    ],
    "clean": [
        FieldSpec("model", "Model", "enum", required=True),
        FieldSpec("bucket", "Type of run", "enum", default="all", choices=("all", "history", "benchmark", "ml-data", "legacy")),
        FieldSpec("keep", "Keep Newest N", "text", default="5"),
        FieldSpec("apply", "Apply", "bool"),
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
    "bucket": "--bucket",
    "keep": "--keep",
    "apply": "--apply",
}


class WorkflowGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Reserv ML Workflow")
        self.root.geometry("980x720")

        self.models = list_models()
        self._proc: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._field_vars: dict[str, StringVar | BooleanVar] = {}

        top = ttk.Frame(root, padding=10)
        top.pack(fill="both", expand=True)

        header = ttk.Frame(top)
        header.pack(fill="x")

        ttk.Label(header, text="Mode").pack(side="left")
        self.mode_var = StringVar(value="history-run")
        modes = sorted(MODE_SPECS.keys())
        self.mode_combo = ttk.Combobox(header, textvariable=self.mode_var, values=modes, state="readonly", width=24)
        self.mode_combo.pack(side="left", padx=(8, 16))
        self.mode_combo.bind("<<ComboboxSelected>>", self._render_fields)

        self.preview_var = StringVar(value="")
        ttk.Label(header, textvariable=self.preview_var).pack(side="left", fill="x", expand=True)

        self.form = ttk.LabelFrame(top, text="Arguments", padding=10)
        self.form.pack(fill="x", pady=(10, 10))

        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=(0, 8))
        self.run_btn = ttk.Button(actions, text="Run", command=self.run_command)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop_command, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Copy Command", command=self.copy_command).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Clear Output", command=self.clear_output).pack(side="left", padx=(8, 0))

        self.output = ScrolledText(top, height=24)
        self.output.pack(fill="both", expand=True)

        self._render_fields()
        self._tick_output()

    def _render_fields(self, *_args: object) -> None:
        for child in self.form.winfo_children():
            child.destroy()
        self._field_vars.clear()

        specs = MODE_SPECS[self.mode_var.get()]
        row = 0
        for spec in specs:
            ttk.Label(self.form, text=spec.label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            if spec.kind == "bool":
                var = BooleanVar(value=False)
                widget = ttk.Checkbutton(self.form, variable=var, command=self._update_preview)
                widget.grid(row=row, column=1, sticky="w", pady=4)
                self._field_vars[spec.key] = var
            elif spec.kind == "enum":
                if spec.key == "model":
                    values = tuple(self.models)
                    default = spec.default or (self.models[0] if self.models else "")
                else:
                    values = spec.choices
                    default = spec.default or (values[0] if values else "")
                var = StringVar(value=default)
                combo = ttk.Combobox(self.form, textvariable=var, values=values, state="readonly", width=42)
                combo.grid(row=row, column=1, sticky="ew", pady=4)
                combo.bind("<<ComboboxSelected>>", lambda _e: self._update_preview())
                self._field_vars[spec.key] = var
            else:
                var = StringVar(value=spec.default)
                entry = ttk.Entry(self.form, textvariable=var, width=45)
                entry.grid(row=row, column=1, sticky="ew", pady=4)
                entry.bind("<KeyRelease>", lambda _e: self._update_preview())
                self._field_vars[spec.key] = var
            row += 1

        self.form.columnconfigure(1, weight=1)
        self._update_preview()

    def _build_command(self) -> list[str]:
        mode = self.mode_var.get()
        cmd = [str(WORKFLOW), mode]
        specs = MODE_SPECS[mode]

        for spec in specs:
            flag = ARG_FLAG[spec.key]
            raw = self._field_vars[spec.key]
            if spec.kind == "bool":
                assert isinstance(raw, BooleanVar)
                if raw.get():
                    cmd.append(flag)
                continue

            assert isinstance(raw, StringVar)
            value = raw.get().strip()
            if spec.required and not value:
                raise ValueError(f"Missing required field: {spec.label}")
            if not value:
                continue
            if mode == "clean" and spec.key == "bucket" and value == "all":
                continue

            cmd.extend([flag, value])

        return cmd

    def _update_preview(self) -> None:
        try:
            cmd = self._build_command()
            pretty = " ".join(self._quote(part) for part in cmd)
        except ValueError as exc:
            pretty = f"Invalid: {exc}"
        self.preview_var.set(pretty)

    @staticmethod
    def _quote(v: str) -> str:
        if " " in v or "\t" in v:
            return '"' + v.replace('"', '\\"') + '"'
        return v

    def copy_command(self) -> None:
        try:
            cmd = self._build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid command", str(exc))
            return
        text = " ".join(self._quote(part) for part in cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def clear_output(self) -> None:
        self.output.delete("1.0", "end")

    def run_command(self) -> None:
        if self._proc is not None:
            return
        try:
            cmd = self._build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid command", str(exc))
            return

        self.output.insert("end", f"$ {' '.join(self._quote(p) for p in cmd)}\n")
        self.output.see("end")

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def worker() -> None:
            env = dict(os.environ)
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self._queue.put(line)
            rc = proc.wait()
            self._queue.put(f"\n[exit] code={rc}\n")
            self._proc = None

        threading.Thread(target=worker, daemon=True).start()

    def stop_command(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()

    def _tick_output(self) -> None:
        updated = False
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            self.output.insert("end", line)
            updated = True
        if updated:
            self.output.see("end")
            if self._proc is None:
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
        self.root.after(100, self._tick_output)


def main() -> int:
    if not WORKFLOW.exists():
        print(f"workflow script not found: {WORKFLOW}")
        return 2
    root = Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    WorkflowGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
