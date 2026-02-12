#!/usr/bin/env python3
"""
GUI for configuring external multimodal LLM API for image analysis.
Uses tkinter for cross-platform GUI.
Supports: API URL, API Key, test connection, fetch models, select model,
multi-threading toggle, and persistent config.
"""

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import urllib.request
import urllib.error
import ssl
import base64

CONFIG_FILENAME = "llm_api_config.json"


def _get_config_path(workspace_dir=None):
    """Get config file path in .tmp/cache/"""
    if workspace_dir:
        return os.path.join(workspace_dir, ".tmp", "cache", CONFIG_FILENAME)
    return os.path.join(os.getcwd(), ".tmp", "cache", CONFIG_FILENAME)


def load_config(workspace_dir=None):
    """Load saved LLM API config."""
    path = _get_config_path(workspace_dir)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "api_url": "http://localhost:4141/",
        "api_key": "",
        "model": "",
        "models_list": [],
        "enable_multithreading": False,
        "max_threads": 3,
    }


def save_config(config, workspace_dir=None):
    """Save LLM API config."""
    path = _get_config_path(workspace_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _http_request(url, method="GET", data=None, headers=None, timeout=15):
    """Simple HTTP request helper using urllib (no external deps)."""
    if headers is None:
        headers = {}
    if data is not None:
        if isinstance(data, dict):
            data = json.dumps(data).encode('utf-8')
            headers.setdefault('Content-Type', 'application/json')
        elif isinstance(data, str):
            data = data.encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    # Allow both http and https
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode('utf-8')
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace') if e.fp else str(e)
        return e.code, body
    except Exception as e:
        return 0, str(e)


def test_connection(api_url, api_key=""):
    """Test API connection. Returns (success, message)."""
    url = api_url.rstrip('/')
    # Try OpenAI-compatible /v1/models endpoint
    models_url = url + "/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    status, body = _http_request(models_url, headers=headers)
    if status == 200:
        return True, f"连接成功 (HTTP {status})"
    elif status == 401:
        return False, f"认证失败 (HTTP 401): 请检查 API Key"
    elif status == 0:
        return False, f"连接失败: {body}"
    else:
        return False, f"连接异常 (HTTP {status}): {body[:200]}"


def fetch_models(api_url, api_key=""):
    """Fetch available models from API. Returns (success, models_list_or_error)."""
    url = api_url.rstrip('/') + "/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    status, body = _http_request(url, headers=headers)
    if status != 200:
        return False, f"获取模型列表失败 (HTTP {status}): {body[:200]}"
    try:
        data = json.loads(body)
        models = []
        if "data" in data:
            for m in data["data"]:
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
        if not models:
            return False, "API 返回了空的模型列表"
        return True, models
    except Exception as e:
        return False, f"解析模型列表失败: {e}"


def call_llm_vision(api_url, api_key, model, image_b64, mime_type, prompt, timeout=120):
    """Call external LLM with image for vision analysis.
    Returns (success, result_text_or_error).
    """
    url = api_url.rstrip('/') + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    status, body = _http_request(url, method="POST", data=payload, timeout=timeout)
    if status != 200:
        return False, f"LLM 调用失败 (HTTP {status}): {body[:500]}"
    try:
        data = json.loads(body)
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return True, content
        return False, f"LLM 返回了空的响应: {body[:300]}"
    except Exception as e:
        return False, f"解析 LLM 响应失败: {e}"


class LLMConfigGUI:
    """Tkinter GUI for configuring external multimodal LLM API."""

    def __init__(self, workspace_dir=None):
        self.workspace_dir = workspace_dir
        self.config = load_config(workspace_dir)
        self.result = None  # Will be set to config dict if user confirms
        self._build_ui()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("外部多模态 LLM API 配置")
        self.root.geometry("620x580")
        self.root.resizable(False, False)

        # Try to center on screen
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="配置外部多模态 LLM API 进行图片解析",
                                font=("", 12, "bold"))
        title_label.pack(anchor=tk.W, pady=(0, 10))

        # --- API URL ---
        url_frame = ttk.LabelFrame(main_frame, text="API 地址", padding=8)
        url_frame.pack(fill=tk.X, pady=4)

        self.url_var = tk.StringVar(value=self.config.get("api_url", "http://localhost:4141/"))
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=60)
        url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        test_btn = ttk.Button(url_frame, text="测试连接", command=self._on_test_connection)
        test_btn.pack(side=tk.RIGHT)

        # --- API Key ---
        key_frame = ttk.LabelFrame(main_frame, text="API Key（非必填）", padding=8)
        key_frame.pack(fill=tk.X, pady=4)

        self.key_var = tk.StringVar(value=self.config.get("api_key", ""))
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, width=60, show="*")
        key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.show_key_var = tk.BooleanVar(value=False)
        show_key_cb = ttk.Checkbutton(key_frame, text="显示", variable=self.show_key_var,
                                       command=lambda: key_entry.config(show="" if self.show_key_var.get() else "*"))
        show_key_cb.pack(side=tk.RIGHT)

        # --- Model Selection ---
        model_frame = ttk.LabelFrame(main_frame, text="模型选择", padding=8)
        model_frame.pack(fill=tk.X, pady=4)

        model_row = ttk.Frame(model_frame)
        model_row.pack(fill=tk.X)

        self.model_var = tk.StringVar(value=self.config.get("model", ""))
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, width=45)
        saved_models = self.config.get("models_list", [])
        if saved_models:
            self.model_combo['values'] = saved_models
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        fetch_btn = ttk.Button(model_row, text="获取模型列表", command=self._on_fetch_models)
        fetch_btn.pack(side=tk.RIGHT)

        # --- Multi-threading ---
        thread_frame = ttk.LabelFrame(main_frame, text="多线程设置", padding=8)
        thread_frame.pack(fill=tk.X, pady=4)

        self.mt_var = tk.BooleanVar(value=self.config.get("enable_multithreading", False))
        mt_cb = ttk.Checkbutton(thread_frame, text="启用多线程并发处理图片",
                                 variable=self.mt_var, command=self._on_mt_toggle)
        mt_cb.pack(anchor=tk.W)

        thread_count_row = ttk.Frame(thread_frame)
        thread_count_row.pack(fill=tk.X, pady=(4, 0))

        ttk.Label(thread_count_row, text="并发线程数:").pack(side=tk.LEFT)
        self.thread_var = tk.IntVar(value=self.config.get("max_threads", 3))
        self.thread_spin = ttk.Spinbox(thread_count_row, from_=2, to=10,
                                        textvariable=self.thread_var, width=5)
        self.thread_spin.pack(side=tk.LEFT, padx=8)
        self.thread_spin.config(state="normal" if self.mt_var.get() else "disabled")

        # --- Status / Log ---
        log_frame = ttk.LabelFrame(main_frame, text="状态日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=70,
                                                   state=tk.DISABLED, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        cancel_btn = ttk.Button(btn_frame, text="取消", command=self._on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=4)

        confirm_btn = ttk.Button(btn_frame, text="确认并保存", command=self._on_confirm)
        confirm_btn.pack(side=tk.RIGHT, padx=4)

    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _on_mt_toggle(self):
        self.thread_spin.config(state="normal" if self.mt_var.get() else "disabled")

    def _on_test_connection(self):
        url = self.url_var.get().strip()
        key = self.key_var.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入 API 地址")
            return
        self._log(f"正在测试连接: {url} ...")

        def _do():
            ok, msg = test_connection(url, key)
            self.root.after(0, lambda: self._log(f"{'✓' if ok else '✗'} {msg}"))
            if ok:
                self.root.after(0, lambda: messagebox.showinfo("连接测试", msg))
            else:
                self.root.after(0, lambda: messagebox.showerror("连接测试", msg))

        threading.Thread(target=_do, daemon=True).start()

    def _on_fetch_models(self):
        url = self.url_var.get().strip()
        key = self.key_var.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入 API 地址")
            return
        self._log(f"正在获取模型列表 ...")

        def _do():
            ok, result = fetch_models(url, key)
            if ok:
                models = result
                self.root.after(0, lambda: self._update_models(models))
                self.root.after(0, lambda: self._log(f"✓ 获取到 {len(models)} 个模型"))
            else:
                self.root.after(0, lambda: self._log(f"✗ {result}"))
                self.root.after(0, lambda: messagebox.showerror("获取模型", result))

        threading.Thread(target=_do, daemon=True).start()

    def _update_models(self, models):
        self.model_combo['values'] = models
        if models and not self.model_var.get():
            # Auto-select first model with vision capability hints
            vision_hints = ['gpt-4o', 'gpt-4-vision', 'claude', 'gemini', 'qwen-vl', 'glm-4v']
            for m in models:
                ml = m.lower()
                if any(h in ml for h in vision_hints):
                    self.model_var.set(m)
                    break
            if not self.model_var.get():
                self.model_var.set(models[0])

    def _on_confirm(self):
        url = self.url_var.get().strip()
        model = self.model_var.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入 API 地址")
            return
        if not model:
            messagebox.showwarning("提示", "请选择或输入模型名称")
            return

        self.config = {
            "api_url": url,
            "api_key": self.key_var.get().strip(),
            "model": model,
            "models_list": list(self.model_combo['values']) if self.model_combo['values'] else [],
            "enable_multithreading": self.mt_var.get(),
            "max_threads": self.thread_var.get(),
        }
        save_config(self.config, self.workspace_dir)
        self.result = self.config
        self.root.destroy()

    def _on_cancel(self):
        self.result = None
        self.root.destroy()

    def run(self):
        """Show GUI and block until user confirms or cancels. Returns config or None."""
        self.root.mainloop()
        return self.result


def show_config_gui(workspace_dir=None):
    """Show the LLM config GUI. Returns config dict or None if cancelled."""
    gui = LLMConfigGUI(workspace_dir)
    return gui.run()


# CLI entry point for testing
if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    result = show_config_gui(workspace)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Cancelled")
        sys.exit(1)
