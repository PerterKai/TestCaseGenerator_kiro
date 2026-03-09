#!/usr/bin/env python3
"""
Copilot API helper for external multimodal LLM image analysis.
Replaces the previous GUI-based configuration with copilot-api CLI integration.

Flow:
1. Check if copilot-api is installed
2. Start copilot-api service
3. Check login status; if not logged in, extract device code for user
4. Auto-detect API URL once running
5. Use gpt-4o for image analysis
"""

import json
import os
import sys
import subprocess
import time
import re
import urllib.request
import urllib.error
import ssl
import base64
import threading

CONFIG_FILENAME = "llm_api_config.json"

# Default copilot-api endpoint
DEFAULT_COPILOT_API_URL = "http://localhost:4141"


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
        "api_url": DEFAULT_COPILOT_API_URL,
        "api_key": "",
        "model": "gpt-4o",
        "enable_multithreading": True,
        "max_threads": 8,
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

    # Allow both http and https; disable cert verification for local APIs
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


def check_copilot_api_installed():
    """Check if copilot-api CLI is installed. Also checks Node.js prerequisite.
    Returns (installed: bool, path_or_error: str).
    """
    which_cmd = "where" if sys.platform == "win32" else "which"

    # First check Node.js
    try:
        node_result = subprocess.run(
            [which_cmd, "node"],
            capture_output=True, text=True, timeout=10,
            encoding='utf-8', errors='replace'
        )
        if node_result.returncode != 0:
            return False, "Node.js 未安装，请先安装最新版 Node.js (https://nodejs.org/)，然后运行: npm install -g copilot-api"
    except Exception:
        return False, "Node.js 未安装，请先安装最新版 Node.js (https://nodejs.org/)，然后运行: npm install -g copilot-api"

    # Then check copilot-api
    try:
        result = subprocess.run(
            [which_cmd, "copilot-api"],
            capture_output=True, text=True, timeout=10,
            encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            path = result.stdout.strip().split('\n')[0].strip()
            return True, path
        return False, "copilot-api 未找到，请运行: npm install -g copilot-api"
    except Exception as e:
        return False, f"检测 copilot-api 失败: {e}"


def check_copilot_api_running(api_url=None):
    """Check if copilot-api service is running and accessible.
    Returns (running: bool, message: str).
    """
    url = (api_url or DEFAULT_COPILOT_API_URL).rstrip('/')
    models_url = url + "/v1/models"
    status, body = _http_request(models_url, timeout=5)
    if status == 200:
        return True, "copilot-api 服务运行中"
    elif status == 401:
        return False, "copilot-api 服务运行中但未认证（可能需要登录）"
    elif status == 0:
        return False, f"copilot-api 服务未运行或无法连接: {body}"
    else:
        return False, f"copilot-api 服务异常 (HTTP {status}): {body[:200]}"


def start_copilot_api():
    """Start copilot-api service in background.
    Returns (success: bool, message: str, process_or_none).
    
    Captures initial output to detect login requirements.
    """
    try:
        # Start copilot-api as a background process
        if sys.platform == "win32":
            # On Windows, use CREATE_NEW_PROCESS_GROUP to detach
            proc = subprocess.Popen(
                ["copilot-api", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            proc = subprocess.Popen(
                ["copilot-api", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

        # Read initial output for a few seconds to check for login prompts
        output_lines = []
        login_code = None
        login_url = None

        def _read_output():
            nonlocal login_code, login_url
            try:
                for line in proc.stdout:
                    line = line.rstrip('\n')
                    output_lines.append(line)
                    sys.stderr.write(f"[copilot-api] {line}\n")
                    sys.stderr.flush()
                    
                    # Detect device code patterns
                    # Common patterns: "code: XXXX-XXXX", "device code: XXXX-XXXX",
                    # "Enter code: XXXX-XXXX", "one-time code: XXXX-XXXX"
                    code_match = re.search(r'(?:code|Code)[:\s]+([A-Z0-9]{4}-[A-Z0-9]{4})', line)
                    if code_match:
                        login_code = code_match.group(1)
                    
                    # Detect URL patterns for device login
                    url_match = re.search(r'(https?://\S*github\S*device\S*)', line, re.IGNORECASE)
                    if not url_match:
                        url_match = re.search(r'(https?://\S*login\S*)', line, re.IGNORECASE)
                    if url_match:
                        login_url = url_match.group(1)
            except Exception:
                pass

        reader = threading.Thread(target=_read_output, daemon=True)
        reader.start()

        # Wait up to 8 seconds for initial output
        reader.join(timeout=8)

        # Check if process is still running (good sign)
        if proc.poll() is not None:
            # Process exited — likely an error
            reader.join(timeout=2)
            full_output = "\n".join(output_lines)
            return False, f"copilot-api 启动后立即退出 (exit {proc.returncode}):\n{full_output}", None

        if login_code:
            return False, f"NEED_LOGIN:{login_code}:{login_url or ''}", proc

        # Check if service is accessible now
        time.sleep(2)
        running, msg = check_copilot_api_running()
        if running:
            return True, "copilot-api 服务已启动", proc

        # Still starting up, give it more time
        for _ in range(5):
            time.sleep(2)
            running, msg = check_copilot_api_running()
            if running:
                return True, "copilot-api 服务已启动", proc

        full_output = "\n".join(output_lines[-20:])  # Last 20 lines
        return False, f"copilot-api 已启动但服务未就绪:\n{full_output}", proc

    except FileNotFoundError:
        return False, "copilot-api 命令未找到，请先安装最新版 Node.js，然后运行: npm install -g copilot-api", None
    except Exception as e:
        return False, f"启动 copilot-api 失败: {e}", None


def test_connection(api_url, api_key=""):
    """Test API connection. Returns (success, message)."""
    url = api_url.rstrip('/')
    models_url = url + "/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    status, body = _http_request(models_url, headers=headers)
    if status == 200:
        return True, f"连接成功 (HTTP {status})"
    elif status == 401:
        return False, f"认证失败 (HTTP 401): 请检查 API Key 或登录状态"
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
    Returns (success, result_text_or_error, token_usage_or_none).
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
        return False, f"LLM 调用失败 (HTTP {status}): {body[:500]}", None
    try:
        data = json.loads(body)
        usage = data.get("usage")
        token_usage = None
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return True, content, token_usage
        return False, f"LLM 返回了空的响应: {body[:300]}", token_usage
    except Exception as e:
        return False, f"解析 LLM 响应失败: {e}", None


# Keep backward compatibility — old code imports from gui_llm_config
# This module provides the same interface
