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
    
    Searches both PATH and common installation directories to handle
    environments where PATH is incomplete (e.g., MCP server subprocess,
    nvm/fnm managed Node.js, GUI-launched processes on macOS).
    """
    
    def _find_executable(name):
        """Find an executable by name. Tries PATH first, then common locations."""
        which_cmd = "where" if sys.platform == "win32" else "which"
        
        # 1. Try standard PATH lookup
        try:
            result = subprocess.run(
                [which_cmd, name],
                capture_output=True, text=True, timeout=10,
                encoding='utf-8', errors='replace'
            )
            if result.returncode == 0:
                path = result.stdout.strip().split('\n')[0].strip()
                if path:
                    return path
        except Exception:
            pass
        
        # 2. Try running directly (may work if shell profile sets PATH)
        try:
            result = subprocess.run(
                [name, "--version"],
                capture_output=True, text=True, timeout=10,
                encoding='utf-8', errors='replace'
            )
            if result.returncode == 0:
                return name  # Found in some PATH
        except Exception:
            pass
        
        # 3. Search common installation directories
        home = os.path.expanduser("~")
        common_dirs = []
        
        if sys.platform == "darwin":
            # macOS common locations
            common_dirs = [
                "/usr/local/bin",
                "/opt/homebrew/bin",                          # Apple Silicon Homebrew
                os.path.join(home, ".nvm/versions/node"),     # nvm (search subdirs)
                os.path.join(home, ".fnm/node-versions"),     # fnm (search subdirs)
                os.path.join(home, ".local/share/fnm/node-versions"),  # fnm alt
                os.path.join(home, ".volta/bin"),              # volta
                os.path.join(home, ".npm-global/bin"),         # npm global custom
                "/usr/local/lib/node_modules/.bin",
            ]
        elif sys.platform != "win32":
            # Linux common locations
            common_dirs = [
                "/usr/local/bin",
                "/usr/bin",
                os.path.join(home, ".nvm/versions/node"),
                os.path.join(home, ".fnm/node-versions"),
                os.path.join(home, ".local/share/fnm/node-versions"),
                os.path.join(home, ".volta/bin"),
                os.path.join(home, ".npm-global/bin"),
                "/usr/local/lib/node_modules/.bin",
            ]
        else:
            # Windows common locations
            appdata = os.environ.get("APPDATA", "")
            common_dirs = [
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "nodejs"),
                os.path.join(appdata, "npm") if appdata else "",
                os.path.join(appdata, "nvm") if appdata else "",
                os.path.join(home, ".volta", "bin"),
            ]
        
        for d in common_dirs:
            if not d or not os.path.isdir(d):
                # For nvm/fnm, search version subdirectories
                if "nvm/versions/node" in d or "fnm" in d:
                    parent = d
                    if os.path.isdir(parent):
                        try:
                            versions = sorted(os.listdir(parent), reverse=True)
                            for v in versions:
                                bin_dir = os.path.join(parent, v, "bin")
                                candidate = os.path.join(bin_dir, name)
                                if sys.platform == "win32":
                                    for ext in [".cmd", ".exe", ""]:
                                        if os.path.isfile(candidate + ext):
                                            return candidate + ext
                                elif os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                                    return candidate
                        except Exception:
                            pass
                continue
            
            candidate = os.path.join(d, name)
            if sys.platform == "win32":
                for ext in [".cmd", ".exe", ""]:
                    if os.path.isfile(candidate + ext):
                        return candidate + ext
            else:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
            
            # Also check nvm/fnm version subdirs if this is a version root
            if "nvm/versions/node" in d or "fnm" in d:
                try:
                    versions = sorted(os.listdir(d), reverse=True)
                    for v in versions:
                        bin_dir = os.path.join(d, v, "bin")
                        candidate = os.path.join(bin_dir, name)
                        if sys.platform == "win32":
                            for ext in [".cmd", ".exe", ""]:
                                if os.path.isfile(candidate + ext):
                                    return candidate + ext
                        elif os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                            return candidate
                except Exception:
                    pass
        
        return None

    # First check Node.js
    node_path = _find_executable("node")
    if not node_path:
        return False, "Node.js 未安装。请安装 Node.js (https://nodejs.org/)，然后运行: npm install -g copilot-api\n如果已安装但未检测到，可能是 PATH 未包含 Node.js 路径（常见于 nvm/fnm 环境）。"

    # Then check copilot-api
    copilot_path = _find_executable("copilot-api")
    if not copilot_path:
        return False, f"Node.js 已找到 ({node_path})，但 copilot-api 未安装。请运行: npm install -g copilot-api"
    
    return True, copilot_path


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


def start_copilot_api(copilot_path=None):
    """Start copilot-api service in background.
    Returns (success: bool, message: str, process_or_none).
    
    Args:
        copilot_path: Full path to copilot-api executable. If None, uses "copilot-api" from PATH.
    
    Captures initial output to detect login requirements.
    """
    cmd = copilot_path or "copilot-api"
    try:
        # Start copilot-api as a background process
        if sys.platform == "win32":
            # On Windows, use CREATE_NEW_PROCESS_GROUP to detach
            proc = subprocess.Popen(
                [cmd, "start"],
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
                [cmd, "start"],
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
        return False, f"copilot-api 命令未找到 ({cmd})。请确认安装路径正确，或手动指定 API 地址。", None
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
