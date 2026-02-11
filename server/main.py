#!/usr/bin/env python3
"""
Test Case Generator MCP Server v7.0
- Phase-based workflow with file cache for cross-session resume
- Splits document reading into summary + sections to reduce context
- All state persisted to .tmp/cache/ for recovery
- v7.0: Refactored image analysis flow with position tracking and validation
"""

import sys
import json
import os
import glob
import zipfile
import re
import subprocess
import base64
import hashlib
import shutil
from io import BytesIO
from xml.etree import ElementTree as ET

# ============================================================
# Windows binary mode for stdin/stdout
# ============================================================
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# Auto-install helper
# ============================================================

def _ensure_pkg(import_name, pip_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except Exception:
            return False

# ============================================================
# MCP Protocol (stdio JSON-RPC)
# ============================================================

def send_response(rid, result):
    msg = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}, ensure_ascii=False)
    raw = msg.encode('utf-8')
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def send_error(rid, code, message):
    msg = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}, ensure_ascii=False)
    raw = msg.encode('utf-8')
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


# ============================================================
# Constants & Paths
# ============================================================

CACHE_PHASE_STATE = "phase_state.json"
CACHE_IMAGE_PROGRESS = "image_progress.json"
CACHE_TESTCASES = "testcases.json"
CACHE_DOC_SUMMARY = "doc_summary.json"

def _resolve_initial_workspace():
    """Determine workspace root directory at startup.
    
    Priority:
    1. --workspace CLI arg (set by launcher in mcp.json)
    2. KIRO_WORKSPACE env var (if Kiro sets it)
    3. cwd (fallback, may not be correct when running as installed power)
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--workspace" and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
    # Try env var
    env_ws = os.environ.get("KIRO_WORKSPACE")
    if env_ws and os.path.isdir(env_ws):
        return os.path.abspath(env_ws)
    return os.getcwd()

_INITIAL_WORKSPACE = _resolve_initial_workspace()

WORKSPACE_DIR = None
TMP_DOC_DIR = os.path.join(_INITIAL_WORKSPACE, ".tmp", "doc_mk")
TMP_PIC_DIR = os.path.join(_INITIAL_WORKSPACE, ".tmp", "picture")
TMP_CACHE_DIR = os.path.join(_INITIAL_WORKSPACE, ".tmp", "cache")


def _update_workspace(directory):
    global WORKSPACE_DIR, TMP_DOC_DIR, TMP_PIC_DIR, TMP_CACHE_DIR
    WORKSPACE_DIR = directory
    TMP_DOC_DIR = os.path.join(directory, ".tmp", "doc_mk")
    TMP_PIC_DIR = os.path.join(directory, ".tmp", "picture")
    TMP_CACHE_DIR = os.path.join(directory, ".tmp", "cache")


def _workspace():
    return WORKSPACE_DIR or _INITIAL_WORKSPACE


def _resolve_md_path(md_info):
    """Resolve full path for a markdown file entry."""
    # Support both old format (with 'path') and new format (filename only)
    if "path" in md_info and os.path.isfile(md_info["path"]):
        return md_info["path"]
    return os.path.join(TMP_DOC_DIR, md_info["name"])


def _resolve_img_path(img_info):
    """Resolve full path for an image file entry."""
    # Support both old format (with 'path') and new format (filename only)
    if "path" in img_info and os.path.isfile(img_info["path"]):
        return img_info["path"]
    return os.path.join(TMP_PIC_DIR, img_info["filename"])

# ============================================================
# Cache / Persistence Layer
# ============================================================

def _cache_path(filename):
    return os.path.join(TMP_CACHE_DIR, filename)


def _save_cache(filename, data):
    os.makedirs(TMP_CACHE_DIR, exist_ok=True)
    path = _cache_path(filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_cache(filename, default=None):
    path = _cache_path(filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else {}


def _save_phase_state(phase, status, extra=None):
    """Update phase_state.json with current progress."""
    state = _load_cache(CACHE_PHASE_STATE, {
        "current_phase": phase,
        "workspace_dir": _workspace(),
        "phases": {}
    })
    state["current_phase"] = phase
    state["workspace_dir"] = _workspace()
    state.setdefault("phases", {})
    if phase not in state["phases"]:
        state["phases"][phase] = {}
    state["phases"][phase]["status"] = status
    if extra:
        state["phases"][phase].update(extra)
    _save_cache(CACHE_PHASE_STATE, state)
    return state


def _reset_phase_state():
    """Clear all phase state for a fresh start."""
    state = {
        "current_phase": "init",
        "workspace_dir": _workspace(),
        "phases": {}
    }
    _save_cache(CACHE_PHASE_STATE, state)
    return state

# ============================================================
# Image helpers
# ============================================================

def _img_mime(ext):
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff",
            "emf": "image/emf", "wmf": "image/wmf"}.get(ext.lower().lstrip('.'), "image/png")


def _generate_image_id(doc_name, img_name):
    """Generate a stable, extension-free image ID."""
    raw = f"{doc_name}_{img_name}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    base = os.path.splitext(img_name)[0]
    return f"{base}_{short_hash}"


def _resize_image(img_data, ext):
    final_data = img_data
    final_mime = _img_mime(ext)
    try:
        from PIL import Image
        img_obj = Image.open(BytesIO(img_data))
        w, h = img_obj.size

        if w < 60 or h < 60:
            return final_data, final_mime

        MAX_DIM = 1568
        if w > MAX_DIM or h > MAX_DIM:
            ratio = min(MAX_DIM / w, MAX_DIM / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_obj = img_obj.resize((new_w, new_h), Image.LANCZOS)

        if img_obj.mode not in ('L', 'LA'):
            img_obj = img_obj.convert('L')

        buf = BytesIO()
        if img_obj.mode in ('RGBA', 'P', 'LA'):
            img_obj.save(buf, format='PNG', optimize=True)
            final_mime = "image/png"
        else:
            img_obj.save(buf, format='JPEG', quality=65)
            final_mime = "image/jpeg"
        final_data = buf.getvalue()
    except Exception:
        pass
    return final_data, final_mime

# ============================================================
# DOCX → Markdown + Images extraction (with position tracking)
# ============================================================

def _build_rid_to_media(filepath):
    rid_to_media = {}
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            rels_path = 'word/_rels/document.xml.rels'
            if rels_path in z.namelist():
                rels_root = ET.fromstring(z.read(rels_path))
                for rel in rels_root:
                    target = rel.get('Target', '')
                    rid = rel.get('Id', '')
                    if 'media/' in target:
                        rid_to_media[rid] = target.split('/')[-1]
    except Exception:
        pass
    return rid_to_media


def _find_images_in_element(elem, rid_to_media):
    A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    V_NS = 'urn:schemas-microsoft-com:vml'
    found = []
    seen = set()
    for blip in elem.iter(f'{{{A_NS}}}blip'):
        rid = blip.get(f'{{{R_NS}}}embed', '')
        if rid and rid in rid_to_media:
            name = rid_to_media[rid]
            if name not in seen:
                seen.add(name)
                found.append(name)
    for pict in elem.iter(f'{{{W_NS}}}pict'):
        for child in pict.iter():
            rid = child.get(f'{{{R_NS}}}id', '')
            if rid and rid in rid_to_media:
                name = rid_to_media[rid]
                if name not in seen:
                    seen.add(name)
                    found.append(name)
    for obj in elem.iter(f'{{{W_NS}}}object'):
        for child in obj.iter(f'{{{V_NS}}}imagedata'):
            rid = child.get(f'{{{R_NS}}}id', '')
            if rid and rid in rid_to_media:
                name = rid_to_media[rid]
                if name not in seen:
                    seen.add(name)
                    found.append(name)
    return found


def _detect_heading_level(para):
    W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    if para.style and para.style.name and para.style.name.startswith('Heading'):
        lvl_str = para.style.name.replace('Heading ', '').replace('Heading', '1')
        try:
            return int(lvl_str)
        except ValueError:
            return 1
    try:
        pPr = para._element.find(f'{W_NS}pPr')
        if pPr is not None:
            outlineLvl = pPr.find(f'{W_NS}outlineLvl')
            if outlineLvl is not None:
                val = int(outlineLvl.get(f'{W_NS}val', '-1'))
                if val >= 0:
                    return val + 1
    except Exception:
        pass
    max_size = 0
    is_bold = False
    for run in para.runs:
        if run.bold:
            is_bold = True
        if run.font.size:
            sz = run.font.size / 12700
            if sz > max_size:
                max_size = sz
    if max_size >= 22:
        return 1
    if max_size >= 17 and is_bold:
        return 1
    if max_size >= 15 and is_bold:
        return 2
    if max_size >= 14 and is_bold:
        return 3
    return 0

def _table_to_markdown(table, rid_to_media, doc_name, image_registry, image_id_counter, image_positions, current_section):
    """Convert table to markdown, tracking image positions."""
    rows = []
    for row in table.rows:
        row_data = []
        for cell in row.cells:
            cell_text = cell.text.strip().replace('\n', ' ').replace('|', '\\|')
            cell_images = []
            for para in cell.paragraphs:
                imgs = _find_images_in_element(para._element, rid_to_media)
                cell_images.extend(imgs)
            if cell_images:
                for img_name in cell_images:
                    img_id = _generate_image_id(doc_name, img_name)
                    image_registry[img_name] = img_id
                    image_id_counter[img_id] = image_id_counter.get(img_id, 0) + 1
                    count = image_id_counter[img_id]
                    placeholder_id = img_id if count == 1 else f"{img_id}__dup{count}"
                    cell_text += f" {{{{IMG:{placeholder_id}}}}}"
                    # Track position info
                    image_positions[placeholder_id] = {
                        "base_id": img_id,
                        "section": current_section,
                        "context": "table_cell",
                        "occurrence": count
                    }
            row_data.append(cell_text)
        rows.append(row_data)
    if not rows:
        return ""

    if len(rows[0]) == 1:
        content = rows[0][0].replace('\\|', '|')
        code_indicators = [
            'sequenceDiagram', 'stateDiagram', 'erDiagram', 'flowchart',
            'classDiagram', 'gantt', 'pie', 'graph ',
            'SELECT ', 'INSERT ', 'CREATE TABLE', 'ALTER TABLE', 'DROP TABLE',
            'DELETE FROM', 'UPDATE ',
        ]
        lang_hints = {
            'sequenceDiagram': 'mermaid', 'stateDiagram': 'mermaid',
            'erDiagram': 'mermaid', 'flowchart': 'mermaid',
            'classDiagram': 'mermaid', 'gantt': 'mermaid', 'pie': 'mermaid',
            'graph ': 'mermaid',
            'SELECT ': 'sql', 'INSERT ': 'sql', 'CREATE TABLE': 'sql',
            'ALTER TABLE': 'sql', 'DROP TABLE': 'sql', 'DELETE FROM': 'sql',
            'UPDATE ': 'sql',
        }
        for indicator in code_indicators:
            if indicator in content:
                lang = lang_hints.get(indicator, '')
                raw_text = table.rows[0].cells[0].text.strip()
                return f"```{lang}\n{raw_text}\n```"

        code_patterns = [
            ('Java ', 'java'), ('JSON ', 'json'), ('XML ', 'xml'),
            ('Plaintext ', 'text'), ('String ', 'java'),
            ('public ', 'java'), ('private ', 'java'),
            ('if (', 'java'), ('if(', 'java'),
        ]
        for pattern, lang in code_patterns:
            if content.startswith(pattern) or content.startswith(pattern.lower()):
                raw_text = table.rows[0].cells[0].text.strip()
                return f"```{lang}\n{raw_text}\n```"

    md_lines = []
    md_lines.append("| " + " | ".join(rows[0]) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for row in rows[1:]:
        while len(row) < len(rows[0]):
            row.append("")
        md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
    return "\n".join(md_lines)

def convert_docx_to_markdown(filepath):
    """Convert docx to markdown with image position tracking."""
    doc_name = os.path.splitext(os.path.basename(filepath))[0]
    rid_to_media = _build_rid_to_media(filepath)
    image_registry = {}
    image_data_map = {}
    image_id_counter = {}
    image_positions = {}  # Track position info for each placeholder
    md_lines = [f"# {doc_name}", ""]
    current_section = doc_name  # Track current section heading

    try:
        from docx import Document
        doc = Document(filepath)
    except ImportError:
        return _convert_docx_raw(filepath)
    except Exception:
        return _convert_docx_raw(filepath)

    for child in doc.element.body:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'p':
            para = None
            for p in doc.paragraphs:
                if p._element is child:
                    para = p
                    break
            if para is None:
                continue
            text = para.text.strip()
            para_images = _find_images_in_element(para._element, rid_to_media)
            if text:
                level = _detect_heading_level(para)
                if level > 0:
                    current_section = text  # Update current section
                    md_lines.append("")
                    md_lines.append(f"{'#' * (level + 1)} {text}")
                    md_lines.append("")
                else:
                    md_lines.append(text)
                    md_lines.append("")
            if para_images:
                for img_name in para_images:
                    img_id = _generate_image_id(doc_name, img_name)
                    image_registry[img_name] = img_id
                    image_id_counter[img_id] = image_id_counter.get(img_id, 0) + 1
                    count = image_id_counter[img_id]
                    placeholder_id = img_id if count == 1 else f"{img_id}__dup{count}"
                    # Record line number where placeholder will be inserted
                    line_num = len(md_lines)
                    md_lines.append(f"{{{{IMG:{placeholder_id}}}}}")
                    md_lines.append("")
                    # Track position info
                    image_positions[placeholder_id] = {
                        "base_id": img_id,
                        "section": current_section,
                        "context": "paragraph",
                        "line_num": line_num,
                        "occurrence": count
                    }
        elif tag == 'tbl':
            tbl = None
            for t in doc.tables:
                if t._element is child:
                    tbl = t
                    break
            if tbl is not None:
                md_lines.append("")
                table_md = _table_to_markdown(tbl, rid_to_media, doc_name, image_registry, 
                                              image_id_counter, image_positions, current_section)
                if table_md:
                    md_lines.append(table_md)
                    md_lines.append("")

    # Extract and filter images
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            media = [n for n in z.namelist() if n.startswith('word/media/')]
            skipped_ids = []
            for img_path in media:
                name = os.path.basename(img_path)
                if name in image_registry:
                    img_id = image_registry[name]
                    ext = os.path.splitext(name)[1].lstrip('.')
                    img_data = z.read(img_path)
                    if len(img_data) >= 500 and ext.lower() not in ('emf', 'wmf'):
                        try:
                            from PIL import Image as _Img
                            _tmp = _Img.open(BytesIO(img_data))
                            _w, _h = _tmp.size
                            if _w < 60 or _h < 60:
                                skipped_ids.append(img_id)
                                continue
                        except Exception:
                            pass
                        image_data_map[img_id] = (img_data, ext)
                    else:
                        skipped_ids.append(img_id)
            if skipped_ids:
                md_text = "\n".join(md_lines)
                for sid in skipped_ids:
                    placeholder = f"{{{{IMG:{sid}}}}}"
                    annotation = f"<!-- 已跳过小图标: {sid} -->"
                    md_text = md_text.replace(placeholder, annotation)
                    dup_pat = re.escape(f"{{{{IMG:{sid}__dup") + r"\d+" + re.escape("}}")
                    md_text = re.sub(dup_pat, annotation, md_text)
                    # Remove skipped images from position tracking
                    for pid in list(image_positions.keys()):
                        if pid == sid or pid.startswith(f"{sid}__dup"):
                            del image_positions[pid]
                md_lines = md_text.split("\n")
    except Exception:
        pass

    return "\n".join(md_lines), image_registry, image_data_map, image_positions

def _convert_docx_raw(filepath):
    """Fallback raw XML parsing with position tracking."""
    doc_name = os.path.splitext(os.path.basename(filepath))[0]
    rid_to_media = _build_rid_to_media(filepath)
    image_registry = {}
    image_data_map = {}
    image_id_counter = {}
    image_positions = {}
    md_lines = [f"# {doc_name}", ""]
    current_section = doc_name

    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            if 'word/document.xml' in z.namelist():
                root = ET.fromstring(z.read('word/document.xml'))
                body = root.find(f'{W}body')
                if body is None:
                    body = root
                for child in body:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag == 'p':
                        texts = [t.text for t in child.iter(f'{W}t') if t.text]
                        text = ''.join(texts).strip()
                        is_heading = False
                        if text:
                            pPr = child.find(f'{W}pPr')
                            if pPr is not None:
                                pStyle = pPr.find(f'{W}pStyle')
                                if pStyle is not None:
                                    sv = pStyle.get(f'{W}val', '')
                                    if 'Heading' in sv or 'heading' in sv:
                                        is_heading = True
                                        m = re.search(r'\d+', sv)
                                        level = int(m.group()) if m else 1
                                        current_section = text
                                        md_lines.append("")
                                        md_lines.append(f"{'#' * (level + 1)} {text}")
                                        md_lines.append("")
                            if not is_heading:
                                md_lines.append(text)
                                md_lines.append("")
                        para_images = _find_images_in_element(child, rid_to_media)
                        for img_name in para_images:
                            img_id = _generate_image_id(doc_name, img_name)
                            image_registry[img_name] = img_id
                            image_id_counter[img_id] = image_id_counter.get(img_id, 0) + 1
                            count = image_id_counter[img_id]
                            placeholder_id = img_id if count == 1 else f"{img_id}__dup{count}"
                            line_num = len(md_lines)
                            md_lines.append(f"{{{{IMG:{placeholder_id}}}}}")
                            md_lines.append("")
                            image_positions[placeholder_id] = {
                                "base_id": img_id,
                                "section": current_section,
                                "context": "paragraph",
                                "line_num": line_num,
                                "occurrence": count
                            }
                    elif tag == 'tbl':
                        rows = []
                        for tr in child.iter(f'{W}tr'):
                            row = []
                            for tc in tr.iter(f'{W}tc'):
                                cell_text = ''.join(t.text for t in tc.iter(f'{W}t') if t.text).strip()
                                cell_images = _find_images_in_element(tc, rid_to_media)
                                for img_name in cell_images:
                                    img_id = _generate_image_id(doc_name, img_name)
                                    image_registry[img_name] = img_id
                                    image_id_counter[img_id] = image_id_counter.get(img_id, 0) + 1
                                    count = image_id_counter[img_id]
                                    placeholder_id = img_id if count == 1 else f"{img_id}__dup{count}"
                                    cell_text += f" {{{{IMG:{placeholder_id}}}}}"
                                    image_positions[placeholder_id] = {
                                        "base_id": img_id,
                                        "section": current_section,
                                        "context": "table_cell",
                                        "occurrence": count
                                    }
                                row.append(cell_text.replace('|', '\\|'))
                            if row:
                                rows.append(row)
                        if rows:
                            md_lines.append("")
                            md_lines.append("| " + " | ".join(rows[0]) + " |")
                            md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                            for row in rows[1:]:
                                while len(row) < len(rows[0]):
                                    row.append("")
                                md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
                            md_lines.append("")
            media = [n for n in z.namelist() if n.startswith('word/media/')]
            skipped_ids = []
            for img_path in media:
                name = os.path.basename(img_path)
                if name in image_registry:
                    img_id = image_registry[name]
                    ext = os.path.splitext(name)[1].lstrip('.')
                    img_data = z.read(img_path)
                    if len(img_data) >= 500 and ext.lower() not in ('emf', 'wmf'):
                        try:
                            from PIL import Image as _Img
                            _tmp = _Img.open(BytesIO(img_data))
                            _w, _h = _tmp.size
                            if _w < 60 or _h < 60:
                                skipped_ids.append(img_id)
                                continue
                        except Exception:
                            pass
                        image_data_map[img_id] = (img_data, ext)
                    else:
                        skipped_ids.append(img_id)
            if skipped_ids:
                md_text = "\n".join(md_lines)
                for sid in skipped_ids:
                    placeholder = f"{{{{IMG:{sid}}}}}"
                    annotation = f"<!-- 已跳过小图标: {sid} -->"
                    md_text = md_text.replace(placeholder, annotation)
                    dup_pat = re.escape(f"{{{{IMG:{sid}__dup") + r"\d+" + re.escape("}}")
                    md_text = re.sub(dup_pat, annotation, md_text)
                    for pid in list(image_positions.keys()):
                        if pid == sid or pid.startswith(f"{sid}__dup"):
                            del image_positions[pid]
                md_lines = md_text.split("\n")
    except Exception as e:
        md_lines.append(f"\n> 解析错误: {e}\n")

    return "\n".join(md_lines), image_registry, image_data_map, image_positions

# ============================================================
# In-memory Store (backed by cache files)
# ============================================================

testcase_store = {
    "modules": [],
    "pending_images": [],
    "md_files": [],
    "session_image_count": 0,
    "_current_image_id": None,
}


def _sync_store_to_cache():
    """Persist critical store data to cache files."""
    _save_cache(CACHE_IMAGE_PROGRESS, {
        "pending_images": testcase_store["pending_images"],
        "md_files": testcase_store["md_files"],
    })
    _save_cache(CACHE_TESTCASES, {
        "modules": testcase_store["modules"],
    })


def _restore_store_from_cache():
    """Restore store from cache files (for session resume).
    
    Handles both old format (absolute 'path' field) and new format (filename only).
    """
    img_data = _load_cache(CACHE_IMAGE_PROGRESS)
    if img_data:
        testcase_store["pending_images"] = img_data.get("pending_images", [])
        testcase_store["md_files"] = img_data.get("md_files", [])

    tc_data = _load_cache(CACHE_TESTCASES)
    if tc_data:
        testcase_store["modules"] = tc_data.get("modules", [])


# ============================================================
# Document Section Parser
# ============================================================

def _parse_md_sections(md_content):
    """Parse markdown into sections by headings."""
    lines = md_content.split('\n')
    sections = []
    current = None
    in_code_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith('#') and not stripped.startswith('#!'):
            hashes = len(stripped) - len(stripped.lstrip('#'))
            title = stripped.lstrip('#').strip()
            if not title:
                continue
            if current:
                current["end"] = i
                current["char_count"] = sum(len(lines[j]) for j in range(current["start"], i))
            current = {"heading": title, "level": hashes, "start": i, "end": len(lines), "char_count": 0}
            sections.append(current)

    if current:
        current["end"] = len(lines)
        current["char_count"] = sum(len(lines[j]) for j in range(current["start"], current["end"]))

    if not sections:
        sections.append({
            "heading": "(全文)",
            "level": 1,
            "start": 0,
            "end": len(lines),
            "char_count": len(md_content)
        })

    return sections


def _build_doc_summary():
    """Build summary of all markdown docs."""
    md_files = testcase_store.get("md_files", [])
    summary = {"documents": [], "total_chars": 0, "total_sections": 0}

    for md_info in md_files:
        try:
            with open(_resolve_md_path(md_info), 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        sections = _parse_md_sections(content)
        doc_entry = {
            "name": md_info["name"],
            "total_chars": len(content),
            "sections": []
        }
        for sec in sections:
            doc_entry["sections"].append({
                "heading": sec["heading"],
                "level": sec["level"],
                "char_count": sec["char_count"],
                "line_start": sec["start"],
                "line_end": sec["end"],
            })
        summary["documents"].append(doc_entry)
        summary["total_chars"] += len(content)
        summary["total_sections"] += len(sections)

    _save_cache(CACHE_DOC_SUMMARY, summary)
    return summary

# ============================================================
# XMind Export
# ============================================================

def _esc(text):
    if not text:
        return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def create_xmind_file(modules, output_path):
    counter = [0]
    def nid():
        counter[0] += 1
        return f"t{counter[0]}"
    def topic(title, children=None):
        x = f'<topic id="{nid()}"><title>{_esc(title)}</title>'
        if children:
            x += '<children><topics type="attached">' + ''.join(children) + '</topics></children>'
        return x + '</topic>'

    mod_topics = []
    for m in modules:
        sub_topics = []
        for s in m.get("sub_modules", []):
            case_topics = []
            for c in s.get("test_cases", []):
                inner = None
                if c.get("expected_result"):
                    inner = topic(f"预期结果: {c['expected_result']}")
                steps = c.get("steps", [])
                if steps:
                    steps_text = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
                    inner = topic(f"执行步骤:\n{steps_text}", [inner] if inner else None)
                if c.get("preconditions"):
                    inner = topic(f"前置条件: {c['preconditions']}", [inner] if inner else None)
                case_topics.append(topic(c.get("title", "未命名用例"), [inner] if inner else None))
            sub_topics.append(topic(s.get("name", ""), case_topics))
        mod_topics.append(topic(m.get("name", ""), sub_topics))

    content_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<xmap-content xmlns="urn:xmind:xmap:xmlns:content:2.0"
  xmlns:fo="http://www.w3.org/1999/XSL/Format"
  xmlns:svg="http://www.w3.org/2000/svg"
  xmlns:xhtml="http://www.w3.org/1999/xhtml"
  xmlns:xlink="http://www.w3.org/1999/xlink" version="2.0">
  <sheet id="sheet_1"><title>测试用例</title>{topic("测试用例", mod_topics)}</sheet>
</xmap-content>'''

    manifest_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<manifest xmlns="urn:xmind:xmap:xmlns:manifest:1.0">
  <file-entry full-path="content.xml" media-type="text/xml"/>
  <file-entry full-path="META-INF/" media-type=""/>
  <file-entry full-path="META-INF/manifest.xml" media-type="text/xml"/>
</manifest>'''

    meta_xml = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<meta xmlns="urn:xmind:xmap:xmlns:meta:2.0" version="2.0">'
                '<Author><Name>TestCase Generator</Name></Author></meta>')

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('content.xml', content_xml)
        zf.writestr('META-INF/manifest.xml', manifest_xml)
        zf.writestr('meta.xml', meta_xml)
    return output_path

# ============================================================
# MCP Tool Definitions
# ============================================================

TOOLS = [
    {
        "name": "setup_environment",
        "description": "启动检查: 1) 检查并安装Python依赖 2) 检查并创建工作目录 3) 检测缓存任务。如检测到缓存任务，返回 has_cache=true，agent 需询问用户是继续上次任务还是开始新任务。",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "clear_cache",
        "description": "清除所有缓存任务数据，用于开始全新的用例生成任务。",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "parse_documents",
        "description": "Parse .docx files: convert to markdown, extract images with position tracking. Returns file list and pending image count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory containing .docx files (default: cwd)"},
                "file_patterns": {"type": "string", "description": "Glob pattern (default: *.docx)"},
                "force": {"type": "boolean", "description": "Force re-parse (default: false)"}
            },
            "required": []
        }
    },
    {
        "name": "get_pending_image",
        "description": "Get the next unprocessed image for vision analysis. Returns image with position context (section, line number).",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "submit_image_result",
        "description": "Submit vision analysis result. Writes result to markdown at the correct position, replacing placeholder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "The unique image ID"},
                "analysis": {"type": "string", "description": "The analysis result text"}
            },
            "required": ["image_id", "analysis"]
        }
    },
    {
        "name": "verify_image_positions",
        "description": "Verify all image analysis results are in correct positions in markdown files. Returns validation report.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_workflow_state",
        "description": "Get current workflow state for session resume.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Workspace directory (default: cwd)"}
            },
            "required": []
        }
    },
    {
        "name": "get_doc_summary",
        "description": "Get document structure summary (heading tree + char counts).",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_doc_section",
        "description": "Read a specific section of a markdown document by heading name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_name": {"type": "string", "description": "Markdown filename"},
                "section_heading": {"type": "string", "description": "Heading text to match"},
                "include_subsections": {"type": "boolean", "description": "Include child sections (default: true)"}
            },
            "required": ["doc_name"]
        }
    },
    {
        "name": "get_parsed_markdown",
        "description": "Read all processed markdown files. WARNING: may be large.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "save_testcases",
        "description": "Save test cases. Supports incremental save via append_module.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "modules": {"type": "array", "description": "Test case module list", "items": {"type": "object"}},
                "append_module": {"type": "object", "description": "Single module to append"}
            },
            "required": []
        }
    },
    {
        "name": "get_testcases",
        "description": "Get all current test cases.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "export_xmind",
        "description": "Export test cases to XMind format.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Output file path"},
                "requirement_name": {"type": "string", "description": "Requirement name for file naming"}
            },
            "required": []
        }
    },
    {
        "name": "review_module_structure",
        "description": "Review test case module structure for quality issues.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "export_report",
        "description": "Generate test case report as markdown file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "requirement_name": {"type": "string", "description": "Requirement name"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "questions": {"type": "array", "items": {"type": "string"}, "description": "Requirement questions"}
            },
            "required": []
        }
    }
]

# ============================================================
# Tool Handlers
# ============================================================

def handle_setup_environment(args):
    results = []
    all_ok = True
    results.append(f"Python {sys.version.split()[0]}")

    deps = {"docx": "python-docx", "PIL": "Pillow"}
    for imp, pip_name in deps.items():
        try:
            __import__(imp)
            results.append(f"  [ok] {pip_name}")
        except ImportError:
            results.append(f"  [installing] {pip_name}...")
            if _ensure_pkg(imp, pip_name):
                results.append(f"  [ok] {pip_name} installed")
            else:
                results.append(f"  [FAIL] {pip_name}")
                all_ok = False

    workspace = _workspace()
    dirs_to_check = {
        "doc": os.path.join(workspace, "doc"),
        ".tmp/doc_mk": TMP_DOC_DIR,
        ".tmp/picture": TMP_PIC_DIR,
        ".tmp/cache": TMP_CACHE_DIR,
    }
    results.append("")
    results.append("工作目录检查:")
    for label, dir_path in dirs_to_check.items():
        if os.path.isdir(dir_path):
            results.append(f"  [ok] {label}/")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                results.append(f"  [created] {label}/")
            except Exception as e:
                results.append(f"  [FAIL] {label}/ — {e}")
                all_ok = False

    has_cache = False
    cache_info = {}
    existing_state = _load_cache(CACHE_PHASE_STATE)
    if existing_state and existing_state.get("phases"):
        has_cache = True
        _restore_store_from_cache()
        pending = testcase_store.get("pending_images", [])
        total_imgs = len(pending)
        processed_imgs = sum(1 for p in pending if p["processed"])
        modules = testcase_store.get("modules", [])
        total_cases = sum(len(s.get("test_cases", [])) for m in modules for s in m.get("sub_modules", []))
        current_phase = existing_state.get("current_phase", "unknown")

        cache_info = {
            "current_phase": current_phase,
            "total_images": total_imgs,
            "processed_images": processed_imgs,
            "unprocessed_images": total_imgs - processed_imgs,
            "module_count": len(modules),
            "total_cases": total_cases,
        }
        results.append("")
        results.append("⚠️ 检测到缓存任务:")
        results.append(f"  当前阶段: {current_phase}")
        if total_imgs > 0:
            results.append(f"  图片处理: {processed_imgs}/{total_imgs}")
        if modules:
            results.append(f"  已生成用例: {len(modules)} 模块, {total_cases} 用例")

    results.append("")
    results.append("OK - environment ready" if all_ok else "WARN - some deps failed")
    return {
        "content": [{"type": "text", "text": "\n".join(results)}],
        "all_ok": all_ok,
        "has_cache": has_cache,
        "cache_info": cache_info,
    }


def handle_clear_cache(args):
    """Clear all cached task data."""
    cleared = []

    for cache_file in (CACHE_PHASE_STATE, CACHE_IMAGE_PROGRESS, CACHE_TESTCASES, CACHE_DOC_SUMMARY):
        cache_fp = _cache_path(cache_file)
        if os.path.exists(cache_fp):
            os.remove(cache_fp)
            cleared.append(cache_file)

    if os.path.isdir(TMP_DOC_DIR):
        shutil.rmtree(TMP_DOC_DIR)
        os.makedirs(TMP_DOC_DIR, exist_ok=True)
        cleared.append(".tmp/doc_mk/*")

    if os.path.isdir(TMP_PIC_DIR):
        shutil.rmtree(TMP_PIC_DIR)
        os.makedirs(TMP_PIC_DIR, exist_ok=True)
        cleared.append(".tmp/picture/*")

    testcase_store["modules"] = []
    testcase_store["pending_images"] = []
    testcase_store["md_files"] = []
    testcase_store["session_image_count"] = 0

    msg = "✓ 缓存已清除:\n  " + "\n  ".join(cleared) if cleared else "没有需要清除的缓存。"
    return {"content": [{"type": "text", "text": msg}]}

def handle_parse_documents(args):
    """Parse documents with image position tracking."""
    directory = args.get("directory", _workspace())
    pattern = args.get("file_patterns", "*.docx")
    force = args.get("force", False)
    _update_workspace(directory)

    doc_dir = os.path.join(directory, "doc")
    search_dir = doc_dir if os.path.isdir(doc_dir) else directory

    if not force:
        existing_state = _load_cache(CACHE_PHASE_STATE)
        if existing_state:
            phases = existing_state.get("phases", {})
            img_phase = phases.get("image_analysis", {})
            if img_phase.get("status") == "in_progress":
                processed = img_phase.get("processed", 0)
                total = img_phase.get("total", 0)
                return {"content": [{"type": "text", "text": (
                    f"⚠️ 检测到未完成的图片处理进度 ({processed}/{total})。\n"
                    f"如确认要重新开始，请传入 force=true 参数。"
                )}], "blocked": True}

    files = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
    if not files:
        files = glob.glob(os.path.join(search_dir, pattern))
    if not files:
        return {"content": [{"type": "text", "text": f"No .docx files found in {search_dir}"}]}

    for d in (TMP_DOC_DIR, TMP_PIC_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    os.makedirs(TMP_CACHE_DIR, exist_ok=True)

    for cache_file in (CACHE_PHASE_STATE, CACHE_IMAGE_PROGRESS, CACHE_TESTCASES, CACHE_DOC_SUMMARY):
        cache_fp = _cache_path(cache_file)
        if os.path.exists(cache_fp):
            os.remove(cache_fp)

    testcase_store["modules"] = []
    testcase_store["pending_images"] = []
    testcase_store["md_files"] = []
    testcase_store["session_image_count"] = 0

    _reset_phase_state()

    all_md_files = []
    all_pending_images = []
    content_parts = []

    for fpath in files:
        try:
            doc_name = os.path.splitext(os.path.basename(fpath))[0]
            md_text, image_registry, image_data_map, image_positions = convert_docx_to_markdown(fpath)

            md_filename = f"{doc_name}.md"
            md_path = os.path.join(TMP_DOC_DIR, md_filename)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            # Store only filename, resolve full path dynamically via _workspace()
            all_md_files.append({"name": md_filename})

            skipped = 0
            for img_id, (img_data, ext) in image_data_map.items():
                resized_data, mime = _resize_image(img_data, ext)
                out_ext = ".png" if "png" in mime else ".jpg"
                img_filename = img_id + out_ext
                img_path = os.path.join(TMP_PIC_DIR, img_filename)
                with open(img_path, 'wb') as f:
                    f.write(resized_data)
                
                # Find position info for this image
                pos_info = None
                for pid, pinfo in image_positions.items():
                    if pinfo.get("base_id") == img_id and pinfo.get("occurrence", 1) == 1:
                        pos_info = pinfo
                        break
                
                all_pending_images.append({
                    "id": img_id, 
                    "filename": img_filename, 
                    "mime": mime, 
                    "size": len(img_data), 
                    "source_doc": doc_name,
                    "processed": False,
                    "position": pos_info or {"section": doc_name, "context": "unknown"}
                })

            for orig_name, uid in image_registry.items():
                if uid not in image_data_map:
                    skipped += 1

            content_parts.append({
                "type": "text",
                "text": (f"✓ {os.path.basename(fpath)} → {md_filename}\n"
                         f"  图片: {len(image_data_map)} 张提取, {skipped} 张跳过")
            })
        except Exception as e:
            content_parts.append({"type": "text", "text": f"✗ Error parsing {fpath}: {e}"})

    testcase_store["pending_images"] = all_pending_images
    testcase_store["md_files"] = all_md_files

    _save_phase_state("parse", "completed", {
        "file_count": len(all_md_files),
        "total_images": len(all_pending_images)
    })
    _sync_store_to_cache()

    total_imgs = len(all_pending_images)
    summary = (f"\n转换完成:\n"
               f"  Markdown 文件: {len(all_md_files)} 个 → {TMP_DOC_DIR}\n"
               f"  图片文件: {total_imgs} 张 → {TMP_PIC_DIR}\n")
    if total_imgs > 0:
        summary += f"\n请调用 get_pending_image 开始处理图片。"
    else:
        summary += "\n无需处理图片，可直接调用 get_doc_summary。"

    content_parts.append({"type": "text", "text": summary})
    return {"content": content_parts, "file_count": len(all_md_files), "total_images": total_imgs}

IMAGE_ANALYSIS_PROMPT = (
    "你是一位资深测试开发专家，正在从需求文档中提取测试用例设计所需的信息。\n"
    "请先判断这张图片属于以下哪种类型，然后按对应规则提取具体内容：\n\n"
    "1. 数据表/字段定义 → 用 markdown 表格逐行提取每个字段的：字段名、数据类型、长度、是否必填、默认值、描述。\n"
    "2. 流程图/状态机 → 列出所有节点和转换条件，用 A --[条件]--> B 格式描述每条路径。\n"
    "3. ER图/架构图 → 列出所有实体及属性，标注实体间关系和外键。\n"
    "4. UI界面/原型图 → 列出所有表单字段、按钮、表格列头及示例数据。\n"
    "5. 接口/参数定义 → 用 markdown 表格逐个提取参数名、类型、是否必填、取值范围、描述。\n"
    "6. 其他 → 提取所有可见文字和关键信息。\n\n"
    "【输出格式】先用一行标注图片类型，然后输出提取的具体内容。\n"
    "【核心原则】只提取具体数据，禁止笼统概括。"
)


def handle_get_pending_image(args):
    """Get next pending image with position context."""
    pending = testcase_store.get("pending_images", [])
    if not pending:
        _restore_store_from_cache()
        pending = testcase_store.get("pending_images", [])
    if not pending:
        return {"content": [{"type": "text", "text": "No documents parsed yet. Call parse_documents first."}]}

    current_lock = testcase_store.get("_current_image_id")
    if current_lock:
        for img in pending:
            if img["id"] == current_lock and not img["processed"]:
                return {"content": [{"type": "text", "text": (
                    f"⚠️ 图片 {current_lock} 正在等待分析结果提交。\n"
                    f"请先调用 submit_image_result(image_id=\"{current_lock}\", analysis=\"...\") 提交结果。"
                )}]}

    next_img = None
    for img in pending:
        if not img["processed"]:
            next_img = img
            break

    if next_img is None:
        testcase_store["_current_image_id"] = None
        _save_phase_state("image_analysis", "completed")
        return {
            "content": [{"type": "text", "text": "所有图片已处理完毕！请调用 verify_image_positions 验证图片位置，然后调用 get_doc_summary 获取文档结构。"}],
            "all_processed": True
        }

    testcase_store["_current_image_id"] = next_img["id"]

    total = len(pending)
    processed = sum(1 for p in pending if p["processed"])
    remaining = total - processed - 1

    try:
        img_path = _resolve_img_path(next_img)
        with open(img_path, 'rb') as f:
            img_data = f.read()
        b64 = base64.b64encode(img_data).decode('ascii')
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error reading image {next_img.get('filename', next_img.get('path', '?'))}: {e}"}]}

    # Include position context in the prompt
    pos_info = next_img.get("position", {})
    section = pos_info.get("section", "未知章节")
    context = pos_info.get("context", "unknown")
    
    text_info = (
        f"[{processed + 1}/{total}] 图片ID: {next_img['id']}\n"
        f"📍 位置: {section} ({context})\n"
        f"📄 来源文档: {next_img['source_doc']}\n\n"
        f"{IMAGE_ANALYSIS_PROMPT}\n\n"
        f"分析完成后调用 submit_image_result(image_id=\"{next_img['id']}\", analysis=\"你的分析结果\")"
    )

    content_parts = [
        {"type": "text", "text": text_info},
        {"type": "image", "data": b64, "mimeType": next_img["mime"]}
    ]

    if remaining > 0:
        content_parts.append({"type": "text", "text": f"提交后还剩 {remaining} 张待处理。"})

    return {
        "content": content_parts,
        "image_id": next_img["id"],
        "position": pos_info,
        "total_images": total,
        "processed_count": processed,
        "remaining": remaining + 1,
    }

def handle_submit_image_result(args):
    """Submit image analysis result with position validation."""
    image_id = args.get("image_id", "")
    analysis = args.get("analysis", "")

    if not image_id:
        return {"content": [{"type": "text", "text": "Missing required parameter: image_id"}]}
    if not analysis:
        return {"content": [{"type": "text", "text": "Missing required parameter: analysis"}]}

    current_lock = testcase_store.get("_current_image_id")
    if current_lock and image_id != current_lock:
        return {"content": [{"type": "text", "text": (
            f"❌ 提交的 image_id={image_id} 与当前正在处理的图片 {current_lock} 不匹配！"
        )}]}

    pending = testcase_store.get("pending_images", [])
    if not pending:
        _restore_store_from_cache()
        pending = testcase_store.get("pending_images", [])

    target = None
    for img in pending:
        if img["id"] == image_id:
            target = img
            break

    if target is None:
        return {"content": [{"type": "text", "text": f"Image ID not found: {image_id}"}]}
    if target["processed"]:
        return {"content": [{"type": "text", "text": f"Image {image_id} already processed."}]}

    source_doc = target["source_doc"]
    md_path = os.path.join(TMP_DOC_DIR, f"{source_doc}.md")

    if not os.path.exists(md_path):
        return {"content": [{"type": "text", "text": f"Markdown file not found: {md_path}"}]}

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        placeholder = f"{{{{IMG:{image_id}}}}}"
        
        # Create replacement with position marker for verification
        pos_info = target.get("position", {})
        section = pos_info.get("section", "未知")
        replacement = (
            f"<!-- 图片分析开始: {image_id} | 章节: {section} -->\n"
            f"{analysis}\n"
            f"<!-- 图片分析结束: {image_id} -->"
        )

        if placeholder in md_content:
            # Find the line number where placeholder exists (for verification)
            lines = md_content.split('\n')
            placeholder_line = -1
            for i, line in enumerate(lines):
                if placeholder in line:
                    placeholder_line = i
                    break
            
            md_content = md_content.replace(placeholder, replacement, 1)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            target["processed"] = True
            target["result_line"] = placeholder_line
        else:
            # Try matching __dup variants
            found = False
            dup_match = re.search(re.escape(f"{{{{IMG:{image_id}__dup") + r"\d+" + re.escape("}}"), md_content)
            if dup_match:
                dup_replacement = (
                    f"<!-- 图片分析开始: {image_id} (重复引用) | 章节: {section} -->\n"
                    f"{analysis}\n"
                    f"<!-- 图片分析结束: {image_id} -->"
                )
                md_content = md_content[:dup_match.start()] + dup_replacement + md_content[dup_match.end():]
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                target["processed"] = True
                found = True
            if not found:
                target["processed"] = True
                return {"content": [{"type": "text", "text": f"Warning: placeholder not found. Marked as processed."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error updating markdown: {e}"}]}

    _sync_store_to_cache()
    testcase_store["session_image_count"] += 1
    testcase_store["_current_image_id"] = None
    
    total = len(pending)
    processed = sum(1 for p in pending if p["processed"])
    remaining = total - processed
    
    _save_phase_state("image_analysis", "in_progress", {
        "total": total,
        "processed": processed,
    })

    msg = f"✓ 已将图片 [{image_id}] 的分析结果写入 {source_doc}.md ({processed}/{total} 已处理)"
    if remaining > 0:
        msg += f"\n请继续调用 get_pending_image 获取下一张图片。"
    else:
        msg += "\n所有图片已处理完毕！请调用 verify_image_positions 验证位置正确性。"

    return {
        "content": [{"type": "text", "text": msg}],
        "processed_count": processed,
        "total_images": total,
        "remaining": remaining
    }

def handle_verify_image_positions(args):
    """Verify all image analysis results are in correct positions."""
    pending = testcase_store.get("pending_images", [])
    if not pending:
        _restore_store_from_cache()
        pending = testcase_store.get("pending_images", [])
    
    if not pending:
        return {"content": [{"type": "text", "text": "没有图片需要验证。"}]}
    
    results = []
    errors = []
    warnings = []
    
    # Group images by source document
    by_doc = {}
    for img in pending:
        doc = img["source_doc"]
        if doc not in by_doc:
            by_doc[doc] = []
        by_doc[doc].append(img)
    
    for doc_name, images in by_doc.items():
        md_path = os.path.join(TMP_DOC_DIR, f"{doc_name}.md")
        if not os.path.exists(md_path):
            errors.append(f"❌ 文档不存在: {doc_name}.md")
            continue
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            lines = md_content.split('\n')
        except Exception as e:
            errors.append(f"❌ 读取文档失败: {doc_name}.md - {e}")
            continue
        
        results.append(f"\n📄 {doc_name}.md:")
        
        for img in images:
            img_id = img["id"]
            pos_info = img.get("position", {})
            expected_section = pos_info.get("section", "未知")
            
            if not img["processed"]:
                warnings.append(f"  ⚠️ {img_id}: 未处理")
                continue
            
            # Find the analysis block in markdown
            start_marker = f"<!-- 图片分析开始: {img_id}"
            end_marker = f"<!-- 图片分析结束: {img_id} -->"
            
            start_line = -1
            end_line = -1
            actual_section = None
            
            for i, line in enumerate(lines):
                if start_marker in line:
                    start_line = i
                    # Extract section from marker
                    if "章节:" in line:
                        actual_section = line.split("章节:")[1].split("-->")[0].strip()
                if end_marker in line:
                    end_line = i
                    break
            
            if start_line == -1:
                # Check for unprocessed placeholder
                placeholder = f"{{{{IMG:{img_id}}}}}"
                if placeholder in md_content:
                    errors.append(f"  ❌ {img_id}: 占位符未被替换")
                else:
                    errors.append(f"  ❌ {img_id}: 分析结果未找到")
            else:
                # Verify section context
                if actual_section and actual_section != expected_section:
                    warnings.append(f"  ⚠️ {img_id}: 章节可能不匹配 (期望: {expected_section}, 实际: {actual_section})")
                else:
                    results.append(f"  ✓ {img_id}: 位置正确 (行 {start_line+1}, 章节: {expected_section})")
    
    # Summary
    total = len(pending)
    processed = sum(1 for p in pending if p["processed"])
    
    summary_lines = [
        "=" * 50,
        "📊 图片位置验证报告",
        "=" * 50,
        f"总计: {total} 张图片, {processed} 张已处理",
    ]
    
    if errors:
        summary_lines.append(f"\n❌ 错误 ({len(errors)}):")
        summary_lines.extend(errors)
    
    if warnings:
        summary_lines.append(f"\n⚠️ 警告 ({len(warnings)}):")
        summary_lines.extend(warnings)
    
    summary_lines.extend(results)
    
    if not errors and not warnings:
        summary_lines.append("\n✅ 所有图片分析结果位置正确！")
    
    return {
        "content": [{"type": "text", "text": "\n".join(summary_lines)}],
        "total_images": total,
        "processed": processed,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "all_valid": len(errors) == 0
    }

def handle_get_workflow_state(args):
    """Return current workflow state for session resume."""
    directory = args.get("directory", _workspace())
    _update_workspace(directory)

    _restore_store_from_cache()

    state = _load_cache(CACHE_PHASE_STATE)
    if not state:
        return {"content": [{"type": "text", "text": "没有找到已保存的工作流状态。"}], "has_state": False}

    pending = testcase_store.get("pending_images", [])
    total_imgs = len(pending)
    processed_imgs = sum(1 for p in pending if p["processed"])
    unprocessed_imgs = total_imgs - processed_imgs

    md_files = testcase_store.get("md_files", [])
    modules = testcase_store.get("modules", [])
    total_cases = sum(len(s.get("test_cases", [])) for m in modules for s in m.get("sub_modules", []))

    phases = state.get("phases", {})
    lines = ["📋 工作流状态恢复:", ""]

    parse_status = phases.get("parse", {}).get("status", "pending")
    lines.append(f"  阶段1 文档解析: {parse_status}")
    if md_files:
        lines.append(f"    - {len(md_files)} 个 Markdown 文件")

    img_status = phases.get("image_analysis", {}).get("status", "pending")
    if img_status == "in_progress" and total_imgs > 0 and unprocessed_imgs == 0:
        img_status = "completed"
        _save_phase_state("image_analysis", "completed")
    lines.append(f"  阶段2 图片分析: {img_status}")
    if total_imgs > 0:
        lines.append(f"    - {processed_imgs}/{total_imgs} 张已处理")

    gen_status = phases.get("generation", {}).get("status", "pending")
    lines.append(f"  阶段3 用例生成: {gen_status}")
    if modules:
        lines.append(f"    - {len(modules)} 个模块, {total_cases} 个用例")

    export_status = phases.get("export", {}).get("status", "pending")
    lines.append(f"  阶段4 导出: {export_status}")

    lines.append("")
    if unprocessed_imgs > 0:
        lines.append(f"▶ 继续: 调用 get_pending_image 处理剩余 {unprocessed_imgs} 张图片")
    elif (img_status == "completed" or total_imgs == 0) and not modules:
        lines.append("▶ 继续: 调用 get_doc_summary 获取文档结构")
    elif modules and export_status != "completed":
        lines.append("▶ 继续: 调用 export_xmind 和 export_report 导出")
    else:
        lines.append("▶ 工作流已完成")

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "has_state": True,
        "current_phase": state.get("current_phase"),
        "unprocessed_images": unprocessed_imgs,
    }


def handle_get_doc_summary(args):
    """Get document structure summary."""
    if not testcase_store.get("md_files"):
        _restore_store_from_cache()

    md_files = testcase_store.get("md_files", [])
    if not md_files:
        return {"content": [{"type": "text", "text": "No markdown files found."}]}

    summary = _build_doc_summary()

    lines = ["📚 文档结构概览", ""]
    for doc in summary["documents"]:
        lines.append(f"📄 {doc['name']} ({doc['total_chars']} 字符)")
        for sec in doc["sections"]:
            indent = "  " * sec["level"]
            lines.append(f"{indent}{'#' * sec['level']} {sec['heading']} ({sec['char_count']} 字符)")
        lines.append("")

    lines.append(f"总计: {len(summary['documents'])} 个文档, {summary['total_sections']} 个章节, {summary['total_chars']} 字符")

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "summary": summary,
    }

def handle_get_doc_section(args):
    """Read a specific section of a markdown document."""
    doc_name = args.get("doc_name", "")
    section_heading = args.get("section_heading", "")
    include_sub = args.get("include_subsections", True)

    if not doc_name:
        return {"content": [{"type": "text", "text": "Missing required parameter: doc_name"}]}

    if not testcase_store.get("md_files"):
        _restore_store_from_cache()

    md_files = testcase_store.get("md_files", [])
    md_path = None
    for md_info in md_files:
        if md_info["name"] == doc_name or doc_name in md_info["name"]:
            md_path = _resolve_md_path(md_info)
            break

    if not md_path:
        available = [m["name"] for m in md_files]
        return {"content": [{"type": "text", "text": f"Document not found: {doc_name}\nAvailable: {available}"}]}

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error reading {md_path}: {e}"}]}

    if not section_heading:
        return {"content": [{"type": "text", "text": f"FILE: {doc_name}\n\n{content}"}]}

    lines = content.split('\n')
    sections = _parse_md_sections(content)

    target_sec = None
    for sec in sections:
        if section_heading in sec["heading"] or sec["heading"] in section_heading:
            target_sec = sec
            break
    if not target_sec:
        section_lower = section_heading.lower()
        for sec in sections:
            if section_lower in sec["heading"].lower() or sec["heading"].lower() in section_lower:
                target_sec = sec
                break

    if not target_sec:
        available_headings = [s["heading"] for s in sections]
        return {"content": [{"type": "text", "text": (
            f"Section not found: '{section_heading}'\nAvailable: {available_headings}"
        )}]}

    start = target_sec["start"]
    if include_sub:
        end = len(lines)
        for sec in sections:
            if sec["start"] > start and sec["level"] <= target_sec["level"]:
                end = sec["start"]
                break
    else:
        end = target_sec["end"]

    section_content = '\n'.join(lines[start:end])

    return {
        "content": [{"type": "text", "text": f"SECTION: {target_sec['heading']}\n\n{section_content}"}],
        "section_heading": target_sec["heading"],
        "char_count": len(section_content),
    }


def handle_get_parsed_markdown(args):
    """Read all markdown files."""
    if not testcase_store.get("md_files"):
        _restore_store_from_cache()

    md_files = testcase_store.get("md_files", [])
    if not md_files:
        return {"content": [{"type": "text", "text": "No markdown files found."}]}

    content_parts = []
    total_chars = 0

    for md_info in md_files:
        try:
            resolved_path = _resolve_md_path(md_info)
            with open(resolved_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            total_chars += len(md_content)
            content_parts.append({
                "type": "text",
                "text": f"\n{'='*60}\nFILE: {md_info['name']}\n{'='*60}\n\n{md_content}"
            })
        except Exception as e:
            content_parts.append({"type": "text", "text": f"Error reading {md_info['name']}: {e}"})

    summary = f"\n共 {len(md_files)} 个文档 ({total_chars} 字符)。"
    if total_chars > 30000:
        summary += "\n⚠️ 文档较大，建议使用 get_doc_summary + get_doc_section 分段读取。"

    content_parts.append({"type": "text", "text": summary})
    return {"content": content_parts, "file_count": len(md_files), "total_chars": total_chars}

def handle_save_testcases(args):
    """Save test cases with incremental support."""
    modules = args.get("modules", None)
    append_module = args.get("append_module", None)

    if append_module:
        if not isinstance(append_module, dict):
            return {"content": [{"type": "text", "text": "Error: append_module must be a JSON object."}]}
        if "name" not in append_module:
            return {"content": [{"type": "text", "text": "Error: append_module must have a 'name' field."}]}
        if "sub_modules" not in append_module:
            append_module["sub_modules"] = []

        if not testcase_store["modules"]:
            _restore_store_from_cache()
        
        mod_name = append_module.get("name", "")
        replaced = False
        for i, existing in enumerate(testcase_store["modules"]):
            if existing.get("name") == mod_name:
                testcase_store["modules"][i] = append_module
                replaced = True
                break
        if not replaced:
            testcase_store["modules"].append(append_module)
        _sync_store_to_cache()
        _save_phase_state("generation", "in_progress", {"module_count": len(testcase_store["modules"])})
        total = sum(len(s.get("test_cases", [])) for m in testcase_store["modules"] for s in m.get("sub_modules", []))
        action = "替换" if replaced else "追加"
        return {"content": [{"type": "text", "text": f"✓ {action}模块 '{mod_name}', 共 {len(testcase_store['modules'])} 个模块, {total} 个用例"}]}

    if modules is None:
        return {"content": [{"type": "text", "text": "Missing parameter: modules or append_module."}]}

    if not isinstance(modules, list):
        return {"content": [{"type": "text", "text": "Error: modules must be a JSON array."}]}

    for i, m in enumerate(modules):
        if not isinstance(m, dict):
            return {"content": [{"type": "text", "text": f"Error: modules[{i}] must be a JSON object."}]}
        if "name" not in m:
            return {"content": [{"type": "text", "text": f"Error: modules[{i}] must have a 'name' field."}]}
        if "sub_modules" not in m:
            m["sub_modules"] = []

    testcase_store["modules"] = modules
    _sync_store_to_cache()
    _save_phase_state("generation", "completed", {"module_count": len(modules)})

    total = sum(len(s.get("test_cases", [])) for m in modules for s in m.get("sub_modules", []))
    return {"content": [{"type": "text", "text": f"Saved {len(modules)} modules, {total} test cases."}]}


def handle_get_testcases(args):
    """Get all current test cases."""
    if not testcase_store["modules"]:
        _restore_store_from_cache()
    total = sum(len(s.get("test_cases", [])) for m in testcase_store["modules"] for s in m.get("sub_modules", []))
    return {
        "content": [{"type": "text", "text": json.dumps(testcase_store["modules"], ensure_ascii=False, indent=2)}],
        "module_count": len(testcase_store["modules"]),
        "total_cases": total,
    }


def _get_requirement_name():
    """Extract requirement name from parsed documents."""
    md_files = testcase_store.get("md_files", [])
    if not md_files:
        _restore_store_from_cache()
        md_files = testcase_store.get("md_files", [])
    for md_info in md_files:
        name = md_info.get("name", "")
        name = os.path.splitext(name)[0]
        if "需求" in name or "requirement" in name.lower():
            name = re.sub(r'^\[.*?\]', '', name).strip()
            if name:
                return name
    if md_files:
        name = os.path.splitext(md_files[0].get("name", "test_cases"))[0]
        name = re.sub(r'^\[.*?\]', '', name).strip()
        return name or "test_cases"
    return "test_cases"

def handle_review_module_structure(args):
    """Review test case module structure."""
    if not testcase_store["modules"]:
        _restore_store_from_cache()
    modules = testcase_store["modules"]
    if not modules:
        return {"content": [{"type": "text", "text": "没有测试用例可供审查。"}]}

    issues = []
    suggestions = []
    stats = []

    for m in modules:
        subs = m.get("sub_modules", [])
        if not subs:
            issues.append(f"⚠️ 模块 '{m['name']}' 没有子模块")
        for s in subs:
            cases = s.get("test_cases", [])
            if not cases:
                issues.append(f"⚠️ 子模块 '{m['name']} > {s['name']}' 没有用例")

    module_sizes = []
    for m in modules:
        total = sum(len(s.get("test_cases", [])) for s in m.get("sub_modules", []))
        module_sizes.append((m["name"], total))
        stats.append(f"  📦 {m['name']}: {len(m.get('sub_modules', []))} 子模块, {total} 用例")

    if module_sizes:
        max_name, max_size = max(module_sizes, key=lambda x: x[1])
        min_name, min_size = min(module_sizes, key=lambda x: x[1])

        if max_size > 0 and min_size > 0 and max_size / max(min_size, 1) > 5:
            suggestions.append(f"💡 模块大小不均衡: '{max_name}'({max_size}) vs '{min_name}'({min_size})")

        for m in modules:
            for s in m.get("sub_modules", []):
                case_count = len(s.get("test_cases", []))
                if case_count > 15:
                    suggestions.append(f"💡 子模块 '{m['name']} > {s['name']}' 有 {case_count} 个用例，建议拆分")

    mod_names = [m["name"] for m in modules]
    seen_names = {}
    for name in mod_names:
        key = name.strip().lower()
        if key in seen_names:
            issues.append(f"⚠️ 重复模块名: '{name}'")
        seen_names[key] = name

    missing_preconditions = 0
    missing_expected = 0
    empty_steps = 0
    for m in modules:
        for s in m.get("sub_modules", []):
            for c in s.get("test_cases", []):
                if not c.get("preconditions", "").strip():
                    missing_preconditions += 1
                if not c.get("expected_result", "").strip():
                    missing_expected += 1
                if not c.get("steps") or all(not step.strip() for step in c.get("steps", [])):
                    empty_steps += 1

    if missing_preconditions > 0:
        issues.append(f"⚠️ {missing_preconditions} 个用例缺少前置条件")
    if missing_expected > 0:
        issues.append(f"⚠️ {missing_expected} 个用例缺少预期结果")
    if empty_steps > 0:
        issues.append(f"⚠️ {empty_steps} 个用例缺少执行步骤")

    total_cases = sum(s[1] for s in module_sizes)
    lines = [f"📊 模块结构审查报告", "", f"总计: {len(modules)} 个模块, {total_cases} 个用例", "", "模块统计:"]
    lines.extend(stats)

    if issues:
        lines.append(f"\n发现 {len(issues)} 个问题:")
        lines.extend(issues)

    if suggestions:
        lines.append(f"\n优化建议:")
        lines.extend(suggestions)

    if not issues and not suggestions:
        lines.append("\n✅ 模块结构合理。")

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "module_count": len(modules),
        "total_cases": total_cases,
        "issue_count": len(issues),
    }

def handle_export_report(args):
    """Generate test case report."""
    if not testcase_store["modules"]:
        _restore_store_from_cache()
    modules = testcase_store["modules"]
    if not modules:
        return {"content": [{"type": "text", "text": "没有测试用例可供生成报告。"}]}

    req_name = args.get("requirement_name") or _get_requirement_name()
    output_dir = args.get("output_dir", _workspace())
    questions = args.get("questions", [])

    total_cases = sum(len(s.get("test_cases", [])) for m in modules for s in m.get("sub_modules", []))
    total_subs = sum(len(m.get("sub_modules", [])) for m in modules)

    lines = [
        f"# 测试用例生成报告", "",
        f"## 基本信息", "",
        f"| 项目 | 内容 |",
        f"|------|------|",
        f"| 需求名称 | {req_name} |",
        f"| 模块数量 | {len(modules)} |",
        f"| 子模块数量 | {total_subs} |",
        f"| 用例总数 | {total_cases} |",
        "", f"## 用例覆盖概览", "",
    ]

    for m in modules:
        subs = m.get("sub_modules", [])
        mod_total = sum(len(s.get("test_cases", [])) for s in subs)
        lines.append(f"### {m['name']} ({mod_total} 个用例)")
        lines.append("")
        lines.append(f"| 子模块 | 用例数 |")
        lines.append(f"|--------|--------|")
        for s in subs:
            lines.append(f"| {s['name']} | {len(s.get('test_cases', []))} |")
        lines.append("")

    if questions:
        lines.append(f"## 需求疑问点")
        lines.append("")
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    lines.append("---")
    lines.append("*报告由 TestCase Generator 自动生成*")

    report_content = "\n".join(lines)
    report_filename = f"{req_name}_testCaseReport.md"
    report_path = os.path.join(output_dir, report_filename)

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return {
            "content": [{"type": "text", "text": f"✓ 报告已生成: {report_path}"}],
            "report_path": report_path,
        }
    except Exception as e:
        return {"content": [{"type": "text", "text": f"报告生成失败: {e}"}]}


def handle_export_xmind(args):
    """Export test cases to XMind."""
    if not testcase_store["modules"]:
        _restore_store_from_cache()
    if not testcase_store["modules"]:
        return {"content": [{"type": "text", "text": "No test cases to export."}]}

    req_name = args.get("requirement_name") or _get_requirement_name()
    default_filename = f"{req_name}_testCase.xmind"
    p = args.get("output_path", os.path.join(_workspace(), default_filename))

    try:
        create_xmind_file(testcase_store["modules"], p)
        total = sum(len(s.get("test_cases", [])) for m in testcase_store["modules"] for s in m.get("sub_modules", []))
        _save_phase_state("export", "completed")
        return {"content": [{"type": "text", "text": f"Exported: {p}\n{len(testcase_store['modules'])} modules, {total} cases"}],
                "xmind_path": p}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Export failed: {e}"}]}

# ============================================================
# MCP Main Loop
# ============================================================

HANDLERS = {
    "setup_environment": handle_setup_environment,
    "clear_cache": handle_clear_cache,
    "parse_documents": handle_parse_documents,
    "get_pending_image": handle_get_pending_image,
    "submit_image_result": handle_submit_image_result,
    "verify_image_positions": handle_verify_image_positions,
    "get_workflow_state": handle_get_workflow_state,
    "get_doc_summary": handle_get_doc_summary,
    "get_doc_section": handle_get_doc_section,
    "get_parsed_markdown": handle_get_parsed_markdown,
    "save_testcases": handle_save_testcases,
    "get_testcases": handle_get_testcases,
    "export_xmind": handle_export_xmind,
    "review_module_structure": handle_review_module_structure,
    "export_report": handle_export_report,
}


def handle_request(req):
    method = req.get("method", "")
    rid = req.get("id")
    params = req.get("params", {})

    sys.stderr.write(f"[MCP] Received: method={method} id={rid}\n")
    sys.stderr.flush()

    # Ensure workspace is set from cache if not yet initialized
    if WORKSPACE_DIR is None:
        cached_state = _load_cache(CACHE_PHASE_STATE)
        if cached_state and cached_state.get("workspace_dir"):
            cached_ws = cached_state["workspace_dir"]
            if os.path.isdir(cached_ws):
                _update_workspace(cached_ws)

    if method == "initialize":
        send_response(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "testcase-generator", "version": "7.0.0"}
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send_response(rid, {"tools": TOOLS})
    elif method == "tools/call":
        name = params.get("name", "")
        handler = HANDLERS.get(name)
        if handler:
            try:
                result = handler(params.get("arguments", {}))
                send_response(rid, result)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                sys.stderr.write(f"[MCP] Tool error in {name}: {tb}\n")
                sys.stderr.flush()
                send_response(rid, {"content": [{"type": "text", "text": f"Error in {name}: {e}"}], "isError": True})
        else:
            send_error(rid, -32601, f"Unknown tool: {name}")
    elif method == "ping":
        send_response(rid, {})
    else:
        if rid is not None:
            send_error(rid, -32601, f"Method not found: {method}")


def main():
    sys.stderr.write("TestCase Generator MCP Server v7.0 starting...\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.buffer.readline()
            if not line:
                return

            stripped = line.strip()
            if not stripped:
                continue

            decoded = stripped.decode('utf-8', errors='replace')

            if decoded.startswith('{'):
                try:
                    request = json.loads(decoded)
                    handle_request(request)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"JSON parse error: {e}\n")
                    sys.stderr.flush()
            elif decoded.lower().startswith('content-length'):
                try:
                    cl = int(decoded.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    continue
                sys.stdin.buffer.readline()
                body = b""
                while len(body) < cl:
                    chunk = sys.stdin.buffer.read(cl - len(body))
                    if not chunk:
                        break
                    body += chunk
                if body:
                    try:
                        request = json.loads(body.decode('utf-8'))
                        handle_request(request)
                    except json.JSONDecodeError as e:
                        sys.stderr.write(f"JSON parse error: {e}\n")
                        sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Error in main loop: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
