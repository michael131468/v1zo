#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Phase 3.5 of the macOS AI Montage Suite — The Curator.

Reads scored_snippets.json (Phase 2 / critic output) and
final_sequence.json (Phase 4 / director output), generates one preview
frame per non-discarded snippet, then writes an interactive HTML review
page (curator_review.html) that lets the user pin clips for
force-inclusion or exclude clips from future director runs, and export a
curator_overrides.json file.

Usage:
    uv run curator.py \\
        --snippets /path/to/critic/scored_snippets.json \\
        --sequence /path/to/director/final_sequence.json \\
        [--output  /path/to/curator/] \\
        [--workers 8] \\
        [--skip-frames]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console(stderr=True)

PHOTO_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".heic", ".png", ".tiff", ".tif",
    ".bmp", ".gif", ".webp", ".raw", ".dng",
})


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_preview_frame(source_file: str, time_s: float, out_path: Path) -> bool:
    """Extract a single JPEG frame from a video at time_s using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(time_s),
                "-i", source_file,
                "-frames:v", "1",
                "-q:v", "3",
                str(out_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return result.returncode == 0 and out_path.exists()
    except Exception:
        return False


def copy_photo_preview(source_file: str, out_path: Path) -> bool:
    """Copy a photo directly to out_path as the preview."""
    try:
        shutil.copy2(source_file, out_path)
        return out_path.exists()
    except Exception:
        return False


def _extract_one(snippet: dict, previews_dir: Path) -> tuple[str, str | None]:
    """Worker: extract or copy preview for one snippet. Returns (snippet_id, rel_path|None)."""
    sid = snippet.get("snippet_id", "")
    source_file = snippet.get("source_file") or snippet.get("source_file_rel") or ""
    out_path = previews_dir / f"{sid}_preview.jpg"

    # Already exists — skip
    if out_path.exists():
        return sid, f"previews/{out_path.name}"

    if not source_file:
        return sid, None

    ext = Path(source_file).suffix.lower()
    if ext in PHOTO_EXTENSIONS:
        ok = copy_photo_preview(source_file, out_path)
    else:
        bw = snippet.get("best_window") or {}
        time_s = bw.get("mid_time_s") or bw.get("start_time_s") or 0.0
        ok = extract_preview_frame(source_file, float(time_s), out_path)

    if ok:
        return sid, f"previews/{out_path.name}"
    return sid, None


def extract_all_previews(
    snippets: list[dict],
    output_dir: Path,
    workers: int,
) -> dict[str, str | None]:
    """Extract/copy previews for all snippets. Returns {snippet_id -> relative_path | None}."""
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, str | None] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting preview frames", total=len(snippets))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_one, snip, previews_dir): snip.get("snippet_id", "")
                for snip in snippets
            }
            for future in as_completed(futures):
                sid, rel = future.result()
                result[sid] = rel
                progress.advance(task)

    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_curator_html(
    snippets: list[dict],
    in_cut_ids: set[str],
    preview_map: dict[str, str | None],
    output_path: Path,
    snippets_path: Path,
) -> None:
    """Generate the self-contained curator_review.html."""

    # Attach preview paths to snippet objects for JS embedding
    js_snippets: list[dict] = []
    for s in snippets:
        sid = s.get("snippet_id", "")
        obj: dict[str, Any] = {
            "snippet_id":     sid,
            "scene_id":       s.get("scene_id"),
            "type":           s.get("type"),
            "source_file_rel": s.get("source_file_rel") or s.get("source_file") or "",
            "best_window":    s.get("best_window"),
            "scores":         s.get("scores", {}),
            "face_count":     s.get("face_count", 0),
            "discarded":      s.get("discarded", False),
            "discard_reason": s.get("discard_reason"),
            "person_ids":     s.get("person_ids", []),
            "preview_path":   preview_map.get(sid),
        }
        js_snippets.append(obj)

    n_in_cut  = sum(1 for s in js_snippets if s["snippet_id"] in in_cut_ids)
    n_not_cut = sum(1 for s in js_snippets if s["snippet_id"] not in in_cut_ids)
    gen_at    = datetime.now().strftime("%Y-%m-%d")

    snippets_json = json.dumps(js_snippets, ensure_ascii=False, separators=(",", ":"))
    in_cut_json   = json.dumps(sorted(in_cut_ids), ensure_ascii=False, separators=(",", ":"))

    # Short hash of snippets_path for localStorage key namespacing
    path_hash = hashlib.md5(str(snippets_path.resolve()).encode()).hexdigest()[:8]

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Curator Review</title>\n"
        "<style>\n"
        "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#111;color:#eee}\n"
        "header{padding:16px 24px 10px;border-bottom:1px solid #333;position:sticky;"
        "top:0;background:#111;z-index:100}\n"
        "h1{font-size:1.4rem;font-weight:600;margin-bottom:4px}\n"
        ".meta{color:#888;font-size:.8rem;margin-bottom:10px}\n"
        ".controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}\n"
        ".tab-btn{background:#222;border:1px solid #444;color:#ccc;padding:5px 14px;"
        "border-radius:20px;cursor:pointer;font-size:.8rem;transition:all .15s;font-weight:500}\n"
        ".tab-btn:hover{background:#333}\n"
        ".tab-btn.active{background:#30d158;border-color:#30d158;color:#000}\n"
        ".tab-btn.tab-notcut.active{background:#0a84ff;border-color:#0a84ff;color:#fff}\n"
        ".filter-btn{background:#222;border:1px solid #444;color:#ccc;padding:5px 12px;"
        "border-radius:20px;cursor:pointer;font-size:.8rem;transition:all .15s}\n"
        ".filter-btn:hover{background:#333}\n"
        ".filter-btn.active{background:#0a84ff;border-color:#0a84ff;color:#fff}\n"
        ".sort-select{background:#222;border:1px solid #444;color:#ccc;padding:5px 10px;"
        "border-radius:20px;font-size:.8rem;outline:none;cursor:pointer}\n"
        ".sort-select:focus{border-color:#0a84ff}\n"
        "#search-box{background:#222;border:1px solid #444;color:#eee;padding:5px 12px;"
        "border-radius:20px;font-size:.8rem;outline:none;min-width:180px;margin-left:auto}\n"
        "#search-box:focus{border-color:#0a84ff}\n"
        ".export-btn{background:#ff9f0a;border:1px solid #ff9f0a;color:#000;"
        "padding:5px 14px;border-radius:20px;cursor:pointer;font-size:.8rem;"
        "font-weight:600;transition:all .15s}\n"
        ".export-btn:hover{background:#ffb340}\n"
        ".sep{width:1px;height:20px;background:#333;margin:0 2px}\n"
        "#scene-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
        "gap:12px;padding:16px}\n"
        ".scene-card{background:#1c1c1e;border:1px solid #2c2c2e;border-radius:10px;"
        "overflow:hidden;transition:transform .15s,box-shadow .15s;position:relative;"
        "border-left:3px solid transparent}\n"
        ".scene-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.5)}\n"
        ".scene-card.in-cut{border-left-color:#30d158}\n"
        ".scene-card.pinned{border-left-color:#ffd60a!important}\n"
        ".scene-card.excluded{opacity:.4;border-left-color:#ff453a!important;"
        "border:1px solid #ff453a;border-left-width:3px}\n"
        ".card-thumb-wrap{position:relative;height:120px;background:#2c2c2e;cursor:pointer}\n"
        ".card-thumb-wrap img{width:100%;height:120px;object-fit:cover;display:block}\n"
        ".card-thumb-placeholder{width:100%;height:120px;background:#2c2c2e;"
        "display:flex;align-items:center;justify-content:center;color:#555;font-size:.75rem}\n"
        ".card-actions{position:absolute;top:6px;right:6px;display:flex;gap:4px}\n"
        ".action-btn{background:rgba(0,0,0,.6);border:none;border-radius:6px;"
        "width:28px;height:28px;cursor:pointer;font-size:.9rem;display:flex;"
        "align-items:center;justify-content:center;transition:background .15s;padding:0}\n"
        ".action-btn:hover{background:rgba(0,0,0,.85)}\n"
        ".action-btn.pin-active{background:rgba(255,214,10,.25)}\n"
        ".action-btn.excl-active{background:rgba(255,69,58,.25)}\n"
        ".card-meta{padding:7px 10px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}\n"
        ".snippet-id{font-size:.65rem;color:#666;font-family:monospace;width:100%}\n"
        ".score-wrap{display:flex;align-items:center;gap:4px}\n"
        ".score-bar-bg{background:#333;border-radius:3px;height:5px;width:60px;"
        "display:inline-block}\n"
        ".score-bar-fg{background:#0a84ff;height:5px;border-radius:3px}\n"
        ".score-val{font-size:.68rem;color:#aaa}\n"
        ".duration{font-size:.72rem;color:#aaa}\n"
        ".face-badge{font-size:.68rem;padding:2px 6px;border-radius:10px;"
        "background:#1d3557;color:#a8d8ea}\n"
        ".filename{font-size:.68rem;color:#666;white-space:nowrap;overflow:hidden;"
        "text-overflow:ellipsis;max-width:160px;width:100%}\n"
        "#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;"
        "display:flex;align-items:center;justify-content:center}\n"
        "#lightbox.hidden{display:none}\n"
        ".lb-inner{background:#1c1c1e;border-radius:14px;max-width:780px;width:90vw;"
        "max-height:90vh;overflow-y:auto;padding:24px;position:relative}\n"
        ".lb-close{position:absolute;top:12px;right:16px;background:none;border:none;"
        "color:#888;font-size:1.5rem;cursor:pointer}\n"
        ".lb-close:hover{color:#eee}\n"
        ".lb-img{width:100%;border-radius:8px;margin-bottom:16px;"
        "max-height:320px;object-fit:contain;background:#000}\n"
        ".lb-table{width:100%;border-collapse:collapse;font-size:.78rem}\n"
        ".lb-table td{padding:4px 8px;border-bottom:1px solid #2c2c2e}\n"
        ".lb-table td:first-child{color:#888;width:38%}\n"
        ".lb-section{color:#888;font-size:.72rem;text-transform:uppercase;"
        "letter-spacing:.08em;margin:14px 0 6px}\n"
        ".score-chart{display:flex;gap:6px;align-items:flex-end;height:60px;"
        "margin-top:8px}\n"
        ".score-chart-bar{display:flex;flex-direction:column;align-items:center;gap:3px;"
        "flex:1}\n"
        ".score-chart-bar-inner{width:100%;background:#0a84ff;border-radius:3px 3px 0 0;"
        "min-height:2px}\n"
        ".score-chart-label{font-size:.6rem;color:#888;text-align:center}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "<h1>Curator Review</h1>\n"
        f'<p class="meta">{n_in_cut} in cut &nbsp;&middot;&nbsp; '
        f'{n_not_cut} not in cut &nbsp;&middot;&nbsp; Generated: {gen_at}</p>\n'
        '<div class="controls">\n'
        f'<button class="tab-btn tab-incut active" data-tab="incut">In Cut ({n_in_cut})</button>\n'
        f'<button class="tab-btn tab-notcut" data-tab="notcut">Not in Cut ({n_not_cut})</button>\n'
        '<div class="sep"></div>\n'
        '<button class="filter-btn active" data-face="All">All</button>\n'
        '<button class="filter-btn" data-face="Faces">Faces</button>\n'
        '<button class="filter-btn" data-face="No Faces">No Faces</button>\n'
        '<div class="sep"></div>\n'
        '<select class="sort-select" id="sort-select">\n'
        '  <option value="score-desc">Score \u2193</option>\n'
        '  <option value="score-asc">Score \u2191</option>\n'
        '  <option value="dur-desc">Duration \u2193</option>\n'
        '  <option value="dur-asc">Duration \u2191</option>\n'
        '</select>\n'
        '<input type="search" id="search-box" placeholder="Search by filename\u2026">\n'
        '<button class="export-btn" id="export-btn">Export Overrides</button>\n'
        "</div>\n"
        "</header>\n"
        '\n<main id="scene-grid"></main>\n'
        '\n<div id="lightbox" class="hidden">\n'
        '  <div class="lb-inner">\n'
        '    <button class="lb-close" id="lb-close">&times;</button>\n'
        '    <img class="lb-img" id="lb-img" src="" alt="">\n'
        '    <p class="lb-section">Metadata</p>\n'
        '    <table class="lb-table" id="lb-table"></table>\n'
        '    <p class="lb-section">Score Breakdown</p>\n'
        '    <div class="score-chart" id="lb-scores"></div>\n'
        "  </div>\n"
        "</div>\n"
        "\n<script>\n"
        f"const SNIPPETS={snippets_json};\n"
        f"const IN_CUT_IDS=new Set({in_cut_json});\n"
        f"const PATH_HASH='{path_hash}';\n"
        "\n"
        "// State\n"
        "let pinnedIds=new Set();\n"
        "let excludedIds=new Set();\n"
        "let currentTab='incut';\n"
        "let currentFace='All';\n"
        "let currentSort='score-desc';\n"
        "let currentSearch='';\n"
        "\n"
        "// Persist state\n"
        "function saveState(){\n"
        "  localStorage.setItem('curator_pins_'+PATH_HASH,JSON.stringify([...pinnedIds]));\n"
        "  localStorage.setItem('curator_excl_'+PATH_HASH,JSON.stringify([...excludedIds]));\n"
        "}\n"
        "function loadState(){\n"
        "  try{\n"
        "    const p=localStorage.getItem('curator_pins_'+PATH_HASH);\n"
        "    if(p)pinnedIds=new Set(JSON.parse(p));\n"
        "    const e=localStorage.getItem('curator_excl_'+PATH_HASH);\n"
        "    if(e)excludedIds=new Set(JSON.parse(e));\n"
        "  }catch(err){}\n"
        "}\n"
        "\n"
        "function getScore(s){return(s.scores&&s.scores.composite!=null)?s.scores.composite:0;}\n"
        "function getDur(s){\n"
        "  const bw=s.best_window||{};\n"
        "  if(bw.start_time_s!=null&&bw.end_time_s!=null)return bw.end_time_s-bw.start_time_s;\n"
        "  return 0;\n"
        "}\n"
        "\n"
        "function filteredSorted(){\n"
        "  let list=SNIPPETS.filter(s=>{\n"
        "    if(currentTab==='incut'&&!IN_CUT_IDS.has(s.snippet_id))return false;\n"
        "    if(currentTab==='notcut'&&IN_CUT_IDS.has(s.snippet_id))return false;\n"
        "    if(currentFace==='Faces'&&!(s.face_count>0))return false;\n"
        "    if(currentFace==='No Faces'&&s.face_count>0)return false;\n"
        "    if(currentSearch){\n"
        "      const fn=(s.source_file_rel||'').toLowerCase();\n"
        "      if(!fn.includes(currentSearch))return false;\n"
        "    }\n"
        "    return true;\n"
        "  });\n"
        "  list.sort((a,b)=>{\n"
        "    if(currentSort==='score-desc')return getScore(b)-getScore(a);\n"
        "    if(currentSort==='score-asc')return getScore(a)-getScore(b);\n"
        "    if(currentSort==='dur-desc')return getDur(b)-getDur(a);\n"
        "    if(currentSort==='dur-asc')return getDur(a)-getDur(b);\n"
        "    return 0;\n"
        "  });\n"
        "  return list;\n"
        "}\n"
        "\n"
        "function renderCards(){\n"
        "  const grid=document.getElementById('scene-grid');\n"
        "  grid.innerHTML='';\n"
        "  const list=filteredSorted();\n"
        "  const frag=document.createDocumentFragment();\n"
        "  list.forEach(s=>{\n"
        "    const sid=s.snippet_id;\n"
        "    const inCut=IN_CUT_IDS.has(sid);\n"
        "    const isPinned=pinnedIds.has(sid);\n"
        "    const isExcl=excludedIds.has(sid);\n"
        "    const score=getScore(s);\n"
        "    const dur=getDur(s);\n"
        "    const bw=s.best_window||{};\n"
        "    const fn=(s.source_file_rel||'').split('/').pop();\n"
        "    const pct=Math.round(score*100);\n"
        "    let cls='scene-card';\n"
        "    if(inCut)cls+=' in-cut';\n"
        "    if(isPinned)cls+=' pinned';\n"
        "    if(isExcl)cls+=' excluded';\n"
        "    const art=document.createElement('article');\n"
        "    art.className=cls;\n"
        "    art.dataset.id=sid;\n"
        "    const thumbHtml=s.preview_path\n"
        "      ?`<img src='${s.preview_path}' loading='lazy' alt='' style='cursor:pointer'>`\n"
        "      :`<div class='card-thumb-placeholder'>No Preview</div>`;\n"
        "    const faceHtml=s.face_count>0?`<span class='face-badge'>\\u{1F464} ${s.face_count}</span>`:'';\n"
        "    const pinEmoji=isPinned?'\\u2B50':'\\u2606';\n"
        "    const pinCls='action-btn pin-btn'+(isPinned?' pin-active':'');\n"
        "    const exclCls='action-btn excl-btn'+(isExcl?' excl-active':'');\n"
        "    art.innerHTML=`<div class='card-thumb-wrap'>`\n"
        "      +thumbHtml\n"
        "      +`<div class='card-actions'>`\n"
        "      +`<button class='${pinCls}' title='Pin for force-include'>${pinEmoji}</button>`\n"
        "      +`<button class='${exclCls}' title='Exclude from director'>\\uD83D\\uDEAB</button>`\n"
        "      +`</div></div>`\n"
        "      +`<div class='card-meta'>`\n"
        "      +`<span class='snippet-id'>${sid}</span>`\n"
        "      +`<span class='score-wrap'>`\n"
        "      +`<span class='score-bar-bg'><span class='score-bar-fg' style='width:${pct}%'></span></span>`\n"
        "      +`<span class='score-val'>${score.toFixed(2)}</span>`\n"
        "      +`</span>`\n"
        "      +(dur>0?`<span class='duration'>${dur.toFixed(1)}s</span>`:'')\n"
        "      +faceHtml\n"
        "      +`<span class='filename' title='${s.source_file_rel||''}'>${fn}</span>`\n"
        "      +`</div>`;\n"
        "    // Thumb click -> lightbox\n"
        "    const thumbWrap=art.querySelector('.card-thumb-wrap');\n"
        "    if(thumbWrap)thumbWrap.addEventListener('click',()=>openLightbox(s));\n"
        "    // Pin button\n"
        "    const pinBtn=art.querySelector('.pin-btn');\n"
        "    if(pinBtn)pinBtn.addEventListener('click',e=>{\n"
        "      e.stopPropagation();\n"
        "      if(pinnedIds.has(sid)){pinnedIds.delete(sid);}else{pinnedIds.add(sid);}\n"
        "      saveState();renderCards();\n"
        "    });\n"
        "    // Exclude button\n"
        "    const exclBtn=art.querySelector('.excl-btn');\n"
        "    if(exclBtn)exclBtn.addEventListener('click',e=>{\n"
        "      e.stopPropagation();\n"
        "      if(excludedIds.has(sid)){excludedIds.delete(sid);}else{excludedIds.add(sid);}\n"
        "      saveState();renderCards();\n"
        "    });\n"
        "    frag.appendChild(art);\n"
        "  });\n"
        "  grid.appendChild(frag);\n"
        "}\n"
        "\n"
        "// Tab switching\n"
        "document.querySelectorAll('.tab-btn').forEach(btn=>{\n"
        "  btn.addEventListener('click',()=>{\n"
        "    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));\n"
        "    btn.classList.add('active');\n"
        "    currentTab=btn.dataset.tab;\n"
        "    renderCards();\n"
        "  });\n"
        "});\n"
        "\n"
        "// Face filter\n"
        "document.querySelectorAll('.filter-btn').forEach(btn=>{\n"
        "  btn.addEventListener('click',()=>{\n"
        "    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));\n"
        "    btn.classList.add('active');\n"
        "    currentFace=btn.dataset.face;\n"
        "    renderCards();\n"
        "  });\n"
        "});\n"
        "\n"
        "// Sort\n"
        "document.getElementById('sort-select').addEventListener('change',function(){\n"
        "  currentSort=this.value;\n"
        "  renderCards();\n"
        "});\n"
        "\n"
        "// Search\n"
        "document.getElementById('search-box').addEventListener('input',function(){\n"
        "  currentSearch=this.value.toLowerCase();\n"
        "  renderCards();\n"
        "});\n"
        "\n"
        "// Export\n"
        "function exportOverrides(){\n"
        "  const overrides={\n"
        "    force_include:[...pinnedIds],\n"
        "    force_exclude:[...excludedIds]\n"
        "  };\n"
        "  const blob=new Blob([JSON.stringify(overrides,null,2)],{type:'application/json'});\n"
        "  const a=document.createElement('a');\n"
        "  a.href=URL.createObjectURL(blob);\n"
        "  a.download='curator_overrides.json';\n"
        "  a.click();\n"
        "}\n"
        "document.getElementById('export-btn').addEventListener('click',exportOverrides);\n"
        "\n"
        "// Lightbox\n"
        "function openLightbox(s){\n"
        "  const img=document.getElementById('lb-img');\n"
        "  img.src=s.preview_path||'';\n"
        "  img.style.display=s.preview_path?'':'none';\n"
        "  const sc=s.scores||{};\n"
        "  const bw=s.best_window||{};\n"
        "  const dur=(bw.start_time_s!=null&&bw.end_time_s!=null)\n"
        "    ?(bw.end_time_s-bw.start_time_s).toFixed(2)+'s':'—';\n"
        "  const rows=[\n"
        "    ['Snippet ID',s.snippet_id],\n"
        "    ['Scene ID',s.scene_id||'—'],\n"
        "    ['Type',s.type||'—'],\n"
        "    ['File',s.source_file_rel||'—'],\n"
        "    ['Duration',dur],\n"
        "    ['Composite',sc.composite!=null?sc.composite.toFixed(3):'—'],\n"
        "    ['Aesthetic',sc.aesthetic!=null?sc.aesthetic.toFixed(3):'—'],\n"
        "    ['Saliency',sc.saliency_coverage!=null?sc.saliency_coverage.toFixed(3):'—'],\n"
        "    ['Smile',sc.smile!=null?sc.smile.toFixed(3):'—'],\n"
        "    ['Face Count',s.face_count!=null?s.face_count:'—'],\n"
        "    ['Person IDs',(s.person_ids&&s.person_ids.length)?s.person_ids.join(', '):'—'],\n"
        "    ['Discarded',s.discarded?'Yes':'No'],\n"
        "    ['Discard Reason',s.discard_reason||'—'],\n"
        "  ];\n"
        "  document.getElementById('lb-table').innerHTML=rows.map(([k,v])=>\n"
        "    `<tr><td>${k}</td><td>${v}</td></tr>`).join('');\n"
        "  // Score bar chart\n"
        "  const scoreKeys=['composite','aesthetic','saliency_coverage','smile'];\n"
        "  const scoreLabels=['Composite','Aesthetic','Saliency','Smile'];\n"
        "  const chartEl=document.getElementById('lb-scores');\n"
        "  chartEl.innerHTML=scoreKeys.map((k,i)=>{\n"
        "    const val=sc[k]!=null?sc[k]:0;\n"
        "    const h=Math.round(Math.max(2,val*56));\n"
        "    return `<div class='score-chart-bar'>`\n"
        "      +`<div style='font-size:.65rem;color:#aaa;margin-bottom:2px'>${val.toFixed(2)}</div>`\n"
        "      +`<div class='score-chart-bar-inner' style='height:${h}px'></div>`\n"
        "      +`<div class='score-chart-label'>${scoreLabels[i]}</div>`\n"
        "      +`</div>`;\n"
        "  }).join('');\n"
        "  document.getElementById('lightbox').classList.remove('hidden');\n"
        "}\n"
        "\n"
        "document.getElementById('lb-close').addEventListener('click',()=>{\n"
        "  document.getElementById('lightbox').classList.add('hidden');\n"
        "});\n"
        "document.getElementById('lightbox').addEventListener('click',function(e){\n"
        "  if(e.target===this)this.classList.add('hidden');\n"
        "});\n"
        "document.addEventListener('keydown',e=>{\n"
        "  if(e.key==='Escape')document.getElementById('lightbox').classList.add('hidden');\n"
        "});\n"
        "\n"
        "loadState();\n"
        "renderCards();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_curator(
    snippets_path: Path,
    sequence_path: Path,
    output_dir: Path,
    workers: int,
    skip_frames: bool,
) -> None:
    console.print("[bold]Curator[/] — Phase 3.5 of the macOS AI Montage Suite\n")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    console.print("[bold]Stage 1/3[/] Loading inputs …")

    snippets_data = json.loads(snippets_path.read_text(encoding="utf-8"))
    all_snippets: list[dict] = snippets_data.get("snippets", [])
    console.print(f"  Snippets loaded: {len(all_snippets)}")

    sequence_data = json.loads(sequence_path.read_text(encoding="utf-8"))
    in_cut_ids: set[str] = set()
    for entry in sequence_data.get("timeline", []) + sequence_data.get("memory_dump", []):
        sid = entry.get("snippet_id")
        if sid:
            in_cut_ids.add(sid)
    console.print(f"  In-cut snippet IDs: {len(in_cut_ids)}")

    # Only include non-discarded snippets in the review
    review_snippets = [s for s in all_snippets if not s.get("discarded", False)]
    console.print(f"  Non-discarded snippets for review: {len(review_snippets)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------
    if skip_frames:
        console.print("\n[bold]Stage 2/3[/] Skipping frame extraction (--skip-frames) …")
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        preview_map: dict[str, str | None] = {}
        for s in review_snippets:
            sid = s.get("snippet_id", "")
            candidate = previews_dir / f"{sid}_preview.jpg"
            preview_map[sid] = f"previews/{candidate.name}" if candidate.exists() else None
    else:
        console.print(f"\n[bold]Stage 2/3[/] Extracting preview frames ({workers} workers) …")
        preview_map = extract_all_previews(review_snippets, output_dir, workers)

    n_ok = sum(1 for v in preview_map.values() if v is not None)
    console.print(f"  Previews ready: {n_ok}/{len(review_snippets)}")

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 3/3[/] Generating HTML review …")
    html_path = output_dir / "curator_review.html"
    generate_curator_html(
        snippets=review_snippets,
        in_cut_ids=in_cut_ids,
        preview_map=preview_map,
        output_path=html_path,
        snippets_path=snippets_path,
    )

    console.print(f"\n[bold green]✓[/] curator_review.html → [cyan]{html_path}[/]")
    console.print(f"[bold green]✓[/] Previews directory   → [cyan]{output_dir / 'previews'}[/]")
    console.print(
        f"\n[bold]Done.[/] {len(review_snippets)} snippets · "
        f"{len(in_cut_ids)} in cut · {n_ok} previews"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--snippets", "-s", "snippets_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to scored_snippets.json (Phase 2 / critic output).",
)
@click.option(
    "--sequence", "-q", "sequence_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to final_sequence.json (Phase 4 / director output).",
)
@click.option(
    "--output", "-o", "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help='Output directory. Defaults to sibling "curator" dir next to --snippets.',
)
@click.option(
    "--workers",
    default=8, show_default=True,
    help="Number of parallel ffmpeg workers for frame extraction.",
)
@click.option(
    "--skip-frames",
    is_flag=True, default=False,
    help="Skip frame extraction; use any existing previews only.",
)
def main(
    snippets_path: Path,
    sequence_path: Path,
    output_dir: Path | None,
    workers: int,
    skip_frames: bool,
) -> None:
    """Curator — Phase 3.5 of the macOS AI Montage Suite.

    Generates preview frames and an interactive HTML review page for
    curating clips before or after the Director run.  Exports
    curator_overrides.json with force_include / force_exclude lists.
    """
    if output_dir is None:
        output_dir = snippets_path.parent.parent / "curator"

    run_curator(
        snippets_path=snippets_path,
        sequence_path=sequence_path,
        output_dir=output_dir,
        workers=workers,
        skip_frames=skip_frames,
    )


if __name__ == "__main__":
    main()
