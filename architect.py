#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.0",
#   "rich>=13.0",
#   "pillow>=10.0",
# ]
# ///
"""Phase 5 of the macOS AI Montage Suite — The Architect.

Reads final_sequence.json (Phase 4) and produces a montage.fcpxml file that
can be imported directly into Final Cut Pro or DaVinci Resolve.

Usage:
    uv run architect.py --sequence ./director_output/final_sequence.json \\
                        --beats    ./maestro_output/beat_grid.json       \\
                        [OPTIONS]
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import urllib.parse
import hashlib
import uuid
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from rich.table import Table

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ArchitectError(Exception):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FCPXML_VERSION   = "1.11"
TIMEBASE         = 2500      # universal rational denominator
DEFAULT_WIDTH    = 1920
DEFAULT_HEIGHT   = 1080
DEFAULT_FPS_NUM  = 30000
DEFAULT_FPS_DEN  = 1001      # ~29.97 fps
DEFAULT_FPS_LABEL = "2997"

# Keyword thresholds (must match director.py)
KW_AI_PICKS_MIN_SCORE  = 0.70
KW_HIGH_SMILES_MIN     = 0.50

# Still-image file extensions — these assets have no audio stream and must be
# placed as connected <video> clips over a <gap> rather than as <asset-clip>
# elements directly in the spine.  FCP's audio-preflight crashes at
# addAssetClip:toObject:parentFormatID: when it encounters an <asset-clip>
# whose asset has hasAudio="0".
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".heic", ".png", ".tiff", ".tif",
    ".bmp", ".gif", ".webp", ".raw", ".dng", ".mp",
}


def is_still_image(path: str) -> bool:
    """Return True if path is a still image (no audio stream expected)."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Title card image renderer
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFCompact.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Futura.ttc",
]


def render_title_card(
    title: str,
    subtitle: str,
    width: int,
    height: int,
    output_path: Path,
    overlay_opacity: float = 0.45,
) -> None:
    """Render a title card PNG using Pillow. Black background with centred text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Overlay
    if overlay_opacity > 0:
        overlay = Image.new("RGBA", (width, height),
                            (0, 0, 0, int(overlay_opacity * 255)))
        img.paste(overlay, mask=overlay)

    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    title_size    = round(height * 0.075)
    subtitle_size = round(height * 0.038)
    title_font    = _load_font(title_size)
    subtitle_font = _load_font(subtitle_size)

    gap = round(height * 0.025)

    def _text_size(text: str, font: Any) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    t_w, t_h = _text_size(title, title_font)

    if subtitle:
        s_w, s_h = _text_size(subtitle, subtitle_font)
        total_h = t_h + gap + s_h
        t_y = (height - total_h) // 2
        s_y = t_y + t_h + gap
        t_x = (width - t_w) // 2
        s_x = (width - s_w) // 2
        draw.text((t_x + 3, t_y + 3), title,    font=title_font,    fill=(0, 0, 0, 180))
        draw.text((t_x,     t_y),     title,    font=title_font,    fill=(255, 255, 255))
        draw.text((s_x + 2, s_y + 2), subtitle, font=subtitle_font, fill=(0, 0, 0, 180))
        draw.text((s_x,     s_y),     subtitle, font=subtitle_font, fill=(230, 230, 230))
    else:
        t_x = (width - t_w) // 2
        t_y = (height - t_h) // 2
        draw.text((t_x + 3, t_y + 3), title, font=title_font, fill=(0, 0, 0, 180))
        draw.text((t_x,     t_y),     title, font=title_font, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_checks(output_dir: Path) -> str:
    """Verify ffprobe is available, create output dir. Returns ffprobe path."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        ffprobe_bin = shutil.which("ffprobe")  # may be in ffmpeg bundle
    if ffprobe_bin is None:
        raise SystemExit(
            "ffprobe not found. Install with: brew install ffmpeg\n"
            "(ffprobe is included with ffmpeg)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return ffprobe_bin

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_final_sequence(path: Path) -> dict:
    if not path.exists():
        raise ArchitectError(f"final_sequence.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "timeline" not in data:
        raise ArchitectError("final_sequence.json missing 'timeline' key")
    return data


def resolve_audio_path(
    audio_flag: Path | None,
    beats_path: Path | None,
    sequence_data: dict,
) -> Path | None:
    """Determine path to master_mix audio file from CLI flags or beat_grid.json."""
    if audio_flag and audio_flag.exists():
        return audio_flag

    if beats_path and beats_path.exists():
        try:
            bg = json.loads(beats_path.read_text(encoding="utf-8"))
            mix_path = bg.get("metadata", {}).get("master_mix_path")
            if mix_path:
                p = Path(mix_path)
                if p.exists():
                    return p
                # Try relative to beat_grid dir
                p2 = beats_path.parent / p.name
                if p2.exists():
                    return p2
        except Exception:
            pass

    # Try to infer from sequence metadata
    scored = sequence_data.get("metadata", {}).get("scored_snippets_json", "")
    if scored:
        candidate = Path(scored).parent / "master_mix.mp3"
        if candidate.exists():
            return candidate

    return None

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def rational(seconds: float) -> str:
    """Convert seconds to FCPXML rational time string using TIMEBASE."""
    if seconds == 0.0 or seconds < 0.0:
        return "0s"
    n = round(seconds * TIMEBASE)
    if n == 0:
        return "0s"
    g = math.gcd(n, TIMEBASE)
    num, den = n // g, TIMEBASE // g
    if den == 1:
        return f"{num}s"
    return f"{num}/{den}s"


def rational_pair(num: int, den: int) -> str:
    """Format integer numerator/denominator as simplified FCPXML rational."""
    if den == 0 or num == 0:
        return "0s"
    g = math.gcd(abs(num), abs(den))
    n, d = num // g, den // g
    if d == 1:
        return f"{n}s"
    return f"{n}/{d}s"


def file_uri(path: Path | str) -> str:
    """Convert absolute path to file:// URI with proper encoding."""
    p = Path(path).resolve()
    # quote everything except the leading slash(es) and colon
    encoded = urllib.parse.quote(str(p), safe="/:")
    return f"file://{encoded}"


def make_uid(path: str | None = None) -> str:
    """Return a stable, deterministic UID for a media asset.

    Using the resolved absolute path as the seed means the same file always
    gets the same UID across architect.py runs, so FCP's library won't reject
    re-imports with 'different unique identifier' errors.
    """
    if path:
        return hashlib.md5(str(Path(path).resolve()).encode()).hexdigest().upper()
    return uuid.uuid4().hex.upper()

# ---------------------------------------------------------------------------
# Media probing
# ---------------------------------------------------------------------------

def _default_probe() -> dict:
    return {
        "duration_s":     0.0,
        "width":          DEFAULT_WIDTH,
        "height":         DEFAULT_HEIGHT,
        "fps_num":        DEFAULT_FPS_NUM,
        "fps_den":        DEFAULT_FPS_DEN,
        "has_video":      True,
        "has_audio":      False,
        "audio_channels": 2,
        "audio_rate":     44100,
    }


def probe_media(path: Path | str, ffprobe_bin: str) -> dict:
    """Probe a media file with ffprobe. Returns dict with format/stream info."""
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except Exception as exc:
        console.print(f"[yellow]ffprobe failed for {Path(path).name}: {exc}[/]")
        return _default_probe()

    streams = data.get("streams", [])
    fmt     = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    try:
        duration_s = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0

    has_video = video_stream is not None
    has_audio = audio_stream is not None

    width  = int(video_stream.get("width",  DEFAULT_WIDTH))  if has_video else 0
    height = int(video_stream.get("height", DEFAULT_HEIGHT)) if has_video else 0

    fps_num, fps_den = DEFAULT_FPS_NUM, DEFAULT_FPS_DEN
    if has_video:
        r = video_stream.get("r_frame_rate", f"{DEFAULT_FPS_NUM}/{DEFAULT_FPS_DEN}")
        try:
            parts = r.split("/")
            fps_num = int(parts[0])
            fps_den = int(parts[1]) if len(parts) > 1 else 1
        except (ValueError, IndexError):
            pass

    audio_channels = int(audio_stream.get("channels",    2))     if has_audio else 2
    audio_rate     = int(audio_stream.get("sample_rate", 44100)) if has_audio else 44100

    return {
        "duration_s":     duration_s,
        "width":          width,
        "height":         height,
        "fps_num":        fps_num,
        "fps_den":        fps_den,
        "has_video":      has_video,
        "has_audio":      has_audio,
        "audio_channels": audio_channels,
        "audio_rate":     audio_rate,
    }


def probe_all(
    source_files: list[str],
    audio_path: Path | None,
    ffprobe_bin: str,
    workers: int,
) -> dict[str, dict]:
    """Probe all unique source files in parallel. Returns {abs_path_str: probe_dict}."""
    # Normalise all paths to absolute strings so that lookups are consistent
    # regardless of whether the caller used a relative or absolute path.
    paths_to_probe = [str(Path(p).resolve()) for p in source_files]
    if audio_path:
        paths_to_probe.append(str(audio_path.resolve()))

    cache: dict[str, dict] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Probing media …", total=len(paths_to_probe))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(probe_media, p, ffprobe_bin): p
                for p in paths_to_probe
            }
            for future in as_completed(futures):
                p = futures[future]
                try:
                    cache[p] = future.result()
                except Exception as exc:
                    console.print(f"[yellow]Probe error for {Path(p).name}: {exc}[/]")
                    cache[p] = _default_probe()
                progress.advance(task)

    return cache


def detect_format_info(probe_cache: dict[str, dict]) -> dict:
    """Derive <format> element parameters from the first probed video file.

    Prefers files that have both video and audio (real video clips) over
    still images, which also report has_video=True but have no audio and
    often odd portrait dimensions.
    """
    # Two-pass: prefer video+audio clips (real video), fall back to video-only
    for want_audio in (True, False):
        for probe in probe_cache.values():
            if not (probe["has_video"] and probe["width"] > 0):
                continue
            if want_audio and not probe["has_audio"]:
                continue

            fps_num   = probe["fps_num"]
            fps_den   = probe["fps_den"]
            fps_float = fps_num / fps_den if fps_den else 30.0
            w = probe["width"]
            h = probe["height"]

            # Map common frame rates to standard labels + exact frame durations
            if   abs(fps_float - 23.976) < 0.01:
                label = "2398"; fd_n, fd_d = 1001, 24000
            elif abs(fps_float - 24.0)   < 0.01:
                label = "24";   fd_n, fd_d = 1, 24
            elif abs(fps_float - 25.0)   < 0.01:
                label = "25";   fd_n, fd_d = 1, 25
            elif abs(fps_float - 29.97)  < 0.01:
                label = "2997"; fd_n, fd_d = 1001, 30000
            elif abs(fps_float - 30.0)   < 0.01:
                label = "30";   fd_n, fd_d = 1, 30
            elif abs(fps_float - 50.0)   < 0.01:
                label = "50";   fd_n, fd_d = 1, 50
            elif abs(fps_float - 59.94)  < 0.01:
                label = "5994"; fd_n, fd_d = 1001, 60000
            elif abs(fps_float - 60.0)   < 0.01:
                label = "60";   fd_n, fd_d = 1, 60
            else:
                label = str(round(fps_float))
                fd_n, fd_d = fps_den, fps_num

            return {
                "width":        w,
                "height":       h,
                "fps_float":    fps_float,
                "frame_dur_n":  fd_n,
                "frame_dur_d":  fd_d,
                "name":         f"FFVideoFormat{h}p{label}",
                "color_space":  "1-1-1 (Rec. 709)",
            }

    # Fallback: 1080p 29.97
    return {
        "width":       DEFAULT_WIDTH,
        "height":      DEFAULT_HEIGHT,
        "fps_float":   DEFAULT_FPS_NUM / DEFAULT_FPS_DEN,
        "frame_dur_n": DEFAULT_FPS_DEN,
        "frame_dur_d": DEFAULT_FPS_NUM,
        "name":        f"FFVideoFormat{DEFAULT_HEIGHT}p{DEFAULT_FPS_LABEL}",
        "color_space": "1-1-1 (Rec. 709)",
    }


def apply_orientation(format_info: dict, orientation: str) -> dict:
    """Override width/height in format_info to match a target orientation.

    orientation: 'landscape' → 1920×1080  |  'portrait' → 1080×1920
    The fps and color_space are preserved from the detected format.
    The format name is rebuilt using FCP's FFVideoFormat{height}p{label} convention.
    """
    if orientation == "auto":
        return format_info

    if orientation == "landscape":
        w, h = 1920, 1080
    else:  # portrait
        w, h = 1080, 1920

    fps_float = format_info["fps_float"]
    if   abs(fps_float - 23.976) < 0.01: label = "2398"
    elif abs(fps_float - 24.0)   < 0.01: label = "24"
    elif abs(fps_float - 25.0)   < 0.01: label = "25"
    elif abs(fps_float - 29.97)  < 0.01: label = "2997"
    elif abs(fps_float - 30.0)   < 0.01: label = "30"
    elif abs(fps_float - 50.0)   < 0.01: label = "50"
    elif abs(fps_float - 59.94)  < 0.01: label = "5994"
    elif abs(fps_float - 60.0)   < 0.01: label = "60"
    else: label = str(round(fps_float))

    return {
        **format_info,
        "width":  w,
        "height": h,
        "name":   f"FFVideoFormat{h}p{label}",
    }

# ---------------------------------------------------------------------------
# Resource builder
# ---------------------------------------------------------------------------

def build_resources(
    all_entries: list[dict],
    audio_path: Path | None,
    probe_cache: dict[str, dict],
    format_info: dict,
) -> tuple[ET.Element, dict[str, str], str | None]:
    """
    Build the <resources> element.

    Returns:
        (resources_element, asset_id_map, audio_asset_id)
        asset_id_map: {source_file_str: asset_id}
        audio_asset_id: asset id for master mix, or None
    """
    resources = ET.Element("resources")
    counter   = [2]  # asset IDs start at r2; r1 is the format

    def next_id() -> str:
        rid = f"r{counter[0]}"
        counter[0] += 1
        return rid

    # Format element
    fmt = ET.SubElement(resources, "format")
    fmt.set("id", "r1")
    fmt.set("name", format_info["name"])
    fmt.set("frameDuration", rational_pair(format_info["frame_dur_n"], format_info["frame_dur_d"]))
    fmt.set("width",  str(format_info["width"]))
    fmt.set("height", str(format_info["height"]))
    fmt.set("colorSpace", format_info["color_space"])

    # Build asset for each unique source file
    asset_id_map: dict[str, str] = {}

    seen: set[str] = set()
    for entry in all_entries:
        src = entry.get("source_file", "")
        if not src or src in seen:
            continue
        seen.add(src)

        probe   = probe_cache.get(src, _default_probe())
        asset_id = next_id()
        asset_id_map[src] = asset_id

        name = Path(src).name
        asset = ET.SubElement(resources, "asset")
        asset.set("id",       asset_id)
        asset.set("name",     name)
        asset.set("uid",      make_uid(src))
        asset.set("start",    "0s")
        # For still images ffprobe reports a tiny duration (e.g. 1/25s = 1 frame).
        # FCP clips the spine element to the asset duration, so we set a large
        # value for stills so the clip length is governed entirely by the spine.
        asset_dur = 86400.0 if is_still_image(src) else probe["duration_s"]
        asset.set("duration", rational(asset_dur))
        asset.set("hasVideo", "1" if probe["has_video"] else "0")
        asset.set("hasAudio", "1" if probe["has_audio"] else "0")
        if probe["has_audio"]:
            asset.set("audioSources",  "1")
            asset.set("audioChannels", str(probe["audio_channels"]))
            asset.set("audioRate",     str(probe["audio_rate"]))

        media_rep = ET.SubElement(asset, "media-rep")
        media_rep.set("kind", "original-media")
        media_rep.set("src",  file_uri(src))

    # Audio asset (master mix)
    audio_asset_id: str | None = None
    if audio_path:
        audio_str = str(audio_path.resolve())
        probe = probe_cache.get(audio_str, _default_probe())
        audio_asset_id = next_id()

        asset = ET.SubElement(resources, "asset")
        asset.set("id",       audio_asset_id)
        asset.set("name",     audio_path.name)
        asset.set("uid",      make_uid(str(audio_path)))
        asset.set("start",    "0s")
        asset.set("duration", rational(probe["duration_s"]))
        asset.set("hasVideo", "0")
        asset.set("hasAudio", "1")
        asset.set("audioSources",  "1")
        asset.set("audioChannels", str(probe.get("audio_channels", 2)))
        asset.set("audioRate",     str(probe.get("audio_rate", 44100)))

        media_rep = ET.SubElement(asset, "media-rep")
        media_rep.set("kind", "original-media")
        media_rep.set("src",  file_uri(audio_path))

    return resources, asset_id_map, audio_asset_id

# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def get_keywords(entry: dict) -> list[str]:
    """Return keyword labels for a timeline entry."""
    keywords: list[str] = []
    scores    = entry.get("scores", {})
    composite = scores.get("composite") or 0.0
    smile     = scores.get("smile")     or 0.0
    zone      = entry.get("pacing_zone", "")
    stype     = entry.get("slot_type",   "video")

    if composite >= KW_AI_PICKS_MIN_SCORE:
        keywords.append("AI Picks")
    if smile >= KW_HIGH_SMILES_MIN:
        keywords.append("High Smiles")
    if zone == "chorus":
        keywords.append("Action")
    if stype == "memory_dump":
        keywords.append("Memory Dump")

    return keywords


# ---------------------------------------------------------------------------
# Clip element builder
# ---------------------------------------------------------------------------

def build_clip_element(
    entry: dict,
    asset_id_map: dict[str, str],
    probe_cache: dict[str, dict],
    format_info: dict,
) -> tuple[ET.Element, float] | tuple[None, float]:
    """
    Build one <asset-clip> element for a timeline or memory-dump entry.

    Returns (element, actual_duration_s). element is None if the source file
    has no registered asset id or has no usable content at the given start point.
    actual_duration_s is the clamped duration written into the element.
    Note: audio attachment is handled separately in build_fcpxml_string.
    """
    src = entry.get("source_file", "")
    asset_id = asset_id_map.get(src)
    if not asset_id:
        return None, 0.0

    stype      = entry.get("slot_type", "video")
    name       = Path(src).name
    offset_s   = entry.get("timeline_start_s", 0.0)
    dur_s      = entry.get("duration_s",        2.0)
    src_in_s   = entry.get("source_in_s")

    # Clamp duration to available content — only for real video clips.
    # For photo/burst_group entries source_in_s is the display frame offset,
    # not the start of a video range, so clamping against (file_dur - src_in_s)
    # would shrink burst entries near the end of their source to near-zero.
    if stype == "video":
        probe = probe_cache.get(str(Path(src).resolve()))
        if probe:
            file_dur = probe.get("duration_s", 0.0)
            start    = src_in_s if src_in_s is not None else 0.0
            available = file_dur - start
            if available <= 0:
                return None, 0.0
            dur_s = min(dur_s, available)

    clip = ET.Element("asset-clip")
    clip.set("name",     name)
    clip.set("ref",      asset_id)
    clip.set("offset",   rational(offset_s))
    clip.set("duration", rational(dur_s))
    clip.set("start",    rational(src_in_s) if src_in_s is not None else "0s")
    # Do NOT set format on asset-clip: it is #IMPLIED and applying the sequence
    # video format to photo clips crashes FCP's importer.

    # Keywords (range-based metadata — must follow connected clips in DTD)
    kws = get_keywords(entry)
    if kws:
        kw_el = ET.SubElement(clip, "keyword")
        kw_el.set("start",    "0s")
        kw_el.set("duration", rational(dur_s))
        kw_el.set("value",    ", ".join(kws))

    return clip, dur_s


def _attach_music(
    parent_el: ET.Element,
    audio_asset_id: str,
    audio_duration_s: float,
    timeline_offset_s: float = 0.0,
) -> None:
    """
    Attach the master mix as a connected <audio> element on lane -1.

    The <audio> element is the correct FCPXML primitive for connecting an
    external audio asset as a clip.  The offset is an absolute timeline
    position (not relative to the parent); FCP supports connected clips that
    start before their parent anchor clip.

    The element is inserted before any <keyword> children so that anchor
    items precede marker items, as the DTD requires.
    """
    ac = ET.Element("audio")
    ac.set("ref",      audio_asset_id)
    ac.set("lane",     "-1")
    ac.set("offset",   rational(timeline_offset_s))
    ac.set("duration", rational(audio_duration_s))
    ac.set("start",    "0s")
    ac.set("role",     "music.music-1")

    # DTD order: anchor_items before marker_items.
    # Insert before the first <keyword> child (if any).
    insert_pos = len(parent_el)
    for i, child in enumerate(parent_el):
        if child.tag == "keyword":
            insert_pos = i
            break
    parent_el.insert(insert_pos, ac)

# ---------------------------------------------------------------------------
# FCPXML assembly
# ---------------------------------------------------------------------------

def build_fcpxml_string(
    resources: ET.Element,
    timeline_entries: list[dict],
    memory_dump_entries: list[dict],
    asset_id_map: dict[str, str],
    probe_cache: dict[str, dict],
    format_info: dict,
    audio_asset_id: str | None,
    audio_duration_s: float,
    total_dur_s: float,
    event_name: str,
    project_name: str,
    output_dir: Path,
    title_text: str = "",
    subtitle_text: str = "",
    title_duration_s: float = 0.0,
) -> str:
    """Construct the full FCPXML document and return as a UTF-8 string."""

    root = ET.Element("fcpxml")
    root.set("version", FCPXML_VERSION)
    root.append(resources)

    library = ET.SubElement(root, "library")
    library.set("location", file_uri(output_dir) + "/")

    event = ET.SubElement(library, "event")
    event.set("name", event_name)

    project = ET.SubElement(event, "project")
    project.set("name", project_name)

    all_entries = sorted(
        timeline_entries + memory_dump_entries,
        key=lambda e: e.get("timeline_start_s", 0.0),
    )

    sequence = ET.SubElement(project, "sequence")
    sequence.set("format",      "r1")
    sequence.set("duration",    "0s")   # placeholder; updated after spine loop
    sequence.set("tcStart",     "0s")
    sequence.set("tcFormat",    "NDF")
    sequence.set("audioLayout", "stereo")
    sequence.set("audioRate",   "48k")

    spine = ET.SubElement(sequence, "spine")

    all_keywords_used: set[str] = set()
    music_attached = False
    seen_clips: set[tuple] = set()  # (source_file, source_in_s) dedup key

    # Prepend title card as the first spine element (rendered PNG via Pillow)
    spine_pos_s = 0.0
    if title_text and title_duration_s > 0:
        title_png = output_dir / "title_card.png"
        render_title_card(
            title=title_text,
            subtitle=subtitle_text,
            width=format_info["width"],
            height=format_info["height"],
            output_path=title_png,
        )
        # Register PNG asset
        existing_ids = {child.get("id") for child in resources if child.get("id")}
        counter = 2
        while f"r{counter}" in existing_ids:
            counter += 1
        title_asset_id = f"r{counter}"
        asset_el = ET.SubElement(resources, "asset")
        asset_el.set("id",       title_asset_id)
        asset_el.set("name",     "Title Card")
        asset_el.set("uid",      title_asset_id)
        asset_el.set("start",    "0s")
        asset_el.set("duration", rational(title_duration_s))
        asset_el.set("hasVideo", "1")
        asset_el.set("hasAudio", "0")
        media_rep = ET.SubElement(asset_el, "media-rep")
        media_rep.set("kind", "original-media")
        media_rep.set("src",  file_uri(title_png))

        video_el = ET.SubElement(spine, "video")
        video_el.set("ref",      title_asset_id)
        video_el.set("name",     "Title Card")
        video_el.set("offset",   rational(0.0))
        video_el.set("duration", rational(title_duration_s))
        video_el.set("start",    "0s")

        if audio_asset_id and audio_duration_s > 0:
            _attach_music(video_el, audio_asset_id, audio_duration_s,
                          timeline_offset_s=0.0)
            music_attached = True

        spine_pos_s = title_duration_s

    for entry in all_entries:
        dur_s   = entry.get("duration_s", 0.0)
        src     = entry.get("source_file", "")
        in_s    = entry.get("source_in_s")
        clip_key = (src, in_s)
        if clip_key in seen_clips:
            continue
        seen_clips.add(clip_key)

        if is_still_image(src):
            # <asset-clip> with hasAudio="0" crashes FCP's audio preflight
            # (nil-receiver ObjC forwarding fault in addAssetClip:toObject:parentFormatID:).
            # Fix: use <video> directly in the spine — a different importer path
            # that skips audio preflight entirely.  <video> without a lane
            # attribute is treated as a primary storyline element (inline with
            # video clips), not a connected clip.
            asset_id = asset_id_map.get(src)
            if asset_id:
                video_el = ET.SubElement(spine, "video")
                video_el.set("ref",      asset_id)
                video_el.set("name",     Path(src).name)
                video_el.set("offset",   rational(spine_pos_s))
                video_el.set("duration", rational(dur_s))
                video_el.set("start",    "0s")

                if not music_attached and audio_asset_id and audio_duration_s > 0:
                    _attach_music(video_el, audio_asset_id, audio_duration_s,
                                  timeline_offset_s=0.0)
                    music_attached = True

                all_keywords_used.update(get_keywords(entry))
                spine_pos_s += dur_s
        else:
            clip_el, actual_dur_s = build_clip_element(
                entry        = entry,
                asset_id_map = asset_id_map,
                probe_cache  = probe_cache,
                format_info  = format_info,
            )
            if clip_el is not None:
                # Override offset to pack clips with no inter-clip gaps.
                clip_el.set("offset", rational(spine_pos_s))
                if not music_attached and audio_asset_id and audio_duration_s > 0:
                    _attach_music(clip_el, audio_asset_id, audio_duration_s,
                                  timeline_offset_s=0.0)
                    music_attached = True
                spine.append(clip_el)
                all_keywords_used.update(get_keywords(entry))
                spine_pos_s += actual_dur_s

    # Set sequence duration to the actual packed spine length (spine_pos_s).
    # Using spine_pos_s (not pre-computed JSON durations) ensures probe-clamped
    # video clips don't inflate the sequence beyond its real content.
    sequence.set("duration", rational(spine_pos_s))

    # Keyword collections (FCP bins)
    kw_order = ["AI Picks", "High Smiles", "Action", "Memory Dump"]
    for kw in kw_order:
        if kw in all_keywords_used:
            kc = ET.SubElement(event, "keyword-collection")
            kc.set("name", kw)

    # Serialize via minidom for pretty-printing
    rough_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough_str)
    pretty = dom.toprettyxml(indent="    ", encoding=None)

    # minidom adds its own <?xml?> declaration on the first line; strip it
    lines = pretty.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]

    header = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
    return header + "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--sequence", "-s", "sequence_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to final_sequence.json (Phase 4 output).",
)
@click.option(
    "--audio", "-a", "audio_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to master_mix audio file. Overrides --beats inference.",
)
@click.option(
    "--beats", "-b", "beats_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to beat_grid.json (used to infer master_mix_path).",
)
@click.option(
    "--output", "-o", "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. Defaults to same dir as --sequence.",
)
@click.option(
    "--project-name",
    default="Holiday Montage", show_default=True,
    help="Final Cut Pro project name.",
)
@click.option(
    "--event-name",
    default=None,
    help='FCP event name. Defaults to "AI Montage — YYYY-MM-DD".',
)
@click.option(
    "--workers",
    default=4, show_default=True,
    help="Parallel ffprobe workers for media probing.",
)
@click.option(
    "--orientation",
    type=click.Choice(["auto", "landscape", "portrait"], case_sensitive=False),
    default="auto", show_default=True,
    help="Force FCP sequence orientation. 'landscape' sets 1920×1080 (16:9); "
         "'portrait' sets 1080×1920 (9:16). 'auto' detects from media (default).",
)
@click.option(
    "--title", "title_text",
    default="",
    help="Title card text (e.g. 'Grindelwald 2026'). Generates a native FCP title in the timeline.",
)
@click.option(
    "--subtitle", "subtitle_text",
    default="",
    help="Optional subtitle text beneath the title.",
)
@click.option(
    "--title-duration",
    default=5.0, show_default=True,
    help="Duration of the title card in seconds.",
)
@click.option(
    "--skip-audio",
    is_flag=True, default=False,
    help="Don't attach an audio track to the timeline.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Print detailed progress.",
)
def main(
    sequence_path: Path,
    audio_path: Path | None,
    beats_path: Path | None,
    output_dir: Path | None,
    project_name: str,
    event_name: str | None,
    workers: int,
    orientation: str,
    title_text: str,
    subtitle_text: str,
    title_duration: float,
    skip_audio: bool,
    verbose: bool,
) -> None:
    """Architect — Phase 5 of the macOS AI Montage Suite.

    Converts final_sequence.json into a montage.fcpxml file for Final Cut Pro
    or DaVinci Resolve, with Ken Burns photo animation and keyword collections.
    """
    # Resolve defaults
    if output_dir is None:
        output_dir = sequence_path.parent
    if event_name is None:
        event_name = f"AI Montage — {datetime.now().strftime('%Y-%m-%d')}"

    console.print("[bold]Architect[/] — Phase 5 of the macOS AI Montage Suite\n")

    # Preflight
    ffprobe_bin = preflight_checks(output_dir)

    # ------------------------------------------------------------------
    # Stage 1: Load inputs
    # ------------------------------------------------------------------
    console.print("[bold]Stage 1/5[/] Loading final_sequence.json …")
    seq_data = load_final_sequence(sequence_path)

    timeline_entries:    list[dict] = seq_data.get("timeline",    [])
    memory_dump_entries: list[dict] = seq_data.get("memory_dump", [])
    seq_meta: dict = seq_data.get("metadata", {})

    total_dur_s: float = seq_meta.get("total_duration_with_dump_s",
                                      seq_meta.get("total_duration_s", 0.0))

    console.print(f"  Timeline slots: {len(timeline_entries)}")
    console.print(f"  Memory dump:    {len(memory_dump_entries)}")
    console.print(f"  Total duration: {total_dur_s:.1f}s")

    # Resolve audio
    resolved_audio: Path | None = None
    if not skip_audio:
        resolved_audio = resolve_audio_path(audio_path, beats_path, seq_data)
        if resolved_audio:
            console.print(f"  Audio: {resolved_audio.name}")
        else:
            console.print("[yellow]  Audio: not found — timeline will have no music track[/]")

    # ------------------------------------------------------------------
    # Stage 2: Probe media
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 2/5[/] Probing media files …")

    all_entries = timeline_entries + memory_dump_entries
    unique_sources = list({
        e["source_file"]
        for e in all_entries
        if e.get("source_file")
    })

    console.print(f"  Unique source files: {len(unique_sources)}")

    probe_cache = probe_all(unique_sources, resolved_audio, ffprobe_bin, workers)
    format_info = detect_format_info(probe_cache)
    format_info = apply_orientation(format_info, orientation)

    console.print(
        f"  Detected format: {format_info['name']} "
        f"{format_info['width']}×{format_info['height']} "
        f"@ {format_info['fps_float']:.3f}fps"
        + (f"  [yellow](orientation forced: {orientation})[/]" if orientation != "auto" else "")
    )

    # ------------------------------------------------------------------
    # Stage 3: Build resources
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 3/5[/] Building FCPXML resources …")

    resources_el, asset_id_map, audio_asset_id = build_resources(
        all_entries  = all_entries,
        audio_path   = resolved_audio,
        probe_cache  = probe_cache,
        format_info  = format_info,
    )

    if title_text:
        console.print(f"  Title card:  '{title_text}'"
                      + (f" / '{subtitle_text}'" if subtitle_text else "")
                      + f"  ({title_duration}s, rendered PNG)")

    n_assets = len(asset_id_map) + (1 if audio_asset_id else 0)
    console.print(f"  Assets registered: {n_assets}  (1 format + {len(asset_id_map)} media"
                  f"{' + 1 audio' if audio_asset_id else ''})")

    # ------------------------------------------------------------------
    # Stage 4: Build FCPXML document
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 4/5[/] Building FCPXML spine …")

    audio_duration_s = 0.0
    if resolved_audio:
        audio_probe = probe_cache.get(str(resolved_audio.resolve()))
        if audio_probe:
            audio_duration_s = audio_probe["duration_s"]

    fcpxml_str = build_fcpxml_string(
        resources           = resources_el,
        timeline_entries    = timeline_entries,
        memory_dump_entries = memory_dump_entries,
        asset_id_map        = asset_id_map,
        probe_cache         = probe_cache,
        format_info         = format_info,
        audio_asset_id      = audio_asset_id,
        audio_duration_s    = audio_duration_s,
        total_dur_s         = total_dur_s,
        event_name          = event_name,
        project_name        = project_name,
        output_dir          = output_dir,
        title_text          = title_text,
        subtitle_text       = subtitle_text,
        title_duration_s    = title_duration if title_text else 0.0,
    )

    n_spine_clips = len(all_entries)
    console.print(f"  Spine clips:  {n_spine_clips}")
    console.print(f"  With keywords: {sum(1 for e in all_entries if get_keywords(e))}")

    # ------------------------------------------------------------------
    # Stage 5: Write output
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 5/5[/] Writing montage.fcpxml …")

    out_path = output_dir / "montage.fcpxml"
    out_path.write_text(fcpxml_str, encoding="utf-8")

    file_size_kb = out_path.stat().st_size // 1024

    console.print(f"\n[bold green]✓[/] montage.fcpxml → [cyan]{out_path}[/]  ({file_size_kb} KB)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    table = Table(title="FCPXML Summary", show_header=True, header_style="bold")
    table.add_column("Property",  style="cyan")
    table.add_column("Value",     justify="right")

    table.add_row("FCPXML version",  FCPXML_VERSION)
    table.add_row("Format",          format_info["name"])
    table.add_row("Resolution",      f"{format_info['width']}×{format_info['height']}")
    table.add_row("Timeline clips",  str(len(timeline_entries)))
    table.add_row("Memory dump",     str(len(memory_dump_entries)))
    table.add_row("Total duration",  f"{total_dur_s:.1f}s")
    table.add_row("Audio attached",  "yes" if audio_asset_id else "no")
    table.add_row("File size",       f"{file_size_kb} KB")
    console.print(table)

    console.print(
        f"\n[bold]Import instructions:[/]\n"
        f"  Final Cut Pro:     File → Import → XML …  →  select [cyan]montage.fcpxml[/]\n"
        f"  DaVinci Resolve:   File → Import → Timeline (AAF, EDL, XML, FCPXML…)\n"
    )


if __name__ == "__main__":
    main()
