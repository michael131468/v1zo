#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "scenedetect[opencv]>=0.6.0",
#   "pyobjc-core>=9.0",
#   "pyobjc-framework-Cocoa>=9.0",
#   "pyobjc-framework-Vision>=9.0",
#   "Pillow>=10.0",
#   "piexif>=1.1.3",
#   "numpy>=1.24",
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Phase 1 of the macOS AI Montage Suite.

Scans a media directory, detects video scenes via PySceneDetect,
analyses frames with the macOS Vision framework, groups photo bursts,
and writes scenes.json + scout_review.html.

Usage:
    uv run scout.py --input /path/to/media [OPTIONS]
"""
from __future__ import annotations

import functools
import itertools
import json
import platform
import re
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import piexif
from PIL import Image
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

try:
    import Vision
    from Foundation import NSURL  # noqa: F401
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Unique temp IDs for preview file names (replaced by scene IDs in JSON output)
_temp_counter = itertools.count(1)


def _next_temp_id() -> str:
    return f"t{next(_temp_counter):07d}"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ScoutError(Exception):
    pass


class VisionError(ScoutError):
    pass


class FFmpegError(ScoutError):
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = frozenset({".mov", ".mp4", ".m4v", ".avi", ".mkv"})
PHOTO_EXTENSIONS = frozenset({".heic", ".jpg", ".jpeg", ".png", ".tiff", ".tif"})
MIN_SCENE_DURATION_S = 0.5
MIN_FILE_SIZE_BYTES = 10 * 1024  # 10 KB

# Maps Apple Vision identifier substrings to top-level category buckets.
# Sorted so more-specific keys match before broader ones.
CATEGORY_MAP: list[tuple[str, str]] = [
    ("portrait",      "People"),
    ("crowd",         "People"),
    ("person",        "People"),
    ("people",        "People"),
    ("face",          "People"),
    ("food",          "Food"),
    ("beverage",      "Food"),
    ("drink",         "Food"),
    ("meal",          "Food"),
    ("plant_life",    "Nature"),
    ("plant",         "Nature"),
    ("animal",        "Nature"),
    ("beach",         "Nature"),
    ("forest",        "Nature"),
    ("mountain",      "Nature"),
    ("landscape",     "Nature"),
    ("nature",        "Nature"),
    ("water",         "Nature"),
    ("sky",           "Nature"),
    ("outdoor",       "Nature"),
    ("architecture",  "Urban"),
    ("building",      "Urban"),
    ("vehicle",       "Urban"),
    ("street",        "Urban"),
    ("road",          "Urban"),
    ("urban",         "Urban"),
    ("city",          "Urban"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_to_category(observations: list[dict]) -> str:
    for obs in observations:
        label = obs["label"].lower()
        for key, category in CATEGORY_MAP:
            if key in label:
                return category
    return "Other"


def compute_fp_distance(fp_a: list[float], fp_b: list[float]) -> float:
    a = np.array(fp_a, dtype=np.float32)
    b = np.array(fp_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def _sanitize(s: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)[:max_len]


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight_checks(output_dir: Path) -> Path:
    """Validate system requirements. Returns ffmpeg Path."""
    ver_str = platform.mac_ver()[0]
    if ver_str:
        parts = ver_str.split(".")
        major = int(parts[0]) if parts else 0
        if major < 13:
            raise SystemExit(
                f"macOS 13 Ventura or later required (detected {ver_str})."
            )

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")

    if not VISION_AVAILABLE:
        raise SystemExit(
            "PyObjC Vision framework unavailable. "
            "Ensure pyobjc-framework-Vision is installed."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)
    test = output_dir / ".write_test"
    test.touch()
    test.unlink()

    return Path(ffmpeg_bin)


# ---------------------------------------------------------------------------
# Media discovery
# ---------------------------------------------------------------------------

def discover_media(
    input_dir: Path,
    extensions: list[str],
) -> dict[str, list[Path]]:
    ext_set = {f".{e.lower().lstrip('.')}" for e in extensions}
    video_exts = ext_set & VIDEO_EXTENSIONS
    photo_exts = ext_set & PHOTO_EXTENSIONS

    videos: list[Path] = []
    photos: list[Path] = []

    for path in sorted(input_dir.rglob("*")):
        if any(part.startswith(".") for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            if path.stat().st_size < MIN_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue
        suffix = path.suffix.lower()
        if suffix in video_exts:
            videos.append(path)
        elif suffix in photo_exts:
            photos.append(path)

    return {"video": videos, "photo": photos}


# ---------------------------------------------------------------------------
# Video info
# ---------------------------------------------------------------------------

def get_video_info(video_path: Path) -> dict:
    """Return fps and resolution via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        data = json.loads(r.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w = int(stream.get("width", 0))
                h = int(stream.get("height", 0))
                fps_str = stream.get("r_frame_rate", "30/1")
                num, denom = fps_str.split("/")
                fps = float(num) / float(denom) if float(denom) != 0 else 30.0
                return {"resolution": [w, h], "fps": round(fps, 3)}
    except Exception:
        pass
    return {"resolution": [0, 0], "fps": 30.0}


# ---------------------------------------------------------------------------
# Media normalisation
# ---------------------------------------------------------------------------

NORMALISE_TARGETS: dict[str, tuple[int, int]] = {
    "landscape": (1920, 1080),
    "portrait":  (1080, 1920),
}

PHOTO_EXTENSIONS_SET = frozenset({".heic", ".jpg", ".jpeg", ".png", ".tiff", ".tif"})


def _normalise_vf(target: str, mode: str) -> str:
    """Return an ffmpeg -vf filter string for the given target + mode.

    target: 'landscape' or 'portrait'
    mode:   'pad'  → letterbox/pillarbox with black bars
            'crop' → scale to fill then centre-crop
    """
    tw, th = NORMALISE_TARGETS[target]
    if mode == "pad":
        return (
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
    else:  # crop
        return (
            f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},setsar=1"
        )


def _normalise_one(
    src: Path,
    dst: Path,
    vf: str,
    ffmpeg_bin: str,
    is_photo: bool,
) -> tuple[Path, bool, str]:
    """Transcode/convert one file. Returns (dst, success, error_msg).

    Already-existing dst files are treated as done (resume-friendly).
    """
    if dst.exists() and dst.stat().st_size > 0:
        return dst, True, ""

    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_photo:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(src),
            "-vf", vf,
            "-q:v", "2",       # JPEG quality
            str(dst),
        ]
    else:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return dst, True, ""
    except subprocess.CalledProcessError as exc:
        return dst, False, exc.stderr.decode(errors="replace")[-200:]
    except Exception as exc:
        return dst, False, str(exc)


def normalise_media(
    video_files: list[Path],
    photo_files: list[Path],
    input_dir: Path,
    normalise_dir: Path,
    target: str,
    mode: str,
    ffmpeg_bin: str,
    workers: int,
) -> tuple[list[Path], list[Path]]:
    """Normalise all media to a common aspect ratio/resolution.

    Returns new (video_files, photo_files) lists pointing to normalised copies.
    Source files that fail to normalise are dropped with a warning.
    """
    normalise_dir.mkdir(parents=True, exist_ok=True)
    vf = _normalise_vf(target, mode)
    tw, th = NORMALISE_TARGETS[target]

    console.print(
        f"\n[bold]Normalising media[/] → {target} {tw}×{th} "
        f"({'black bars' if mode == 'pad' else 'crop to fill'}) …"
    )
    console.print(f"  Output directory: {normalise_dir}")

    def _dst_path(src: Path, is_photo: bool) -> Path:
        try:
            rel = src.relative_to(input_dir)
        except ValueError:
            rel = Path(src.name)
        dst = normalise_dir / rel
        if is_photo:
            dst = dst.with_suffix(".jpg")
        return dst

    jobs: list[tuple[Path, Path, bool]] = [
        (src, _dst_path(src, False), False) for src in video_files
    ] + [
        (src, _dst_path(src, True), True) for src in photo_files
    ]

    new_videos: list[Path] = []
    new_photos: list[Path] = []
    n_skip = 0
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Transcoding…", total=len(jobs)
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_normalise_one, src, dst, vf, str(ffmpeg_bin), is_photo): (src, is_photo)
                for src, dst, is_photo in jobs
            }
            for fut in as_completed(futs):
                src, is_photo = futs[fut]
                try:
                    dst, ok, err = fut.result()
                except Exception as exc:
                    ok, err, dst = False, str(exc), _dst_path(src, is_photo)

                if ok:
                    if is_photo:
                        new_photos.append(dst)
                    else:
                        new_videos.append(dst)
                else:
                    n_skip += 1
                    errors.append(f"{src.name}: {err}")
                progress.advance(task)

    console.print(f"  Done: {len(new_videos)} videos, {len(new_photos)} photos normalised.")
    if n_skip:
        console.print(f"  [yellow]Skipped {n_skip} file(s) due to errors.[/]")
        for e in errors[:5]:
            console.print(f"    [red]{e}[/]")

    return new_videos, new_photos


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def _sharpness(image_path: Path) -> float:
    """Laplacian-variance sharpness score for a JPEG frame. Higher = sharper."""
    try:
        with Image.open(image_path) as img:
            arr = np.array(img.convert("L"), dtype=np.float32)
        # Approximate Laplacian via finite differences
        lap = (
            np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
            np.roll(arr, 1, 1) + np.roll(arr, -1, 1) -
            4 * arr
        )
        return float(np.var(lap))
    except Exception:
        return 0.0


def find_best_moment(
    video_path: Path,
    scene_start: float,
    scene_end: float,
    snippet_dur: float,
    sample_interval: float,
    ffmpeg_bin: str,
    tmp_dir: Path,
) -> float:
    """Sample frames across [scene_start, scene_end] and return the timestamp
    of the sharpest frame, clamped so a snippet of snippet_dur fits inside."""
    half = snippet_dur / 2.0
    # Clamp sample range so the extracted clip stays inside the scene
    lo = scene_start + half
    hi = scene_end   - half
    if lo >= hi:
        # Scene is shorter than snippet_dur — return centre
        return (scene_start + scene_end) / 2.0

    times: list[float] = []
    t = lo
    while t <= hi:
        times.append(t)
        t += sample_interval
    if not times:
        times = [lo]

    best_t = times[0]
    best_score = -1.0

    for t in times:
        frame_path = tmp_dir / f"smp_{video_path.stem}_{t:.2f}.jpg"
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", f"{t:.4f}", "-i", str(video_path),
            "-vframes", "1", "-q:v", "5", "-vf", "scale=320:-1",
            str(frame_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=15)
        except Exception:
            continue
        score = _sharpness(frame_path)
        if score > best_score:
            best_score = score
            best_t = t
        try:
            frame_path.unlink()
        except Exception:
            pass

    return best_t


def extract_snippet(
    video_path: Path,
    center_t: float,
    duration: float,
    out_path: Path,
    ffmpeg_bin: str,
) -> bool:
    """Extract a duration-second clip centred on center_t from video_path."""
    start_t = max(0.0, center_t - duration / 2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-y",
        "-ss", f"{start_t:.4f}", "-i", str(video_path),
        "-t", f"{duration:.4f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _snippet_worker(
    sd: dict,
    snippet_dur: float,
    snippet_dir: Path,
    sample_interval: float,
    ffmpeg_bin: str,
    tmp_dir: Path,
    scene_counter: int,
    max_per_scene: int,
) -> list[dict]:
    """Divide one long scene into non-overlapping windows and extract the sharpest
    snippet_dur-second clip from each window.

    Returns a list of new scene dicts (one per successfully extracted snippet).
    The original scene dict is not modified.
    """
    video_path: Path = sd["_video_path"]
    scene_start: float = sd["start_time_s"]
    scene_end:   float = sd["end_time_s"]
    scene_dur:   float = sd["duration_s"]
    fps: float = sd.get("fps", 30.0) or 30.0

    # Number of non-overlapping snippet-sized windows that fit in the scene
    n = max(1, int(scene_dur / snippet_dur))
    if max_per_scene > 0:
        n = min(n, max_per_scene)

    window_dur = scene_dur / n
    results: list[dict] = []

    for i in range(n):
        win_start = scene_start + i * window_dur
        win_end   = win_start + window_dur

        center_t = find_best_moment(
            video_path, win_start, win_end,
            snippet_dur, sample_interval, ffmpeg_bin, tmp_dir,
        )

        snippet_name = f"{video_path.stem}_snip{scene_counter:04d}_{i:02d}.mp4"
        out_path = snippet_dir / snippet_name

        if out_path.exists() and out_path.stat().st_size > 0:
            ok = True
        else:
            ok = extract_snippet(video_path, center_t, snippet_dur, out_path, ffmpeg_bin)

        if not ok:
            continue

        # Build a new scene dict derived from the original
        new_sd: dict = {
            **sd,
            "original_source_file": sd["source_file"],
            "source_file":    str(out_path),
            "_video_path":    out_path,
            "_temp_id":       _next_temp_id(),
            "start_time_s":   0.0,
            "end_time_s":     snippet_dur,
            "duration_s":     snippet_dur,
            "start_frame":    0,
            "end_frame":      round(fps * snippet_dur),
            "snippet_center_t":  round(center_t, 3),
            "snippet_index":     i,
            "snippet_extracted": True,
            "previews": {"start": None, "mid": None, "end": None, "preview_gif": None},
            "previews_generated": False,
            "errors": list(sd.get("errors", [])),
        }
        results.append(new_sd)

    return results


def extract_scene_snippets(
    all_video_scenes: list[dict],
    snippet_dur: float,
    snippet_dir: Path,
    min_scene_dur: float,
    sample_interval: float,
    ffmpeg_bin: str,
    workers: int,
    max_per_scene: int,
) -> list[dict]:
    """Replace long scenes with one or more extracted snippets.

    Scenes shorter than min_scene_dur pass through unchanged.
    Long scenes are divided into non-overlapping windows; the sharpest
    snippet_dur-second clip is extracted from each window.

    Returns the rebuilt scene list.
    """
    short_scenes = [sd for sd in all_video_scenes if sd["duration_s"] <= min_scene_dur]
    long_scenes  = [sd for sd in all_video_scenes if sd["duration_s"] >  min_scene_dur]

    if not long_scenes:
        return all_video_scenes

    snippet_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = snippet_dir / "_tmp_samples"
    tmp_dir.mkdir(exist_ok=True)

    n_windows = sum(
        min(max(1, int(sd["duration_s"] / snippet_dur)), max_per_scene) if max_per_scene > 0
        else max(1, int(sd["duration_s"] / snippet_dur))
        for sd in long_scenes
    )
    console.print(
        f"\n[bold]Extracting snippets[/] from {len(long_scenes)} long scene(s) "
        f"(>{min_scene_dur}s → up to {n_windows} × {snippet_dur}s clips) …"
    )

    extracted: dict[int, list[dict]] = {}  # scene index → snippet list

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Snippets…", total=len(long_scenes))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _snippet_worker, sd, snippet_dur, snippet_dir,
                    sample_interval, str(ffmpeg_bin), tmp_dir, i, max_per_scene,
                ): i
                for i, sd in enumerate(long_scenes)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    extracted[i] = fut.result()
                except Exception as exc:
                    extracted[i] = []
                    long_scenes[i]["errors"].append({
                        "stage": "snippet_extraction",
                        "message": str(exc),
                        "detail": "",
                    })
                progress.advance(task)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Rebuild: short scenes first (preserve original order), then snippets
    # interleaved at the position of the original long scene
    result: list[dict] = []
    short_iter = iter(short_scenes)
    long_idx = 0

    for sd in all_video_scenes:
        if sd["duration_s"] <= min_scene_dur:
            result.append(next(short_iter))
        else:
            snippets = extracted.get(long_idx, [])
            if snippets:
                result.extend(snippets)
            else:
                # Extraction failed — keep original scene so pipeline continues
                result.append(sd)
            long_idx += 1

    n_ok = sum(len(v) for v in extracted.values())
    console.print(f"  Extracted {n_ok} snippet(s) from {len(long_scenes)} scene(s) → {snippet_dir}")
    return result


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_scenes(video_path: Path, threshold: float) -> list[dict]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    mgr = SceneManager()
    mgr.add_detector(ContentDetector(threshold=threshold))
    mgr.detect_scenes(video=video, show_progress=False)
    scene_list = mgr.get_scene_list()

    if not scene_list:
        duration = video.duration
        total_s = duration.get_seconds() if duration else 0.0
        total_frames = duration.get_frames() if duration else 0
        return [{
            "scene_index": 0,
            "start_frame": 0,
            "end_frame": total_frames,
            "start_time_s": 0.0,
            "end_time_s": total_s,
            "duration_s": total_s,
        }]

    results = []
    for i, (start_tc, end_tc) in enumerate(scene_list):
        duration_s = end_tc.get_seconds() - start_tc.get_seconds()
        if duration_s < MIN_SCENE_DURATION_S:
            continue
        results.append({
            "scene_index": i,
            "start_frame": start_tc.get_frames(),
            "end_frame": end_tc.get_frames(),
            "start_time_s": start_tc.get_seconds(),
            "end_time_s": end_tc.get_seconds(),
            "duration_s": duration_s,
        })
    return results


# ---------------------------------------------------------------------------
# FFmpeg frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    scene: dict,
    previews_dir: Path,
    temp_id: str,
    ffmpeg_bin: Path,
) -> tuple[dict[str, str | None], list[dict]]:
    errors: list[dict] = []
    previews: dict[str, str | None] = {
        "start": None, "mid": None, "end": None, "preview_gif": None,
    }

    s = scene["start_time_s"]
    e = scene["end_time_s"]
    # Clamp to avoid black frames at exact cut points
    t_start = s + min(0.1, (e - s) * 0.05)
    t_end   = e - min(0.1, (e - s) * 0.05)
    t_mid   = (t_start + t_end) / 2.0

    for label, t in [("start", t_start), ("mid", t_mid), ("end", t_end)]:
        out = previews_dir / f"{temp_id}_{label}.jpg"
        cmd = [
            str(ffmpeg_bin),
            "-ss", f"{t:.3f}", "-i", str(video_path),
            "-vframes", "1", "-q:v", "3", "-vf", "scale=640:-1",
            str(out), "-y",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            previews[label] = f"previews/{out.name}"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = (
                exc.stderr.decode(errors="replace")
                if hasattr(exc, "stderr") and exc.stderr
                else str(exc)
            )
            errors.append({
                "stage": "extract_frames",
                "message": f"ffmpeg failed for {label} frame",
                "detail": detail,
            })

    # 1-second GIF micro-preview centred on mid point
    gif_start = max(0.0, t_mid - 0.5)
    gif_out = previews_dir / f"{temp_id}_preview.gif"
    cmd = [
        str(ffmpeg_bin),
        "-ss", f"{gif_start:.3f}", "-t", "1.0", "-i", str(video_path),
        "-vf",
        "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", str(gif_out), "-y",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        previews["preview_gif"] = f"previews/{gif_out.name}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (
            exc.stderr.decode(errors="replace")
            if hasattr(exc, "stderr") and exc.stderr
            else str(exc)
        )
        errors.append({
            "stage": "extract_frames",
            "message": "ffmpeg failed for GIF preview",
            "detail": detail,
        })

    return previews, errors


def make_photo_thumbnail(photo_path: Path, output_path: Path, width: int = 640) -> bool:
    try:
        with Image.open(photo_path) as img:
            img = img.convert("RGB")
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=85)
        return True
    except Exception:
        return False


def get_photo_resolution(photo_path: Path) -> list[int]:
    try:
        with Image.open(photo_path) as img:
            return [img.width, img.height]
    except Exception:
        return [0, 0]


# ---------------------------------------------------------------------------
# macOS Vision framework
# ---------------------------------------------------------------------------

def _analyse_single_image(image_path: Path) -> dict:
    """Run VNClassifyImageRequest + VNGenerateImageFeaturePrintRequest.

    Creates its own VNImageRequestHandler — safe to call from threads.
    """
    result: dict[str, Any] = {
        "classifications": [],
        "category": "Other",
        "feature_print": None,
        "errors": [],
    }

    try:
        url = NSURL.fileURLWithPath_(str(image_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})

        classify_req = Vision.VNClassifyImageRequest.alloc().init()
        fp_req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
        fp_req.setImageCropAndScaleOption_(0)  # VNImageCropAndScaleOptionScaleFit

        # Batch both requests in one call — image is decoded once
        success, err = handler.performRequests_error_(
            [classify_req, fp_req], None
        )
        if not success:
            raise VisionError(f"performRequests_error_ failed: {err}")

        # Classifications
        classifications = []
        for obs in (classify_req.results() or []):
            conf = float(obs.confidence())
            if conf > 0.05:
                classifications.append({
                    "label": str(obs.identifier()),
                    "confidence": conf,
                })
        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        result["classifications"] = classifications
        result["category"] = map_to_category(classifications)

        # Feature print → raw float vector via NSData
        fp_obs_list = fp_req.results() or []
        if fp_obs_list:
            obs = fp_obs_list[0]
            buf = bytes(obs.data())
            count = obs.elementCount()
            etype = obs.elementType()  # 1 = float32, 2 = float64
            dtype = np.float32 if etype == 1 else np.float64
            vector = np.frombuffer(buf, dtype=dtype, count=count)
            result["feature_print"] = vector.tolist()

    except Exception as exc:
        result["errors"].append({
            "stage": "vision",
            "message": str(exc),
            "detail": traceback.format_exc(),
        })

    return result


def analyse_images_parallel(
    image_paths: list[Path],
    workers: int,
) -> dict[str, dict]:
    """Analyse images in parallel. Returns {str(path): analysis_dict}."""
    out: dict[str, dict] = {}
    if not image_paths:
        return out
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_analyse_single_image, p): p for p in image_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                out[str(path)] = future.result()
            except Exception as exc:
                out[str(path)] = {
                    "classifications": [], "category": "Other",
                    "feature_print": None,
                    "errors": [{"stage": "vision", "message": str(exc), "detail": ""}],
                }
    return out


# ---------------------------------------------------------------------------
# EXIF reading
# ---------------------------------------------------------------------------

def _gps_to_decimal(coords: tuple, ref: str) -> float:
    d = coords[0][0] / coords[0][1]
    m = coords[1][0] / coords[1][1]
    s = coords[2][0] / coords[2][1]
    val = d + m / 60 + s / 3600
    return round(-val if ref in ("S", "W") else val, 7)


def read_exif_metadata(photo_path: Path) -> dict:
    result: dict[str, Any] = {
        "datetime_original": None,
        "subsec_time": None,
        "full_timestamp_unix": None,
        "gps_lat": None,
        "gps_lon": None,
        "make": "",
        "model": "",
        "orientation": 1,
    }

    # ---- Try piexif (JPEG/TIFF) ----
    exif_dict: dict | None = None
    try:
        exif_dict = piexif.load(str(photo_path))
    except Exception:
        pass

    # ---- Fallback: Pillow _getexif (works for HEIC on macOS via ImageIO) ----
    if exif_dict is None or not exif_dict.get("Exif"):
        try:
            with Image.open(photo_path) as img:
                raw = getattr(img, "_getexif", lambda: None)()
                if raw:
                    TAG_DATETIME_ORIGINAL = 36867
                    TAG_SUBSEC = 37521
                    dt_raw = raw.get(TAG_DATETIME_ORIGINAL)
                    if dt_raw:
                        s = dt_raw if isinstance(dt_raw, str) else dt_raw.decode()
                        try:
                            result["datetime_original"] = datetime.strptime(
                                s, "%Y:%m:%d %H:%M:%S"
                            )
                        except ValueError:
                            pass
                    subsec = raw.get(TAG_SUBSEC)
                    if subsec is not None:
                        result["subsec_time"] = str(subsec).strip()
        except Exception:
            pass

    if exif_dict:
        exif_ifd = exif_dict.get("Exif", {})
        zeroth = exif_dict.get("0th", {})
        gps = exif_dict.get("GPS", {})

        if result["datetime_original"] is None:
            raw_dt = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"")
            if raw_dt:
                try:
                    dt_str = raw_dt.decode() if isinstance(raw_dt, bytes) else raw_dt
                    result["datetime_original"] = datetime.strptime(
                        dt_str, "%Y:%m:%d %H:%M:%S"
                    )
                except (ValueError, UnicodeDecodeError):
                    pass

        if result["subsec_time"] is None:
            raw_sub = exif_ifd.get(piexif.ExifIFD.SubSecTimeOriginal, b"")
            if raw_sub:
                try:
                    result["subsec_time"] = (
                        raw_sub.decode() if isinstance(raw_sub, bytes) else str(raw_sub)
                    ).strip()
                except Exception:
                    pass

        for attr, tag in [("make", piexif.ImageIFD.Make), ("model", piexif.ImageIFD.Model)]:
            val = zeroth.get(tag, b"")
            if isinstance(val, bytes):
                result[attr] = val.decode(errors="replace").strip("\x00").strip()

        result["orientation"] = zeroth.get(piexif.ImageIFD.Orientation, 1)

        if piexif.GPSIFD.GPSLatitude in gps:
            try:
                result["gps_lat"] = _gps_to_decimal(
                    gps[piexif.GPSIFD.GPSLatitude],
                    gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode(),
                )
                result["gps_lon"] = _gps_to_decimal(
                    gps[piexif.GPSIFD.GPSLongitude],
                    gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode(),
                )
            except Exception:
                pass

    # ---- macOS Spotlight fallback for datetime ----
    if result["datetime_original"] is None:
        try:
            proc = subprocess.run(
                ["mdls", "-name", "kMDItemContentCreationDate", "-raw", str(photo_path)],
                capture_output=True, text=True, timeout=5,
            )
            val = proc.stdout.strip()
            if val and val != "(null)":
                for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        result["datetime_original"] = datetime.strptime(val, fmt)
                        break
                    except ValueError:
                        pass
        except Exception:
            pass

    # ---- Compute unix timestamp with sub-second precision ----
    if result["datetime_original"] is not None:
        ts = result["datetime_original"].timestamp()
        subsec = (result.get("subsec_time") or "").strip()
        if subsec and subsec != "0":
            try:
                ts += float("0." + subsec)
            except ValueError:
                pass
        result["full_timestamp_unix"] = round(ts, 6)
        result["datetime_original"] = result["datetime_original"].strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    return result


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------

def detect_burst_groups(
    photo_records: list[dict],
    time_window_s: float,
    min_count: int,
    similarity_threshold: float,
) -> list[dict]:
    burst_groups: list[dict] = []
    in_burst: set[str] = set()  # source_file strings

    # ---- Pass 1: Temporal grouping ----
    timestamped = [
        (p["exif"]["full_timestamp_unix"], p)
        for p in photo_records
        if p.get("exif", {}).get("full_timestamp_unix") is not None
    ]
    timestamped.sort(key=lambda x: x[0])

    i = 0
    while i < len(timestamped):
        ts_i, p_i = timestamped[i]
        if p_i["source_file"] in in_burst:
            i += 1
            continue

        group: list[tuple[float, dict]] = [(ts_i, p_i)]
        j = i + 1
        while j < len(timestamped):
            ts_j, p_j = timestamped[j]
            if ts_j - ts_i <= time_window_s:
                group.append((ts_j, p_j))
                j += 1
            else:
                break

        if len(group) >= min_count:
            members = [p for _, p in group]
            for p in members:
                in_burst.add(p["source_file"])
            rep = _best_representative(members)
            burst_groups.append({
                "_members": members,
                "burst_type": "temporal",
                "representative_source_file": rep["source_file"],
                "time_span_s": round(group[-1][0] - group[0][0], 3),
            })
            i = j
        else:
            i += 1

    # ---- Pass 2: Visual similarity (non-burst photos only) ----
    non_burst = [p for p in photo_records if p["source_file"] not in in_burst]
    used_visual: set[str] = set()

    for idx_a, pa in enumerate(non_burst):
        if pa["source_file"] in used_visual:
            continue
        fp_a = pa.get("vision", {}).get("feature_print")
        if fp_a is None:
            continue

        group_v = [pa]
        for pb in non_burst[idx_a + 1:]:
            if pb["source_file"] in used_visual:
                continue
            fp_b = pb.get("vision", {}).get("feature_print")
            if fp_b is None:
                continue
            if compute_fp_distance(fp_a, fp_b) < similarity_threshold:
                group_v.append(pb)

        if len(group_v) >= 2:
            for p in group_v:
                used_visual.add(p["source_file"])
            rep = _best_representative(group_v)
            burst_groups.append({
                "_members": group_v,
                "burst_type": "visual",
                "representative_source_file": rep["source_file"],
                "time_span_s": None,
            })

    return burst_groups


def _best_representative(members: list[dict]) -> dict:
    return max(
        members,
        key=lambda p: max(
            (c["confidence"] for c in p.get("vision", {}).get("classifications", [])),
            default=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Scene list assembly
# ---------------------------------------------------------------------------

def build_scenes_list(
    video_scenes: list[dict],
    photo_records: list[dict],
    burst_groups: list[dict],
    input_dir: Path,
) -> list[dict]:
    scene_counter = itertools.count(1)
    burst_counter = itertools.count(1)
    result: list[dict] = []
    burst_source_files: set[str] = set()

    def make_sid() -> str:
        return f"S{next(scene_counter):05d}"

    def make_bid() -> str:
        return f"B{next(burst_counter):05d}"

    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(input_dir))
        except ValueError:
            return Path(p).name

    # ---- Burst groups + their members ----
    burst_records: list[dict] = []
    for bg in burst_groups:
        burst_id = make_bid()
        members = bg["_members"]
        member_ids: list[str] = []

        for m in members:
            sid = make_sid()
            m["scene_id"] = sid
            m["burst_group_id"] = burst_id
            m["source_file_rel"] = rel(m["source_file"])
            burst_source_files.add(m["source_file"])
            member_ids.append(sid)
            result.append(m)

        rep = next(
            (m for m in members
             if m["source_file"] == bg["representative_source_file"]),
            members[0],
        )
        vision = rep.get("vision", {})
        burst_records.append({
            "scene_id": burst_id,
            "type": "burst_group",
            "burst_type": bg["burst_type"],
            "member_scene_ids": member_ids,
            "member_count": len(members),
            "representative_scene_id": rep["scene_id"],
            "time_span_s": bg["time_span_s"],
            "source_file": rep.get("source_file", ""),
            "source_file_rel": rel(rep.get("source_file", "")),
            "previews": {
                "start": members[0].get("previews", {}).get("start"),
                "mid": rep.get("previews", {}).get("start"),
                "end": members[-1].get("previews", {}).get("start"),
                "preview_gif": None,
            },
            "vision": {
                "category": vision.get("category", "Other"),
                "feature_print": vision.get("feature_print"),
            },
            "burst_group_id": burst_id,
            "errors": [],
        })

    # ---- Standalone photos (not in any burst) ----
    for pr in photo_records:
        if pr["source_file"] not in burst_source_files:
            pr["scene_id"] = make_sid()
            pr["burst_group_id"] = None
            pr["source_file_rel"] = rel(pr["source_file"])
            result.append(pr)

    # ---- Video scenes ----
    for vs in video_scenes:
        vs["scene_id"] = make_sid()
        vs["burst_group_id"] = None
        vs["source_file_rel"] = rel(vs["source_file"])
        result.append(vs)

    result.extend(burst_records)
    return result


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_scenes_json(
    scenes: list[dict],
    output_path: Path,
    metadata: dict,
) -> None:
    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in scenes]
    output_path.write_text(
        json.dumps({"metadata": metadata, "scenes": clean}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# HTML review
# ---------------------------------------------------------------------------

def generate_review_html(scenes: list[dict], output_path: Path) -> None:
    displayable = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in scenes
        if s.get("type") in ("video_scene", "photo", "burst_group")
    ]
    scenes_json = json.dumps(displayable, ensure_ascii=False, separators=(",", ":"))

    n_video  = sum(1 for s in scenes if s.get("type") == "video_scene")
    n_photo  = sum(1 for s in scenes if s.get("type") == "photo")
    n_burst  = sum(1 for s in scenes if s.get("type") == "burst_group")
    gen_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Scout Review</title>\n"
        "<style>\n"
        "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#111;color:#eee}\n"
        "header{padding:20px 24px 12px;border-bottom:1px solid #333;position:sticky;"
        "top:0;background:#111;z-index:100}\n"
        "h1{font-size:1.4rem;font-weight:600;margin-bottom:4px}\n"
        ".meta{color:#888;font-size:.8rem;margin-bottom:12px}\n"
        ".controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}\n"
        ".filter-btn{background:#222;border:1px solid #444;color:#ccc;padding:5px 12px;"
        "border-radius:20px;cursor:pointer;font-size:.8rem;transition:all .15s}\n"
        ".filter-btn:hover{background:#333}\n"
        ".filter-btn.active{background:#0a84ff;border-color:#0a84ff;color:#fff}\n"
        "#search-box{background:#222;border:1px solid #444;color:#eee;padding:5px 12px;"
        "border-radius:20px;font-size:.8rem;outline:none;min-width:200px;margin-left:auto}\n"
        "#search-box:focus{border-color:#0a84ff}\n"
        "#scene-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));"
        "gap:12px;padding:16px}\n"
        ".scene-card{background:#1c1c1e;border:1px solid #2c2c2e;border-radius:10px;"
        "overflow:hidden;cursor:pointer;transition:transform .15s,box-shadow .15s}\n"
        ".scene-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.5)}\n"
        ".card-filmstrip{display:flex;gap:2px;height:120px}\n"
        ".card-filmstrip img,.gif-wrap img{flex:1;width:33%;height:120px;object-fit:cover}\n"
        ".gif-wrap{flex:1;overflow:hidden}\n"
        ".gif-wrap img{width:100%;height:120px}\n"
        ".card-meta{padding:8px 10px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}\n"
        ".scene-id{font-size:.7rem;color:#888;font-family:monospace}\n"
        ".category-badge{font-size:.68rem;padding:2px 7px;border-radius:10px;"
        "font-weight:600;text-transform:uppercase}\n"
        ".Nature{background:#2d6a4f;color:#d8f3dc}\n"
        ".Urban{background:#4a4e69;color:#c9c9ff}\n"
        ".Food{background:#9c4221;color:#ffd7ba}\n"
        ".People{background:#1d3557;color:#a8d8ea}\n"
        ".Other{background:#3a3a3c;color:#aaa}\n"
        ".duration{font-size:.72rem;color:#aaa}\n"
        ".filename{font-size:.7rem;color:#666;white-space:nowrap;overflow:hidden;"
        "text-overflow:ellipsis;max-width:140px}\n"
        ".type-badge{font-size:.65rem;padding:1px 5px;border-radius:6px;"
        "background:#3a3a3c;color:#aaa}\n"
        ".burst-card{position:relative}\n"
        ".burst-card::before,.burst-card::after{content:'';position:absolute;"
        "border:1px solid #3a3a3c;border-radius:10px;background:#161618;z-index:-1}\n"
        ".burst-card::before{top:-4px;left:4px;right:4px;bottom:4px}\n"
        ".burst-card::after{top:-8px;left:8px;right:8px;bottom:8px}\n"
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
        ".conf-wrap{background:#333;border-radius:3px;height:6px;width:100px;"
        "display:inline-block;vertical-align:middle}\n"
        ".conf-bar{background:#0a84ff;height:6px;border-radius:3px}\n"
        "#sparkline{width:100%;height:40px;margin-top:8px}\n"
        ".lb-section{color:#888;font-size:.72rem;text-transform:uppercase;"
        "letter-spacing:.08em;margin:14px 0 6px}\n"
        ".error-count{color:#ff453a;font-size:.7rem}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "<h1>Scout Review</h1>\n"
        f'<p class="meta">Generated: {gen_at} &nbsp;&middot;&nbsp; '
        f"{n_video} video scenes &nbsp;&middot;&nbsp; "
        f"{n_photo} photos &nbsp;&middot;&nbsp; "
        f"{n_burst} bursts</p>\n"
        '<div class="controls">\n'
        '<button class="filter-btn active" data-category="All">All</button>\n'
        '<button class="filter-btn" data-category="Nature">Nature</button>\n'
        '<button class="filter-btn" data-category="Urban">Urban</button>\n'
        '<button class="filter-btn" data-category="Food">Food</button>\n'
        '<button class="filter-btn" data-category="People">People</button>\n'
        '<button class="filter-btn" data-category="Other">Other</button>\n'
        '<input type="search" id="search-box" placeholder="Search by filename\u2026">\n'
        "</div>\n"
        "</header>\n"
        '\n<main id="scene-grid"></main>\n'
        '\n<div id="lightbox" class="hidden">\n'
        '  <div class="lb-inner">\n'
        '    <button class="lb-close" id="lb-close">&times;</button>\n'
        '    <img class="lb-img" id="lb-img" src="" alt="">\n'
        '    <p class="lb-section">Metadata</p>\n'
        '    <table class="lb-table" id="lb-table"></table>\n'
        '    <p class="lb-section">Top Classifications</p>\n'
        '    <table class="lb-table" id="lb-cls"></table>\n'
        '    <p class="lb-section">Feature Print</p>\n'
        '    <canvas id="sparkline" width="700" height="40"></canvas>\n'
        "  </div>\n"
        "</div>\n"
        "\n<script>\n"
        f"const SCENES={scenes_json};\n"
        "\n"
        "function catClass(c){return c||'Other';}\n"
        "\n"
        "function renderCards(){\n"
        "  const grid=document.getElementById('scene-grid');\n"
        "  const frag=document.createDocumentFragment();\n"
        "  SCENES.forEach(scene=>{\n"
        "    const cat=(scene.vision&&scene.vision.category)||'Other';\n"
        "    const isBurst=scene.type==='burst_group';\n"
        "    const art=document.createElement('article');\n"
        "    art.className='scene-card'+(isBurst?' burst-card':'');\n"
        "    art.dataset.id=scene.scene_id;\n"
        "    art.dataset.category=cat;\n"
        "    art.dataset.type=scene.type||'';\n"
        "    const p=scene.previews||{};\n"
        "    const s0=p.start||''; const s1=p.mid||p.start||''; const s2=p.end||p.start||'';\n"
        "    const thumbs=`<img class='thumb' src='${s0}' loading='lazy' alt=''>`\n"
        "      +`<img class='thumb' src='${s1}' loading='lazy' alt=''>`\n"
        "      +`<img class='thumb' src='${s2}' loading='lazy' alt=''>`;\n"
        "    const durLabel=scene.duration_s!=null?scene.duration_s.toFixed(1)+'s':\n"
        "      (scene.type==='photo'?'photo':'');\n"
        "    const fn=(scene.source_file_rel||scene.source_file||'').split('/').pop();\n"
        "    const typeLabel=isBurst?`burst \u00d7${scene.member_count}`:\n"
        "      (scene.type==='photo'?'photo':'video');\n"
        "    const errs=(scene.errors||[]).length;\n"
        "    art.innerHTML=`<div class='card-filmstrip'>${thumbs}</div>`\n"
        "      +`<div class='card-meta'>`\n"
        "      +`<span class='scene-id'>${scene.scene_id}</span>`\n"
        "      +`<span class='category-badge ${cat}'>${cat}</span>`\n"
        "      +`<span class='type-badge'>${typeLabel}</span>`\n"
        "      +(durLabel?`<span class='duration'>${durLabel}</span>`:'')\n"
        "      +`<span class='filename' title='${scene.source_file||''}'>${fn}</span>`\n"
        "      +(errs?`<span class='error-count'>${errs} err</span>`:'')\n"
        "      +'</div>';\n"
        "    const filmstrip=art.querySelector('.card-filmstrip');\n"
        "    art.addEventListener('mouseenter',()=>{\n"
        "      if(p.preview_gif)filmstrip.innerHTML=`<div class='gif-wrap'><img src='${p.preview_gif}' alt=''></div>`;\n"
        "    });\n"
        "    art.addEventListener('mouseleave',()=>{filmstrip.innerHTML=thumbs;});\n"
        "    art.addEventListener('click',()=>openLightbox(scene));\n"
        "    frag.appendChild(art);\n"
        "  });\n"
        "  grid.appendChild(frag);\n"
        "}\n"
        "\n"
        "document.querySelectorAll('.filter-btn').forEach(btn=>{\n"
        "  btn.addEventListener('click',()=>{\n"
        "    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));\n"
        "    btn.classList.add('active');\n"
        "    const cat=btn.dataset.category;\n"
        "    document.querySelectorAll('.scene-card').forEach(card=>{\n"
        "      card.style.display=(cat==='All'||card.dataset.category===cat)?'':'none';\n"
        "    });\n"
        "  });\n"
        "});\n"
        "\n"
        "document.getElementById('search-box').addEventListener('input',function(){\n"
        "  const term=this.value.toLowerCase();\n"
        "  document.querySelectorAll('.scene-card').forEach(card=>{\n"
        "    const scene=SCENES.find(s=>s.scene_id===card.dataset.id);\n"
        "    if(!scene)return;\n"
        "    const name=(scene.source_file_rel||scene.source_file||'').toLowerCase();\n"
        "    card.style.display=name.includes(term)?'':'none';\n"
        "  });\n"
        "});\n"
        "\n"
        "function openLightbox(scene){\n"
        "  const p=scene.previews||{};\n"
        "  document.getElementById('lb-img').src=p.mid||p.start||p.end||'';\n"
        "  const v=scene.vision||{};\n"
        "  const e=scene.exif||null;\n"
        "  const rows=[\n"
        "    ['Scene ID',scene.scene_id],\n"
        "    ['Type',scene.type||''],\n"
        "    ['File',scene.source_file_rel||scene.source_file||''],\n"
        "    ['Category',v.category||'Other'],\n"
        "    ['Duration',scene.duration_s!=null?scene.duration_s.toFixed(2)+'s':'—'],\n"
        "    ['FPS',scene.fps!=null?scene.fps:'—'],\n"
        "    ['Resolution',scene.resolution?scene.resolution.join(' \u00d7 '):'—'],\n"
        "    ...(e?[['Camera',(e.make||'')+' '+(e.model||'')]]:[]),\n"
        "    ...(e&&e.datetime_original?[['Taken',e.datetime_original]]:[]),\n"
        "    ...(e&&e.gps_lat!=null?[['GPS',e.gps_lat.toFixed(5)+', '+e.gps_lon.toFixed(5)]]:[]),\n"
        "    ...(scene.burst_group_id?[['Burst Group',scene.burst_group_id]]:[]),\n"
        "    ...(scene.type==='burst_group'?[['Members',(scene.member_scene_ids||[]).join(', ')]]:[]),\n"
        "    ...((scene.errors||[]).length?[['Errors',(scene.errors||[]).length+' (see scenes.json)']]:[]),\n"
        "  ];\n"
        "  document.getElementById('lb-table').innerHTML=rows.map(([k,v])=>\n"
        "    `<tr><td>${k}</td><td>${v}</td></tr>`).join('');\n"
        "  const cls=(v.classifications||[]).slice(0,5);\n"
        "  document.getElementById('lb-cls').innerHTML=cls.map(c=>{\n"
        "    const pct=Math.round(c.confidence*100);\n"
        "    return `<tr><td>${c.label}</td><td>`\n"
        "      +`<div class='conf-wrap'><div class='conf-bar' style='width:${pct}%'></div></div>`\n"
        "      +` ${pct}%</td></tr>`;\n"
        "  }).join('')||'<tr><td colspan=2 style=color:#666>No classifications</td></tr>';\n"
        "  const fp=v.feature_print;\n"
        "  const cv=document.getElementById('sparkline');\n"
        "  const ctx=cv.getContext('2d');\n"
        "  ctx.clearRect(0,0,cv.width,cv.height);\n"
        "  if(fp&&fp.length>0){\n"
        "    const step=fp.length>128?Math.floor(fp.length/128):1;\n"
        "    const sample=fp.filter((_,i)=>i%step===0);\n"
        "    const min=Math.min(...sample),max=Math.max(...sample),range=max-min||1;\n"
        "    const bw=cv.width/sample.length;\n"
        "    ctx.fillStyle='#0a84ff';\n"
        "    sample.forEach((v,i)=>{\n"
        "      const h=((v-min)/range)*cv.height;\n"
        "      ctx.fillRect(i*bw,cv.height-h,Math.max(1,bw-1),h);\n"
        "    });\n"
        "  }\n"
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
        "renderCards();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input", "-i", "input_dir",
    required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root directory to scan recursively for media.",
)
@click.option(
    "--output", "-o", "output_dir",
    default="./scout_output", show_default=True, type=click.Path(path_type=Path),
    help="Directory for all generated artifacts.",
)
@click.option(
    "--extensions",
    default="mov,mp4,heic,jpg,jpeg,png", show_default=True,
    help="Comma-separated file extensions to ingest.",
)
@click.option(
    "--scene-threshold", default=27.0, show_default=True,
    help="PySceneDetect ContentDetector threshold (0–100).",
)
@click.option(
    "--burst-window", default=3.0, show_default=True,
    help="Seconds window for temporal burst grouping.",
)
@click.option(
    "--burst-min-count", default=3, show_default=True,
    help="Minimum photos in a time window to form a burst.",
)
@click.option(
    "--similarity-threshold", default=0.15, show_default=True,
    help="Max feature-print L2 distance for visual similarity bursts.",
)
@click.option(
    "--workers", default=4, show_default=True,
    help="Concurrent threads for Vision analysis.",
)
@click.option(
    "--extract-snippets", "extract_snippets",
    is_flag=True, default=False,
    help="For scenes longer than --snippet-min-scene, divide them into windows "
         "and extract the sharpest --snippet-duration clip from each window.",
)
@click.option(
    "--snippet-duration",
    default=5.0, show_default=True,
    help="Duration (seconds) of each extracted snippet.",
)
@click.option(
    "--snippet-min-scene",
    default=10.0, show_default=True,
    help="Scenes shorter than this (seconds) are left untouched; "
         "longer scenes are divided into snippet-sized windows.",
)
@click.option(
    "--snippet-max-per-scene",
    default=3, show_default=True,
    help="Maximum snippets to extract from one scene (0 = unlimited, "
         "extract one per snippet-duration window).",
)
@click.option(
    "--snippet-dir", "snippet_dir",
    default=None, type=click.Path(path_type=Path),
    help="Directory for extracted snippet clips. "
         "Defaults to {output_dir}/snippets.",
)
@click.option(
    "--normalise",
    type=click.Choice(["none", "landscape", "portrait"], case_sensitive=False),
    default="none", show_default=True,
    help="Pre-render all media to a common aspect ratio before analysis. "
         "'landscape' targets 1920×1080 (16:9); 'portrait' targets 1080×1920 (9:16).",
)
@click.option(
    "--normalise-dir", "normalise_dir",
    default=None, type=click.Path(path_type=Path),
    help="Output directory for normalised media. "
         "Defaults to {output_dir}/normalised.",
)
@click.option(
    "--normalise-mode",
    type=click.Choice(["pad", "crop"], case_sensitive=False),
    default="pad", show_default=True,
    help="How to handle aspect-ratio mismatch: "
         "'pad' adds black bars; 'crop' scales to fill and centre-crops.",
)
@click.option(
    "--skip-previews", is_flag=True, default=False,
    help="Skip GIF/thumbnail generation (faster, no preview images).",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Show verbose debug output.",
)
@click.option(
    "--resume", is_flag=True, default=False,
    help="Skip source files already present in scenes.json.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    extensions: str,
    scene_threshold: float,
    burst_window: float,
    burst_min_count: int,
    similarity_threshold: float,
    workers: int,
    extract_snippets: bool,
    snippet_duration: float,
    snippet_min_scene: float,
    snippet_max_per_scene: int,
    snippet_dir: Path | None,
    normalise: str,
    normalise_dir: Path | None,
    normalise_mode: str,
    skip_previews: bool,
    verbose: bool,
    resume: bool,
) -> None:
    """Scout — Phase 1 of the macOS AI Montage Suite.

    Scans INPUT_DIR for video and photo files, runs scene detection and
    Vision framework analysis, groups photo bursts, and writes
    scenes.json + scout_review.html into OUTPUT_DIR.
    """
    previews_dir = output_dir / "previews"
    ffmpeg_bin = preflight_checks(output_dir)
    ext_list = [e.strip() for e in extensions.split(",") if e.strip()]

    # ------------------------------------------------------------------
    # Resume: load already-processed source files
    # ------------------------------------------------------------------
    already_processed: set[str] = set()
    scenes_json_path = output_dir / "scenes.json"
    if resume and scenes_json_path.exists():
        try:
            existing = json.loads(scenes_json_path.read_text())
            for s in existing.get("scenes", []):
                sf = s.get("source_file")
                if sf:
                    already_processed.add(sf)
            console.print(
                f"[yellow]Resume:[/] skipping {len(already_processed)} already-processed files."
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/] could not read existing scenes.json: {exc}")

    # ------------------------------------------------------------------
    # Discover media
    # ------------------------------------------------------------------
    console.print("[bold]Scanning for media…[/]")
    media = discover_media(input_dir, ext_list)
    video_files = [p for p in media["video"] if str(p) not in already_processed]
    photo_files = [p for p in media["photo"] if str(p) not in already_processed]
    console.print(
        f"Found [cyan]{len(video_files)}[/] video files, "
        f"[cyan]{len(photo_files)}[/] photo files."
    )

    # ------------------------------------------------------------------
    # Stage 0: Normalise media (optional)
    # ------------------------------------------------------------------
    if normalise != "none":
        if normalise_dir is None:
            normalise_dir = output_dir / "normalised"
        video_files, photo_files = normalise_media(
            video_files=video_files,
            photo_files=photo_files,
            input_dir=input_dir,
            normalise_dir=normalise_dir,
            target=normalise,
            mode=normalise_mode,
            ffmpeg_bin=ffmpeg_bin,
            workers=workers,
        )

    # ------------------------------------------------------------------
    # Stage 1: Scene detection for all videos
    # ------------------------------------------------------------------
    all_video_scenes: list[dict] = []

    if video_files:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Detecting scenes…", total=len(video_files)
            )
            for video_path in video_files:
                progress.update(task, description=f"[cyan]Detecting: {video_path.name}")
                vid_info = get_video_info(video_path)
                try:
                    raw_scenes = detect_scenes(video_path, scene_threshold)
                except Exception as exc:
                    console.print(
                        f"[red]Scene detection failed[/] for {video_path.name}: {exc}"
                    )
                    progress.advance(task)
                    continue

                for raw in raw_scenes:
                    temp_id = _next_temp_id()
                    scene_dict: dict[str, Any] = {
                        "type": "video_scene",
                        "source_file": str(video_path),
                        "scene_index": raw["scene_index"],
                        "start_frame": raw["start_frame"],
                        "end_frame": raw["end_frame"],
                        "start_time_s": raw["start_time_s"],
                        "end_time_s": raw["end_time_s"],
                        "duration_s": raw["duration_s"],
                        "fps": vid_info["fps"],
                        "resolution": vid_info["resolution"],
                        "previews": {
                            "start": None, "mid": None, "end": None, "preview_gif": None,
                        },
                        "previews_generated": False,
                        "vision": {
                            "classifications": [],
                            "category": "Other",
                            "feature_print": None,
                            "feature_print_frame": "mid",
                        },
                        "exif": None,
                        "errors": [],
                        "_temp_id": temp_id,
                        "_video_path": video_path,
                    }
                    all_video_scenes.append(scene_dict)

                progress.advance(task)

    # ------------------------------------------------------------------
    # Stage 1.5: Snippet extraction for long scenes (optional)
    # ------------------------------------------------------------------
    if extract_snippets and all_video_scenes:
        if snippet_dir is None:
            snippet_dir = output_dir / "snippets"
        all_video_scenes = extract_scene_snippets(
            all_video_scenes=all_video_scenes,
            snippet_dur=snippet_duration,
            snippet_dir=snippet_dir,
            min_scene_dur=snippet_min_scene,
            sample_interval=2.0,
            ffmpeg_bin=ffmpeg_bin,
            workers=workers,
            max_per_scene=snippet_max_per_scene,
        )

    # ------------------------------------------------------------------
    # Stage 2: Frame extraction for video scenes
    # ------------------------------------------------------------------
    if all_video_scenes and not skip_previews:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Extracting frames…", total=len(all_video_scenes)
            )
            for sd in all_video_scenes:
                vp = sd.pop("_video_path")
                tid = sd["_temp_id"]
                scene_raw = {
                    "start_time_s": sd["start_time_s"],
                    "end_time_s": sd["end_time_s"],
                }
                progress.update(
                    task,
                    description=f"[cyan]Frames: {vp.name} scene {sd['scene_index']}",
                )
                previews, errs = extract_frames(
                    vp, scene_raw, previews_dir, tid, ffmpeg_bin
                )
                sd["previews"] = previews
                sd["previews_generated"] = any(v for v in previews.values())
                sd["errors"].extend(errs)
                progress.advance(task)
    else:
        # Remove internal _video_path key even when skipping previews
        for sd in all_video_scenes:
            sd.pop("_video_path", None)

    # ------------------------------------------------------------------
    # Stage 3: Vision analysis on video mid-frames
    # ------------------------------------------------------------------
    if all_video_scenes:
        mid_frame_paths: list[Path] = []
        for sd in all_video_scenes:
            mid_rel = sd["previews"].get("mid")
            if mid_rel:
                mid_abs = output_dir / mid_rel
                if mid_abs.exists():
                    mid_frame_paths.append(mid_abs)
                    sd["_vision_key"] = str(mid_abs)

        if mid_frame_paths:
            console.print(
                f"[cyan]Running Vision on {len(mid_frame_paths)} video frames "
                f"({workers} threads)…[/]"
            )
            video_analysis = analyse_images_parallel(mid_frame_paths, workers)
            for sd in all_video_scenes:
                key = sd.pop("_vision_key", None)
                if key and key in video_analysis:
                    a = video_analysis[key]
                    sd["vision"]["classifications"] = a.get("classifications", [])
                    sd["vision"]["category"] = a.get("category", "Other")
                    sd["vision"]["feature_print"] = a.get("feature_print")
                    sd["errors"].extend(a.get("errors", []))
        else:
            for sd in all_video_scenes:
                sd.pop("_vision_key", None)

    # Clean up remaining internal keys
    for sd in all_video_scenes:
        sd.pop("_temp_id", None)
        sd.pop("_vision_key", None)
        sd.pop("_video_path", None)

    # ------------------------------------------------------------------
    # Stage 4: Photo thumbnails + EXIF
    # ------------------------------------------------------------------
    photo_records: list[dict] = []

    if photo_files:
        console.print(f"[green]Processing {len(photo_files)} photos…[/]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Photos…", total=len(photo_files))
            for photo_path in photo_files:
                progress.update(task, description=f"[green]{photo_path.name}")
                thumb_rel: str | None = None
                if not skip_previews:
                    tid = _next_temp_id()
                    thumb_out = previews_dir / f"{tid}_thumb.jpg"
                    if make_photo_thumbnail(photo_path, thumb_out):
                        thumb_rel = f"previews/{thumb_out.name}"
                exif_data = read_exif_metadata(photo_path)
                resolution = get_photo_resolution(photo_path)
                photo_records.append({
                    "type": "photo",
                    "source_file": str(photo_path),
                    "scene_index": None,
                    "start_frame": None,
                    "end_frame": None,
                    "start_time_s": None,
                    "end_time_s": None,
                    "duration_s": None,
                    "fps": None,
                    "resolution": resolution,
                    "previews": {
                        "start": thumb_rel, "mid": None, "end": None, "preview_gif": None,
                    },
                    "previews_generated": thumb_rel is not None,
                    "vision": {
                        "classifications": [],
                        "category": "Other",
                        "feature_print": None,
                        "feature_print_frame": "full",
                    },
                    "exif": exif_data,
                    "errors": [],
                    "burst_group_id": None,
                })
                progress.advance(task)

    # ------------------------------------------------------------------
    # Stage 5: Vision analysis on photos
    # ------------------------------------------------------------------
    if photo_records:
        console.print(
            f"[green]Running Vision on {len(photo_files)} photos "
            f"({workers} threads)…[/]"
        )
        photo_analysis = analyse_images_parallel(photo_files, workers)
        for pr in photo_records:
            a = photo_analysis.get(pr["source_file"], {})
            pr["vision"]["classifications"] = a.get("classifications", [])
            pr["vision"]["category"] = a.get("category", "Other")
            pr["vision"]["feature_print"] = a.get("feature_print")
            pr["errors"].extend(a.get("errors", []))

    # ------------------------------------------------------------------
    # Stage 6: Burst detection
    # ------------------------------------------------------------------
    console.print("[yellow]Detecting burst groups…[/]")
    burst_groups = detect_burst_groups(
        photo_records, burst_window, burst_min_count, similarity_threshold
    )
    console.print(f"Found [yellow]{len(burst_groups)}[/] burst groups.")

    # ------------------------------------------------------------------
    # Stage 7: Assemble final scene list + assign IDs
    # ------------------------------------------------------------------
    console.print("[bold]Assembling scene list…[/]")
    scenes = build_scenes_list(all_video_scenes, photo_records, burst_groups, input_dir)

    # ------------------------------------------------------------------
    # Stage 8: Write outputs
    # ------------------------------------------------------------------
    n_scenes = sum(1 for s in scenes if s.get("type") in ("video_scene", "photo"))
    n_bursts = sum(1 for s in scenes if s.get("type") == "burst_group")
    n_errors = sum(len(s.get("errors", [])) for s in scenes)

    metadata: dict[str, Any] = {
        "scout_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_scenes": n_scenes,
        "total_videos": len(video_files),
        "total_photos": len(photo_files),
        "total_bursts": n_bursts,
        "scene_threshold": scene_threshold,
        "burst_window_s": burst_window,
        "burst_min_count": burst_min_count,
        "similarity_threshold": similarity_threshold,
    }

    write_scenes_json(scenes, scenes_json_path, metadata)
    console.print(f"[bold green]\u2713[/] scenes.json   \u2192 [cyan]{scenes_json_path}[/]")

    html_path = output_dir / "scout_review.html"
    generate_review_html(scenes, html_path)
    console.print(f"[bold green]\u2713[/] scout_review.html \u2192 [cyan]{html_path}[/]")

    err_style = "[red]" if n_errors else ""
    err_end   = "[/]"  if n_errors else ""
    console.print(
        f"\n[bold]Done.[/] {n_scenes} scenes · {n_bursts} bursts · "
        f"{err_style}{n_errors} errors{err_end}"
    )
    console.print(f"Open review in browser: file://{html_path}")


if __name__ == "__main__":
    main()
