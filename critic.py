#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyobjc-core>=9.0",
#   "pyobjc-framework-Cocoa>=9.0",
#   "pyobjc-framework-Vision>=9.0",
#   "opencv-python-headless>=4.8",
#   "numpy>=1.24",
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Phase 2 of the macOS AI Montage Suite — The Critic.

Reads scenes.json produced by scout.py, evaluates every scene and photo
using a sliding-window strategy and the macOS Vision framework, and writes
scored_snippets.json.

Usage:
    uv run critic.py --scenes ./scout_output/scenes.json [OPTIONS]
"""
from __future__ import annotations

import itertools
import json
import math
import platform
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
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

_snippet_counter = itertools.count(1)


def _next_cid() -> str:
    return f"C{next(_snippet_counter):05d}"


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SIZE_S = 3.0
DEFAULT_WINDOW_STEP_S = 0.5
DEFAULT_BLUR_THRESHOLD = 50.0
DEFAULT_TILT_THRESHOLD = 15.0

# Composite score weights (must sum to 1.0)
W_AESTHETIC  = 0.40
W_SALIENCY   = 0.20
W_SMILE      = 0.20
W_BLUR       = 0.10
W_HORIZON    = 0.10

# Whether VNGenerateImageAestheticsScoresRequest is available (macOS 14+)
_AESTHETICS_AVAILABLE: bool | None = None


def _check_aesthetics_api() -> bool:
    global _AESTHETICS_AVAILABLE
    if _AESTHETICS_AVAILABLE is None:
        _AESTHETICS_AVAILABLE = (
            VISION_AVAILABLE
            and hasattr(Vision, "VNCalculateImageAestheticsScoresRequest")
        )
    return bool(_AESTHETICS_AVAILABLE)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_checks(output_dir: Path) -> Path:
    """Validate system requirements. Returns ffmpeg Path."""
    ver_str = platform.mac_ver()[0]
    if ver_str:
        parts = ver_str.split(".")
        if int(parts[0]) < 13:
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

    return Path(ffmpeg_bin)


# ---------------------------------------------------------------------------
# Load scenes.json
# ---------------------------------------------------------------------------

def load_scenes_json(scenes_path: Path) -> tuple[dict, list[dict]]:
    """Return (metadata, scenes_list)."""
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    return data.get("metadata", {}), data.get("scenes", [])


def filter_scorable_scenes(scenes: list[dict]) -> list[dict]:
    """Return scenes that the critic should score.

    Burst group records are skipped — their representative member
    is a regular photo record and will be scored directly.
    """
    return [s for s in scenes if s.get("type") in ("video_scene", "photo")]


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def compute_windows(
    start_time_s: float,
    end_time_s: float,
    window_size_s: float,
    window_step_s: float,
) -> list[dict]:
    """Return list of window dicts covering [start_time_s, end_time_s]."""
    duration = end_time_s - start_time_s
    if duration <= 0:
        return []

    if duration <= window_size_s:
        mid = (start_time_s + end_time_s) / 2.0
        return [{
            "start_time_s": start_time_s,
            "end_time_s": end_time_s,
            "duration_s": duration,
            "mid_time_s": mid,
        }]

    windows: list[dict] = []
    w_start = start_time_s
    while w_start + window_size_s <= end_time_s + 1e-9:
        w_end = min(w_start + window_size_s, end_time_s)
        windows.append({
            "start_time_s": round(w_start, 4),
            "end_time_s": round(w_end, 4),
            "duration_s": round(w_end - w_start, 4),
            "mid_time_s": round((w_start + w_end) / 2.0, 4),
        })
        w_start += window_step_s

    return windows


# ---------------------------------------------------------------------------
# FFmpeg frame extraction
# ---------------------------------------------------------------------------

def extract_frame(
    source_file: Path,
    time_s: float,
    output_path: Path,
    ffmpeg_bin: Path,
) -> bool:
    """Extract a single JPEG frame at time_s. Returns True on success."""
    cmd = [
        str(ffmpeg_bin),
        "-ss", f"{time_s:.4f}",
        "-i", str(source_file),
        "-vframes", "1",
        "-q:v", "3",
        "-vf", "scale=640:-1",
        str(output_path), "-y",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Vision scoring
# ---------------------------------------------------------------------------

def _cgRect_xywh(bbox: Any) -> tuple[float, float, float, float]:
    """Unpack a CGRect into (x, y, w, h), handling struct or tuple forms."""
    try:
        return (
            float(bbox.origin.x),
            float(bbox.origin.y),
            float(bbox.size.width),
            float(bbox.size.height),
        )
    except AttributeError:
        # PyObjC may return ((x, y), (w, h))
        try:
            (x, y), (w, h) = bbox[0], bbox[1]
            return float(x), float(y), float(w), float(h)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


def score_frame_vision(image_path: Path) -> dict:
    """Run Vision scoring requests on a single frame image.

    Returns a dict with keys:
        aesthetic, saliency_coverage, saliency_centroid, salient_objects,
        smile, face_count, horizon_angle_deg, errors
    """
    result: dict[str, Any] = {
        "aesthetic": 0.5,        # neutral default when unavailable
        "saliency_coverage": 0.0,
        "saliency_centroid": None,
        "salient_objects": [],
        "smile": 0.0,
        "face_count": 0,
        "horizon_angle_deg": 0.0,
        "errors": [],
    }

    if not VISION_AVAILABLE:
        result["errors"].append({
            "stage": "vision_scoring",
            "message": "Vision framework not available",
            "detail": "",
        })
        return result

    try:
        url = NSURL.fileURLWithPath_(str(image_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})

        requests: list[Any] = []

        # Aesthetics (macOS 14+)
        aesthetics_req = None
        if _check_aesthetics_api():
            try:
                aesthetics_req = Vision.VNCalculateImageAestheticsScoresRequest.alloc().init()
                requests.append(aesthetics_req)
            except Exception:
                pass

        # Attention-based saliency
        saliency_req = Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
        requests.append(saliency_req)

        # Horizon detection
        horizon_req = Vision.VNDetectHorizonRequest.alloc().init()
        requests.append(horizon_req)

        # Face landmarks (includes expressions on macOS 14+)
        face_req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        requests.append(face_req)

        success, err = handler.performRequests_error_(requests, None)
        if not success:
            result["errors"].append({
                "stage": "vision_scoring",
                "message": f"performRequests_error_ failed: {err}",
                "detail": "",
            })
            return result

        # ---- Aesthetics ----
        if aesthetics_req is not None:
            obs_list = aesthetics_req.results() or []
            if obs_list:
                # overallScore is in [-1, 1]; normalise to [0, 1]
                raw = float(obs_list[0].overallScore())
                result["aesthetic"] = (raw + 1.0) / 2.0

        # ---- Saliency ----
        sal_obs_list = saliency_req.results() or []
        if sal_obs_list:
            sal_obs = sal_obs_list[0]
            salient = sal_obs.salientObjects() or []
            total_area = 0.0
            weighted_cx, weighted_cy, weight_sum = 0.0, 0.0, 0.0
            for obj in salient:
                x, y, w, h = _cgRect_xywh(obj.boundingBox())
                area = w * h
                total_area += area
                cx = x + w / 2.0
                cy = y + h / 2.0
                weighted_cx += cx * area
                weighted_cy += cy * area
                weight_sum += area
                result["salient_objects"].append({
                    "x": round(x, 4), "y": round(y, 4),
                    "w": round(w, 4), "h": round(h, 4),
                    "confidence": round(float(obj.confidence()), 4),
                })
            result["saliency_coverage"] = round(min(1.0, total_area), 4)
            if weight_sum > 0:
                result["saliency_centroid"] = [
                    round(weighted_cx / weight_sum, 4),
                    round(weighted_cy / weight_sum, 4),
                ]

        # ---- Horizon ----
        hz_obs_list = horizon_req.results() or []
        if hz_obs_list:
            angle_rad = float(hz_obs_list[0].angle())
            result["horizon_angle_deg"] = round(math.degrees(abs(angle_rad)), 3)

        # ---- Face landmarks + optional expressions ----
        face_obs_list = face_req.results() or []
        result["face_count"] = len(face_obs_list)
        for face_obs in face_obs_list:
            # expressions() available macOS 14+ on VNFaceObservation
            try:
                exprs = face_obs.expressions()
                if exprs is not None:
                    # happiness() returns float 0-1
                    happiness = float(exprs.happiness())
                    result["smile"] = max(result["smile"], happiness)
            except Exception:
                # Treat face presence as weak smile signal if no expressions API
                result["smile"] = max(result["smile"], 0.2)

    except Exception as exc:
        result["errors"].append({
            "stage": "vision_scoring",
            "message": str(exc),
            "detail": traceback.format_exc(),
        })

    return result


# ---------------------------------------------------------------------------
# Blur scoring (OpenCV)
# ---------------------------------------------------------------------------

def compute_blur(image_path: Path) -> float:
    """Return Laplacian variance of the image. Higher = sharper."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-frame scoring (Vision + blur combined, runs in a thread)
# ---------------------------------------------------------------------------

def score_frame(image_path: Path) -> dict:
    """Full scoring for one frame: Vision + Laplacian blur."""
    vision = score_frame_vision(image_path)
    blur_var = compute_blur(image_path)
    return {**vision, "blur_variance": blur_var}


def score_frames_parallel(
    frame_paths: list[Path],
    workers: int,
) -> dict[str, dict]:
    """Score a list of frame images in parallel. Returns {str(path): scores}."""
    out: dict[str, dict] = {}
    if not frame_paths:
        return out
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score_frame, p): p for p in frame_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                out[str(path)] = future.result()
            except Exception as exc:
                out[str(path)] = {
                    "aesthetic": 0.5, "saliency_coverage": 0.0,
                    "saliency_centroid": None, "salient_objects": [],
                    "smile": 0.0, "face_count": 0,
                    "horizon_angle_deg": 0.0, "blur_variance": 0.0,
                    "errors": [{"stage": "score_frame", "message": str(exc), "detail": ""}],
                }
    return out


# ---------------------------------------------------------------------------
# Score composition
# ---------------------------------------------------------------------------

def compute_composite(
    scores: dict,
    blur_threshold: float,
    tilt_threshold: float,
) -> tuple[float, float, float, bool, str | None]:
    """Return (composite, blur_score, horizon_score, discarded, discard_reason)."""
    blur_var = scores.get("blur_variance", 0.0)
    horizon_deg = scores.get("horizon_angle_deg", 0.0)

    blur_score = min(1.0, blur_var / 500.0)
    horizon_score = max(0.0, 1.0 - horizon_deg / max(tilt_threshold, 1e-9))

    composite = (
        W_AESTHETIC * scores.get("aesthetic", 0.5)
        + W_SALIENCY * scores.get("saliency_coverage", 0.0)
        + W_SMILE    * scores.get("smile", 0.0)
        + W_BLUR     * blur_score
        + W_HORIZON  * horizon_score
    )

    discarded = False
    discard_reason: str | None = None
    if blur_var < blur_threshold:
        discarded = True
        discard_reason = "blurry"
    elif horizon_deg > tilt_threshold:
        discarded = True
        discard_reason = "tilted"

    return round(composite, 4), round(blur_score, 4), round(horizon_score, 4), discarded, discard_reason


def find_peak_window(
    windows: list[dict],
    frame_scores: dict[str, dict],   # {str(frame_path): raw_scores}
    frame_paths: list[Path],          # parallel with windows
    blur_threshold: float,
    tilt_threshold: float,
) -> tuple[dict, dict, Path | None]:
    """Return (best_window, best_scores, best_frame_path).

    Prefers non-discarded windows. Falls back to best discarded window.
    """
    best_passing: tuple[float, int] | None = None   # (composite, idx)
    best_failing: tuple[float, int] | None = None

    enriched: list[tuple[float, dict, bool, str | None]] = []
    for i, (w, fp) in enumerate(zip(windows, frame_paths)):
        raw = frame_scores.get(str(fp), {})
        composite, blur_score, horizon_score, disc, reason = compute_composite(
            raw, blur_threshold, tilt_threshold
        )
        enriched.append((composite, raw, disc, reason, blur_score, horizon_score, i))
        if not disc:
            if best_passing is None or composite > best_passing[0]:
                best_passing = (composite, i)
        else:
            if best_failing is None or composite > best_failing[0]:
                best_failing = (composite, i)

    chosen_idx = (best_passing or best_failing or (0.0, 0))[1]
    composite, raw, disc, reason, blur_score, horizon_score, _ = enriched[chosen_idx]

    best_window = windows[chosen_idx]
    best_path = frame_paths[chosen_idx] if chosen_idx < len(frame_paths) else None

    best_scores = {
        "aesthetic":        round(float(raw.get("aesthetic", 0.5)), 4),
        "saliency_coverage": round(float(raw.get("saliency_coverage", 0.0)), 4),
        "smile":            round(float(raw.get("smile", 0.0)), 4),
        "blur_variance":    round(float(raw.get("blur_variance", 0.0)), 2),
        "blur_score":       blur_score,
        "horizon_angle_deg": round(float(raw.get("horizon_angle_deg", 0.0)), 3),
        "horizon_score":    horizon_score,
        "composite":        composite,
    }

    return best_window, best_scores, best_path, disc, reason, raw


# ---------------------------------------------------------------------------
# Score a video scene (sliding window)
# ---------------------------------------------------------------------------

def score_video_scene(
    scene: dict,
    output_dir: Path,
    ffmpeg_bin: Path,
    window_size_s: float,
    window_step_s: float,
    blur_threshold: float,
    tilt_threshold: float,
    workers: int,
    skip_previews: bool,
) -> dict:
    """Evaluate a video scene. Returns a snippet dict (without snippet_id)."""
    errors: list[dict] = []

    source_file = Path(scene["source_file"])
    start_s = scene.get("start_time_s", 0.0) or 0.0
    end_s   = scene.get("end_time_s", 0.0) or 0.0

    windows = compute_windows(start_s, end_s, window_size_s, window_step_s)
    if not windows:
        return _null_snippet(scene, errors, reason="zero_duration")

    with tempfile.TemporaryDirectory(prefix="critic_") as tmpdir:
        tmp = Path(tmpdir)
        frame_paths: list[Path] = []

        # Extract one frame per window (sequential; fast due to input-seek)
        for i, w in enumerate(windows):
            fp = tmp / f"frame_{i:04d}.jpg"
            ok = extract_frame(source_file, w["mid_time_s"], fp, ffmpeg_bin)
            if not ok:
                errors.append({
                    "stage": "extract_frame",
                    "message": f"FFmpeg failed for window {i} at {w['mid_time_s']:.2f}s",
                    "detail": "",
                })
            frame_paths.append(fp if ok else None)

        # Score all extracted frames in parallel
        valid_frames = [fp for fp in frame_paths if fp is not None and fp.exists()]
        raw_scores = score_frames_parallel(valid_frames, workers)

        # Re-map Nones so indices align with windows
        full_frame_paths = [
            fp if (fp is not None and fp.exists()) else None
            for fp in frame_paths
        ]

        # Substitute empty scores for missing frames
        final_scores: dict[str, dict] = {}
        for fp in full_frame_paths:
            if fp is not None:
                final_scores[str(fp)] = raw_scores.get(str(fp), {})

        # Filter to windows that have a scored frame
        scored_windows  = [w for w, fp in zip(windows, full_frame_paths) if fp is not None]
        scored_paths    = [fp for fp in full_frame_paths if fp is not None]

        if not scored_windows:
            return _null_snippet(scene, errors, reason="all_frames_failed")

        best_window, best_scores, best_frame, discarded, discard_reason, raw = (
            find_peak_window(
                scored_windows, final_scores, scored_paths,
                blur_threshold, tilt_threshold,
            )
        )

        # Save best-frame preview
        preview_rel: str | None = None
        if not skip_previews and best_frame is not None and best_frame.exists():
            cid_placeholder = "__CID__"  # replaced in build_snippets_list
            preview_name = f"{cid_placeholder}_best.jpg"
            preview_out = output_dir / "previews" / preview_name
            try:
                shutil.copy2(best_frame, preview_out)
                preview_rel = f"previews/{preview_name}"
            except Exception as exc:
                errors.append({
                    "stage": "save_preview",
                    "message": str(exc),
                    "detail": "",
                })

    return {
        "scene_id": scene["scene_id"],
        "type": scene["type"],
        "source_file": scene["source_file"],
        "source_file_rel": scene.get("source_file_rel", ""),
        "best_window": best_window,
        "scores": best_scores,
        "saliency_centroid": raw.get("saliency_centroid"),
        "salient_objects": raw.get("salient_objects", []),
        "face_count": raw.get("face_count", 0),
        "windows_evaluated": len(scored_windows),
        "discarded": discarded,
        "discard_reason": discard_reason,
        "_preview_rel": preview_rel,
        "errors": errors + raw.get("errors", []),
    }


# ---------------------------------------------------------------------------
# Score a photo (no sliding window)
# ---------------------------------------------------------------------------

def score_photo(
    scene: dict,
    output_dir: Path,
    ffmpeg_bin: Path,
    blur_threshold: float,
    tilt_threshold: float,
    skip_previews: bool,
) -> dict:
    """Score a photo scene. Returns a snippet dict (without snippet_id)."""
    errors: list[dict] = []
    source_file = Path(scene["source_file"])

    # For photos, use the existing thumbnail if available, else the original
    thumb_rel = (scene.get("previews") or {}).get("start")
    if thumb_rel:
        score_path = output_dir / thumb_rel
        if not score_path.exists():
            score_path = source_file
    else:
        score_path = source_file

    if not score_path.exists():
        return _null_snippet(scene, errors, reason="file_not_found")

    raw = score_frame(score_path)
    errors.extend(raw.get("errors", []))

    composite, blur_score, horizon_score, discarded, discard_reason = compute_composite(
        raw, blur_threshold, tilt_threshold
    )

    best_scores = {
        "aesthetic":         round(float(raw.get("aesthetic", 0.5)), 4),
        "saliency_coverage": round(float(raw.get("saliency_coverage", 0.0)), 4),
        "smile":             round(float(raw.get("smile", 0.0)), 4),
        "blur_variance":     round(float(raw.get("blur_variance", 0.0)), 2),
        "blur_score":        blur_score,
        "horizon_angle_deg": round(float(raw.get("horizon_angle_deg", 0.0)), 3),
        "horizon_score":     horizon_score,
        "composite":         composite,
    }

    # Save preview
    preview_rel: str | None = None
    if not skip_previews:
        if thumb_rel:
            preview_rel = thumb_rel  # reuse existing thumbnail
        # Otherwise the original photo is large; skip copying

    return {
        "scene_id": scene["scene_id"],
        "type": scene["type"],
        "source_file": scene["source_file"],
        "source_file_rel": scene.get("source_file_rel", ""),
        "best_window": {
            "start_time_s": None, "end_time_s": None,
            "duration_s": None, "mid_time_s": None,
        },
        "scores": best_scores,
        "saliency_centroid": raw.get("saliency_centroid"),
        "salient_objects": raw.get("salient_objects", []),
        "face_count": raw.get("face_count", 0),
        "windows_evaluated": 1,
        "discarded": discarded,
        "discard_reason": discard_reason,
        "_preview_rel": preview_rel,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Null snippet helper
# ---------------------------------------------------------------------------

def _null_snippet(scene: dict, errors: list[dict], reason: str) -> dict:
    return {
        "scene_id": scene.get("scene_id", ""),
        "type": scene.get("type", ""),
        "source_file": scene.get("source_file", ""),
        "source_file_rel": scene.get("source_file_rel", ""),
        "best_window": {
            "start_time_s": None, "end_time_s": None,
            "duration_s": None, "mid_time_s": None,
        },
        "scores": {
            "aesthetic": 0.0, "saliency_coverage": 0.0, "smile": 0.0,
            "blur_variance": 0.0, "blur_score": 0.0,
            "horizon_angle_deg": 0.0, "horizon_score": 0.0, "composite": 0.0,
        },
        "saliency_centroid": None,
        "salient_objects": [],
        "face_count": 0,
        "windows_evaluated": 0,
        "discarded": True,
        "discard_reason": reason,
        "_preview_rel": None,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Build snippets list (assign IDs)
# ---------------------------------------------------------------------------

def build_snippets_list(
    raw_snippets: list[dict],
    output_dir: Path,
) -> list[dict]:
    """Assign snippet IDs and rename placeholder preview files."""
    result: list[dict] = []
    previews_dir = output_dir / "previews"

    for raw in raw_snippets:
        cid = _next_cid()
        raw["snippet_id"] = cid

        # Rename placeholder preview file
        prev_rel = raw.pop("_preview_rel", None)
        if prev_rel and "__CID__" in prev_rel:
            old_name = prev_rel.split("/")[-1]
            new_name = old_name.replace("__CID__", cid)
            old_path = previews_dir / old_name
            new_path = previews_dir / new_name
            if old_path.exists():
                try:
                    old_path.rename(new_path)
                    prev_rel = f"previews/{new_name}"
                except Exception:
                    pass
        raw["preview_frame"] = prev_rel

        result.append(raw)

    return result


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_scored_snippets_json(
    snippets: list[dict],
    output_path: Path,
    metadata: dict,
) -> None:
    payload = {"metadata": metadata, "snippets": snippets}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--scenes", "-s", "scenes_path",
    required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to scenes.json produced by scout.py.",
)
@click.option(
    "--output", "-o", "output_dir",
    default=None, type=click.Path(path_type=Path),
    help="Output directory (defaults to the directory containing scenes.json).",
)
@click.option(
    "--window-size", default=DEFAULT_WINDOW_SIZE_S, show_default=True,
    help="Sliding window length in seconds.",
)
@click.option(
    "--window-step", default=DEFAULT_WINDOW_STEP_S, show_default=True,
    help="Sliding window step in seconds.",
)
@click.option(
    "--blur-threshold", default=DEFAULT_BLUR_THRESHOLD, show_default=True,
    help="Laplacian variance below this value → discard as blurry.",
)
@click.option(
    "--tilt-threshold", default=DEFAULT_TILT_THRESHOLD, show_default=True,
    help="Horizon angle in degrees above this value → discard as tilted.",
)
@click.option(
    "--workers", default=4, show_default=True,
    help="Concurrent Vision/scoring threads.",
)
@click.option(
    "--skip-previews", is_flag=True, default=False,
    help="Skip saving best-frame JPEG previews.",
)
@click.option(
    "--resume", is_flag=True, default=False,
    help="Skip scene_ids already present in scored_snippets.json.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Verbose output.",
)
def main(
    scenes_path: Path,
    output_dir: Path | None,
    window_size: float,
    window_step: float,
    blur_threshold: float,
    tilt_threshold: float,
    workers: int,
    skip_previews: bool,
    resume: bool,
    verbose: bool,
) -> None:
    """Critic — Phase 2 of the macOS AI Montage Suite.

    Evaluates every scene and photo from SCENES_PATH using a sliding-window
    strategy and the macOS Vision framework, then writes scored_snippets.json.
    """
    if output_dir is None:
        output_dir = scenes_path.parent

    ffmpeg_bin = preflight_checks(output_dir)

    # ------------------------------------------------------------------
    # Load scenes
    # ------------------------------------------------------------------
    console.print(f"[bold]Loading[/] {scenes_path.name} …")
    scout_meta, all_scenes = load_scenes_json(scenes_path)
    scenes = filter_scorable_scenes(all_scenes)
    console.print(f"Loaded [cyan]{len(scenes)}[/] scorable scenes.")

    # Resume: skip already-scored scene IDs
    already_scored: set[str] = set()
    snippets_path = output_dir / "scored_snippets.json"
    if resume and snippets_path.exists():
        try:
            existing = json.loads(snippets_path.read_text())
            for sn in existing.get("snippets", []):
                sid = sn.get("scene_id")
                if sid:
                    already_scored.add(sid)
            console.print(
                f"[yellow]Resume:[/] skipping {len(already_scored)} already-scored scenes."
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/] could not read existing scored_snippets.json: {exc}")

    scenes_to_process = [s for s in scenes if s.get("scene_id") not in already_scored]
    console.print(f"Processing [cyan]{len(scenes_to_process)}[/] scenes.")

    if not _check_aesthetics_api():
        console.print(
            "[yellow]Note:[/] VNGenerateImageAestheticsScoresRequest not available "
            "(requires macOS 14). Aesthetic score will default to 0.5."
        )

    # ------------------------------------------------------------------
    # Score each scene
    # ------------------------------------------------------------------
    video_scenes = [s for s in scenes_to_process if s.get("type") == "video_scene"]
    photo_scenes = [s for s in scenes_to_process if s.get("type") == "photo"]

    raw_snippets: list[dict] = []

    # ---- Video scenes ----
    if video_scenes:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Scoring video scenes…", total=len(video_scenes)
            )
            for scene in video_scenes:
                fname = Path(scene.get("source_file", "")).name
                progress.update(
                    task,
                    description=f"[cyan]{fname} "
                    f"[dim]scene {scene.get('scene_index', '?')}[/]",
                )
                snippet = score_video_scene(
                    scene=scene,
                    output_dir=output_dir,
                    ffmpeg_bin=ffmpeg_bin,
                    window_size_s=window_size,
                    window_step_s=window_step,
                    blur_threshold=blur_threshold,
                    tilt_threshold=tilt_threshold,
                    workers=workers,
                    skip_previews=skip_previews,
                )
                raw_snippets.append(snippet)
                if verbose and snippet.get("errors"):
                    for e in snippet["errors"]:
                        console.print(
                            f"  [red]err[/] {scene['scene_id']} {e['stage']}: {e['message']}"
                        )
                progress.advance(task)

    # ---- Photo scenes ----
    if photo_scenes:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[green]Scoring photos…", total=len(photo_scenes)
            )
            for scene in photo_scenes:
                fname = Path(scene.get("source_file", "")).name
                progress.update(task, description=f"[green]{fname}")
                snippet = score_photo(
                    scene=scene,
                    output_dir=output_dir,
                    ffmpeg_bin=ffmpeg_bin,
                    blur_threshold=blur_threshold,
                    tilt_threshold=tilt_threshold,
                    skip_previews=skip_previews,
                )
                raw_snippets.append(snippet)
                progress.advance(task)

    # ------------------------------------------------------------------
    # Assign IDs + merge with any previously-scored snippets
    # ------------------------------------------------------------------
    final_snippets = build_snippets_list(raw_snippets, output_dir)

    if resume and already_scored and snippets_path.exists():
        try:
            existing = json.loads(snippets_path.read_text())
            old = [
                sn for sn in existing.get("snippets", [])
                if sn.get("scene_id") in already_scored
            ]
            final_snippets = old + final_snippets
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    n_total     = len(final_snippets)
    n_discarded = sum(1 for sn in final_snippets if sn.get("discarded"))
    n_errors    = sum(len(sn.get("errors", [])) for sn in final_snippets)

    metadata: dict[str, Any] = {
        "critic_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenes_json": str(scenes_path),
        "output_dir": str(output_dir),
        "window_size_s": window_size,
        "window_step_s": window_step,
        "blur_threshold": blur_threshold,
        "tilt_threshold_deg": tilt_threshold,
        "total_snippets": n_total,
        "discarded_snippets": n_discarded,
    }

    write_scored_snippets_json(final_snippets, snippets_path, metadata)
    console.print(
        f"[bold green]\u2713[/] scored_snippets.json \u2192 [cyan]{snippets_path}[/]"
    )

    err_tag = "[red]" if n_errors else ""
    err_end = "[/]"  if n_errors else ""
    console.print(
        f"\n[bold]Done.[/] {n_total} snippets · "
        f"[yellow]{n_discarded} discarded[/] · "
        f"{err_tag}{n_errors} errors{err_end}"
    )


if __name__ == "__main__":
    main()
