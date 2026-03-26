#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.24",
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Phase 4 of the macOS AI Montage Suite — The Director.

Reads scored_snippets.json (Phase 2) and beat_grid.json (Phase 3), applies
constraint-solving and narrative logic to produce a beat-aligned video timeline,
and writes final_sequence.json.

Usage:
    uv run director.py --snippets ./critic_output/scored_snippets.json \\
                       --beats    ./maestro_output/beat_grid.json      \\
                       [OPTIONS]
"""
from __future__ import annotations

import json
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
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
from rich.table import Table

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DirectorError(Exception):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_DUMP_SLOT_DURATION_S = 1.0
MIN_SLOT_DURATION_S         = 0.5   # slots shorter than this are merged
KEN_BURNS_ZOOM_FACTOR       = 1.20  # 20% zoom for Ken Burns end rect
RECENT_FINGERPRINT_WINDOW   = 5     # how many recent fingerprints to compare

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_scored_snippets(path: Path) -> dict:
    """Load and validate scored_snippets.json."""
    if not path.exists():
        raise DirectorError(f"scored_snippets.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "snippets" not in data:
        raise DirectorError("scored_snippets.json missing 'snippets' key")
    return data


def load_beat_grid(path: Path) -> dict:
    """Load and validate beat_grid.json."""
    if not path.exists():
        raise DirectorError(f"beat_grid.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "beat_grid" not in data or "pacing_zones" not in data:
        raise DirectorError("beat_grid.json missing required keys")
    return data


def load_scenes_json(path: Path | None) -> dict[str, list[float]]:
    """Load scene feature prints from scenes.json. Returns {scene_id: fp_array}."""
    if path is None:
        return {}
    if not path.exists():
        console.print(f"[yellow]Warning: scenes.json not found at {path} — skipping feature-print dedup[/]")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    fps: dict[str, list[float]] = {}
    for scene in data.get("scenes", []):
        sid = scene.get("scene_id")
        fp = scene.get("feature_print")
        if sid and isinstance(fp, list) and fp:
            fps[sid] = fp
    console.print(f"  Loaded {len(fps)} feature prints from {path.name}")
    return fps

# ---------------------------------------------------------------------------
# Slot grid construction
# ---------------------------------------------------------------------------

def build_slot_grid(beat_grid_data: dict, min_clip_duration: float = MIN_SLOT_DURATION_S) -> list[dict]:
    """
    Build a list of time-range slots from pacing zones + beat times.

    Each slot:
        {start_s, end_s, duration_s, pacing_zone, beat_aligned}
    """
    pacing_zones: list[dict] = beat_grid_data.get("pacing_zones", [])
    beat_times:   list[float] = beat_grid_data["beat_grid"].get("beat_times", [])
    downbeat_times: list[float] = beat_grid_data["beat_grid"].get("downbeat_times", [])

    beats_arr     = np.array(beat_times, dtype=float) if beat_times else np.array([])
    downbeats_arr = np.array(downbeat_times, dtype=float) if downbeat_times else np.array([])

    slots: list[dict] = []

    for zone in pacing_zones:
        z_start = zone["start_s"]
        z_end   = zone["end_s"]
        z_name  = zone["zone"]
        cut_n   = zone.get("cut_every_n_beats", 2)

        # Find downbeats within this zone
        if len(downbeats_arr) > 0:
            mask = (downbeats_arr >= z_start) & (downbeats_arr < z_end)
            zone_downbeats = downbeats_arr[mask]
        else:
            zone_downbeats = np.array([])

        if len(zone_downbeats) == 0:
            # No downbeats: single slot covering the whole zone
            dur = z_end - z_start
            if dur >= MIN_SLOT_DURATION_S:
                slots.append({
                    "start_s":    round(z_start, 4),
                    "end_s":      round(z_end, 4),
                    "duration_s": round(dur, 4),
                    "pacing_zone": z_name,
                    "beat_aligned": False,
                })
            continue

        # Walk downbeats in steps of cut_every_n_beats
        zone_db = list(zone_downbeats)
        i = 0
        while i < len(zone_db):
            slot_start = zone_db[i]
            next_i = i + cut_n
            if next_i < len(zone_db):
                slot_end = zone_db[next_i]
            else:
                slot_end = z_end
            # Clamp to zone bounds
            slot_end = min(slot_end, z_end)
            dur = slot_end - slot_start
            if dur >= MIN_SLOT_DURATION_S:
                slots.append({
                    "start_s":    round(slot_start, 4),
                    "end_s":      round(slot_end, 4),
                    "duration_s": round(dur, 4),
                    "pacing_zone": z_name,
                    "beat_aligned": True,
                })
            i += cut_n

    # Sort by start time (should already be ordered)
    slots.sort(key=lambda s: s["start_s"])

    # Merge consecutive slots that are shorter than min_clip_duration into the
    # next slot, so no timeline positions are lost — the clip just plays longer.
    if min_clip_duration > MIN_SLOT_DURATION_S and slots:
        merged: list[dict] = []
        i = 0
        while i < len(slots):
            slot = dict(slots[i])
            while slot["duration_s"] < min_clip_duration and i + 1 < len(slots):
                i += 1
                slot["end_s"]      = slots[i]["end_s"]
                slot["duration_s"] = round(slot["end_s"] - slot["start_s"], 4)
            merged.append(slot)
            i += 1
        slots = merged

    return slots


# ---------------------------------------------------------------------------
# Snippet pools
# ---------------------------------------------------------------------------

def partition_snippets(
    snippets: list[dict],
    min_score: float,
    people_boost: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """
    Split snippets into (video_pool, photo_pool).

    Video pool: type=video_scene, non-discarded, composite >= min_score,
                sorted by effective score desc.
    Photo pool: type=photo or burst_group, non-discarded, composite >= min_score,
                sorted composite desc.

    people_boost: added to composite score for any clip with face_count > 0,
                  pushing people-focused clips ahead of landscape/nature scenes.
    """
    video_pool: list[dict] = []
    photo_pool: list[dict] = []

    for s in snippets:
        if s.get("discarded", False):
            continue
        score = s.get("scores", {}).get("composite", 0.0)
        if score < min_score:
            continue
        t = s.get("type", "video_scene")
        if t == "video_scene":
            video_pool.append(s)
        elif t in ("photo", "burst_group"):
            photo_pool.append(s)

    def _effective_score(snippet: dict) -> float:
        base = snippet["scores"]["composite"]
        if people_boost > 0 and snippet.get("face_count", 0) > 0:
            return base + people_boost
        return base

    video_pool.sort(key=_effective_score, reverse=True)
    photo_pool.sort(key=lambda x: x["scores"]["composite"], reverse=True)
    return video_pool, photo_pool


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def l2_distance(fp_a: list[float], fp_b: list[float]) -> float:
    """L2 distance between two feature print vectors."""
    a = np.array(fp_a, dtype=float)
    b = np.array(fp_b, dtype=float)
    return float(np.linalg.norm(a - b))


def passes_diversity(
    candidate: dict,
    recent_sources: deque,
    recent_fingerprints: deque,
    fingerprint_map: dict[str, list[float]],
    feature_distance: float,
) -> bool:
    """Return True if candidate passes source-file and feature-print diversity."""
    src = candidate.get("source_file", "")
    if src in recent_sources:
        return False

    if fingerprint_map and feature_distance > 0:
        scene_id = candidate.get("scene_id", "")
        fp = fingerprint_map.get(scene_id)
        if fp:
            for rfp in recent_fingerprints:
                if l2_distance(fp, rfp) < feature_distance:
                    return False
    return True


def record_diversity(
    snippet: dict,
    recent_sources: deque,
    recent_fingerprints: deque,
    fingerprint_map: dict[str, list[float]],
) -> None:
    """Update diversity tracking after assigning a snippet to a slot."""
    src = snippet.get("source_file", "")
    if src:
        recent_sources.append(src)

    scene_id = snippet.get("scene_id", "")
    fp = fingerprint_map.get(scene_id)
    if fp:
        recent_fingerprints.append(fp)


# ---------------------------------------------------------------------------
# Ken Burns annotation
# ---------------------------------------------------------------------------

def compute_ken_burns(snippet: dict) -> dict | None:
    """Compute Ken Burns start/end rects for a photo/burst snippet."""
    t = snippet.get("type", "")
    if t not in ("photo", "burst_group"):
        return None

    centroid = snippet.get("saliency_centroid")
    if centroid and len(centroid) == 2:
        cx, cy = float(centroid[0]), float(centroid[1])
    else:
        cx, cy = 0.5, 0.5

    z = KEN_BURNS_ZOOM_FACTOR
    half_w = (1.0 / z) / 2.0
    half_h = (1.0 / z) / 2.0
    end_w = 1.0 / z
    end_h = 1.0 / z

    end_x = max(0.0, min(cx - half_w, 1.0 - end_w))
    end_y = max(0.0, min(cy - half_h, 1.0 - end_h))

    return {
        "start_rect": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "end_rect":   {"x": round(end_x, 4), "y": round(end_y, 4),
                       "w": round(end_w, 4), "h": round(end_h, 4)},
        "centroid":   [round(cx, 4), round(cy, 4)],
    }


# ---------------------------------------------------------------------------
# Slot assembly helper
# ---------------------------------------------------------------------------

def _make_slot(
    slot_id: str,
    slot_type: str,
    slot: dict,
    snippet: dict,
    override_start_s: float | None = None,
) -> dict:
    """Build a final slot dict from a grid slot + assigned snippet."""
    t_start = override_start_s if override_start_s is not None else slot["start_s"]
    t_end   = t_start + slot["duration_s"]

    bw = snippet.get("best_window") or {}
    src_in  = bw.get("start_time_s")
    src_out = bw.get("end_time_s")

    # For video: trim best_window to slot duration if needed
    if slot_type == "video" and src_in is not None and src_out is not None:
        bw_dur = src_out - src_in
        slot_dur = slot["duration_s"]
        if bw_dur > slot_dur:
            # Centre the window around mid_time
            mid = bw.get("mid_time_s", src_in + bw_dur / 2.0)
            src_in  = round(max(src_in, mid - slot_dur / 2.0), 4)
            src_out = round(src_in + slot_dur, 4)

    sc = snippet.get("scores", {})
    return {
        "slot_id":        slot_id,
        "slot_type":      slot_type,
        "timeline_start_s": round(t_start, 4),
        "timeline_end_s":   round(t_end, 4),
        "duration_s":       round(slot["duration_s"], 4),
        "snippet_id":       snippet.get("snippet_id"),
        "scene_id":         snippet.get("scene_id"),
        "source_file":      snippet.get("source_file"),
        "source_file_rel":  snippet.get("source_file_rel"),
        "source_in_s":      round(src_in, 4)  if src_in  is not None else None,
        "source_out_s":     round(src_out, 4) if src_out is not None else None,
        "pacing_zone":      slot.get("pacing_zone", "verse"),
        "beat_aligned":     slot.get("beat_aligned", False),
        "transition":       "cut",
        "scores": {
            "aesthetic":         sc.get("aesthetic"),
            "saliency_coverage": sc.get("saliency_coverage"),
            "smile":             sc.get("smile"),
            "composite":         sc.get("composite"),
        },
        "ken_burns": compute_ken_burns(snippet),
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Fill video slots (greedy)
# ---------------------------------------------------------------------------

def fill_slots(
    slots: list[dict],
    video_pool: list[dict],
    photo_pool: list[dict],
    photo_bridge_interval: int,
    diversity_window: int,
    feature_distance: float,
    fingerprint_map: dict[str, list[float]],
    rng: random.Random,
    people_ratio: float = 0.9,
    max_clips_per_source: int = 0,
    photo_max_duration: float | None = None,
    no_repeat: bool = False,
    photo_burst_size: int = 3,
    pinned_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Assign a video or photo snippet to every slot.

    people_ratio: max fraction of video slots that may contain clips with
                  detected faces (0.0 = no people, 1.0 = uncapped).
                  When the quota is reached the passes skip face clips and
                  fall through to landscape/nature ones instead.
    max_clips_per_source: maximum number of clips from any single source file
                  (0 = uncapped).  Limits any one person's videos from
                  dominating when they happen to have many high-scoring clips.

    Returns (timeline_entries, used_snippet_ids).
    """
    recent_sources      = deque(maxlen=diversity_window)
    recent_fingerprints = deque(maxlen=RECENT_FINGERPRINT_WINDOW)
    used_ids:   set[str] = set()
    used_photos: set[str] = set()
    clips_from_source: dict[str, int] = {}
    clips_per_person: dict[str, int] = {}
    timeline: list[dict] = []
    _pinned: set[str] = pinned_ids or set()

    video_candidates = list(video_pool)
    photo_candidates = list(photo_pool)

    # Pre-assign pinned (force_include) video items to random, evenly-spread
    # slot positions so they are guaranteed to appear without clustering.
    pinned_slot_assignments: dict[int, dict] = {}
    if _pinned and video_candidates:
        pinned_items = [v for v in video_candidates if v.get("snippet_id", "") in _pinned]
        if pinned_items:
            rng.shuffle(pinned_items)
            # Only target non-bridge slots
            eligible = [
                i for i in range(len(slots))
                if not (photo_bridge_interval > 0 and (i + 1) % photo_bridge_interval == 0)
            ] or list(range(len(slots)))
            n_e = len(eligible)
            k = min(len(pinned_items), n_e)
            step = n_e / k
            offset = rng.randint(0, max(0, round(step / 2) - 1)) if k > 0 else 0
            assigned_positions: set[int] = set()
            for j, item in enumerate(pinned_items[:k]):
                raw = int(j * step + offset) % n_e
                # Nudge forward if collision
                while eligible[raw % n_e] in assigned_positions:
                    raw += 1
                pos = eligible[raw % n_e]
                assigned_positions.add(pos)
                pinned_slot_assignments[pos] = item

    # Face-clip quota: how many of the video slots may contain people.
    n_video_slots = len(slots) - (len(slots) // photo_bridge_interval if photo_bridge_interval > 0 else 0)
    face_quota     = round(n_video_slots * max(0.0, min(1.0, people_ratio)))
    face_slots_used = 0

    def _face_allowed(v: dict) -> bool:
        """Return False when the face quota is full and this clip has faces."""
        return face_slots_used < face_quota or v.get("face_count", 0) == 0

    def _source_allowed(v: dict) -> bool:
        """Return False when this source file has hit the per-source cap.
        force_include (pinned) items always bypass the cap."""
        if max_clips_per_source <= 0:
            return True
        if v.get("snippet_id", "") in _pinned:
            return True
        return clips_from_source.get(v.get("source_file", ""), 0) < max_clips_per_source

    # Build cycling iterator for exhausted pool
    video_cycle_index = 0
    photo_slot_index  = 0

    for slot_idx, slot in enumerate(slots):
        slot_id = f"SL{slot_idx + 1:05d}"

        # Pre-assigned pinned items are force-injected, bypassing photo bridge
        if slot_idx in pinned_slot_assignments:
            forced = pinned_slot_assignments[slot_idx]
            if forced.get("face_count", 0) > 0:
                face_slots_used += 1
            src = forced.get("source_file", "")
            clips_from_source[src] = clips_from_source.get(src, 0) + 1
            for pid in forced.get("person_ids", []):
                clips_per_person[pid] = clips_per_person.get(pid, 0) + 1
            used_ids.add(forced.get("snippet_id", ""))
            record_diversity(forced, recent_sources, recent_fingerprints, fingerprint_map)
            timeline.append(_make_slot(slot_id, "video", slot, forced))
            continue

        # Decide if this should be a photo bridge
        use_photo_bridge = (
            photo_bridge_interval > 0
            and (slot_idx + 1) % photo_bridge_interval == 0
            and photo_candidates
        )

        if use_photo_bridge:
            def _pick_photo() -> dict | None:
                nonlocal photo_slot_index
                for ph in photo_candidates:
                    pid = ph.get("snippet_id", "")
                    if pid in used_photos:
                        continue
                    if passes_diversity(ph, recent_sources, recent_fingerprints,
                                        fingerprint_map, feature_distance):
                        return ph
                for ph in photo_candidates:
                    if ph.get("snippet_id", "") not in used_photos:
                        return ph
                if photo_candidates and not no_repeat:
                    ph = photo_candidates[photo_slot_index % len(photo_candidates)]
                    photo_slot_index += 1
                    return ph
                return None

            burst_dur = photo_max_duration if photo_max_duration is not None else 1.0
            burst_added = 0
            for b in range(photo_burst_size):
                chosen = _pick_photo()
                if chosen is None:
                    break
                used_photos.add(chosen.get("snippet_id", ""))
                record_diversity(chosen, recent_sources, recent_fingerprints, fingerprint_map)
                burst_slot = {**slot, "duration_s": burst_dur,
                              "end_s": round(slot["start_s"] + burst_dur, 4)}
                entry = _make_slot(f"{slot_id}b{b}", "photo_bridge", burst_slot, chosen)
                timeline.append(entry)
                burst_added += 1
            if burst_added:
                continue

        # Person-balance: re-sort video candidates so clips whose people are
        # least represented so far float to the top.  Falls back to composite
        # order for clips with no person_ids.
        def _person_sort_key(v: dict) -> tuple:
            pids = v.get("person_ids", [])
            min_usage = min((clips_per_person.get(p, 0) for p in pids), default=0)
            return (min_usage, -v.get("scores", {}).get("composite", 0.0))

        if any(v.get("person_ids") for v in video_candidates):
            video_candidates_sorted = sorted(video_candidates, key=_person_sort_key)
        else:
            video_candidates_sorted = video_candidates

        # Fill with video
        chosen = None

        # Pass 1: full diversity constraints, unused only
        for v in video_candidates_sorted:
            vid = v.get("snippet_id", "")
            if vid in used_ids:
                continue
            if not _face_allowed(v) or not _source_allowed(v):
                continue
            if passes_diversity(v, recent_sources, recent_fingerprints,
                                fingerprint_map, feature_distance):
                chosen = v
                break

        # Pass 2: relax feature-print constraint
        if chosen is None:
            for v in video_candidates_sorted:
                vid = v.get("snippet_id", "")
                if vid in used_ids:
                    continue
                if not _face_allowed(v) or not _source_allowed(v):
                    continue
                if v.get("source_file", "") not in recent_sources:
                    chosen = v
                    break

        # Pass 3: relax source diversity
        if chosen is None:
            for v in video_candidates_sorted:
                vid = v.get("snippet_id", "")
                if vid not in used_ids and _face_allowed(v) and _source_allowed(v):
                    chosen = v
                    break

        # Pass 4: loop — reuse previously used snippets (still respects quota)
        if chosen is None and video_candidates_sorted and not no_repeat:
            for i in range(len(video_candidates_sorted)):
                candidate = video_candidates_sorted[(video_cycle_index + i) % len(video_candidates_sorted)]
                if _face_allowed(candidate) and _source_allowed(candidate):
                    chosen = candidate
                    video_cycle_index += i + 1
                    break
            else:
                chosen = video_candidates_sorted[video_cycle_index % len(video_candidates_sorted)]
                video_cycle_index += 1

        if chosen is not None:
            if chosen.get("face_count", 0) > 0:
                face_slots_used += 1
            src = chosen.get("source_file", "")
            clips_from_source[src] = clips_from_source.get(src, 0) + 1
            for pid in chosen.get("person_ids", []):
                clips_per_person[pid] = clips_per_person.get(pid, 0) + 1
            used_ids.add(chosen.get("snippet_id", ""))
            record_diversity(chosen, recent_sources, recent_fingerprints, fingerprint_map)
            entry = _make_slot(slot_id, "video", slot, chosen)
            timeline.append(entry)

    return timeline, list(used_ids)


# ---------------------------------------------------------------------------
# Memory dump
# ---------------------------------------------------------------------------

def assemble_memory_dump(
    photo_pool: list[dict],
    used_photo_ids: set[str],
    memory_dump_count: int,
    timeline_end_s: float,
    memory_dump_duration: float = MEMORY_DUMP_SLOT_DURATION_S,
) -> list[dict]:
    """Build Memory Dump entries from unused high-quality photos."""
    unused = [p for p in photo_pool if p.get("snippet_id", "") not in used_photo_ids]
    unused.sort(key=lambda x: x["scores"]["composite"], reverse=True)
    unused = unused[:memory_dump_count]

    slot_dur = memory_dump_duration

    memory_dump: list[dict] = []
    t = timeline_end_s
    for i, photo in enumerate(unused):
        md_slot = {
            "start_s":    t,
            "end_s":      t + slot_dur,
            "duration_s": slot_dur,
            "pacing_zone": "memory_dump",
            "beat_aligned": False,
        }
        entry = _make_slot(f"MD{i + 1:05d}", "memory_dump", md_slot, photo)
        # Override slot_type to memory_dump
        entry["slot_type"] = "memory_dump"
        memory_dump.append(entry)
        t += slot_dur

    return memory_dump


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def count_looped(timeline: list[dict], all_video_snippet_ids: set[str]) -> int:
    """Count slots that reuse an already-used snippet (loop cycles)."""
    seen: set[str] = set()
    loops = 0
    for entry in timeline:
        sid = entry.get("snippet_id", "")
        if sid in seen:
            loops += 1
        seen.add(sid)
    return loops


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def assemble_final_sequence(
    metadata: dict,
    timeline: list[dict],
    memory_dump: list[dict],
) -> dict:
    return {
        "metadata":    metadata,
        "timeline":    timeline,
        "memory_dump": memory_dump,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--snippets", "-s", "snippets_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to scored_snippets.json (Phase 2 output).",
)
@click.option(
    "--beats", "-b", "beats_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to beat_grid.json (Phase 3 output).",
)
@click.option(
    "--scenes",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to scenes.json (Phase 1 output, enables feature-print dedup).",
)
@click.option(
    "--output", "-o", "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. Defaults to same dir as --snippets.",
)
@click.option(
    "--min-score",
    default=0.30, show_default=True,
    help="Minimum composite score for a snippet to be considered.",
)
@click.option(
    "--min-clip-duration",
    default=2.0, show_default=True,
    help="Minimum clip duration in seconds. Beat slots shorter than this are dropped.",
)
@click.option(
    "--photo-bridge-interval",
    default=8, show_default=True,
    help="Insert one photo bridge every N video slots (0 to disable).",
)
@click.option(
    "--photo-max-duration",
    default=0.5, show_default=True,
    help="Duration of each photo in a bridge burst (seconds).",
)
@click.option(
    "--photo-burst-size",
    default=3, show_default=True,
    help="Number of photos to insert per bridge slot (1 = single photo, no burst).",
)
@click.option(
    "--memory-dump-count",
    default=100, show_default=True,
    help="Maximum number of photos in the end-credits Memory Dump.",
)
@click.option(
    "--memory-dump-duration",
    default=MEMORY_DUMP_SLOT_DURATION_S, show_default=True,
    help="Duration of each photo in the end-credits Memory Dump (seconds).",
)
@click.option(
    "--diversity-window",
    default=10, show_default=True,
    help="Avoid repeating the same source_file within last N slots.",
)
@click.option(
    "--feature-distance",
    default=0.15, show_default=True,
    help="Minimum L2 feature-print distance for visual diversity (requires --scenes).",
)
@click.option(
    "--people-boost",
    default=0.2, show_default=True,
    help="Score bonus added to clips with detected faces (face_count > 0). "
         "Raises people/portrait clips above landscape/nature in the selection order. "
         "Typical useful range: 0.1–0.3.",
)
@click.option(
    "--people-ratio",
    default=0.75, show_default=True,
    help="Maximum fraction of video slots that may contain clips with detected faces "
         "(0.0–1.0). Use with --people-boost to prioritise people while guaranteeing "
         "some landscape/nature scenes. E.g. 0.75 = at most 75%% people clips.",
)
@click.option(
    "--max-clips-per-source",
    default=3, show_default=True,
    help="Maximum clips from any single source file in the timeline (0 = uncapped). "
         "Prevents one person's action-heavy video from dominating the montage.",
)
@click.option(
    "--overrides", "overrides_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to curator_overrides.json. force_exclude snippet IDs are removed from "
         "the pool entirely; force_include snippet IDs are promoted to the front of the "
         "pool so they are chosen first.",
)
@click.option(
    "--no-repeat",
    is_flag=True, default=False,
    help="Never show the same clip twice. When the pool is exhausted, slots are skipped "
         "rather than looping.",
)
@click.option(
    "--seed",
    default=42, show_default=True,
    help="Random seed for tie-breaking.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Print detailed progress.",
)
def main(
    snippets_path: Path,
    beats_path: Path,
    scenes: Path | None,
    output_dir: Path | None,
    min_score: float,
    min_clip_duration: float,
    photo_bridge_interval: int,
    photo_max_duration: float,
    photo_burst_size: int,
    memory_dump_count: int,
    memory_dump_duration: float,
    diversity_window: int,
    feature_distance: float,
    people_boost: float,
    people_ratio: float,
    max_clips_per_source: int,
    overrides_path: Path | None,
    no_repeat: bool,
    seed: int,
    verbose: bool,
) -> None:
    """Director — Phase 4 of the macOS AI Montage Suite.

    Combines scored snippets and a beat grid into a final video timeline.
    Writes final_sequence.json with beat-aligned slot assignments, Ken Burns
    photo parameters, and an end-credits Memory Dump.
    """
    rng = random.Random(seed)

    # Resolve output dir
    if output_dir is None:
        output_dir = snippets_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Director[/] — Phase 4 of the macOS AI Montage Suite\n")

    # ------------------------------------------------------------------
    # Stage 1: Load inputs
    # ------------------------------------------------------------------
    console.print("[bold]Stage 1/5[/] Loading inputs …")

    snippets_data = load_scored_snippets(snippets_path)
    beat_grid_data = load_beat_grid(beats_path)
    fingerprint_map = load_scenes_json(scenes)

    all_snippets: list[dict] = snippets_data.get("snippets", [])
    total_duration_s: float = beat_grid_data["metadata"].get("total_duration_s", 0.0)

    console.print(f"  Snippets:     {len(all_snippets)}")
    console.print(f"  Beat-grid mix duration: {total_duration_s:.1f}s")
    console.print(f"  Pacing zones: {len(beat_grid_data.get('pacing_zones', []))}")

    # ------------------------------------------------------------------
    # Stage 2: Build slot grid
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 2/5[/] Building slot grid …")
    slots = build_slot_grid(beat_grid_data, min_clip_duration=min_clip_duration)
    console.print(f"  Slots generated: {len(slots)}")

    if verbose:
        zone_counts: dict[str, int] = {}
        for s in slots:
            z = s["pacing_zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
        for z, c in sorted(zone_counts.items()):
            console.print(f"    {z}: {c} slots")

    # ------------------------------------------------------------------
    # Stage 3: Partition snippets
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Load curator overrides (optional)
    # ------------------------------------------------------------------
    force_include: list[str] = []
    force_exclude: set[str] = set()
    if overrides_path is not None:
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        force_include = overrides.get("force_include", [])
        force_exclude = set(overrides.get("force_exclude", []))
        console.print(
            f"\n[bold]Overrides[/] loaded from {overrides_path.name}: "
            f"{len(force_include)} force-include, {len(force_exclude)} force-exclude"
        )
        if force_exclude:
            before = len(all_snippets)
            all_snippets = [s for s in all_snippets if s.get("snippet_id") not in force_exclude]
            console.print(f"  Removed {before - len(all_snippets)} excluded snippet(s)")

    console.print("\n[bold]Stage 3/5[/] Partitioning snippets …")
    video_pool, photo_pool = partition_snippets(all_snippets, min_score, people_boost)
    console.print(f"  Video pool: {len(video_pool)} clips")
    console.print(f"  Photo pool: {len(photo_pool)} photos")

    # Promote force_include snippets to front of their pool
    if force_include:
        by_id = {s["snippet_id"]: s for s in all_snippets}
        promoted_video, promoted_photo = [], []
        for sid in force_include:
            s = by_id.get(sid)
            if s is None:
                continue
            if s.get("type") in ("photo", "burst_group"):
                promoted_photo.append(s)
            else:
                promoted_video.append(s)
        # Prepend promoted clips, removing duplicates from the rest of the pool
        pinned_ids = set(force_include)
        video_pool = promoted_video + [s for s in video_pool if s["snippet_id"] not in pinned_ids]
        photo_pool = promoted_photo + [s for s in photo_pool if s["snippet_id"] not in pinned_ids]
        console.print(f"  Promoted: {len(promoted_video)} video, {len(promoted_photo)} photo snippet(s) to front of pool")

    discarded = sum(1 for s in all_snippets if s.get("discarded", False))
    below_min = sum(
        1 for s in all_snippets
        if not s.get("discarded", False)
        and s.get("scores", {}).get("composite", 0) < min_score
    )
    if discarded or below_min:
        console.print(f"  Excluded: {discarded} discarded, {below_min} below min-score {min_score}")

    if not video_pool and not slots:
        console.print("[yellow]No video clips available and no slots — nothing to do.[/]")
        raise SystemExit(0)

    if not video_pool:
        console.print("[yellow]Warning: no video clips meet quality threshold; "
                      "timeline will be photo-only.[/]")

    # ------------------------------------------------------------------
    # Stage 4: Fill slots
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 4/5[/] Filling slots …")

    combined_pool = video_pool if video_pool else photo_pool

    timeline, used_video_ids = fill_slots(
        slots=slots,
        video_pool=combined_pool,
        photo_pool=photo_pool,
        photo_bridge_interval=photo_bridge_interval,
        diversity_window=diversity_window,
        feature_distance=feature_distance,
        fingerprint_map=fingerprint_map,
        rng=rng,
        people_ratio=people_ratio,
        max_clips_per_source=max_clips_per_source,
        photo_max_duration=photo_max_duration,
        no_repeat=no_repeat,
        photo_burst_size=photo_burst_size,
        pinned_ids=set(force_include) if force_include else None,
    )

    used_photo_ids: set[str] = {
        e["snippet_id"] for e in timeline
        if e["slot_type"] in ("photo_bridge",) and e.get("snippet_id")
    }

    n_video_slots  = sum(1 for e in timeline if e["slot_type"] == "video")
    n_bridge_slots = sum(1 for e in timeline if e["slot_type"] == "photo_bridge")
    n_looped = count_looped(timeline, set(used_video_ids))

    console.print(f"  Video slots:  {n_video_slots}")
    console.print(f"  Photo bridges: {n_bridge_slots}")
    if n_looped:
        console.print(f"  [yellow]Looped snippets: {n_looped} (pool exhausted)[/]")

    # ------------------------------------------------------------------
    # Stage 5: Memory Dump + assemble output
    # ------------------------------------------------------------------
    console.print("\n[bold]Stage 5/5[/] Assembling Memory Dump and output …")

    tl_end_s = timeline[-1]["timeline_end_s"] if timeline else 0.0
    memory_dump = assemble_memory_dump(
        photo_pool=photo_pool,
        used_photo_ids=used_photo_ids,
        memory_dump_count=memory_dump_count,
        timeline_end_s=tl_end_s,
        memory_dump_duration=memory_dump_duration,
    )

    total_end_s = (
        memory_dump[-1]["timeline_end_s"]
        if memory_dump else tl_end_s
    )

    console.print(f"  Memory Dump: {len(memory_dump)} photos → {tl_end_s:.1f}s → {total_end_s:.1f}s")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "director_version":     "1.0.0",
        "generated_at":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scored_snippets_json": str(snippets_path.resolve()),
        "beat_grid_json":       str(beats_path.resolve()),
        "scenes_json":          str(scenes.resolve()) if scenes else None,
        "output_dir":           str(output_dir.resolve()),
        "total_slots":          len(timeline),
        "total_duration_s":     round(tl_end_s, 3),
        "total_duration_with_dump_s": round(total_end_s, 3),
        "photo_bridge_count":   n_bridge_slots,
        "memory_dump_count":    len(memory_dump),
        "looped_snippets":      n_looped,
        "min_score":            min_score,
        "photo_bridge_interval": photo_bridge_interval,
        "diversity_window":     diversity_window,
        "feature_distance":     feature_distance,
        "seed":                 seed,
    }

    result = assemble_final_sequence(metadata, timeline, memory_dump)

    out_path = output_dir / "final_sequence.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    console.print(f"\n[bold green]✓[/] final_sequence.json → [cyan]{out_path}[/]")

    # ------------------------------------------------------------------
    # Summary table — aggregate by slot_type
    # ------------------------------------------------------------------
    table = Table(title="Timeline Summary", show_header=True, header_style="bold")
    table.add_column("Type",     style="cyan")
    table.add_column("Slots",    justify="right")
    table.add_column("Duration", justify="right")

    type_agg: dict[str, dict] = {}
    for entry in timeline + memory_dump:
        st = entry["slot_type"]
        if st not in type_agg:
            type_agg[st] = {"count": 0, "dur": 0.0}
        type_agg[st]["count"] += 1
        type_agg[st]["dur"]   += entry["duration_s"]

    for st in ("video", "photo_bridge", "memory_dump"):
        if st in type_agg:
            agg = type_agg[st]
            table.add_row(st.replace("_", " "), str(agg["count"]), f"{agg['dur']:.1f}s")

    table.add_row(
        "TOTAL",
        str(len(timeline) + len(memory_dump)),
        f"{total_end_s:.1f}s",
    )
    console.print(table)

    console.print(
        f"\n[bold]Done.[/] {len(timeline)} timeline slots · "
        f"{len(memory_dump)} memory-dump photos · "
        f"{total_end_s:.1f}s total"
    )


if __name__ == "__main__":
    main()
