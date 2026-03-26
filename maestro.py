#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "librosa>=0.10",
#   "pydub>=0.25",
#   "soundfile>=0.12",
#   "numpy>=1.24",
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Phase 3 of the macOS AI Montage Suite — The Maestro.

Analyses one or more music tracks, builds a beat-aligned crossfaded mix, and
writes master_mix.mp3 + beat_grid.json for consumption by director.py.

Usage:
    uv run maestro.py --music track1.mp3 --music track2.mp3 [OPTIONS]

When --snippet-duration is set, each track contributes only its
highest-energy N-second window rather than its full length.
"""
from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
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

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MaestroError(Exception):
    pass

class TrackLoadError(MaestroError):
    pass

class MixError(MaestroError):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CROSSFADE_S      = 3.5
DEFAULT_OUTPUT_DIR       = "./maestro_output"
DEFAULT_MP3_BITRATE      = "192k"
DEFAULT_SR               = 22050   # Hz
HOP_LENGTH               = 512
ENERGY_MAP_RESOLUTION_S  = 0.1

CRESCENDO_PERCENTILE     = 75
BREAKDOWN_PERCENTILE     = 25

PACING_CHORUS_THRESHOLD  = 0.65
PACING_VERSE_THRESHOLD   = 0.35
MIN_ZONE_DURATION_S      = 2.0

BPM_BRIDGE_RATE_MIN      = 0.85
BPM_BRIDGE_RATE_MAX      = 1.15
BPM_BRIDGE_SKIP_DELTA    = 2.0

ONSET_DENSITY_WINDOW_S   = 1.0
SNAP_SEARCH_RADIUS_S     = 4.0

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_checks(output_dir: Path) -> Path:
    """Check ffmpeg, create output directory. Returns ffmpeg Path."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(ffmpeg_bin)

# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def load_track(path: Path, sr: int) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 array at the given sample rate."""
    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, mono=True)
        return y, sr_loaded
    except Exception as exc:
        raise TrackLoadError(f"Failed to load {path.name}: {exc}") from exc

# ---------------------------------------------------------------------------
# Beat analysis
# ---------------------------------------------------------------------------

def analyse_beats(
    y: np.ndarray,
    sr: int,
    target_bpm: float | None,
) -> dict:
    """Return tempo_bpm, beat_times (s), downbeat_times (s)."""
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LENGTH,
        bpm=target_bpm,
        units="frames",
    )
    # librosa >= 0.10 returns tempo as a 1-element ndarray; normalise to scalar
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times: np.ndarray = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=HOP_LENGTH
    )

    # Downbeat detection via PLP (predominant local pulse)
    downbeat_times: list[float] = []
    errors: list[str] = []
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        pulse = librosa.beat.plp(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=HOP_LENGTH,
            tempo_min=float(tempo) * 0.9,
            tempo_max=float(tempo) * 1.1,
        )
        plp_peak_time = librosa.frames_to_time(
            int(np.argmax(pulse)), sr=sr, hop_length=HOP_LENGTH
        )
        phase_idx = int(np.argmin(np.abs(beat_times - plp_peak_time)))
        db_indices = list(range(phase_idx % 4, len(beat_times), 4))
        downbeat_times = beat_times[db_indices].tolist()
    except Exception as exc:
        errors.append(f"PLP downbeat detection failed ({exc}); falling back to every-4th-beat.")
        downbeat_times = beat_times[::4].tolist()

    return {
        "tempo_bpm": float(tempo),
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times,
        "_beat_errors": errors,
    }

# ---------------------------------------------------------------------------
# Energy analysis
# ---------------------------------------------------------------------------

def analyse_energy(
    y: np.ndarray,
    sr: int,
    smoothing_s: float,
) -> dict:
    """Return energy_map, rms_p25, rms_p75, rms_mean, rms_max."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    frame_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH
    )

    # Moving-average smoothing (no scipy dependency)
    window = max(1, int(smoothing_s * sr / HOP_LENGTH))
    kernel = np.ones(window) / window
    rms_smooth = np.convolve(rms, kernel, mode="same")

    p25 = float(np.percentile(rms_smooth, BREAKDOWN_PERCENTILE))
    p75 = float(np.percentile(rms_smooth, CRESCENDO_PERCENTILE))

    stride = max(1, int(ENERGY_MAP_RESOLUTION_S * sr / HOP_LENGTH))
    energy_map = [
        {
            "time_s":       round(float(frame_times[i]), 4),
            "rms":          round(float(rms_smooth[i]), 6),
            "is_crescendo": bool(rms_smooth[i] >= p75),
            "is_breakdown": bool(rms_smooth[i] <= p25),
        }
        for i in range(0, len(rms_smooth), stride)
    ]

    return {
        "energy_map": energy_map,
        "rms_p25":  p25,
        "rms_p75":  p75,
        "rms_mean": float(np.mean(rms_smooth)),
        "rms_max":  float(np.max(rms_smooth)),
    }

# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def analyse_spectral(y: np.ndarray, sr: int) -> dict:
    """Return onset_density list."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH
    )
    win = int(ONSET_DENSITY_WINDOW_S * sr / HOP_LENGTH)
    onset_density = [
        {
            "time_s":         round(float(onset_times[i]), 4),
            "onset_strength": round(float(np.mean(onset_env[i: i + win])), 6),
        }
        for i in range(0, len(onset_env), max(1, win))
    ]
    return {"onset_density": onset_density}

# ---------------------------------------------------------------------------
# Peak-energy window selection
# ---------------------------------------------------------------------------

def find_peak_energy_window(
    y: np.ndarray,
    sr: int,
    duration_s: float,
    energy_map: list[dict],
    downbeat_times: list[float],
    snap_to_downbeat: bool = True,
) -> tuple[float, float, float]:
    """Find the highest-energy contiguous window of `duration_s` seconds.

    Returns (start_s, end_s, mean_rms).
    Optionally snaps the start to the nearest downbeat for clean entry.
    """
    total_s = len(y) / sr

    if duration_s >= total_s:
        mean_rms = float(np.mean([e["rms"] for e in energy_map])) if energy_map else 0.0
        return 0.0, total_s, mean_rms

    times     = np.array([e["time_s"] for e in energy_map])
    rms_vals  = np.array([e["rms"]    for e in energy_map])

    best_start = 0.0
    best_score = -1.0

    for t in times:
        if t + duration_s > total_s:
            break
        mask  = (times >= t) & (times < t + duration_s)
        score = float(np.mean(rms_vals[mask])) if np.any(mask) else 0.0
        if score > best_score:
            best_score = score
            best_start = float(t)

    # Snap to nearest downbeat within search radius
    if snap_to_downbeat and downbeat_times:
        db = np.array(downbeat_times)
        lo = max(0.0, best_start - SNAP_SEARCH_RADIUS_S)
        hi = min(best_start + SNAP_SEARCH_RADIUS_S, total_s - duration_s)
        candidates = db[(db >= lo) & (db <= hi)]
        if len(candidates):
            nearest = float(candidates[np.argmin(np.abs(candidates - best_start))])
            if nearest + duration_s <= total_s:
                best_start = nearest

    return round(best_start, 4), round(best_start + duration_s, 4), round(best_score, 6)


def extract_snippet(track_data: dict, start_s: float, end_s: float) -> dict:
    """Slice the track to [start_s, end_s] and re-zero all timestamps."""
    y  = track_data["y"]
    sr = track_data["sr"]
    y_snip = y[int(start_s * sr): int(end_s * sr)]

    def shift_list(ts: list[float]) -> list[float]:
        return [round(t - start_s, 4) for t in ts if start_s <= t < end_s]

    def shift_map(entries: list[dict], key: str = "time_s") -> list[dict]:
        return [
            {**e, key: round(e[key] - start_s, 4)}
            for e in entries
            if start_s <= e[key] < end_s
        ]

    return {
        **track_data,
        "y":              y_snip,
        "duration_s":     round(end_s - start_s, 4),
        "beat_times":     shift_list(track_data["beat_times"]),
        "downbeat_times": shift_list(track_data["downbeat_times"]),
        "energy_map":     shift_map(track_data["energy_map"]),
        "onset_density":  shift_map(track_data.get("onset_density", [])),
        "snippet_start_s": start_s,
        "snippet_end_s":   end_s,
    }

# ---------------------------------------------------------------------------
# Full track analysis
# ---------------------------------------------------------------------------

def analyse_track(
    path: Path,
    sr: int,
    target_bpm: float | None,
    smoothing_s: float,
    snippet_duration_s: float | None,
    snap_to_downbeat: bool,
) -> dict:
    """Load and fully analyse a track. Applies snippet selection if configured."""
    y, sr = load_track(path, sr)
    duration_s = librosa.get_duration(y=y, sr=sr)

    beat_data     = analyse_beats(y, sr, target_bpm)
    energy_data   = analyse_energy(y, sr, smoothing_s)
    spectral_data = analyse_spectral(y, sr)

    track: dict[str, Any] = {
        "path":        path,
        "y":           y,
        "sr":          sr,
        "duration_s":  duration_s,
        "snippet_start_s": None,
        "snippet_end_s":   None,
        **beat_data,
        **energy_data,
        **spectral_data,
    }

    errors: list[str] = list(beat_data.pop("_beat_errors", []))

    if snippet_duration_s is not None:
        start_s, end_s, mean_rms = find_peak_energy_window(
            y=y,
            sr=sr,
            duration_s=snippet_duration_s,
            energy_map=energy_data["energy_map"],
            downbeat_times=beat_data["downbeat_times"],
            snap_to_downbeat=snap_to_downbeat,
        )
        track = extract_snippet(track, start_s, end_s)
        console.print(
            f"  [dim]snippet: {start_s:.1f}s – {end_s:.1f}s "
            f"(mean RMS {mean_rms:.4f})[/]"
        )

    track["_errors"] = errors
    return track

# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def find_crossfade_start(
    downbeat_times: list[float],
    duration_s: float,
    crossfade_s: float,
) -> float:
    """Last downbeat that leaves at least crossfade_s seconds of audio."""
    candidates = [t for t in downbeat_times if t <= duration_s - crossfade_s]
    return candidates[-1] if candidates else max(0.0, duration_s - crossfade_s)


def apply_bpm_bridge(
    y_out: np.ndarray,
    y_in: np.ndarray,
    sr: int,
    bpm_out: float,
    bpm_in: float,
    crossfade_start_s: float,
) -> tuple[np.ndarray, np.ndarray, bool, float | None]:
    """Time-stretch the outgoing tail to glide BPM toward the incoming track.

    Returns (y_out_bridged, y_in, applied, rate_used).
    """
    if abs(bpm_out - bpm_in) < BPM_BRIDGE_SKIP_DELTA:
        return y_out, y_in, False, None

    rate = float(np.clip(bpm_out / bpm_in, BPM_BRIDGE_RATE_MIN, BPM_BRIDGE_RATE_MAX))
    split = int(crossfade_start_s * sr)
    y_head = y_out[:split]
    y_tail = y_out[split:]

    try:
        y_tail_stretched = librosa.effects.time_stretch(y_tail, rate=rate)
        y_out_bridged = np.concatenate([y_head, y_tail_stretched])
        return y_out_bridged, y_in, True, round(rate, 4)
    except Exception as exc:
        console.print(f"  [yellow]BPM bridge failed ({exc}); skipping.[/]")
        return y_out, y_in, False, None


def constant_power_crossfade(
    y_out: np.ndarray,
    y_in: np.ndarray,
    sr: int,
    crossfade_start_s: float,
    crossfade_s: float,
) -> np.ndarray:
    """Merge y_out and y_in with a constant-power crossfade.

    fade_out² + fade_in² = 1 throughout, preserving perceived loudness.
    """
    n_xfade = int(crossfade_s * sr)
    split   = int(crossfade_start_s * sr)

    y_pre  = y_out[:split]
    tail   = y_out[split: split + n_xfade]
    head   = y_in[:n_xfade]
    y_post = y_in[n_xfade:]

    # Pad to equal length (BPM bridge may have changed tail length)
    n = max(len(tail), len(head))
    tail = np.pad(tail, (0, n - len(tail)))
    head = np.pad(head, (0, n - len(head)))

    t = np.linspace(0, 1, n, endpoint=False)
    xfade = tail * np.cos(t * np.pi / 2) + head * np.sin(t * np.pi / 2)

    return np.concatenate([y_pre, xfade, y_post])


def build_mix(
    tracks: list[dict],
    crossfade_s: float,
    target_bpm: float | None,
    target_duration_s: float | None,
) -> tuple[np.ndarray, int, list[dict]]:
    """Assemble all tracks into a single audio array with crossfades.

    Returns (mix_array, sr, track_boundaries).
    """
    if not tracks:
        raise MixError("No tracks to mix.")

    sr       = tracks[0]["sr"]
    mix      = tracks[0]["y"].copy()
    offset_s = 0.0
    boundaries: list[dict] = []

    for i, track in enumerate(tracks):
        is_last = (i == len(tracks) - 1)

        xfade_out_start: float | None = None
        xfade_out_end:   float | None = None
        bridge_applied = False
        bridge_rate: float | None = None

        if not is_last:
            next_track = tracks[i + 1]

            xfade_start_in_track = find_crossfade_start(
                track["downbeat_times"], track["duration_s"], crossfade_s
            )
            xfade_out_start = offset_s + xfade_start_in_track
            xfade_out_end   = xfade_out_start + crossfade_s

            # Identify which portion of `mix` corresponds to this track's tail
            tail_split = int(xfade_out_start * sr)
            mix_head   = mix[:tail_split]
            mix_tail   = mix[tail_split:]

            # BPM bridge on this track's tail
            mix_tail, _, bridge_applied, bridge_rate = apply_bpm_bridge(
                mix_tail, next_track["y"],
                sr,
                track["tempo_bpm"], next_track["tempo_bpm"],
                crossfade_start_s=0.0,  # tail already starts at the crossfade point
            )
            mix = np.concatenate([mix_head, mix_tail])

            # Constant-power crossfade
            mix = constant_power_crossfade(
                mix, next_track["y"],
                sr,
                crossfade_start_s=xfade_out_start,
                crossfade_s=crossfade_s,
            )

            mix_end_s = xfade_out_start + track["duration_s"] - xfade_start_in_track + len(next_track["y"]) / sr
        else:
            mix_end_s = offset_s + track["duration_s"]

        boundaries.append({
            "track_index":         i,
            "source_file":         str(track["path"]),
            "source_file_rel":     track["path"].name,
            "original_bpm":        round(track["tempo_bpm"], 3),
            "original_duration_s": round(track["duration_s"], 3),
            "mix_start_s":         round(offset_s, 4),
            "mix_end_s":           round(mix_end_s, 4),
            "crossfade_out_start_s": round(xfade_out_start, 4) if xfade_out_start is not None else None,
            "crossfade_out_end_s":   round(xfade_out_end, 4)   if xfade_out_end   is not None else None,
            "bpm_bridge_applied":  bridge_applied,
            "bpm_bridge_rate":     bridge_rate,
            "track_beat_times":    [round(t + offset_s, 4) for t in track["beat_times"]],
            "track_downbeat_times":[round(t + offset_s, 4) for t in track["downbeat_times"]],
            "snippet_start_s":     track.get("snippet_start_s"),
            "snippet_end_s":       track.get("snippet_end_s"),
        })

        if not is_last:
            # Next track's content effectively starts at the crossfade boundary
            offset_s = xfade_out_start  # type: ignore[assignment]

    # Optional target-duration trim/pad
    if target_duration_s is not None:
        target_samples = int(target_duration_s * sr)
        if len(mix) > target_samples:
            fade_len = min(sr, target_samples)  # 1-second fade-out
            mix[target_samples - fade_len: target_samples] *= np.linspace(1.0, 0.0, fade_len)
            mix = mix[:target_samples]
        elif len(mix) < target_samples:
            mix = np.pad(mix, (0, target_samples - len(mix)))

    return mix, sr, boundaries

# ---------------------------------------------------------------------------
# Post-mix: beat grid, energy map, pacing zones
# ---------------------------------------------------------------------------

def merge_beat_grids(
    tracks: list[dict],
    boundaries: list[dict],
    target_bpm: float | None,
) -> dict:
    """Build global beat/downbeat arrays in mix-absolute coordinates."""
    all_beats:     list[float] = []
    all_downbeats: list[float] = []

    for boundary in boundaries:
        all_beats.extend(boundary["track_beat_times"])
        all_downbeats.extend(boundary["track_downbeat_times"])

    # Sort and deduplicate beats within 50 ms of each other
    all_beats.sort()
    deduped: list[float] = []
    for t in all_beats:
        if not deduped or t - deduped[-1] > 0.05:
            deduped.append(t)

    all_downbeats = sorted(set(round(t, 4) for t in all_downbeats))

    if target_bpm is not None:
        effective_bpm = target_bpm
    else:
        total_dur = sum(b["original_duration_s"] for b in boundaries)
        effective_bpm = (
            sum(b["original_bpm"] * b["original_duration_s"] for b in boundaries)
            / total_dur
            if total_dur > 0
            else tracks[0]["tempo_bpm"]
        )

    return {
        "beat_times":      deduped,
        "downbeat_times":  all_downbeats,
        "beat_count":      len(deduped),
        "downbeat_count":  len(all_downbeats),
        "effective_bpm":   round(effective_bpm, 3),
    }


def build_mix_energy_map(
    tracks: list[dict],
    boundaries: list[dict],
) -> dict:
    """Merge per-track energy maps into a single mix-absolute timeline."""
    merged: list[dict] = []
    for track, boundary in zip(tracks, boundaries):
        offset = boundary["mix_start_s"]
        for entry in track["energy_map"]:
            merged.append({
                "time_s":       round(entry["time_s"] + offset, 4),
                "rms":          entry["rms"],
                "is_crescendo": entry["is_crescendo"],
                "is_breakdown": entry["is_breakdown"],
            })

    merged.sort(key=lambda e: e["time_s"])

    # Recompute global thresholds on merged values
    rms_arr = np.array([e["rms"] for e in merged])
    g_p25 = float(np.percentile(rms_arr, BREAKDOWN_PERCENTILE))
    g_p75 = float(np.percentile(rms_arr, CRESCENDO_PERCENTILE))
    g_max = float(np.max(rms_arr))

    for entry in merged:
        entry["is_crescendo"] = entry["rms"] >= g_p75
        entry["is_breakdown"] = entry["rms"] <= g_p25

    return {
        "energy_map": merged,
        "rms_max":    g_max,
        "rms_p25":    g_p25,
        "rms_p75":    g_p75,
    }


def annotate_pacing_zones(
    energy_map: list[dict],
    rms_max: float,
) -> list[dict]:
    """Label each energy entry and run-length encode into pacing zones."""
    if not energy_map or rms_max <= 0:
        return []

    CUT_MAP = {"chorus": 1, "verse": 2, "breakdown": 4}

    def label(rms: float) -> str:
        n = rms / rms_max
        if n >= PACING_CHORUS_THRESHOLD:
            return "chorus"
        if n >= PACING_VERSE_THRESHOLD:
            return "verse"
        return "breakdown"

    # Run-length encode
    zones: list[dict] = []
    for entry in energy_map:
        z = label(entry["rms"])
        if zones and zones[-1]["zone"] == z:
            zones[-1]["end_s"] = entry["time_s"]
        else:
            zones.append({
                "start_s": entry["time_s"],
                "end_s":   entry["time_s"],
                "zone":    z,
            })

    # Merge zones shorter than MIN_ZONE_DURATION_S into their predecessor
    merged: list[dict] = []
    for zone in zones:
        dur = zone["end_s"] - zone["start_s"]
        if merged and dur < MIN_ZONE_DURATION_S:
            merged[-1]["end_s"] = zone["end_s"]
        else:
            merged.append(zone)

    # Finalise
    result: list[dict] = []
    for zone in merged:
        dur = round(zone["end_s"] - zone["start_s"], 4)
        result.append({
            "start_s":          round(zone["start_s"], 4),
            "end_s":            round(zone["end_s"], 4),
            "duration_s":       dur,
            "zone":             zone["zone"],
            "cut_every_n_beats": CUT_MAP[zone["zone"]],
        })
    return result

# ---------------------------------------------------------------------------
# MP3 export
# ---------------------------------------------------------------------------

def export_mp3(
    y: np.ndarray,
    sr: int,
    output_path: Path,
    bitrate: str,
) -> None:
    """Export mix as MP3. Falls back to WAV if PyDub/ffmpeg fails."""
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    try:
        segment = AudioSegment(
            y_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        segment.export(str(output_path), format="mp3", bitrate=bitrate)
    except Exception as exc:
        wav_path = output_path.with_suffix(".wav")
        console.print(
            f"[yellow]MP3 export failed ({exc}); writing WAV fallback → {wav_path}[/]"
        )
        sf.write(str(wav_path), y, sr)

# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------

def assemble_beat_grid_json(
    metadata: dict,
    beat_grid: dict,
    energy_map: list[dict],
    pacing_zones: list[dict],
    track_boundaries: list[dict],
    per_track_analysis: list[dict],
) -> dict:
    return {
        "metadata":           metadata,
        "beat_grid":          beat_grid,
        "energy_map":         energy_map,
        "pacing_zones":       pacing_zones,
        "track_boundaries":   track_boundaries,
        "per_track_analysis": per_track_analysis,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--music", "-m",
    multiple=True, required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input audio file(s). Repeat for multiple tracks. Order = playback order.",
)
@click.option(
    "--output", "-o", "output_dir",
    default=DEFAULT_OUTPUT_DIR, show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory for master_mix.mp3 and beat_grid.json.",
)
@click.option(
    "--crossfade", default=DEFAULT_CROSSFADE_S, show_default=True,
    help="Crossfade duration between tracks (seconds).",
)
@click.option(
    "--target-bpm", default=None, type=float,
    help="Override BPM; crossfade regions are stretched toward this value.",
)
@click.option(
    "--target-duration", default=None, type=float,
    help="Trim or pad final mix to this many seconds.",
)
@click.option(
    "--snippet-duration", default=None, type=float,
    help=(
        "Use only the highest-energy N-second window from each track "
        "instead of the full track."
    ),
)
@click.option(
    "--snap-to-downbeat / --no-snap-to-downbeat",
    default=True, show_default=True,
    help="When using --snippet-duration, snap the snippet start to the nearest downbeat.",
)
@click.option(
    "--energy-smoothing", default=0.5, show_default=True,
    help="Moving-average window in seconds for RMS smoothing.",
)
@click.option(
    "--mp3-bitrate", default=DEFAULT_MP3_BITRATE, show_default=True,
    help="MP3 export bitrate.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Verbose debug output.",
)
def main(
    music: tuple[Path, ...],
    output_dir: Path,
    crossfade: float,
    target_bpm: float | None,
    target_duration: float | None,
    snippet_duration: float | None,
    snap_to_downbeat: bool,
    energy_smoothing: float,
    mp3_bitrate: str,
    verbose: bool,
) -> None:
    """Maestro — Phase 3 of the macOS AI Montage Suite.

    Analyses music tracks, builds a beat-aligned crossfaded mix, and
    writes master_mix.mp3 + beat_grid.json.
    """
    preflight_checks(output_dir)

    music_paths = list(music)
    console.print(
        f"[bold]Maestro[/] · {len(music_paths)} track(s)"
        + (f" · snippet {snippet_duration}s each" if snippet_duration else "")
        + (f" · target BPM {target_bpm}" if target_bpm else "")
    )

    # ------------------------------------------------------------------
    # Stage 1: Analyse tracks
    # ------------------------------------------------------------------
    loaded_tracks: list[dict] = []
    per_track_analysis: list[dict] = []
    failed: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Analysing tracks…", total=len(music_paths))
        for track_idx, path in enumerate(music_paths):
            progress.update(task, description=f"[cyan]{path.name}")
            # First track always plays from the start for a clean intro rhythm;
            # subsequent tracks use peak-energy snippet selection if configured.
            effective_snippet = None if track_idx == 0 else snippet_duration
            try:
                track = analyse_track(
                    path=path,
                    sr=DEFAULT_SR,
                    target_bpm=target_bpm,
                    smoothing_s=energy_smoothing,
                    snippet_duration_s=effective_snippet,
                    snap_to_downbeat=snap_to_downbeat,
                )

                # For the first track, trim from the beginning rather than
                # using the peak-energy window.
                if track_idx == 0 and snippet_duration is not None:
                    end_s = min(float(snippet_duration), track["duration_s"])
                    track = extract_snippet(track, 0.0, end_s)
                    console.print(
                        f"  [dim]snippet: 0.0s – {end_s:.1f}s (from start)[/]"
                    )

                loaded_tracks.append(track)
                errors = track.pop("_errors", [])

                per_track_analysis.append({
                    "track_index":       len(loaded_tracks) - 1,
                    "source_file":       str(path),
                    "original_bpm":      round(track["tempo_bpm"], 3),
                    "original_duration_s": round(track["duration_s"], 3),
                    "snippet_start_s":   track.get("snippet_start_s"),
                    "snippet_end_s":     track.get("snippet_end_s"),
                    "beat_count":        len(track["beat_times"]),
                    "downbeat_count":    len(track["downbeat_times"]),
                    "rms_mean":          round(track["rms_mean"], 6),
                    "rms_max":           round(track["rms_max"], 6),
                    "rms_p25":           round(track["rms_p25"], 6),
                    "rms_p75":           round(track["rms_p75"], 6),
                    "onset_density":     track.get("onset_density", []),
                    "errors":            errors,
                })

                bpm_str = f"{track['tempo_bpm']:.1f} BPM"
                dur_str = f"{track['duration_s']:.1f}s"
                snip_str = (
                    f" [{track['snippet_start_s']:.1f}–{track['snippet_end_s']:.1f}s]"
                    if track.get("snippet_start_s") is not None
                    else ""
                )
                console.print(
                    f"  [green]✓[/] {path.name} — {bpm_str}, {dur_str}{snip_str}"
                )

            except TrackLoadError as exc:
                failed.append(str(path))
                console.print(f"  [yellow]⚠[/] {path.name}: {exc}")
            progress.advance(task)

    if failed:
        console.print(
            f"[yellow]{len(failed)} track(s) skipped:[/] "
            + ", ".join(Path(p).name for p in failed)
        )
    if not loaded_tracks:
        raise SystemExit("No tracks could be loaded. Minimum 1 required.")

    # ------------------------------------------------------------------
    # Stage 2: Build mix
    # ------------------------------------------------------------------
    console.print("[bold]Building mix…[/]")
    mix, sr, boundaries = build_mix(
        loaded_tracks, crossfade, target_bpm, target_duration
    )
    total_duration_s = len(mix) / sr
    console.print(f"  Mix length: [cyan]{total_duration_s:.1f}s[/]")

    # ------------------------------------------------------------------
    # Stage 3: Global beat grid + energy map + pacing zones
    # ------------------------------------------------------------------
    beat_grid_data = merge_beat_grids(loaded_tracks, boundaries, target_bpm)
    mix_energy     = build_mix_energy_map(loaded_tracks, boundaries)
    pacing_zones   = annotate_pacing_zones(
        mix_energy["energy_map"], mix_energy["rms_max"]
    )

    if verbose:
        zone_summary = {}
        for z in pacing_zones:
            zone_summary[z["zone"]] = round(
                zone_summary.get(z["zone"], 0) + z["duration_s"], 1
            )
        console.print(f"  Pacing zones: {zone_summary}")

    # ------------------------------------------------------------------
    # Stage 4: Export MP3
    # ------------------------------------------------------------------
    mix_path = output_dir / "master_mix.mp3"
    console.print(f"[bold]Exporting[/] {mix_path.name} …")
    export_mp3(mix, sr, mix_path, mp3_bitrate)

    # ------------------------------------------------------------------
    # Stage 5: Write beat_grid.json
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "maestro_version":    "1.0.0",
        "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_tracks":      [str(p) for p in music_paths],
        "output_dir":         str(output_dir),
        "master_mix_path":    str(mix_path),
        "total_duration_s":   round(total_duration_s, 3),
        "effective_bpm":      beat_grid_data["effective_bpm"],
        "target_bpm_override": target_bpm,
        "target_duration_s":  target_duration,
        "snippet_duration_s": snippet_duration,
        "snap_to_downbeat":   snap_to_downbeat,
        "crossfade_s":        crossfade,
        "track_count":        len(loaded_tracks),
        "mp3_bitrate":        mp3_bitrate,
    }

    beat_grid_json = assemble_beat_grid_json(
        metadata=metadata,
        beat_grid={
            "beat_times":     beat_grid_data["beat_times"],
            "downbeat_times": beat_grid_data["downbeat_times"],
            "beat_count":     beat_grid_data["beat_count"],
            "downbeat_count": beat_grid_data["downbeat_count"],
        },
        energy_map=mix_energy["energy_map"],
        pacing_zones=pacing_zones,
        track_boundaries=boundaries,
        per_track_analysis=per_track_analysis,
    )

    grid_path = output_dir / "beat_grid.json"
    grid_path.write_text(
        json.dumps(beat_grid_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    console.print(f"[bold green]✓[/] master_mix.mp3  → [cyan]{mix_path}[/]")
    console.print(f"[bold green]✓[/] beat_grid.json  → [cyan]{grid_path}[/]")
    console.print(
        f"\n[bold]Done.[/] {len(loaded_tracks)} tracks · "
        f"{beat_grid_data['beat_count']} beats · "
        f"{beat_grid_data['effective_bpm']:.1f} BPM effective · "
        f"{total_duration_s:.1f}s total"
    )


if __name__ == "__main__":
    main()
