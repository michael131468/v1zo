# Phase 3: The Maestro — Implementation Plan

## Overview

`maestro.py` is a standalone PEP 723 `uv`-managed Python script that constitutes
Phase 3 of the macOS AI Montage Suite. It ingests one or more music tracks,
performs deep audio analysis using Librosa, constructs a beat-aligned mixed
soundtrack using PyDub, and writes two output artefacts: `master_mix.mp3` (the
final audio track) and `beat_grid.json` (a rich pacing map consumed by Phase 4,
`director.py`).

`maestro.py` is deliberately independent of Phase 1 and Phase 2 outputs — it
needs only music files as input.

---

## 1. File Layout

```
v1z0/
├── maestro.py                  ← standalone PEP 723 script
└── <output_dir>/               ← default: ./maestro_output
    ├── master_mix.mp3           ← output: merged, crossfaded audio
    └── beat_grid.json           ← output: full pacing map for director.py
```

---

## 2. PEP 723 Inline Metadata Header

```python
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
```

**System dependencies** (checked in preflight, not managed by uv):
- `ffmpeg` — required by Librosa (MP3 decode via `audioread`) and PyDub (MP3 encode).

**Note on first-run latency:** `librosa` depends on `numba` for JIT compilation
of beat tracking. First-run compilation is slow; subsequent runs are fast.

---

## 3. Constants

```python
DEFAULT_CROSSFADE_S      = 4.0       # seconds
DEFAULT_OUTPUT_DIR       = "./maestro_output"
DEFAULT_MP3_BITRATE      = "192k"
DEFAULT_SR               = 22050     # librosa canonical sample rate (Hz)
HOP_LENGTH               = 512       # librosa default hop length (samples)
ENERGY_MAP_RESOLUTION_S  = 0.1       # seconds between energy_map entries

# Energy thresholds (relative to track's own RMS distribution)
CRESCENDO_PERCENTILE     = 75        # top quartile → crescendo
BREAKDOWN_PERCENTILE     = 25        # bottom quartile → breakdown

# Pacing zone normalised-RMS thresholds
PACING_CHORUS_THRESHOLD  = 0.65      # above → "chorus"
PACING_VERSE_THRESHOLD   = 0.35      # above → "verse", else → "breakdown"

# Minimum pacing zone duration before merging into neighbour
MIN_ZONE_DURATION_S      = 2.0

# BPM bridge: clamp stretch ratio to this range to limit artefacts
BPM_BRIDGE_RATE_MIN      = 0.85
BPM_BRIDGE_RATE_MAX      = 1.15
BPM_BRIDGE_SKIP_DELTA    = 2.0       # BPM difference below this → skip bridge

# Onset density window
ONSET_DENSITY_WINDOW_S   = 1.0
```

---

## 4. Custom Exceptions

```python
class MaestroError(Exception):
    pass

class TrackLoadError(MaestroError):
    pass

class MixError(MaestroError):
    pass
```

`TrackLoadError` is caught per-track; the loop logs a warning and continues.
`MixError` is fatal. If all tracks fail to load, the run exits with a clear
message.

---

## 5. CLI Interface

```
uv run maestro.py --music track1.mp3 --music track2.mp3 [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--music` / `-m` | `PATH` (multiple) | required | Input audio files. Order determines playback order. |
| `--output` / `-o` | `PATH` | `./maestro_output` | Output directory. |
| `--crossfade` | `FLOAT` | `4.0` | Crossfade duration in seconds. |
| `--target-bpm` | `FLOAT` | `None` | Override BPM; all tracks' crossfade regions are stretched toward this value. |
| `--target-duration` | `FLOAT` | `None` | Trim/pad final mix to this many seconds. |
| `--snippet-duration` | `FLOAT` | `None` | If set, extract the highest-energy N-second window from each track instead of using the full track. |
| `--snap-to-downbeat` | `FLAG` | `True` | When using `--snippet-duration`, snap the snippet start to the nearest downbeat for a clean musical entry. |
| `--energy-smoothing` | `FLOAT` | `0.5` | Moving-average window in seconds for RMS smoothing. |
| `--mp3-bitrate` | `TEXT` | `192k` | PyDub MP3 export bitrate. |
| `--verbose` / `-v` | `FLAG` | `False` | Verbose debug output. |

`--music` uses `multiple=True` with `click.Path(exists=True, dir_okay=False, path_type=Path)`.

---

## 6. Preflight Checks

```python
def preflight_checks(output_dir: Path) -> Path:
    """Check ffmpeg, create output dir. Returns ffmpeg Path."""
```

- `shutil.which("ffmpeg")` — exits with `"ffmpeg not found. Install with: brew install ffmpeg"` if absent.
- `output_dir.mkdir(parents=True, exist_ok=True)`.
- No macOS version check (no Vision framework used in this phase).

---

## 7. Per-Track Analysis Pipeline

### 7.1 `load_track(path, sr) → (np.ndarray, int)`

```python
y, sr = librosa.load(str(path), sr=sr, mono=True)
```

Wraps in `try/except` and re-raises as `TrackLoadError`. Always resamples to
`DEFAULT_SR` and converts to mono.

### 7.2 `analyse_beats(y, sr, target_bpm) → dict`

**Beat tracking:**

```python
tempo, beat_frames = librosa.beat.beat_track(
    y=y, sr=sr, hop_length=HOP_LENGTH,
    bpm=target_bpm,   # None → auto-detect; float → use as start estimate
    units="frames",
)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
```

**Downbeat detection via PLP:**

```python
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
pulse = librosa.beat.plp(
    onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
    tempo_min=tempo * 0.9, tempo_max=tempo * 1.1,
)
plp_peak_time = librosa.frames_to_time(np.argmax(pulse), sr=sr, hop_length=HOP_LENGTH)
phase_idx = int(np.argmin(np.abs(beat_times - plp_peak_time)))
downbeat_indices = list(range(phase_idx % 4, len(beat_times), 4))
downbeat_times = beat_times[downbeat_indices].tolist()
```

**Fallback** (if `librosa.beat.plp` raises): every 4th beat from index 0.

Returns `{"tempo_bpm", "beat_times", "downbeat_times"}`.

### 7.3 `analyse_energy(y, sr, smoothing_s) → dict`

```python
rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
```

**Smoothing** (no scipy dependency — NumPy moving average):

```python
window = max(1, int(smoothing_s * sr / HOP_LENGTH))
kernel = np.ones(window) / window
rms_smooth = np.convolve(rms, kernel, mode="same")
```

**Thresholds:**

```python
p25 = np.percentile(rms_smooth, BREAKDOWN_PERCENTILE)
p75 = np.percentile(rms_smooth, CRESCENDO_PERCENTILE)
```

**Energy map** (downsampled to `ENERGY_MAP_RESOLUTION_S`):

```python
stride = max(1, int(ENERGY_MAP_RESOLUTION_S * sr / HOP_LENGTH))
energy_map = [
    {
        "time_s": round(float(frame_times[i]), 4),
        "rms": round(float(rms_smooth[i]), 6),
        "is_crescendo": bool(rms_smooth[i] >= p75),
        "is_breakdown": bool(rms_smooth[i] <= p25),
    }
    for i in range(0, len(rms_smooth), stride)
]
```

Returns `{"energy_map", "rms_p25", "rms_p75", "rms_mean", "rms_max"}`.

### 7.4 `analyse_spectral(y, sr) → dict`

```python
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)
window_frames = int(ONSET_DENSITY_WINDOW_S * sr / HOP_LENGTH)
onset_density = [
    {
        "time_s": round(float(onset_times[i]), 4),
        "onset_strength": round(float(np.mean(onset_env[i:i + window_frames])), 6),
    }
    for i in range(0, len(onset_env), window_frames)
]
```

Returns `{"onset_density"}`.

### 7.5 `find_peak_energy_window(y, sr, duration_s, energy_map, downbeat_times, snap_to_downbeat) → tuple[float, float, float]`

Finds the contiguous `duration_s`-second window with the highest average RMS
energy in a track. Returns `(start_s, end_s, mean_rms)`.

**Algorithm:**

1. If `duration_s >= track_duration`, return the full track (no slicing needed).
2. Build parallel `times` and `rms_values` arrays from `energy_map`.
3. Slide a window across the energy map at `ENERGY_MAP_RESOLUTION_S` steps.
   For each candidate start position, compute `mean(rms[start:start+duration])`.
4. Record the `start_s` with the highest mean RMS.
5. **Snap to downbeat** (if enabled): search for downbeats within ±4 s of
   `best_start`. Pick the nearest one whose resulting window still fits within
   the track. This ensures the snippet begins on a bar boundary so the
   crossfade entry is rhythmically coherent.

```python
def find_peak_energy_window(
    y, sr, duration_s, energy_map, downbeat_times,
    snap_to_downbeat=True,
) -> tuple[float, float, float]:
    total_s = len(y) / sr
    if duration_s >= total_s:
        return 0.0, total_s, mean(rms for entry in energy_map)

    times  = np.array([e["time_s"] for e in energy_map])
    rms    = np.array([e["rms"]    for e in energy_map])

    best_start, best_score = 0.0, -1.0
    for i, t in enumerate(times):
        if t + duration_s > total_s:
            break
        mask = (times >= t) & (times < t + duration_s)
        score = float(np.mean(rms[mask]))
        if score > best_score:
            best_score, best_start = score, t

    if snap_to_downbeat and downbeat_times:
        db = np.array(downbeat_times)
        candidates = db[
            (db >= max(0, best_start - 4.0)) &
            (db <= min(best_start + 4.0, total_s - duration_s))
        ]
        if len(candidates):
            nearest = float(candidates[np.argmin(np.abs(candidates - best_start))])
            if nearest + duration_s <= total_s:
                best_start = nearest

    return round(best_start, 4), round(best_start + duration_s, 4), round(best_score, 6)
```

### 7.6 `extract_snippet(track_data, start_s, end_s) → dict`

Slices the audio array and adjusts all time-indexed data to the new
origin (so `t=0` in the snippet corresponds to `start_s` in the original).

```python
def extract_snippet(track_data, start_s, end_s):
    y     = track_data["y"]
    sr    = track_data["sr"]
    y_snip = y[int(start_s * sr) : int(end_s * sr)]

    def shift(ts):
        return [round(t - start_s, 4) for t in ts if start_s <= t < end_s]

    def shift_map(entries, key="time_s"):
        return [
            {**e, key: round(e[key] - start_s, 4)}
            for e in entries if start_s <= e[key] < end_s
        ]

    return {
        **track_data,
        "y":               y_snip,
        "duration_s":      round(end_s - start_s, 4),
        "beat_times":      shift(track_data["beat_times"]),
        "downbeat_times":  shift(track_data["downbeat_times"]),
        "energy_map":      shift_map(track_data["energy_map"]),
        "onset_density":   shift_map(track_data.get("onset_density", [])),
        "snippet_start_s": start_s,   # stored for beat_grid.json provenance
        "snippet_end_s":   end_s,
    }
```

The `snippet_start_s` / `snippet_end_s` fields are recorded in
`per_track_analysis` so the user knows which part of the source file was used.

### 7.7 `analyse_track(path, sr, target_bpm, smoothing_s) → dict`

Top-level wrapper. Calls `load_track` → `analyse_beats` → `analyse_energy` →
`analyse_spectral` and merges all results. Stores the raw `y` array for mixing
(never written to JSON).

---

## 8. Multi-Track Mixing

### 8.1 `find_crossfade_start(downbeat_times, duration_s, crossfade_s) → float`

Returns the last downbeat time that leaves at least `crossfade_s` seconds
remaining. Falls back to `max(0.0, duration_s - crossfade_s)` if no qualifying
downbeat exists.

### 8.2 `apply_bpm_bridge(y_out, y_in, sr, bpm_out, bpm_in, crossfade_s, crossfade_start_s) → (np.ndarray, np.ndarray)`

Skipped if `abs(bpm_out - bpm_in) < BPM_BRIDGE_SKIP_DELTA`.

1. Extract tail: `y_tail = y_out[int(crossfade_start_s * sr):]`
2. Stretch ratio: `rate = np.clip(bpm_out / bpm_in, BPM_BRIDGE_RATE_MIN, BPM_BRIDGE_RATE_MAX)`
3. Stretch: `y_tail_stretched = librosa.effects.time_stretch(y_tail, rate=rate)`
4. Reassemble: `y_out_bridged = np.concatenate([y_out[:int(crossfade_start_s * sr)], y_tail_stretched])`

Returns `(y_out_bridged, y_in)`. If `target_bpm` is set globally, both tails
are stretched toward the target.

If `librosa.effects.time_stretch` fails, log a warning, return originals
unchanged, set `bpm_bridge_applied: false` in the boundary record.

### 8.3 `constant_power_crossfade(y_out, y_in, sr, crossfade_start_s, crossfade_s) → np.ndarray`

Gain curves over `N = int(crossfade_s * sr)` samples:

```
fade_out[i] = cos( i/N * π/2 )    # 1.0 → 0.0
fade_in[i]  = sin( i/N * π/2 )    # 0.0 → 1.0
```

Satisfies constant-power invariant: `fade_out²[i] + fade_in²[i] = 1`.

```python
t = np.linspace(0, 1, N, endpoint=False)
fade_out_curve = np.cos(t * np.pi / 2)
fade_in_curve  = np.sin(t * np.pi / 2)
xfade = tail * fade_out_curve + head * fade_in_curve
return np.concatenate([y_pre, xfade, y_post])
```

Both `tail` and `head` are zero-padded to the same length before the multiply
(BPM bridge may have changed `y_out`'s tail length).

### 8.4 `build_mix(tracks, crossfade_s, target_bpm, target_duration_s) → (np.ndarray, int, list[dict])`

Iterates through tracks, applying `find_crossfade_start` → `apply_bpm_bridge` →
`constant_power_crossfade` for each adjacent pair. Maintains a running
`offset_s` to track where each track begins in the final timeline.

After assembly, optionally trims or zero-pads to `target_duration_s` with a
1-second fade-out applied before any hard trim point.

Returns `(mix_array, sr, track_boundaries)`.

---

## 9. Post-Mix Processing

### 9.1 `merge_beat_grids(tracks, boundaries, target_bpm) → dict`

Shifts each track's beat/downbeat times by its `mix_start_s`. Deduplicates
beats within 0.05 s of each other in the crossfade region (keeps the one closer
to the theoretical grid). Re-anchors downbeats using the phase of track 0's
first downbeat.

`effective_bpm`: if `target_bpm` is set, use it directly; otherwise compute the
duration-weighted average of all tracks' `tempo_bpm`.

### 9.2 `build_mix_energy_map(tracks, boundaries) → dict`

Shifts each track's `energy_map` entries by `mix_start_s`. Merges into a single
timeline sorted by `time_s`. Recomputes global `p25`/`p75` on the merged RMS
values (individual track thresholds are no longer meaningful after mixing).

### 9.3 `annotate_pacing_zones(energy_map, rms_max) → list[dict]`

1. Normalise each entry's `rms` by `rms_max` → value in `[0, 1]`.
2. Label each entry: `>= PACING_CHORUS_THRESHOLD` → `"chorus"`;
   `>= PACING_VERSE_THRESHOLD` → `"verse"`; else → `"breakdown"`.
3. Run-length encode into contiguous zones.
4. Merge zones shorter than `MIN_ZONE_DURATION_S` into their preceding neighbour
   (or following if it's the first zone).
5. Assign `cut_every_n_beats`: chorus → 1, verse → 2, breakdown → 4.

---

## 10. MP3 Export

```python
def export_mp3(y, sr, output_path, bitrate) -> None:
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    segment = AudioSegment(
        y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1,
    )
    segment.export(str(output_path), format="mp3", bitrate=bitrate)
```

**Fallback:** if PyDub export raises (ffmpeg codec issue), write as WAV via
`soundfile.write(output_path.with_suffix(".wav"), y.T, sr)` and warn the user.

---

## 11. beat_grid.json Schema

```json
{
  "metadata": {
    "maestro_version": "1.0.0",
    "generated_at": "2026-03-25T12:00:00Z",
    "source_tracks": ["/abs/path/track1.mp3"],
    "output_dir": "/abs/path/maestro_output",
    "master_mix_path": "/abs/path/maestro_output/master_mix.mp3",
    "total_duration_s": 213.4,
    "effective_bpm": 124.0,
    "target_bpm_override": null,
    "target_duration_s": null,
    "crossfade_s": 4.0,
    "track_count": 2,
    "mp3_bitrate": "192k"
  },

  "beat_grid": {
    "beat_times": [0.48, 0.97, 1.46],
    "downbeat_times": [0.48, 2.40, 4.32],
    "beat_count": 438,
    "downbeat_count": 110
  },

  "energy_map": [
    {
      "time_s": 0.0,
      "rms": 0.042318,
      "is_crescendo": false,
      "is_breakdown": false
    }
  ],

  "pacing_zones": [
    {
      "start_s": 0.0,
      "end_s": 32.4,
      "duration_s": 32.4,
      "zone": "verse",
      "cut_every_n_beats": 2
    }
  ],

  "track_boundaries": [
    {
      "track_index": 0,
      "source_file": "/abs/path/track1.mp3",
      "source_file_rel": "track1.mp3",
      "original_bpm": 120.3,
      "original_duration_s": 214.8,
      "mix_start_s": 0.0,
      "mix_end_s": 118.2,
      "crossfade_out_start_s": 114.2,
      "crossfade_out_end_s": 118.2,
      "bpm_bridge_applied": true,
      "bpm_bridge_rate": 0.97,
      "track_beat_times": [0.48, 0.97],
      "track_downbeat_times": [0.48, 2.40]
    },
    {
      "track_index": 1,
      "source_file": "/abs/path/track2.mp3",
      "source_file_rel": "track2.mp3",
      "original_bpm": 126.1,
      "original_duration_s": 198.3,
      "mix_start_s": 114.2,
      "mix_end_s": 213.4,
      "crossfade_out_start_s": null,
      "crossfade_out_end_s": null,
      "bpm_bridge_applied": true,
      "bpm_bridge_rate": 1.02,
      "track_beat_times": [114.68, 115.16],
      "track_downbeat_times": [114.68, 116.60]
    }
  ],

  "per_track_analysis": [
    {
      "track_index": 0,
      "source_file": "/abs/path/track1.mp3",
      "original_bpm": 120.3,
      "original_duration_s": 214.8,
      "beat_count": 430,
      "downbeat_count": 108,
      "rms_mean": 0.041,
      "rms_max": 0.112,
      "rms_p25": 0.022,
      "rms_p75": 0.068,
      "onset_density": [
        {"time_s": 0.0, "onset_strength": 0.31}
      ],
      "snippet_start_s": 48.2,           // float | null — source-file start of selected window (null = full track used)
      "snippet_end_s": 168.2,            // float | null — source-file end of selected window
      "errors": []
    }
  ]
}
```

### Schema notes

| Field | Coordinate system | Used by Phase 4 |
|---|---|---|
| `beat_grid.beat_times` | Mix-absolute seconds | Primary cut-point source |
| `beat_grid.downbeat_times` | Mix-absolute seconds | Scene boundary alignment |
| `energy_map[].time_s` | Mix-absolute seconds | Clip-to-energy correlation |
| `pacing_zones[].zone` + `cut_every_n_beats` | Mix-absolute seconds | Controls cut frequency |
| `track_boundaries[].track_beat_times` | Mix-absolute seconds | Debug / per-source correlation |
| `per_track_analysis[].onset_density[].time_s` | Source-file seconds | Debug only |

All timestamps in `beat_grid`, `energy_map`, and `pacing_zones` are
**mix-absolute** (seconds from the start of `master_mix.mp3`). Only
`per_track_analysis.onset_density` uses source-file-relative coordinates.

---

## 12. Main Orchestration Sequence

```
preflight_checks(output_dir)

[Rich Progress] Analyse tracks
    for each track:
        analyse_track() → {y, tempo_bpm, beat_times, downbeat_times,
                            energy_map, rms_*, onset_density}
        TrackLoadError → skip + warn
    ≥1 loaded track required (else SystemExit)

build_mix(tracks, crossfade_s, target_bpm, target_duration_s)
    → (mix_array, sr, track_boundaries)

merge_beat_grids(tracks, boundaries, target_bpm)
    → {beat_times, downbeat_times, effective_bpm}

build_mix_energy_map(tracks, boundaries)
    → {energy_map, rms_max}

annotate_pacing_zones(energy_map, rms_max)
    → pacing_zones

export_mp3(mix_array, sr, output_dir/"master_mix.mp3", bitrate)
    → WAV fallback on PyDub failure

write beat_grid.json

console.print summary
```

---

## 13. Error Handling

| Situation | Behaviour |
|---|---|
| Single track fails to load | Log warning, skip, continue with remaining tracks |
| All tracks fail | `SystemExit("No tracks could be loaded.")` |
| BPM bridge ratio outside `[0.85, 1.15]` | Skip bridge, log warning recommending `--target-bpm` |
| `librosa.effects.time_stretch` raises | Skip bridge, `bpm_bridge_applied: false`, warn |
| `librosa.beat.plp` unavailable | Fall back to every-4th-beat downbeats, log to `errors[]` |
| PyDub MP3 export fails | Write WAV via soundfile, warn user |

---

## 14. Implementation Sequence

1. Scaffolding: PEP 723 header, imports, constants, exceptions, `preflight_checks`, `main()` stub.
2. `load_track` + `analyse_beats` — verify beat timestamps on a known-BPM file.
3. `analyse_energy` — check energy map entries, percentile thresholds.
4. `analyse_spectral` — onset density windows.
5. `analyse_track` integration.
6. `find_crossfade_start`.
7. `apply_bpm_bridge` — test with two tracks of known BPM difference.
8. `constant_power_crossfade` — verify `fade_out² + fade_in² ≈ 1` numerically.
9. `build_mix` — multi-track assembly, trim/pad.
10. `merge_beat_grids` — offset arithmetic, deduplication.
11. `build_mix_energy_map` — coordinate shift, global threshold recompute.
12. `annotate_pacing_zones` — run-length encoding, zone merging.
13. `export_mp3` + WAV fallback.
14. `beat_grid.json` assembly and write.
15. End-to-end test: two tracks, different BPMs; verify `master_mix.mp3` plays cleanly, `beat_grid.json` is valid.
