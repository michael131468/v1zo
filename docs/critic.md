# Phase 2: The Critic — Implementation Plan

## Overview

`critic.py` reads `scenes.json` from Phase 1, evaluates the quality of every
scene and photo, and writes `scored_snippets.json`. It uses a sliding-window
strategy to find the single best 3-second moment inside each video scene
rather than scoring the whole clip.

---

## 1. File Layout

```
v1z0/
├── critic.py                  ← standalone PEP 723 script
└── <output_dir>/
    ├── scenes.json            ← input (from scout.py)
    ├── scored_snippets.json   ← output
    └── previews/
        └── C{n}_best.jpg      ← best-window frame per snippet
```

`critic.py` defaults `--output` to the same directory that contains
`scenes.json`.

---

## 2. CLI

```
uv run critic.py --scenes ./scout_output/scenes.json [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--scenes` / `-s` | required | Path to `scenes.json` |
| `--output` / `-o` | same dir as `--scenes` | Output directory |
| `--window-size` | `3.0` | Sliding window length (seconds) |
| `--window-step` | `0.5` | Sliding window step (seconds) |
| `--blur-threshold` | `50.0` | Min Laplacian variance; lower → discard as blurry |
| `--tilt-threshold` | `15.0` | Max horizon angle (degrees); higher → discard as tilted |
| `--workers` | `4` | Concurrent Vision/scoring threads |
| `--skip-previews` | flag | Skip saving best-frame JPEGs |
| `--resume` | flag | Skip scene_ids already in scored_snippets.json |

---

## 3. Pipeline

```
load_scenes_json()
    └─> for each video_scene:
    |       compute_windows()
    |       extract_window_frames()   [FFmpeg, parallel per scene]
    |       score_frames_parallel()   [Vision + blur, thread pool]
    |       find_peak_window()
    |       save_best_frame()
    └─> for each photo / burst representative:
    |       score_single_photo()
    └─> build_snippets_list()
    └─> write_scored_snippets_json()
```

---

## 4. Sliding Window

For a scene with `duration_s = D`, `window_size = W`, `window_step = S`:

- If `D <= W`: one window covering the full scene.
- Else: windows at `start = 0, S, 2S, …` until `start + W >= D`.
- Frame extracted at `mid = start + W/2` for scoring.

---

## 5. Vision Requests (per frame, batched in one handler call)

| Request | Result | Accessor | Notes |
|---|---|---|---|
| `VNCalculateImageAestheticsScoresRequest` | `VNImageAestheticsScoresObservation` | `.overallScore()` → float [−1, 1] | check `hasattr` |
| `VNGenerateAttentionBasedSaliencyImageRequest` | `VNSaliencyImageObservation` | `.salientObjects()` → [VNRectangleObservation] | Always available |
| `VNDetectHorizonRequest` | `VNHorizonObservation` | `.angle()` → radians | Always available |
| `VNDetectFaceLandmarksRequest` | [VNFaceObservation] | `.expressions()` (macOS 14+) | Fall back to face count only |

`VNDetectFaceExpressionsRequest` is **not a public API** — face expression data
is accessed via `VNFaceObservation.expressions()` on macOS 14+ from a
`VNDetectFaceLandmarksRequest`.

---

## 6. Blur Detection

Using OpenCV Laplacian variance on the grayscale frame:

```python
gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

Normalize: `blur_score = min(1.0, variance / 500.0)` (500+ = very sharp).
Discard if `variance < blur_threshold`.

---

## 7. Composite Score

```
composite = (
    0.40 * aesthetic_score    +   # Vision aesthetic quality [0, 1]
    0.20 * saliency_score     +   # Coverage of salient objects [0, 1]
    0.20 * smile_score        +   # Happiness from face expressions [0, 1]
    0.10 * blur_score         +   # Laplacian sharpness [0, 1]
    0.10 * horizon_score          # Level horizon [0, 1]
)
```

- **aesthetic_score**: `(overallScore + 1) / 2` (normalise −1..1 → 0..1). Default 0.5 if API unavailable.
- **saliency_score**: `min(1.0, sum(w*h for each salient bbox))`.
- **smile_score**: Happiness confidence from `VNFaceObservation.expressions()`, else 0.
- **blur_score**: `min(1.0, laplacian_variance / 500.0)`.
- **horizon_score**: `max(0.0, 1.0 - abs(angle_deg) / tilt_threshold)`.

### Discard Conditions

| Condition | Reason |
|---|---|
| `laplacian_variance < blur_threshold` | `"blurry"` |
| `abs(horizon_angle_deg) > tilt_threshold` | `"tilted"` |

A snippet is marked `discarded: true` only if **all windows** in a scene fail.
If any window passes, the best passing window is selected. If all fail, the
best-scoring failing window is selected anyway and marked discarded.

---

## 8. scored_snippets.json Schema

```json
{
  "metadata": {
    "critic_version": "1.0.0",
    "generated_at": "...",
    "scenes_json": "/abs/path/to/scenes.json",
    "output_dir": "/abs/path/to/scout_output",
    "window_size_s": 3.0,
    "window_step_s": 0.5,
    "blur_threshold": 50.0,
    "tilt_threshold_deg": 15.0,
    "total_snippets": 142,
    "discarded_snippets": 8
  },
  "snippets": [
    {
      "snippet_id": "C00001",
      "scene_id": "S00001",
      "type": "video_scene",
      "source_file": "/abs/path/to/clip.mov",
      "source_file_rel": "clips/clip.mov",
      "best_window": {
        "start_time_s": 2.5,
        "end_time_s": 5.5,
        "duration_s": 3.0,
        "mid_time_s": 4.0
      },
      "scores": {
        "aesthetic": 0.85,
        "saliency_coverage": 0.72,
        "smile": 0.30,
        "blur_variance": 450.2,
        "blur_score": 0.90,
        "horizon_angle_deg": 1.2,
        "horizon_score": 0.92,
        "composite": 0.78
      },
      "saliency_centroid": [0.45, 0.50],
      "salient_objects": [{"x": 0.3, "y": 0.2, "w": 0.4, "h": 0.6}],
      "face_count": 1,
      "windows_evaluated": 23,
      "discarded": false,
      "discard_reason": null,
      "preview_frame": "previews/C00001_best.jpg",
      "errors": []
    }
  ]
}
```

Photos set `best_window` fields to `null` and `windows_evaluated: 1`.
Burst groups set `type: "burst_group"` and score the representative member.

---

## 9. Dependencies

| Package | Purpose |
|---|---|
| `pyobjc-core>=9.0` | PyObjC bridge |
| `pyobjc-framework-Cocoa>=9.0` | Foundation (NSURL) |
| `pyobjc-framework-Vision>=9.0` | Vision requests |
| `opencv-python-headless>=4.8` | Laplacian blur detection |
| `numpy>=1.24` | Numeric ops |
| `click>=8.0` | CLI |
| `rich>=13.0` | Progress display |

System: `ffmpeg` (Homebrew), macOS 13+.

---

## 10. Implementation Sequence

1. Scaffolding: PEP 723 header, CLI, `preflight_checks`, `load_scenes_json`.
2. Sliding window logic: `compute_windows`.
3. FFmpeg frame extraction: `extract_frame` (single call, input-seek).
4. Vision scoring: `score_frame_vision` (batched requests, graceful degradation).
5. Blur scoring: `compute_blur` (OpenCV).
6. Composite + discard logic: `compute_composite`, `find_peak_window`.
7. Photo scoring (no windows, score original file).
8. Snippet assembly + JSON output.
