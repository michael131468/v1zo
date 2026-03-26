# Phase 1: The Scout — Implementation Plan

## Overview

`scout.py` is a standalone `uv`-managed Python script (PEP 723 inline metadata)
that ingests a directory of raw media, runs scene detection and Vision
framework analysis, and writes `scenes.json` plus `scout_review.html`. It is
the first stage of the macOS AI Montage Suite pipeline and produces all data
consumed by Phase 2 (`critic.py`).

---

## 1. Project Structure and File Layout

Because the suite philosophy is "each phase is a standalone Python script," `scout.py` is a single self-contained file with PEP 723 inline metadata. All generated artifacts go into a per-run output directory.

```
v1z0/
├── docs/
│   ├── architecture.md
│   └── scout.md              ← this plan
├── scout.py                  ← single file, uv PEP 723 script
└── (generated at runtime)
    └── <output_dir>/
        ├── scenes.json
        ├── scout_review.html
        └── previews/
            ├── <scene_id>_start.jpg
            ├── <scene_id>_mid.jpg
            ├── <scene_id>_end.jpg
            └── <scene_id>_preview.gif
```

`scout.py` is run directly via `uv run scout.py [args]`. There is no `src/` layout, no `__init__.py`, no package install step. `uv` manages the ephemeral virtual environment from the inline metadata block.

### PEP 723 Inline Metadata Header

The top of `scout.py` begins with a `# /// script` block that `uv run` reads to resolve dependencies before execution:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "scenedetect[opencv]>=0.6.7",
#   "pyobjc-core>=12.1",
#   "pyobjc-framework-Vision>=12.1",
#   "pyobjc-framework-Quartz>=12.1",
#   "pyobjc-framework-Foundation>=12.1",
#   "Pillow>=12.1",
#   "piexif>=1.1.3",
#   "numpy>=2.2",
#   "click>=8.3",
#   "rich>=14.3",
# ]
# ///
```

This means `uv run scout.py --input /path/to/media` resolves and installs all dependencies into a cached isolated environment on first run, with no manual `pip install` required.

---

## 2. CLI Interface

Built with `click`. All arguments exposed via `@click.command()` decorated `main()`.

```
uv run scout.py [OPTIONS] --input DIR
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` / `-i` | `PATH` | required | Root directory to scan recursively for media |
| `--output` / `-o` | `PATH` | `./scout_output` | Directory for all generated artifacts |
| `--extensions` | `TEXT` | `mov,mp4,heic,jpg,jpeg,png` | Comma-separated file extensions to ingest |
| `--scene-threshold` | `FLOAT` | `27.0` | PySceneDetect ContentDetector threshold (0–100) |
| `--burst-window` | `FLOAT` | `3.0` | Seconds within which photos count as a burst |
| `--burst-min-count` | `INT` | `3` | Minimum photos in a time window to form a burst |
| `--similarity-threshold` | `FLOAT` | `0.15` | Max VNFeaturePrint distance to treat two frames as visually identical |
| `--workers` | `INT` | `4` | Concurrent worker threads for Vision analysis |
| `--skip-previews` | `FLAG` | `False` | Skip GIF/thumbnail generation (dry run mode) |
| `--verbose` / `-v` | `FLAG` | `False` | Rich-formatted debug logging |
| `--resume` | `FLAG` | `False` | Skip files whose scene IDs already appear in scenes.json |

The `main()` function orchestrates the full pipeline by calling each stage in sequence and passing the accumulated state dict forward.

---

## 3. Processing Pipeline Stages

The pipeline runs in this order, each stage returning data that feeds the next:

```
discover_media()
    └─> process_video()  (per video file)
    |       └─> detect_scenes()
    |       └─> extract_frames()
    |       └─> classify_scene()       [Vision]
    |       └─> fingerprint_scene()    [Vision]
    └─> process_photos()  (per photo file)
    |       └─> read_exif_metadata()
    |       └─> fingerprint_photo()    [Vision]
    └─> detect_burst_groups()
    └─> build_scenes_list()
    └─> write_scenes_json()
    └─> generate_review_html()
```

### 3.1 `discover_media`

```python
def discover_media(
    input_dir: Path,
    extensions: list[str],
) -> dict[str, list[Path]]:
    ...
    # Returns: {"video": [...], "photo": [...]}
```

Uses `Path.rglob()` with case-insensitive suffix matching. Skips hidden directories (name starts with `.`). Skips files smaller than 10 KB to avoid corrupt/empty stubs. Returns a dict keyed by media type.

### 3.2 `detect_scenes`

```python
def detect_scenes(
    video_path: Path,
    threshold: float,
) -> list[dict]:
    ...
    # Returns list of scene dicts: {scene_index, start_frame, end_frame,
    #                               start_time_s, end_time_s, duration_s}
```

Wraps PySceneDetect's programmatic API (not CLI subprocess). Uses `open_video()` and `ContentDetector`. See Section 5 for full PySceneDetect integration details.

### 3.3 `extract_frames`

```python
def extract_frames(
    video_path: Path,
    scene: dict,
    output_dir: Path,
    scene_id: str,
) -> dict[str, Path]:
    ...
    # Returns: {"start": Path, "mid": Path, "end": Path, "preview_gif": Path}
```

Calls FFmpeg as a subprocess for three JPEG stills and one GIF. See Section 6.

### 3.4 `classify_scene`

```python
def classify_scene(
    image_path: Path,
) -> list[dict]:
    ...
    # Returns: [{"label": str, "confidence": float}, ...]  sorted descending
```

Uses VNClassifyImageRequest via PyObjC. See Section 4.

### 3.5 `fingerprint_scene`

```python
def fingerprint_scene(
    image_path: Path,
) -> list[float] | None:
    ...
    # Returns the raw feature vector as a Python list[float], or None on failure
```

Uses VNGenerateImageFeaturePrintRequest via PyObjC. See Section 4.

### 3.6 `compute_feature_print_distance`

```python
def compute_feature_print_distance(
    fp_a: list[float],
    fp_b: list[float],
) -> float:
    ...
    # Returns L2 distance (float). Lower = more similar.
```

VNFeaturePrintObservation has an ObjC method `computeDistanceTo_error_()` but PyObjC bridging for out-error patterns is fragile. The reliable fallback is to keep the raw float vectors and compute L2 distance directly with NumPy: `float(np.linalg.norm(np.array(fp_a) - np.array(fp_b)))`. The Vision observation objects must be retained in scope during extraction (see Section 4), then the vectors are serialised to plain Python lists for JSON storage and NumPy comparison.

### 3.7 `read_exif_metadata`

```python
def read_exif_metadata(
    photo_path: Path,
) -> dict:
    ...
    # Returns: {"datetime_original": datetime | None,
    #           "datetime_digitized": datetime | None,
    #           "subsec_time": str | None,
    #           "gps_lat": float | None, "gps_lon": float | None,
    #           "make": str, "model": str,
    #           "orientation": int}
```

See Section 7.

### 3.8 `detect_burst_groups`

```python
def detect_burst_groups(
    photos: list[dict],   # each dict has exif + fingerprint + path
    time_window_s: float,
    min_count: int,
    similarity_threshold: float,
) -> list[dict]:
    ...
    # Returns list of burst group dicts (see scenes.json schema)
```

Two-pass algorithm:
- **Pass 1 (temporal)**: Sort photos by `datetime_original`. Slide a window of `time_window_s` seconds. Any cluster of `min_count` or more photos gets tagged as a temporal burst candidate.
- **Pass 2 (visual dedup within non-temporal singles)**: For photos not yet in a burst cluster, compare their feature print vectors pairwise. If distance < `similarity_threshold`, merge them into a visual similarity burst.

This prevents double-grouping: a photo already in a temporal burst is not re-evaluated for visual similarity bursts.

### 3.9 `build_scenes_list`

```python
def build_scenes_list(
    video_scenes: list[dict],
    photo_records: list[dict],
    burst_groups: list[dict],
) -> list[dict]:
    ...
    # Returns the flat list that becomes scenes.json["scenes"]
```

Merges all three input lists, assigns globally unique `scene_id` values (format: `S{zero_padded_index:05d}`), and ensures burst members reference their `burst_group_id`.

### 3.10 `write_scenes_json`

```python
def write_scenes_json(
    scenes: list[dict],
    output_path: Path,
    metadata: dict,
) -> None:
```

Writes the full JSON structure. See Section 8.

### 3.11 `generate_review_html`

```python
def generate_review_html(
    scenes: list[dict],
    output_path: Path,
    previews_dir: Path,
) -> None:
```

Builds `scout_review.html` from an inline template string (no Jinja2 dependency). See Section 9.

---

## 4. macOS Vision Framework Integration (PyObjC)

### 4.1 Import Block

```python
import Vision
import Quartz
import Foundation
from Foundation import NSURL, NSData
```

PyObjC 12.1 ships `pyobjc-framework-Vision` as a separate package. After installing via the PEP 723 metadata, the above imports work directly.

### 4.2 Creating a VNImageRequestHandler from a file path

Vision requests require either an `NSURL` or a `CVPixelBuffer`. The simplest path for JPEG/PNG files is to pass an `NSURL` directly to `VNImageRequestHandler`.

```python
def _make_request_handler(image_path: Path) -> "Vision.VNImageRequestHandler":
    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        url, {}
    )
    return handler
```

For HEIC files (iPhone photos), `NSURL` also works because macOS ImageIO can decode HEIC natively on macOS 13+.

### 4.3 VNClassifyImageRequest

```python
def classify_scene(image_path: Path) -> list[dict]:
    handler = _make_request_handler(image_path)
    request = Vision.VNClassifyImageRequest.alloc().init()

    # performRequests_error_ returns a tuple: (bool_success, NSError_or_None)
    success, err = handler.performRequests_error_([request], None)

    if not success or err:
        raise RuntimeError(f"VNClassifyImageRequest failed: {err}")

    observations = request.results()
    results = []
    for obs in observations:
        # obs is VNClassificationObservation
        if obs.confidence() > 0.05:
            results.append({
                "label": str(obs.identifier()),
                "confidence": float(obs.confidence()),
            })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
```

**PyObjC out-parameter pattern**: `performRequests_error_` maps the ObjC `NSError **` out-parameter: call as `handler.performRequests_error_(requests, None)` and unpack the return tuple `(bool_success, nsError_or_None)`.

**Semantic label mapping**: Apple's identifiers are hierarchical (e.g., `"plant_life"`, `"outdoor"`, `"food"`, `"person"`). Map these to the architecture's top-level buckets:

```python
CATEGORY_MAP = {
    "person": "People",
    "face": "People",
    "crowd": "People",
    "food": "Food",
    "drink": "Food",
    "plant_life": "Nature",
    "water": "Nature",
    "sky": "Nature",
    "mountain": "Nature",
    "animal": "Nature",
    "urban": "Urban",
    "architecture": "Urban",
    "vehicle": "Urban",
    "street": "Urban",
}

def map_to_category(observations: list[dict]) -> str:
    for obs in observations:  # already sorted descending
        label = obs["label"].lower()
        for key, category in CATEGORY_MAP.items():
            if key in label:
                return category
    return "Other"
```

This map should be treated as a living lookup and extended as unknown identifiers are observed in logs.

### 4.4 VNGenerateImageFeaturePrintRequest

```python
def fingerprint_scene(image_path: Path) -> list[float] | None:
    handler = _make_request_handler(image_path)
    request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    request.setImageCropAndScaleOption_(0)  # VNImageCropAndScaleOptionScaleFit

    success, err = handler.performRequests_error_([request], None)
    if not success or err:
        return None

    observations = request.results()
    if not observations or len(observations) == 0:
        return None

    obs = observations[0]  # VNFeaturePrintObservation
    raw_data = obs.data()      # NSData: raw C array of floats
    element_count = obs.elementCount()
    element_type = obs.elementType()  # 1 = float32, 2 = float64

    import numpy as np
    buf = bytes(raw_data)
    dtype = np.float32 if element_type == 1 else np.float64
    vector = np.frombuffer(buf, dtype=dtype, count=element_count)
    return vector.tolist()  # plain list[float] for JSON serialisability
```

**Why NSData + NumPy instead of `computeDistanceTo_error_`**: The ObjC method requires a pointer-to-float out-parameter which is fragile in PyObjC. Extracting the raw NSData and computing L2 distance in NumPy is simpler, testable in isolation, and lets Phase 4 (Director) use the stored vectors directly for deduplication without re-running Vision.

### 4.5 Batching Requests per Handler

Both requests should be submitted in a **single** `performRequests_error_` call per image — the handler only decodes the image once, halving I/O and ImageIO overhead:

```python
success, err = handler.performRequests_error_([classify_req, fp_req], None)
```

### 4.6 Thread Safety

`VNImageRequestHandler` is **not thread-safe**. Each handler and its requests must be created, executed, and read on the **same thread**. Use `ThreadPoolExecutor` with each worker creating its own handler per image:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def analyse_images_parallel(
    image_paths: list[Path],
    workers: int,
) -> dict[Path, dict]:
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyse_single_image, p): p
            for p in image_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            results[path] = future.result()
    return results
```

The GIL is released during `performRequests_error_` because PyObjC drops the GIL when executing native ObjC calls, so threads provide real concurrency for Vision work.

---

## 5. PySceneDetect Integration

### 5.1 API Usage (programmatic, not CLI subprocess)

Use PySceneDetect's Python API directly — do not shell out to the `scenedetect` CLI binary.

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path: Path, threshold: float) -> list[dict]:
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video=video, show_progress=False, downscale=4)

    scene_list = scene_manager.get_scene_list()
    results = []
    for i, (start_tc, end_tc) in enumerate(scene_list):
        results.append({
            "scene_index": i,
            "start_frame": start_tc.get_frames(),
            "end_frame": end_tc.get_frames(),
            "start_time_s": start_tc.get_seconds(),
            "end_time_s": end_tc.get_seconds(),
            "duration_s": end_tc.get_seconds() - start_tc.get_seconds(),
        })
    return results
```

### 5.2 ContentDetector Threshold Tuning

`ContentDetector` computes the sum of pixel-level differences (HSL) between consecutive frames.

- **Lower (e.g., 15)**: Detects soft transitions, can oversplit a take with inconsistent lighting.
- **Higher (e.g., 40)**: Only hard cuts. Recommended for iPhone footage with heavy auto-exposure correction.
- **Default 27.0**: Good general starting point. Expose as `--scene-threshold` for per-project tuning.

### 5.3 Minimum Scene Duration and Edge Cases

Filter out scenes shorter than 0.5 s (flicker artefacts):

```python
MIN_SCENE_DURATION_S = 0.5
results = [s for s in results if s["duration_s"] >= MIN_SCENE_DURATION_S]
```

If `scene_list` is empty (no cuts detected), treat the entire file as one scene: `start_time_s=0.0`, `end_time_s=video.duration.get_seconds()`.

### 5.4 Downscale for Performance

Pass `downscale=4` to `detect_scenes()` so ContentDetector operates at 1/4 resolution for 4K/6K sources. Scene cut detection does not require full resolution.

---

## 6. FFmpeg Commands

FFmpeg is a **system dependency** installed via `brew install ffmpeg`. The scout script checks for `ffmpeg` at startup:

```python
def _require_ffmpeg() -> Path:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")
    return Path(ffmpeg_bin)
```

All FFmpeg calls use `subprocess.run(..., check=True, capture_output=True)`. On failure, the error is logged and the scene is marked `previews_generated: false` rather than aborting the run.

### 6.1 Extract Three JPEG Frames (Start, Mid, End)

```
ffmpeg -ss {time_s} -i {video_path} -vframes 1 -q:v 3 -vf scale=640:-1 {output_path} -y
```

- `-ss` before `-i` (input-side seeking): avoids decoding the entire video before the target timestamp.
- `-q:v 3`: ~85% quality JPEG.
- `scale=640:-1`: 640 px wide, aspect ratio preserved.

Timestamps per frame:
- `start`: `scene["start_time_s"] + 0.1` (avoid potential black frame at exact cut)
- `mid`: `(scene["start_time_s"] + scene["end_time_s"]) / 2.0`
- `end`: `scene["end_time_s"] - 0.1`

Output naming: `{scene_id}_start.jpg`, `{scene_id}_mid.jpg`, `{scene_id}_end.jpg`.

### 6.2 Micro-Preview GIF

1-second, low-resolution GIF centred on the scene's mid-point, for hover playback in the HTML review:

```
ffmpeg -ss {mid - 0.5} -t 1.0 -i {video_path}
  -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
  -loop 0 {output_path} -y
```

- `fps=10`: 10 frames → ~150–400 KB at 320 px wide.
- `palettegen` + `paletteuse` via `split`: single-command two-pass palette generation for high-quality GIF colour (avoids banding).
- `-loop 0`: loop indefinitely.

### 6.3 Photo Thumbnails (HEIC/JPEG/PNG)

Use Pillow (not FFmpeg) for still photos:

```python
from PIL import Image

def make_photo_thumbnail(photo_path: Path, output_path: Path, width: int = 640) -> Path:
    with Image.open(photo_path) as img:
        img = img.convert("RGB")  # normalise HEIC/CMYK
        ratio = width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((width, new_height), Image.LANCZOS)
        img.save(output_path, "JPEG", quality=85)
    return output_path
```

Pillow on macOS 11+ supports HEIC via the system ImageIO codec.

---

## 7. EXIF Reading for Burst Detection

### 7.1 Library: `piexif`

`piexif` is chosen over alternatives:
- Pure Python, no native dependencies.
- Structured access to `DateTimeOriginal`, `SubSecTimeOriginal`, GPS tags, `Make`, `Model`, `Orientation`.
- Lighter than `exifread` and simpler than `pyexiv2` (which requires libexiv2 native bindings).

### 7.2 Timestamp Extraction

```python
import piexif
from datetime import datetime

def read_exif_metadata(photo_path: Path) -> dict:
    result = {
        "datetime_original": None,
        "subsec_time": None,
        "gps_lat": None, "gps_lon": None,
        "make": "", "model": "",
        "orientation": 1,
    }
    try:
        exif_dict = piexif.load(str(photo_path))
    except Exception:
        return result  # EXIF unreadable; still include file in pipeline

    exif_ifd = exif_dict.get("Exif", {})
    zeroth_ifd = exif_dict.get("0th", {})
    gps_ifd = exif_dict.get("GPS", {})

    raw_dt = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"")
    if raw_dt:
        try:
            result["datetime_original"] = datetime.strptime(
                raw_dt.decode(), "%Y:%m:%d %H:%M:%S"
            )
        except (ValueError, UnicodeDecodeError):
            pass

    raw_subsec = exif_ifd.get(piexif.ExifIFD.SubSecTimeOriginal, b"")
    if raw_subsec:
        result["subsec_time"] = raw_subsec.decode().strip()

    result["make"] = zeroth_ifd.get(piexif.ImageIFD.Make, b"").decode(errors="replace").strip("\x00")
    result["model"] = zeroth_ifd.get(piexif.ImageIFD.Model, b"").decode(errors="replace").strip("\x00")
    result["orientation"] = zeroth_ifd.get(piexif.ImageIFD.Orientation, 1)

    if piexif.GPSIFD.GPSLatitude in gps_ifd:
        result["gps_lat"] = _gps_to_decimal(
            gps_ifd[piexif.GPSIFD.GPSLatitude],
            gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode()
        )
        result["gps_lon"] = _gps_to_decimal(
            gps_ifd[piexif.GPSIFD.GPSLongitude],
            gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode()
        )
    return result
```

### 7.3 Sub-Second Precision for Burst Sorting

iPhone burst shots can fire at ~10 fps. `DateTimeOriginal` only has 1-second resolution; `SubSecTimeOriginal` provides the fractional part (e.g., `"123"` = 0.123 s):

```python
def full_timestamp(exif: dict) -> float | None:
    dt = exif.get("datetime_original")
    if dt is None:
        return None
    ts = dt.timestamp()
    subsec = exif.get("subsec_time") or "0"
    frac = float("0." + subsec.lstrip("0") or "0") if subsec.strip("0") else 0.0
    return ts + frac
```

### 7.4 HEIC Fallback Chain

`piexif.load()` only reads JPEG/TIFF-embedded EXIF. For `.heic` files, fall back in order:
1. Try `piexif` directly.
2. If no `DateTimeOriginal`, try Pillow `img._getexif()` (triggers macOS ImageIO to extract EXIF from HEIC container).
3. If still no date, try `subprocess.run(["mdls", str(photo_path)])` to read `kMDItemContentCreationDate` from Spotlight metadata.

---

## 8. scenes.json Schema

Output file: `{output_dir}/scenes.json`. Top-level keys: `metadata` and `scenes`.

### 8.1 Top-Level Structure

```json
{
  "metadata": {
    "scout_version": "1.0.0",
    "generated_at": "2026-03-24T14:32:00Z",
    "input_dir": "/Users/michael/Videos/trip2026",
    "output_dir": "/Users/michael/Videos/trip2026/scout_output",
    "total_scenes": 142,
    "total_videos": 12,
    "total_photos": 87,
    "total_bursts": 5,
    "scene_threshold": 27.0,
    "burst_window_s": 3.0,
    "burst_min_count": 3,
    "similarity_threshold": 0.15
  },
  "scenes": [ ... ]
}
```

### 8.2 Video Scene Record

```json
{
  "scene_id": "S00001",
  "type": "video_scene",
  "source_file": "/abs/path/to/clip.mov",
  "source_file_rel": "clips/clip.mov",
  "scene_index": 0,
  "start_frame": 0,
  "end_frame": 423,
  "start_time_s": 0.0,
  "end_time_s": 14.1,
  "duration_s": 14.1,
  "fps": 30.0,
  "resolution": [1920, 1080],
  "previews": {
    "start": "previews/S00001_start.jpg",
    "mid": "previews/S00001_mid.jpg",
    "end": "previews/S00001_end.jpg",
    "preview_gif": "previews/S00001_preview.gif"
  },
  "previews_generated": true,
  "vision": {
    "classifications": [
      {"label": "outdoor", "confidence": 0.94},
      {"label": "sky", "confidence": 0.87}
    ],
    "category": "Nature",
    "feature_print": [0.123, -0.456, 0.789, "..."],
    "feature_print_frame": "mid"
  },
  "errors": [],
  "burst_group_id": null
}
```

All paths in `previews` are **relative to `output_dir`** so the HTML and JSON are portable.

### 8.3 Photo Record

```json
{
  "scene_id": "S00087",
  "type": "photo",
  "source_file": "/abs/path/to/IMG_1234.heic",
  "source_file_rel": "photos/IMG_1234.heic",
  "scene_index": null,
  "start_frame": null, "end_frame": null,
  "start_time_s": null, "end_time_s": null, "duration_s": null,
  "fps": null,
  "resolution": [4032, 3024],
  "previews": {
    "start": "previews/S00087_thumb.jpg",
    "mid": null, "end": null, "preview_gif": null
  },
  "previews_generated": true,
  "vision": {
    "classifications": [{"label": "food", "confidence": 0.91}],
    "category": "Food",
    "feature_print": [0.012, 0.345, -0.678, "..."],
    "feature_print_frame": "full"
  },
  "exif": {
    "datetime_original": "2026-03-20T12:34:56",
    "subsec_time": "123",
    "full_timestamp_unix": 1742474096.123,
    "gps_lat": 48.8584,
    "gps_lon": 2.2945,
    "make": "Apple",
    "model": "iPhone 16 Pro",
    "orientation": 1
  },
  "errors": [],
  "burst_group_id": null
}
```

### 8.4 Burst Group Record

```json
{
  "scene_id": "B00001",
  "type": "burst_group",
  "burst_type": "temporal",
  "member_scene_ids": ["S00088", "S00089", "S00090", "S00091"],
  "member_count": 4,
  "representative_scene_id": "S00089",
  "time_span_s": 1.2,
  "previews": {
    "start": "previews/S00088_thumb.jpg",
    "mid": "previews/S00089_thumb.jpg",
    "end": "previews/S00091_thumb.jpg",
    "preview_gif": null
  },
  "vision": {
    "category": "People",
    "feature_print": [0.111, 0.222, "..."]
  },
  "burst_group_id": "B00001"
}
```

- `burst_type`: `"temporal"` (time-based grouping) or `"visual"` (similarity-based).
- `representative_scene_id`: member with the highest average classification confidence — Phase 2 scores this first.
- Individual photo records that are burst members have their `burst_group_id` set to the burst's `scene_id`.

### 8.5 Field Type Reference

| Field | Type | Notes |
|---|---|---|
| `scene_id` | `string` | `S{05d}` for scenes, `B{05d}` for bursts |
| `type` | `"video_scene" \| "photo" \| "burst_group"` | |
| `source_file` | `string` | Absolute POSIX path |
| `source_file_rel` | `string` | Relative to `input_dir` |
| `start_time_s` | `number \| null` | Seconds, float |
| `end_time_s` | `number \| null` | Seconds, float |
| `duration_s` | `number \| null` | Seconds, float |
| `fps` | `number \| null` | Frames per second |
| `resolution` | `[width, height]` | Pixel dimensions |
| `previews` | `object` | Relative paths, null if not generated |
| `previews_generated` | `boolean` | False if FFmpeg/Pillow failed |
| `vision.classifications` | `array[{label, confidence}]` | Sorted descending |
| `vision.category` | `string` | `"Nature" \| "Urban" \| "Food" \| "People" \| "Other"` |
| `vision.feature_print` | `array[float] \| null` | Raw L2-normalisable vector (~2048 floats) |
| `vision.feature_print_frame` | `"start" \| "mid" \| "end" \| "full"` | Which frame was fingerprinted |
| `exif` | `object \| null` | Only present on `photo` records |
| `errors` | `array[{stage, message, detail}]` | Empty array if no errors |
| `burst_group_id` | `string \| null` | References a `burst_group` scene_id |

---

## 9. scout_review.html Structure and Interactivity

A single self-contained HTML file — no external CDN dependencies. All CSS and JS is inlined. Scene data is embedded as `const SCENES = [...]`.

### 9.1 Page Structure

```
<html>
  <head>
    <title>Scout Review — {input_dir_name}</title>
    <style> ... </style>
  </head>
  <body>
    <header>
      <h1>Scout Review</h1>
      <p class="meta">Generated: {datetime} · {n} scenes · {n} videos · {n} photos</p>
      <div class="controls">
        <button class="filter-btn active" data-category="All">All</button>
        <button class="filter-btn" data-category="Nature">Nature</button>
        <button class="filter-btn" data-category="Urban">Urban</button>
        <button class="filter-btn" data-category="Food">Food</button>
        <button class="filter-btn" data-category="People">People</button>
        <button class="filter-btn" data-category="Other">Other</button>
        <input type="search" id="search-box" placeholder="Search by file name...">
      </div>
    </header>
    <main id="scene-grid"> <!-- cards rendered by JS --> </main>
    <div id="lightbox" class="hidden"> ... </div>
    <script>
      const SCENES = [ /* JSON */ ];
      /* all interactivity */
    </script>
  </body>
</html>
```

### 9.2 Scene Card

```html
<article class="scene-card" data-id="S00001" data-category="Nature" data-type="video_scene">
  <div class="card-filmstrip">
    <img class="thumb" src="previews/S00001_start.jpg" loading="lazy">
    <img class="thumb" src="previews/S00001_mid.jpg"   loading="lazy">
    <img class="thumb" src="previews/S00001_end.jpg"   loading="lazy">
  </div>
  <div class="card-meta">
    <span class="scene-id">S00001</span>
    <span class="category-badge nature">Nature</span>
    <span class="duration">14.1s</span>
    <span class="filename" title="/abs/path/clip.mov">clip.mov</span>
  </div>
</article>
```

Burst group cards use `data-type="burst_group"` and a stacked-photo CSS treatment (`box-shadow` offset layers).

### 9.3 Hover Behaviour (GIF Preview)

On `mouseenter`, swap the filmstrip for the GIF preview. On `mouseleave`, restore the three-frame filmstrip:

```javascript
card.addEventListener('mouseenter', () => {
  if (scene.previews.preview_gif) {
    filmstrip.innerHTML = `<img class="gif-preview" src="${scene.previews.preview_gif}">`;
  }
});
card.addEventListener('mouseleave', () => {
  filmstrip.innerHTML = `
    <img class="thumb" src="${scene.previews.start}" loading="lazy">
    <img class="thumb" src="${scene.previews.mid}"   loading="lazy">
    <img class="thumb" src="${scene.previews.end}"   loading="lazy">
  `;
});
```

### 9.4 Category Filter

```javascript
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cat = btn.dataset.category;
    document.querySelectorAll('.scene-card').forEach(card => {
      card.style.display =
        (cat === 'All' || card.dataset.category === cat) ? '' : 'none';
    });
  });
});
```

### 9.5 Search Box

Case-insensitive substring match on `source_file_rel`:

```javascript
document.getElementById('search-box').addEventListener('input', function() {
  const term = this.value.toLowerCase();
  document.querySelectorAll('.scene-card').forEach(card => {
    const scene = SCENES.find(s => s.scene_id === card.dataset.id);
    card.style.display = scene.source_file_rel.toLowerCase().includes(term) ? '' : 'none';
  });
});
```

### 9.6 Lightbox

Clicking a card opens a lightbox showing:
- Full-size mid-frame image
- Metadata table: scene_id, file path, duration, fps, resolution, category, top-3 classification labels with confidence bars, burst membership
- Feature print sparkline: 64 sampled values drawn as a `<canvas>` bar chart

### 9.7 CSS Grid

```css
#scene-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
  padding: 16px;
}
.card-filmstrip { display: flex; gap: 2px; }
.card-filmstrip img { flex: 1; width: 33%; aspect-ratio: 16/9; object-fit: cover; }
.category-badge.nature  { background: #2d6a4f; color: #fff; }
.category-badge.urban   { background: #4a4e69; color: #fff; }
.category-badge.food    { background: #9c4221; color: #fff; }
.category-badge.people  { background: #1d3557; color: #fff; }
.category-badge.other   { background: #555;    color: #fff; }
```

---

## 10. Dependencies

### 10.1 Python Packages (PEP 723 inline metadata, installed by `uv run`)

| Package | Version | Purpose |
|---|---|---|
| `scenedetect[opencv]` | `>=0.6.7` | Video scene detection; `[opencv]` pulls `opencv-python-headless` |
| `pyobjc-core` | `>=12.1` | PyObjC runtime bridge |
| `pyobjc-framework-Vision` | `>=12.1` | VNClassifyImageRequest, VNGenerateImageFeaturePrintRequest |
| `pyobjc-framework-Quartz` | `>=12.1` | CGImage types used by Vision handler options |
| `pyobjc-framework-Foundation` | `>=12.1` | NSURL, NSData |
| `Pillow` | `>=12.1` | Photo thumbnails, HEIC fallback EXIF |
| `piexif` | `>=1.1.3` | EXIF metadata reading for burst detection |
| `numpy` | `>=2.2` | Feature print L2 distance computation |
| `click` | `>=8.3` | CLI argument parsing |
| `rich` | `>=14.3` | Progress bars, formatted console output |

### 10.2 System Dependencies

| Dependency | Install | Notes |
|---|---|---|
| FFmpeg | `brew install ffmpeg` | Frame extraction and GIF generation |
| macOS 13 Ventura or later | — | Required for VNClassifyImageRequest v2 and HEIC ImageIO support |
| Apple Silicon (M-series) | — | ANE-accelerated Vision framework; Intel Macs have Vision but slower |
| Xcode Command Line Tools | `xcode-select --install` | Required for PyObjC native extension compilation |

### 10.3 Python Version

`>=3.12` required for `type | None` union syntax in annotations.

---

## 11. Error Handling Strategy

### 11.1 Principles

- **Never abort the run for a single file failure.** Log the error, mark the scene record with an error field, continue.
- **Fail fast for system prerequisites.** Missing FFmpeg or wrong macOS version → exit before processing anything.
- **Capture all Vision errors as soft failures.** Missing `feature_print` → `feature_print: null`. Phase 4 must handle null gracefully.

### 11.2 Error Fields in scenes.json

```json
"errors": [
  {"stage": "extract_frames", "message": "ffmpeg exited with code 1", "detail": "...stderr..."},
  {"stage": "classify_scene", "message": "VNClassifyImageRequest failed: domain=VNError code=9"}
]
```

### 11.3 Exception Taxonomy

```python
class ScoutError(Exception): ...
class MediaReadError(ScoutError): ...
class VisionError(ScoutError): ...
class FFmpegError(ScoutError): ...
class ExifError(ScoutError): ...
```

All pipeline stage functions are wrapped by a `safe_run()` decorator that catches exceptions, returns an `_error` dict, and allows the caller to append to `errors[]` rather than crash.

### 11.4 Startup Checks (`preflight_checks`)

Run once before any processing:
1. FFmpeg presence (`shutil.which("ffmpeg")`)
2. macOS version >= 13.0 (`platform.mac_ver()`)
3. PyObjC Vision import smoke test
4. Output directory writable (create, write test file, unlink)

---

## 12. Performance Considerations

### 12.1 Batch Vision Requests

Submit both `VNClassifyImageRequest` and `VNGenerateImageFeaturePrintRequest` in a single `performRequests_error_` call — the handler decodes the image once.

### 12.2 ThreadPoolExecutor vs. ProcessPoolExecutor

Use `ThreadPoolExecutor` (not `ProcessPool`) because:
1. Vision ObjC objects are not picklable.
2. The GIL is released during `performRequests_error_` (native ObjC call) and during `subprocess.run()` (I/O wait).
3. Default `--workers 4` is conservative; 8 workers is safe on M2/M3 systems.

### 12.3 Input-Side FFmpeg Seeking

Always place `-ss` before `-i`. For a 30-minute 4K `.mov`, input-side seeking takes ~0.1 s vs. ~45 s for output-side seeking.

### 12.4 Lazy Loading in HTML

All `<img>` tags use `loading="lazy"`. A 500-scene library generates ~1,500 thumbnails; without lazy loading the browser fetches all on page open.

### 12.5 Resume Mode

`--resume` reads the existing `scenes.json`, collects already-processed `source_file` paths, and skips those files. Only new or failed files are re-processed.

### 12.6 Feature Print Storage

A `VNFeaturePrintObservation` vector is typically 2,048 float32 values (~8 KB as JSON per scene). For 1,000 scenes this is ~8 MB in `scenes.json`. A future optimisation could write vectors to a separate `feature_prints.npy` (NumPy binary) with a reference path in `scenes.json`, but inline JSON is acceptable for Phase 1.

### 12.7 Rich Progress Display

Use `rich.progress.Progress` with `SpinnerColumn`, `BarColumn`, `TimeRemainingColumn`. Track:
1. Overall file-level progress (outer bar)
2. Current stage name in a status column (`[Detecting scenes]`, `[Extracting frames]`, `[Running Vision]`)
3. Error count as a red counter

---

## 13. Implementation Sequence

Recommended order for writing the code:

1. **Scaffolding**: PEP 723 header, `main()`, `preflight_checks()`, `discover_media()`. Verify `uv run scout.py --input /tmp/test` executes.
2. **PySceneDetect**: Implement `detect_scenes()`. Verify scene dicts for a test video.
3. **FFmpeg frame extraction**: Implement `extract_frames()` and `make_photo_thumbnail()`. Confirm JPEGs and GIFs appear in `previews/`.
4. **EXIF reading**: Implement `read_exif_metadata()` and `full_timestamp()` with HEIC fallback chain.
5. **Vision framework**: Implement `_make_request_handler()`, `classify_scene()`, `fingerprint_scene()`, parallel `analyse_images_parallel()`. Test with a single known image first.
6. **Burst detection**: Implement `detect_burst_groups()` (temporal + visual passes). Unit-test with synthetic timestamp data.
7. **scenes.json assembly**: Implement `build_scenes_list()` and `write_scenes_json()`. Verify schema.
8. **HTML generation**: Implement `generate_review_html()`. Verify filter buttons and GIF hover in Safari.
9. **Error handling pass**: Wrap all stage functions with `safe_run()`. Feed corrupt files and verify the run completes with errors in JSON, not crashes.
10. **Performance pass**: Add `--resume` logic, `--workers` threading, `downscale` to PySceneDetect. Benchmark against a 100-file library.
