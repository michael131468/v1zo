# Phase 4: The Director — Implementation Plan

## Overview

`director.py` reads `scored_snippets.json` (Phase 2) and `beat_grid.json` (Phase 3),
applies constraint-solving and narrative logic to produce a beat-aligned video
timeline, and writes `final_sequence.json`. It optionally reads `scenes.json`
(Phase 1) to access visual feature prints for deduplication.

---

## 1. File Layout

```
v1z0/
├── director.py                 ← standalone PEP 723 script
└── <output_dir>/
    └── final_sequence.json     ← output
```

`director.py` defaults `--output` to the same directory that contains
`scored_snippets.json`.

---

## 2. CLI

```
uv run director.py --snippets ./critic_output/scored_snippets.json \
                   --beats    ./maestro_output/beat_grid.json     \
                   [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--snippets` / `-s` | required | Path to `scored_snippets.json` |
| `--beats` / `-b` | required | Path to `beat_grid.json` |
| `--scenes` | None | Path to `scenes.json` (enables feature-print dedup) |
| `--output` / `-o` | same dir as `--snippets` | Output directory |
| `--min-score` | `0.30` | Discard snippets below this composite score |
| `--photo-bridge-interval` | `8` | Insert one photo bridge every N video slots |
| `--memory-dump-count` | `20` | Max photos in end-credits Memory Dump |
| `--diversity-window` | `3` | No same `source_file` within last N slots |
| `--feature-distance` | `0.15` | Min L2 distance for visual diversity (needs `--scenes`) |
| `--seed` | `42` | Random seed for tie-breaking |

---

## 3. Pipeline

```
load_inputs()
    ├── load scored_snippets.json
    ├── load beat_grid.json
    └── load scenes.json (optional, for feature prints)

build_slot_grid()
    └── for each pacing_zone in beat_grid:
            determine cut_interval_beats (1/2/4)
            walk downbeat_times to emit slots

fill_video_slots()
    └── greedy assignment:
            sort candidates by composite desc
            for each slot: pick best candidate satisfying diversity constraints
            track used_snippets

insert_photo_bridges()
    └── every photo_bridge_interval slots: replace video slot with photo

assemble_memory_dump()
    └── remaining unused photos sorted by composite desc (up to memory_dump_count)
        annotate each with ken_burns params

annotate_ken_burns()
    └── for every photo slot: compute start/end rect from saliency_centroid

write_final_sequence_json()
```

---

## 4. Slot Grid Construction

The beat_grid `pacing_zones` list drives how many cuts occur per zone:

| Zone type | `cut_interval_beats` | Rationale |
|---|---|---|
| `chorus` | 1 | High energy → cut on every beat |
| `verse` | 2 | Medium energy → cut every 2 beats |
| `breakdown` | 4 | Low energy → hold shots for 4 beats |

For each zone `[zone_start_s, zone_end_s]`:

1. Find all downbeat times within `[zone_start_s, zone_end_s]`.
2. Walk those downbeats in steps of `cut_interval_beats`.
3. Each step `[beat_i, beat_i + cut_interval_beats_dur]` becomes one slot.
4. The last slot in the zone is clamped to `zone_end_s`.

If a zone has no downbeats (e.g. very short zone), emit one slot covering the
full zone.

Minimum slot duration: `0.5 s`. Sub-threshold slots are merged with the next.

---

## 5. Greedy Video Assignment

Candidates = non-discarded video snippets with `composite >= min_score`, sorted
by `composite` descending. Photos are excluded from this pool and handled
separately.

For each slot:

```
for candidate in sorted_candidates:
    if candidate already used: skip
    if source_file in last diversity_window slots: skip
    if feature_distance check enabled:
        if min L2 dist to last 5 fingerprints < feature_distance: skip
    assign candidate to slot
    record source_file in recency window
    record fingerprint in recent_fingerprints
    break
```

If no candidate passes all constraints, relax constraints in order:
1. Drop feature-print constraint.
2. Drop source-file diversity constraint.
3. Allow already-used snippets (loop).

Fallback: if the pool is exhausted, cycle through used snippets again (loop
mode). This ensures every slot always gets a clip.

---

## 6. Photo Bridge Insertion

After video assignment, every `photo_bridge_interval`-th video slot is replaced
with a photo. The displaced video snippet is returned to the pool.

Photo candidates = non-discarded photos sorted by `composite` descending.
Same diversity-window check applies (by source_file).

Photos carry `type: "photo_bridge"`.

---

## 7. Memory Dump Assembly

After main timeline is built, remaining unused photos (sorted by `composite`
desc, limited to `memory_dump_count`) form the `memory_dump` section. These are
rapid-fire slots placed sequentially after `timeline_end_s`:

- Duration: 2.0 s each (one beat at ~120 BPM)
- `type: "memory_dump"`
- `transition: "cut"`

---

## 8. Ken Burns Annotation

Applied to all photo and burst_group slots (including memory dump).

Given `saliency_centroid = [cx, cy]` (normalized 0–1 in the image):

```
zoom_factor = 1.20   # 20% zoom

# Start: full frame
start_rect = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

# End: zoomed in, centred on saliency point
half_w = 1.0 / zoom_factor / 2
half_h = 1.0 / zoom_factor / 2
end_x = clamp(cx - half_w, 0.0, 1.0 - 2*half_w)
end_y = clamp(cy - half_h, 0.0, 1.0 - 2*half_h)
end_rect = {"x": end_x, "y": end_y,
            "w": 1.0/zoom_factor, "h": 1.0/zoom_factor}
```

If `saliency_centroid` is `null` (not available), default centroid = `[0.5, 0.5]`.

---

## 9. Feature Print Deduplication (optional)

If `--scenes` is provided, load `scenes.json` and build a dict:
`{scene_id: feature_print_array}`.

`feature_print` in `scenes.json` is stored as a list of floats under
`"feature_print"`. L2 distance:

```python
dist = np.linalg.norm(fp_a - fp_b)
```

Keep the last 5 assigned clips' fingerprints in a rolling buffer. Skip a
candidate if `min(dist for fp in recent_fps) < feature_distance`.

---

## 10. final_sequence.json Schema

```json
{
  "metadata": {
    "director_version": "1.0.0",
    "generated_at": "...",
    "scored_snippets_json": "/abs/path",
    "beat_grid_json": "/abs/path",
    "scenes_json": "/abs/path or null",
    "output_dir": "/abs/path",
    "total_slots": 87,
    "total_duration_s": 183.5,
    "photo_bridge_count": 10,
    "memory_dump_count": 18,
    "looped_snippets": 3
  },
  "timeline": [
    {
      "slot_id": "SL00001",
      "slot_type": "video",
      "timeline_start_s": 0.0,
      "timeline_end_s": 2.0,
      "duration_s": 2.0,
      "snippet_id": "C00001",
      "scene_id": "S00001",
      "source_file": "/abs/path/to/clip.mov",
      "source_file_rel": "clips/clip.mov",
      "source_in_s": 2.5,
      "source_out_s": 4.5,
      "pacing_zone": "chorus",
      "beat_aligned": true,
      "transition": "cut",
      "scores": {
        "aesthetic": 0.85,
        "saliency_coverage": 0.72,
        "smile": 0.30,
        "composite": 0.78
      },
      "ken_burns": null,
      "errors": []
    },
    {
      "slot_id": "SL00009",
      "slot_type": "photo_bridge",
      "timeline_start_s": 16.0,
      "timeline_end_s": 18.0,
      "duration_s": 2.0,
      "snippet_id": "C00012",
      "scene_id": "S00015",
      "source_file": "/abs/path/to/photo.heic",
      "source_file_rel": "photos/photo.heic",
      "source_in_s": null,
      "source_out_s": null,
      "pacing_zone": "verse",
      "beat_aligned": true,
      "transition": "cut",
      "scores": {
        "aesthetic": 0.91,
        "saliency_coverage": 0.60,
        "smile": 0.80,
        "composite": 0.82
      },
      "ken_burns": {
        "start_rect": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "end_rect":   {"x": 0.08, "y": 0.12, "w": 0.83, "h": 0.83},
        "centroid":   [0.50, 0.45]
      },
      "errors": []
    }
  ],
  "memory_dump": [
    {
      "slot_id": "MD00001",
      "slot_type": "memory_dump",
      "timeline_start_s": 183.5,
      "timeline_end_s": 185.5,
      "duration_s": 2.0,
      "snippet_id": "C00099",
      "scene_id": "S00102",
      "source_file": "/abs/path/to/another.heic",
      "source_file_rel": "photos/another.heic",
      "source_in_s": null,
      "source_out_s": null,
      "pacing_zone": "memory_dump",
      "beat_aligned": false,
      "transition": "cut",
      "scores": {"aesthetic": 0.75, "saliency_coverage": 0.55, "smile": 0.20, "composite": 0.63},
      "ken_burns": {
        "start_rect": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "end_rect":   {"x": 0.04, "y": 0.04, "w": 0.83, "h": 0.83},
        "centroid":   [0.50, 0.50]
      },
      "errors": []
    }
  ]
}
```

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `numpy>=1.24` | Feature print L2 distance |
| `click>=8.0` | CLI |
| `rich>=13.0` | Progress display |

No system dependencies beyond Python 3.12+. FFmpeg and Vision framework are
not needed — all heavy lifting is done by Phase 1–3.

---

## 12. Implementation Sequence

1. CLI scaffolding: flags, `preflight_checks`, `load_inputs`.
2. `build_slot_grid`: pacing zone → slot list.
3. `fill_video_slots`: greedy assignment with diversity constraints.
4. `insert_photo_bridges`: swap every Nth slot.
5. `assemble_memory_dump`: remainder photos.
6. `annotate_ken_burns`: compute rects for photo slots.
7. `write_final_sequence_json`: assemble metadata + timeline + memory_dump.
