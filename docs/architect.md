# Phase 5: The Architect — Implementation Plan

## Overview

`architect.py` reads `final_sequence.json` (Phase 4) and produces a `montage.fcpxml`
file that can be imported directly into Final Cut Pro or DaVinci Resolve. It
builds a magnetic-spine timeline referencing the original media files, attaches
the master mix as a connected audio clip, applies Ken Burns pans/zooms to photos
using saliency centroids, and organises clips into keyword collections ("AI Picks",
"High Smiles", "Action", "Memory Dump").

---

## 1. File Layout

```
v1z0/
├── architect.py               ← standalone PEP 723 script
└── <output_dir>/
    └── montage.fcpxml         ← Final Cut Pro / DaVinci Resolve project
```

---

## 2. CLI

```
uv run architect.py --sequence ./director_output/final_sequence.json [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--sequence` / `-s` | required | Path to `final_sequence.json` |
| `--audio` / `-a` | None | Path to `master_mix.mp3` (overrides beats inference) |
| `--beats` / `-b` | None | Path to `beat_grid.json` (to infer audio path) |
| `--output` / `-o` | same dir as `--sequence` | Output directory |
| `--project-name` | `"Holiday Montage"` | FCP project name |
| `--event-name` | `"AI Montage — YYYY-MM-DD"` | FCP event name |
| `--workers` | `4` | Parallel ffprobe workers |
| `--skip-audio` | flag | Don't attach audio track |

---

## 3. FCPXML 1.11 Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
  <resources>
    <format id="r1" name="FFVideoFormat1080p2997"
            frameDuration="1001/30000s" width="1920" height="1080"
            colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="r2" name="clip.mov" uid="UID"
           start="0s" duration="Xs"
           hasVideo="1" hasAudio="1" audioSources="1"
           audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///abs/path"/>
    </asset>
    <asset id="r3" name="photo.heic" uid="UID"
           start="0s" duration="0s" hasVideo="1" hasAudio="0">
      <media-rep kind="original-media" src="file:///abs/path"/>
    </asset>
    <asset id="r_audio" name="master_mix.mp3" uid="UID"
           start="0s" duration="Xs"
           hasVideo="0" hasAudio="1" audioSources="1"
           audioChannels="1" audioRate="44100">
      <media-rep kind="original-media" src="file:///abs/path"/>
    </asset>
  </resources>

  <library location="file:///output_dir/">
    <event name="AI Montage — 2025-12-25">
      <project name="Holiday Montage">
        <sequence format="r1" duration="Xs" tcStart="0s"
                  tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <!-- First spine element: audio connected as lane=-1 -->
            <asset-clip name="clip.mov" ref="r2" offset="0s"
                        duration="2s" start="2500/2500s" format="r1">
              <audio ref="r_audio" lane="-1" offset="0s"
                     duration="183750/2500s" start="0s" role="music.music-1"/>
              <keyword start="0s" duration="2s" value="AI Picks, Action"/>
            </asset-clip>

            <!-- Photo with Ken Burns -->
            <asset-clip name="photo.heic" ref="r3" offset="5000/2500s"
                        duration="5000/2500s" start="0s" format="r1">
              <!-- Ken Burns: mode="pan", values are % of frame height -->
              <adjust-crop mode="pan" enabled="1">
                <pan-rect left="0"     top="0"    right="0"     bottom="0"/>
                <pan-rect left="14.81" top="8.33" right="14.81" bottom="8.33"/>
              </adjust-crop>
              <keyword start="0s" duration="2s" value="High Smiles, AI Picks"/>
            </asset-clip>
          </spine>
        </sequence>
      </project>

      <!-- Keyword collections appear as bins in FCP -->
      <keyword-collection name="AI Picks"/>
      <keyword-collection name="High Smiles"/>
      <keyword-collection name="Action"/>
      <keyword-collection name="Memory Dump"/>
    </event>
  </library>
</fcpxml>
```

---

## 4. Time Representation

All time values are stored as FCPXML rational strings using **timebase 2500**:

```python
def rational(seconds: float) -> str:
    if seconds == 0:
        return "0s"
    TIMEBASE = 2500
    n = round(seconds * TIMEBASE)
    g = gcd(n, TIMEBASE)
    num, den = n // g, TIMEBASE // g
    return f"{num}s" if den == 1 else f"{num}/{den}s"
```

`2500` is chosen because it is divisible by 25 (PAL), 50 (50fps), 100 (1/100s precision)
and produces clean fractions for 2.0s, 3.0s, 4.0s slots. For 29.97fps content FCP
snaps to its own internal frame grid on import.

---

## 5. Media Probing (ffprobe)

All unique `source_file` paths are probed once in parallel with a `ThreadPoolExecutor`:

```
ffprobe -v quiet -print_format json -show_streams -show_format <path>
```

Fields extracted:

| Field | From |
|---|---|
| `duration_s` | `format.duration` |
| `width`, `height` | first video stream |
| `fps_num / fps_den` | `r_frame_rate` (e.g. `"30000/1001"`) |
| `has_video` / `has_audio` | stream `codec_type` |
| `audio_channels` | audio stream `channels` |
| `audio_rate` | audio stream `sample_rate` |

Photos (`hasVideo=1, duration=0s`) and audio (`hasVideo=0`) are handled separately.

**Format detection**: The `<format>` element in `<resources>` uses the first probed
video file's resolution and frame rate. Supported frame rates: 23.976, 24, 25, 29.97,
30, 60 fps. Falls back to 1080p 29.97 if no video found.

---

## 6. Asset ID Map

```
r1       → <format>
r2..rN   → one <asset> per unique source_file (video + photo)
r(N+1)   → <asset> for master_mix audio
```

`source_file → asset_id` dict built during resource pass, reused when building clips.

---

## 7. Ken Burns via `<adjust-crop mode="pan">` + `<pan-rect>` pairs

From the FCPXML DTD:
```
<!ELEMENT adjust-crop (crop-rect?, trim-rect?, (pan-rect, pan-rect)?)>
<!ATTLIST adjust-crop mode (trim | crop | pan) #REQUIRED>
<!ATTLIST adjust-crop enabled (0 | 1) "1">

<!ELEMENT pan-rect EMPTY>
<!ATTLIST pan-rect left   CDATA "0">
<!ATTLIST pan-rect top    CDATA "0">
<!ATTLIST pan-rect right  CDATA "0">
<!ATTLIST pan-rect bottom CDATA "0">
```

Ken Burns uses `mode="pan"`. Two `<pan-rect>` elements define start and end
crop windows. Values are **percentage of original frame height** for all four
sides.

Conversion from `ken_burns.end_rect` (normalised 0–1) for a `W × H` frame:

```python
left_pct   = end_rect.x         * W / H * 100   # x is relative to width → convert via W/H
top_pct    = end_rect.y                  * 100   # y is relative to height
right_pct  = (1 - end_rect.x - end_rect.w) * W / H * 100
bottom_pct = (1 - end_rect.y - end_rect.h)  * 100
```

`start_rect` is always the full frame → all values = 0. Resulting XML:

```xml
<adjust-crop mode="pan" enabled="1">
  <pan-rect left="0"     top="0"    right="0"     bottom="0"/>
  <pan-rect left="14.81" top="8.33" right="14.81" bottom="8.33"/>
</adjust-crop>
```

---

## 8. Keyword Strategy

| Keyword | Condition |
|---|---|
| `"AI Picks"` | `composite >= 0.70` |
| `"High Smiles"` | `smile >= 0.50` |
| `"Action"` | `pacing_zone == "chorus"` |
| `"Memory Dump"` | `slot_type == "memory_dump"` |

Applied as a `<keyword>` child of each `<asset-clip>`:
```xml
<keyword start="0s" duration="2s" value="AI Picks, Action"/>
```

At event level, one `<keyword-collection name="…"/>` per keyword used.

---

## 9. Audio Attachment

The master mix is attached to the **first spine element** as a connected audio clip:

```xml
<asset-clip name="first_clip.mov" ref="r2" ...>
  <audio ref="r_audio" lane="-1" offset="0s"
         duration="{total_mix_duration}" start="0s" role="music.music-1"/>
</asset-clip>
```

`lane="-1"` places it below the primary storyline (standard for music in FCP).

If `--skip-audio` is set or no audio asset exists, this sub-element is omitted.

---

## 10. XML Generation

Python's `xml.etree.ElementTree` builds the element tree. `xml.dom.minidom` is used
for pretty-printing (4-space indent). The final file is assembled as:

```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
{pretty-printed XML root}
```

`minidom.toprettyxml` emits its own `<?xml?>` declaration, which is stripped before
prepending the DOCTYPE-aware header.

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `click>=8.0` | CLI |
| `rich>=13.0` | Progress display |

System: `ffprobe` (Homebrew `ffmpeg`), macOS/Linux.
No additional Python packages needed — XML and UUID are stdlib.

---

## 12. Implementation Sequence

1. CLI scaffolding + `preflight_checks` (ffprobe check, output dir).
2. `probe_media(path, ffprobe_bin)` + parallel probing for all unique sources.
3. `detect_format_info(probe_cache)` → picks `<format>` from first video.
4. `build_resources(entries, audio_path, probe_cache, format_info)` → `ET.Element`.
5. `rational(seconds)` + `file_uri(path)` helpers.
6. `compute_crop_insets(ken_burns, width, height)` → `(l, r, t, b)`.
7. `get_keywords(entry)` → `list[str]`.
8. `build_clip_element(entry, ...)` → `ET.Element` (handles video + photo + audio attach).
9. `build_spine(all_entries, ...)` → walk entries, accumulate spine.
10. `build_fcpxml_string(...)` → full XML string with DOCTYPE.
11. Write `montage.fcpxml`.
