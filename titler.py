#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""Titler — Phase 0.5 of the macOS AI Montage Suite.

Generates a title-card video clip (MP4) with a background photo, Ken Burns
slow zoom, optional dark overlay, and centred text (title + subtitle).
The output can be passed to architect.py via --title-clip.

Usage:
    uv run titler.py \\
      --title "Grindelwald 2026" \\
      --subtitle "Family Holiday" \\
      --snippets /path/to/critic/scored_snippets.json \\
      --output /path/to/title_card.mp4
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import click
from rich.console import Console

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFCompact.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Futura.ttc",
    "/Library/Fonts/Georgia.ttf",
]


def find_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Photo selection
# ---------------------------------------------------------------------------

def pick_best_photo(snippets_path: Path) -> str | None:
    """Return the source_file of the highest-scoring non-discarded photo."""
    try:
        data = json.loads(snippets_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Cannot read snippets:[/] {exc}")
        return None

    photos = [
        s for s in data.get("snippets", [])
        if not s.get("discarded", False)
        and s.get("type") in ("photo", "burst_group")
    ]
    if not photos:
        return None

    photos.sort(
        key=lambda s: s.get("scores", {}).get("composite", 0.0),
        reverse=True,
    )
    return photos[0].get("source_file")


# ---------------------------------------------------------------------------
# Title card generation
# ---------------------------------------------------------------------------

_PHOTO_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".heic", ".png", ".tiff", ".tif", ".bmp", ".webp",
})


def _escape_drawtext(text: str) -> str:
    """Escape special characters for ffmpeg drawtext filter value."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("'",  "\\'")
        .replace(":",  "\\:")
        .replace("[",  "\\[")
        .replace("]",  "\\]")
    )


def generate_title_card(
    photo_path: Path,
    output_path: Path,
    title: str,
    subtitle: str,
    duration: float,
    width: int,
    height: int,
    fps: int,
    fade_duration: float,
    font_path: str | None,
    overlay_opacity: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")

    is_photo_input = photo_path.suffix.lower() in _PHOTO_EXTENSIONS
    fade_frames = round(fade_duration * fps)
    fade_out_start = duration - fade_duration

    # --- Build the -vf filter chain ---
    filters: list[str] = []

    # 1. Scale + pad photo to target canvas
    filters.append(
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )

    # 2. Dark overlay for text legibility
    if overlay_opacity > 0:
        filters.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:"
            f"color=black@{overlay_opacity:.2f}:t=fill"
        )

    # 4. Title text
    title_esc = _escape_drawtext(title)
    font_arg  = f"fontfile={font_path}:" if font_path else ""
    filters.append(
        f"drawtext={font_arg}"
        f"text='{title_esc}':"
        f"fontsize={round(height * 0.075)}:"
        f"fontcolor=white:"
        f"x=(w-text_w)/2:"
        f"y=(h-text_h)/2{'-' + str(round(height * 0.065)) if subtitle else ''}:"
        f"shadowcolor=black@0.7:shadowx=3:shadowy=3"
    )

    # 5. Subtitle text (if provided)
    if subtitle:
        sub_esc = _escape_drawtext(subtitle)
        filters.append(
            f"drawtext={font_arg}"
            f"text='{sub_esc}':"
            f"fontsize={round(height * 0.038)}:"
            f"fontcolor=white@0.85:"
            f"x=(w-text_w)/2:"
            f"y=(h-text_h)/2+{round(height * 0.065)}:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2"
        )

    # 6. Fade in / fade out
    filters.append(f"fade=in:st=0:d={fade_duration}")
    filters.append(f"fade=out:st={fade_out_start:.4f}:d={fade_duration}")

    vf = ",".join(filters)

    # --- Build ffmpeg command ---
    if is_photo_input:
        cmd = [
            ffmpeg, "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", str(photo_path),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        # Video input — seek to mid-point and grab duration seconds
        cmd = [
            ffmpeg, "-y",
            "-i", str(photo_path),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"  Generating title card → {output_path.name} …")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-400:]
        raise SystemExit(f"ffmpeg failed:\n{err}")

    console.print(f"  [bold green]✓[/] {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--title", "-t", required=True,
    help="Main title text (e.g. 'Grindelwald 2026').",
)
@click.option(
    "--subtitle", "-s", default="",
    help="Optional subtitle text (e.g. 'Family Holiday').",
)
@click.option(
    "--photo", "photo_path",
    default=None, type=click.Path(path_type=Path),
    help="Path to the background photo or video frame. "
         "If omitted, the best-scoring photo from --snippets is used.",
)
@click.option(
    "--snippets", "snippets_path",
    default=None, type=click.Path(exists=True, path_type=Path),
    help="Path to scored_snippets.json — used to auto-select the best photo "
         "when --photo is not given.",
)
@click.option(
    "--output", "-o", "output_path",
    default=None, type=click.Path(path_type=Path),
    help="Output MP4 path. Defaults to title_card.mp4 next to --snippets "
         "(or current directory).",
)
@click.option(
    "--duration", default=5.0, show_default=True,
    help="Title card duration in seconds.",
)
@click.option(
    "--orientation",
    type=click.Choice(["landscape", "portrait"], case_sensitive=False),
    default="landscape", show_default=True,
    help="Output resolution: landscape=1920×1080, portrait=1080×1920.",
)
@click.option(
    "--fps", default=30, show_default=True,
    help="Output frame rate.",
)
@click.option(
    "--fade", "fade_duration", default=0.75, show_default=True,
    help="Fade-in and fade-out duration in seconds.",
)
@click.option(
    "--overlay-opacity", default=0.35, show_default=True,
    help="Dark overlay opacity (0=none, 1=fully black). "
         "Makes text easier to read on bright photos.",
)
def main(
    title: str,
    subtitle: str,
    photo_path: Path | None,
    snippets_path: Path | None,
    output_path: Path | None,
    duration: float,
    orientation: str,
    fps: int,
    fade_duration: float,
    overlay_opacity: float,
) -> None:
    """Titler — generate a title-card MP4 with background photo and text overlay."""

    console.print("[bold]Titler[/] — Phase 0.5 of the macOS AI Montage Suite\n")

    # Resolve photo
    if photo_path is None:
        if snippets_path is None:
            raise click.UsageError("Either --photo or --snippets must be provided.")
        console.print(f"  Auto-selecting best photo from {snippets_path.name} …")
        best = pick_best_photo(snippets_path)
        if best is None:
            raise SystemExit("No suitable photos found in scored_snippets.json.")
        photo_path = Path(best)
        console.print(f"  Selected: {photo_path.name}")
    elif not photo_path.exists():
        raise SystemExit(f"Photo not found: {photo_path}")

    # Resolve output
    if output_path is None:
        base = snippets_path.parent if snippets_path else Path(".")
        output_path = base / "title_card.mp4"

    # Resolve resolution
    if orientation == "landscape":
        width, height = 1920, 1080
    else:
        width, height = 1080, 1920

    # Find font
    font_path = find_font()
    if font_path:
        console.print(f"  Font: {Path(font_path).name}")
    else:
        console.print("[yellow]  No system font found — ffmpeg will use its built-in font.[/]")

    console.print(f"  Title:    {title!r}")
    if subtitle:
        console.print(f"  Subtitle: {subtitle!r}")
    console.print(f"  Canvas:   {width}×{height} @ {fps}fps  {duration}s")

    generate_title_card(
        photo_path=photo_path,
        output_path=output_path,
        title=title,
        subtitle=subtitle,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        fade_duration=fade_duration,
        font_path=font_path,
        overlay_opacity=overlay_opacity,
    )


if __name__ == "__main__":
    main()
