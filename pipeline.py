#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.0",
#   "rich>=13.0",
# ]
# ///
"""v1z0 Pipeline — run all phases end-to-end.

Phases:
  1  scout      — scan media, detect scenes, Vision analysis
  2  critic     — score scenes, find best windows
  2.5 identifier — cluster faces, annotate person_ids  (optional)
  3  maestro    — analyse music, build beat grid + mix
  4  director   — select clips, build timeline
  5  architect  — generate montage.fcpxml for Final Cut Pro

Usage:
    uv run pipeline.py \\
      --input  /path/to/media \\
      --output /path/to/work  \\
      --music  /path/to/track.mp3
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

console = Console()
SCRIPTS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(phase: str, args: list[str], verbose: bool) -> bool:
    """Run one phase script via uv run. Inherits stdout/stderr so rich output
    streams to the terminal. Returns True on success."""
    script = SCRIPTS_DIR / f"{phase}.py"
    cmd = ["uv", "run", str(script)] + args
    console.print(Rule(f"[bold cyan]Phase: {phase}[/]"))
    if verbose:
        console.print(f"[dim]$ {' '.join(str(a) for a in cmd)}[/]\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print(
            f"\n[bold red]✗ {phase} failed (exit code {result.returncode}). "
            "Pipeline stopped.[/]"
        )
        return False
    console.print(f"\n[bold green]✓ {phase} done[/]")
    return True


def _exists(path: Path, label: str) -> bool:
    if not path.exists():
        console.print(f"[red]Expected output not found:[/] {path}  ({label})")
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"max_content_width": 100})

# ── Required ──────────────────────────────────────────────────────────────
@click.option(
    "--input", "-i", "input_dir",
    required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source media directory (photos + videos).",
)
@click.option(
    "--output", "-o", "output_dir",
    required=True, type=click.Path(path_type=Path),
    help="Work directory. Sub-directories are created per phase.",
)
@click.option(
    "--music", "-m", "music_files",
    multiple=True, type=click.Path(exists=True, path_type=Path),
    help="Music file(s) for Maestro. Repeat for multiple tracks. "
         "Required unless --skip-maestro is set.",
)
@click.option(
    "--music-snippet-duration", "music_snippet_duration",
    default=None, type=float,
    help="Trim each track (except the first) to this many seconds at its peak-energy window.",
)

# ── Phase control ──────────────────────────────────────────────────────────
@click.option("--skip-scout",      is_flag=True, help="Skip Phase 1 (scout).")
@click.option("--skip-critic",     is_flag=True, help="Skip Phase 2 (critic).")
@click.option("--skip-identifier", is_flag=True, help="Skip Phase 2.5 (face identifier).")
@click.option("--skip-maestro",    is_flag=True, help="Skip Phase 3 (maestro / music).")
@click.option("--skip-director",   is_flag=True, help="Skip Phase 4 (director).")
@click.option("--skip-architect",  is_flag=True, help="Skip Phase 5 (architect).")
@click.option("--curate",          is_flag=True, help="Run curator after director, open review in browser, and pause before architect.")

# ── Scout options ──────────────────────────────────────────────────────────
@click.option(
    "--extensions", default="mov,mp4,heic,jpg,jpeg,png", show_default=True,
    help="Comma-separated media extensions to scan.",
)
@click.option(
    "--scene-threshold", default=27.0, show_default=True,
    help="PySceneDetect content threshold (0–100).",
)
@click.option(
    "--extract-snippets", "extract_snippets", is_flag=True,
    help="Extract short clips from long scenes (see --snippet-* options).",
)
@click.option("--snippet-duration",     default=5.0,  show_default=True, help="Snippet length (s).")
@click.option("--snippet-min-scene",    default=10.0, show_default=True, help="Min scene length to snippet (s).")
@click.option("--snippet-max-per-scene",default=3,    show_default=True, help="Max snippets per scene (0=unlimited).")
@click.option(
    "--normalise",
    type=click.Choice(["none", "landscape", "portrait"], case_sensitive=False),
    default="none", show_default=True,
    help="Pre-render media to a common aspect ratio.",
)
@click.option(
    "--normalise-mode",
    type=click.Choice(["pad", "crop"], case_sensitive=False),
    default="pad", show_default=True,
    help="'pad' adds black bars; 'crop' scales to fill.",
)

# ── Director options ───────────────────────────────────────────────────────
@click.option("--min-score",            default=0.3,  show_default=True, help="Minimum composite score for clip selection.")
@click.option("--min-clip-duration",    default=2.0,  show_default=True, help="Minimum clip duration in seconds. Shorter beat slots are dropped.")
@click.option("--people-boost",         default=0.0,  show_default=True, help="Score bonus for clips with detected faces.")
@click.option("--people-ratio",         default=0.9,  show_default=True, help="Max fraction of face clips in the timeline.")
@click.option("--max-clips-per-source", default=3,    show_default=True, help="Max clips per source file.")
@click.option("--photo-bridge-interval",default=8,    show_default=True, help="Insert a photo burst every N video slots (0=off).")
@click.option("--photo-max-duration",   default=0.5,  show_default=True,  help="Duration of each photo in a bridge burst (seconds).")
@click.option("--photo-burst-size",     default=3,    show_default=True,  help="Number of photos per bridge burst.")
@click.option("--memory-dump-count",    default=20,   show_default=True, help="Photos in the end-credits memory dump.")
@click.option("--memory-dump-duration", default=1.0,  show_default=True, help="Duration of each end-credits photo in seconds.")
@click.option("--no-repeat",            is_flag=True,                    help="Never show the same clip twice (skips slots rather than looping).")
@click.option("--seed",                 default=42,   show_default=True, help="Random seed.")
@click.option(
    "--overrides", "overrides_path",
    default=None, type=click.Path(exists=True, path_type=Path),
    help="curator_overrides.json — force-include / force-exclude snippets.",
)

# ── Architect options ──────────────────────────────────────────────────────
@click.option("--project-name", default="Holiday Montage", show_default=True, help="FCP project name.")
@click.option(
    "--orientation",
    type=click.Choice(["auto", "landscape", "portrait"], case_sensitive=False),
    default="auto", show_default=True,
    help="FCP sequence orientation.",
)
@click.option(
    "--title", "title_text", default=None,
    help="Title card text (e.g. 'Grindelwald 2026'). Triggers titler.py before architect.",
)
@click.option(
    "--subtitle", "subtitle_text", default="",
    help="Optional title card subtitle text.",
)
@click.option("--skip-audio", is_flag=True, help="Omit audio track from FCPXML.")

# ── General ────────────────────────────────────────────────────────────────
@click.option("--workers", default=4, show_default=True, help="Parallel workers (used by scout, critic, architect).")
@click.option("--verbose", "-v", is_flag=True, help="Pass --verbose to each phase.")
@click.option("--force", is_flag=True, help="Re-run all phases even if outputs already exist.")

def main(
    input_dir: Path,
    output_dir: Path,
    music_files: tuple[Path, ...],
    skip_scout: bool,
    skip_critic: bool,
    skip_identifier: bool,
    skip_maestro: bool,
    skip_director: bool,
    skip_architect: bool,
    curate: bool,
    extensions: str,
    scene_threshold: float,
    extract_snippets: bool,
    snippet_duration: float,
    snippet_min_scene: float,
    snippet_max_per_scene: int,
    normalise: str,
    normalise_mode: str,
    min_score: float,
    min_clip_duration: float,
    people_boost: float,
    people_ratio: float,
    max_clips_per_source: int,
    photo_bridge_interval: int,
    photo_max_duration: float,
    photo_burst_size: int,
    memory_dump_count: int,
    memory_dump_duration: float,
    seed: int,
    no_repeat: bool,
    overrides_path: Path | None,
    project_name: str,
    orientation: str,
    title_text: str | None,
    subtitle_text: str,
    skip_audio: bool,
    music_snippet_duration: float | None,
    workers: int,
    verbose: bool,
    force: bool,
) -> None:
    """v1z0 Pipeline — run all phases end-to-end."""

    if not skip_maestro and not music_files:
        raise click.UsageError(
            "--music is required unless --skip-maestro is set."
        )

    # Resolve output sub-directories
    scout_dir      = output_dir / "scout"
    critic_dir     = output_dir / "critic"
    maestro_dir    = output_dir / "maestro"
    director_dir   = output_dir / "director"
    architect_dir  = output_dir / "architect"

    scenes_json        = scout_dir    / "scenes.json"
    scored_snippets    = critic_dir   / "scored_snippets.json"
    beat_grid_json     = maestro_dir  / "beat_grid.json"
    final_sequence     = director_dir / "final_sequence.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-resume: skip phases whose outputs already exist ──────────────
    if not force:
        _auto_skipped: list[str] = []
        if not skip_scout and scenes_json.exists():
            skip_scout = True
            _auto_skipped.append("scout")
        if not skip_critic and scored_snippets.exists():
            skip_critic = True
            _auto_skipped.append("critic")
        if not skip_identifier and scored_snippets.exists():
            try:
                _snips = json.loads(scored_snippets.read_text(encoding="utf-8")).get("snippets", [])
                if _snips and "person_ids" in _snips[0]:
                    skip_identifier = True
                    _auto_skipped.append("identifier")
            except Exception:
                pass
        if not skip_maestro and beat_grid_json.exists():
            skip_maestro = True
            _auto_skipped.append("maestro")
        if not skip_director and final_sequence.exists():
            skip_director = True
            _auto_skipped.append("director")
        if _auto_skipped:
            console.print(
                f"[dim]Auto-resuming — skipping already-complete phases: "
                f"{', '.join(_auto_skipped)}[/]"
            )
            console.print("[dim](Use --force to re-run all phases)[/]\n")

    # ── Plan summary ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]v1z0 Pipeline[/]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Input",  str(input_dir))
    table.add_row("Output", str(output_dir))
    if music_files:
        table.add_row("Music", ", ".join(str(m) for m in music_files))
    phases = []
    if not skip_scout:      phases.append("1  scout")
    if not skip_critic:     phases.append("2  critic")
    if not skip_identifier: phases.append("2.5 identifier")
    if not skip_maestro:    phases.append("3  maestro")
    if not skip_director:   phases.append("4  director")
    if curate:              phases.append("4.5 curator ⏸")
    if title_text and not skip_architect: phases.append("4.6 title card")
    if not skip_architect:  phases.append("5  architect")
    table.add_row("Phases", " → ".join(phases))
    console.print(table)
    console.print()

    # ── Phase 1: Scout ────────────────────────────────────────────────────
    if not skip_scout:
        args = [
            "--input",  str(input_dir),
            "--output", str(scout_dir),
            "--extensions", extensions,
            "--scene-threshold", str(scene_threshold),
            "--workers", str(workers),
        ]
        if extract_snippets:
            args += [
                "--extract-snippets",
                "--snippet-duration",      str(snippet_duration),
                "--snippet-min-scene",     str(snippet_min_scene),
                "--snippet-max-per-scene", str(snippet_max_per_scene),
            ]
        if normalise != "none":
            args += [
                "--normalise",      normalise,
                "--normalise-mode", normalise_mode,
            ]
        if verbose:
            args.append("--verbose")
        if not _run("scout", args, verbose):
            sys.exit(1)

    if not skip_critic or not skip_director:
        if not _exists(scenes_json, "scout output"):
            sys.exit(1)

    # ── Phase 2: Critic ───────────────────────────────────────────────────
    if not skip_critic:
        args = [
            "--scenes",  str(scenes_json),
            "--output",  str(critic_dir),
            "--workers", str(workers),
        ]
        if verbose:
            args.append("--verbose")
        if not _run("critic", args, verbose):
            sys.exit(1)

    if not skip_identifier or not skip_director:
        if not _exists(scored_snippets, "critic output"):
            sys.exit(1)

    # ── Phase 2.5: Identifier ─────────────────────────────────────────────
    if not skip_identifier:
        args = [
            "--snippets", str(scored_snippets),
        ]
        if verbose:
            args.append("--verbose")
        if not _run("identifier", args, verbose):
            sys.exit(1)

    # ── Phase 3: Maestro ──────────────────────────────────────────────────
    if not skip_maestro:
        args = ["--output", str(maestro_dir)]
        for mf in music_files:
            args += ["--music", str(mf)]
        if music_snippet_duration is not None:
            args += ["--snippet-duration", str(music_snippet_duration)]
        if verbose:
            args.append("--verbose")
        if not _run("maestro", args, verbose):
            sys.exit(1)

    if not skip_director:
        if not _exists(beat_grid_json, "maestro output"):
            sys.exit(1)

    # ── Phase 4: Director ─────────────────────────────────────────────────
    if not skip_director:
        args = [
            "--snippets", str(scored_snippets),
            "--beats",    str(beat_grid_json),
            "--scenes",   str(scenes_json),
            "--output",   str(director_dir),
            "--min-score",             str(min_score),
            "--min-clip-duration",     str(min_clip_duration),
            "--people-boost",          str(people_boost),
            "--people-ratio",          str(people_ratio),
            "--max-clips-per-source",  str(max_clips_per_source),
            "--photo-bridge-interval",  str(photo_bridge_interval),
            "--memory-dump-count",      str(memory_dump_count),
            "--memory-dump-duration",   str(memory_dump_duration),
            "--seed",                   str(seed),
        ]
        args += ["--photo-max-duration", str(photo_max_duration),
                 "--photo-burst-size",   str(photo_burst_size)]
        if no_repeat:
            args.append("--no-repeat")
        if overrides_path:
            args += ["--overrides", str(overrides_path)]
        if verbose:
            args.append("--verbose")
        if not _run("director", args, verbose):
            sys.exit(1)

    if not skip_architect:
        if not _exists(final_sequence, "director output"):
            sys.exit(1)

    # ── Phase 4.5: Curator (optional) ─────────────────────────────────────
    if curate and not skip_architect:
        curator_dir = output_dir / "curator"
        args = [
            "--snippets", str(scored_snippets),
            "--sequence", str(final_sequence),
            "--output",   str(curator_dir),
        ]
        if not _run("curator", args, verbose):
            sys.exit(1)

        review_html = curator_dir / "curator_review.html"
        if review_html.exists():
            subprocess.run(["open", str(review_html)])

        console.print(
            "\n[bold yellow]Curator review opened in your browser.[/]\n"
            f"  Pin or exclude clips, then download [cyan]curator_overrides.json[/]\n"
            f"  and place it at:\n"
            f"  [cyan]{curator_dir / 'curator_overrides.json'}[/]\n"
        )
        click.confirm("  Continue to architect?", default=True, abort=True)

        auto_overrides = curator_dir / "curator_overrides.json"
        if auto_overrides.exists() and overrides_path is None:
            console.print(f"[dim]  Using overrides from {auto_overrides}[/]")
            overrides_path = auto_overrides

        # Re-run director with the new overrides
        if auto_overrides.exists():
            console.print("\n[bold]Re-running director with curator overrides …[/]")
            args = [
                "--snippets", str(scored_snippets),
                "--beats",    str(beat_grid_json),
                "--scenes",   str(scenes_json),
                "--output",   str(director_dir),
                "--min-score",             str(min_score),
                "--min-clip-duration",     str(min_clip_duration),
                "--people-boost",          str(people_boost),
                "--people-ratio",          str(people_ratio),
                "--max-clips-per-source",  str(max_clips_per_source),
                "--photo-bridge-interval",  str(photo_bridge_interval),
                "--memory-dump-count",      str(memory_dump_count),
                "--memory-dump-duration",   str(memory_dump_duration),
                "--seed",                   str(seed),
                "--overrides",              str(auto_overrides),
            ]
            args += ["--photo-max-duration", str(photo_max_duration),
                     "--photo-burst-size",   str(photo_burst_size)]
            if verbose:
                args.append("--verbose")
            if not _run("director", args, verbose):
                sys.exit(1)

    # ── Phase 5: Architect ────────────────────────────────────────────────
    if not skip_architect:
        args = [
            "--sequence",     str(final_sequence),
            "--beats",        str(beat_grid_json),
            "--output",       str(architect_dir),
            "--project-name", project_name,
            "--orientation",  orientation,
            "--workers",      str(workers),
        ]
        if title_text:
            args += ["--title", title_text]
        if subtitle_text:
            args += ["--subtitle", subtitle_text]
        if skip_audio:
            args.append("--skip-audio")
        if verbose:
            args.append("--verbose")
        if not _run("architect", args, verbose):
            sys.exit(1)

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Pipeline complete[/]")
    fcpxml = architect_dir / "montage.fcpxml"
    if fcpxml.exists():
        console.print(f"[bold]FCPXML:[/]  {fcpxml}")
        console.print(f"[dim]Open in Final Cut Pro:[/] open \"{fcpxml}\"")
    review = scout_dir / "scout_review.html"
    if review.exists():
        console.print(f"[bold]Scout review:[/] {review}")
    console.print()


if __name__ == "__main__":
    main()
