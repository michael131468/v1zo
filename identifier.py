#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.0",
#   "rich>=13.0",
#   "deepface>=0.0.93",
#   "tf-keras>=2.16",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
# ]
# ///
"""Phase 2.5 of the macOS AI Montage Suite — The Identifier.

Reads scored_snippets.json (Phase 2), extracts a representative frame for
every snippet that contains faces, clusters the 128-d face encodings with
DBSCAN, and annotates each snippet with a person_ids list.

Usage:
    uv run identifier.py --snippets ./critic_output/scored_snippets.json [OPTIONS]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
import numpy as np
from deepface import DeepFace
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
from sklearn.cluster import DBSCAN

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(
    source_file: str,
    time_s: float,
    out_path: Path,
) -> bool:
    """Extract a single JPEG frame at time_s using ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{time_s:.4f}",
        "-i", source_file,
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Face encoding
# ---------------------------------------------------------------------------

def get_largest_face_encoding(
    image_path: Path,
    model_name: str = "Facenet",
) -> np.ndarray | None:
    """Return the face embedding for the largest face in the image, or None.

    Uses DeepFace with Facenet embeddings and opencv detector.
    Picks the largest face by bounding-box area when multiple faces are detected.
    """
    try:
        results = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            enforce_detection=True,
            detector_backend="opencv",
        )
        if not results:
            return None
        # Pick largest face by facial_area w*h
        largest = max(
            results,
            key=lambda r: r.get("facial_area", {}).get("w", 0)
                        * r.get("facial_area", {}).get("h", 0),
        )
        return np.array(largest["embedding"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-snippet extraction worker (used in ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _extract_worker(
    snippet: dict,
    tmp_dir: Path,
    idx: int,
) -> tuple[str, Path | None]:
    """Extract frame for one snippet. Returns (snippet_id, frame_path | None)."""
    snippet_id = snippet.get("snippet_id", f"idx_{idx}")
    source_file = snippet.get("source_file", "")
    bw = snippet.get("best_window") or {}
    time_s = bw.get("mid_time_s")

    if not source_file or time_s is None:
        return snippet_id, None

    out_path = tmp_dir / f"frame_{idx:06d}.jpg"
    ok = extract_frame(source_file, float(time_s), out_path)
    return snippet_id, (out_path if ok else None)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_identifier(
    snippets_path: Path,
    output_path: Path,
    eps: float,
    min_samples: int,
    workers: int,
    verbose: bool,
) -> None:
    # ------------------------------------------------------------------
    # Load scored_snippets.json
    # ------------------------------------------------------------------
    if not snippets_path.exists():
        raise SystemExit(f"File not found: {snippets_path}")

    data: dict[str, Any] = json.loads(snippets_path.read_text(encoding="utf-8"))
    snippets: list[dict] = data.get("snippets", [])
    console.print(f"[bold]Identifier[/] — Phase 2.5 of the macOS AI Montage Suite\n")
    console.print(f"  Loaded {len(snippets)} snippets from {snippets_path.name}")

    # Identify candidates: face_count > 0 and not discarded
    candidates: list[tuple[int, dict]] = [
        (i, s) for i, s in enumerate(snippets)
        if not s.get("discarded", False) and s.get("face_count", 0) > 0
    ]
    console.print(f"  Candidates with faces: {len(candidates)}")

    if not candidates:
        console.print("[yellow]No face candidates found — writing empty person_ids to all snippets.[/]")
        for s in snippets:
            s["person_ids"] = []
        _write_output(data, output_path)
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="identifier_"))
    try:
        # ------------------------------------------------------------------
        # Stage 1: Parallel frame extraction
        # ------------------------------------------------------------------
        frame_map: dict[str, Path | None] = {}  # snippet_id -> frame path

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Extracting frames…", total=len(candidates)
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_extract_worker, snippet, tmp_dir, idx): snippet_id
                    for idx, (_, snippet) in enumerate(candidates)
                    for snippet_id in [snippet.get("snippet_id", f"idx_{idx}")]
                }
                for future in as_completed(futures):
                    try:
                        sid, frame_path = future.result()
                        frame_map[sid] = frame_path
                    except Exception as exc:
                        sid = futures[future]
                        frame_map[sid] = None
                        if verbose:
                            console.print(f"  [red]Extraction error for {sid}:[/] {exc}")
                    progress.advance(task)

        # ------------------------------------------------------------------
        # Stage 2: Face encoding (sequential; face_recognition is not thread-safe)
        # ------------------------------------------------------------------
        encoding_map: dict[str, np.ndarray] = {}  # snippet_id -> 128-d encoding

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[green]Encoding faces…", total=len(candidates)
            )
            for _, snippet in candidates:
                sid = snippet.get("snippet_id", "")
                frame_path = frame_map.get(sid)
                if frame_path is not None and frame_path.exists():
                    encoding = get_largest_face_encoding(frame_path)
                    if encoding is not None:
                        encoding_map[sid] = encoding
                progress.advance(task)

        console.print(f"  Successfully encoded: {len(encoding_map)} faces")

        if not encoding_map:
            console.print("[yellow]No faces could be encoded — assigning empty person_ids.[/]")
            for s in snippets:
                s["person_ids"] = []
            _write_output(data, output_path)
            return

        # ------------------------------------------------------------------
        # Stage 3: DBSCAN clustering
        # ------------------------------------------------------------------
        ordered_ids = list(encoding_map.keys())
        X = np.array([encoding_map[sid] for sid in ordered_ids])

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
        labels: np.ndarray = db.labels_

        # Map numeric label -> person ID string
        unique_labels = sorted(set(labels))
        label_to_pid: dict[int, str] = {}
        person_counter = 1
        for lbl in unique_labels:
            if lbl == -1:
                label_to_pid[-1] = "unknown"
            else:
                label_to_pid[lbl] = f"P{person_counter:03d}"
                person_counter += 1

        # snippet_id -> list of person IDs
        sid_to_pids: dict[str, list[str]] = {}
        for sid, lbl in zip(ordered_ids, labels):
            sid_to_pids[sid] = [label_to_pid[lbl]]

        # ------------------------------------------------------------------
        # Stage 4: Annotate snippets
        # ------------------------------------------------------------------
        for snippet in snippets:
            sid = snippet.get("snippet_id", "")
            snippet["person_ids"] = sid_to_pids.get(sid, [])

        # ------------------------------------------------------------------
        # Stage 5: Write output
        # ------------------------------------------------------------------
        _write_output(data, output_path)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        n_people = sum(1 for lbl in unique_labels if lbl != -1)
        n_unknown = int(np.sum(labels == -1))

        console.print(f"\n[bold green]✓[/] Identified [cyan]{n_people}[/] unique people")
        if n_unknown:
            console.print(f"  [yellow]{n_unknown}[/] face(s) could not be clustered (unknown)")

        if verbose and n_people > 0:
            # Build clips-per-person table
            clips_per_person: dict[str, int] = {}
            for pids in sid_to_pids.values():
                for pid in pids:
                    clips_per_person[pid] = clips_per_person.get(pid, 0) + 1

            table = Table(title="People Found", show_header=True, header_style="bold")
            table.add_column("Person ID", style="cyan")
            table.add_column("Clips", justify="right")

            for pid in sorted(clips_per_person):
                table.add_row(pid, str(clips_per_person[pid]))
            if n_unknown:
                table.add_row("unknown", str(n_unknown))

            console.print(table)

        console.print(f"[bold]Done.[/] Wrote {output_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_output(data: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    "--output", "-o", "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path. Defaults to overwriting --snippets in-place.",
)
@click.option(
    "--eps",
    default=0.5, show_default=True,
    help="DBSCAN epsilon distance. Clips within this euclidean distance "
         "in face-encoding space are considered the same person.",
)
@click.option(
    "--min-samples",
    default=1, show_default=True,
    help="DBSCAN min_samples. Default 1 means every face gets an ID.",
)
@click.option(
    "--workers",
    default=4, show_default=True,
    help="Parallel ffmpeg frame-extraction workers.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Print per-person clip count table.",
)
def main(
    snippets_path: Path,
    output_path: Path | None,
    eps: float,
    min_samples: int,
    workers: int,
    verbose: bool,
) -> None:
    """Identifier — Phase 2.5 of the macOS AI Montage Suite.

    Clusters face encodings across all scored snippets and annotates each
    snippet with a person_ids list (e.g. ["P001"]).  Snippets with no
    detected faces receive an empty person_ids list.
    """
    if output_path is None:
        output_path = snippets_path

    run_identifier(
        snippets_path=snippets_path,
        output_path=output_path,
        eps=eps,
        min_samples=min_samples,
        workers=workers,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
