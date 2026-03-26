🏗️ macOS AI Montage Suite: Architecture Spec

Suite Philosophy

- Modular: Each phase is a standalone Python script that passes data via
JSON.

- Mac-Native: High-performance analysis using Apple Silicon (M-series)
via the Vision framework.

- Non-Destructive: The suite generates an FCPXML project file; it does
not "bake" the video, allowing for manual fine-tuning in Final Cut Pro
or DaVinci Resolve.

Phase 1: The Scout (scout.py)

Primary Goal: Media Ingest & Categorization.

- Scene Detection: Uses PySceneDetect to split long recordings into individual camera takes.

- Semantic Tagging: Uses VNClassifyImageRequest to identify if a clip is "Nature," "Urban," "Food," or "People."

- Visual Fingerprinting: Generates a VNGenerateImageFeaturePrintRequest (vector) for every scene to identify visually similar content early.

- Burst & Sequence Detection: * Temporal Grouping: Scans photo metadata (Exif) for clusters taken within a short window (e.g., 5 photos in 3 seconds).
  - Visual Similarity Grouping: Uses VNGenerateImageFeaturePrintRequest to identify photos that are visually nearly identical, even if the timestamps are slightly apart.
  - Action: Groups these into a "Burst Scene" rather than individual photos.
    
- The Reviewer: Generates a scout_report.html storyboard, grouping your media by AI-detected categories for quick human oversight.

🛠️ Technical Stack

    Detection: PySceneDetect (Content-aware thresholding).

    Extraction: FFmpeg for high-speed frame grabbing.

    Review UI: A local scout_review.html generated automatically.

📋 Functionality

    Directory Crawl: Scan for .mov, .mp4, .heic (iPhone video).

    Scene Analysis: Detect hard cuts.

    Visual Review (The "Contact Sheet"):

        Extract 3 frames per scene (Start, Middle, End) using FFmpeg.

        Generate a Storyboard HTML: A responsive grid where you can hover over a scene to see its metadata and play a "micro-preview" (a 1s low-res gif/clip).

    Output: scenes.json (List of file paths, timestamps, and paths to preview thumbnails).
    
Phase 2: The Critic (critic.py)

Primary Goal: Quality Ranking & Snippet Extraction.

    Sliding Window Logic: Instead of scoring a whole 15s clip, it scans the clip in 3-second windows (stepping every 0.5s) to find the "peak" moment.

    Aesthetic Scoring: Uses VNGenerateImageAestheticsScoresRequest to rank lighting, composition, and "pro-look."

	Saliency: VNGenerateSaliencyImageRequest (Detects the subject of interest).

    Expression Detection: Uses VNDetectFaceExpressionsRequest to find windows with high "smile" probability.

    Technical Filters: Discards windows with high blur (Laplacian variance) or extreme camera tilt (Horizon detection).
    
    Horizon/Stability: VNDetectHorizonRequest to filter out badly tilted or extremely shaky handheld shots.

Output: scored_snippets.json.

Phase 3: The Maestro (maestro.py)

Primary Goal: Audio Foundation & Pacing Map.

    Beat Tracking: Uses Librosa to map every beat and "downbeat" (the 1 of a 4-beat measure).

    Audio Energy: Identifies "Crescendos" (high volume/energy) and "Breakdowns" (quiet moments).

    Smart Mixing: Crossfades multiple songs using a BPM Bridge (subtle time-stretching) so the beat remains consistent during transitions.
    
Analysis: Librosa to find the BPM and "Downbeats."

Mixer: PyDub to merge tracks with Constant Power Crossfades.

Output: master_mix.mp3 + beat_grid.json.
    
Phase 4: The Director (director.py)

Primary Goal: The Constraint Solver & Narrative Logic.

    The Draft: Matches "High Aesthetic" snippets to beats. High-energy video is mapped to audio crescendos.

    Deduplication: Uses the "Visual Fingerprints" from Phase 1 to ensure we don't play three similar beach shots in a row.

    Photo Strategy: * Intermixed: Photos used as narrative bridges.

        Credits Flow: All unused high-quality photos are bundled into a rapid-fire "Memory Dump" at the end.

    Pacing Engine: Automatically chooses between cutting on every beat (high energy) or every 4th beat (scenic/chill).
    
Selection: Matches "High Aesthetic" + "Smile" snippets to the music's chorus.

Variety Logic: Prevents visual repetition from the same source file.

Pacing: Swaps clips on the "1" and "3" beats of every measure.

Output: final_sequence.json.
    
Phase 5: The Architect (architect.py)

Primary Goal: NLE Project Generation.

    FCPXML Generation: Builds a .fcpxml file that references your original media.

    Magnetic Spine: Uses the FCP "Spine" so the clips stay perfectly synced to the music.

    Procedural Animation: Automatically applies Ken Burns pans/zooms to photos, centering the animation on "Saliency" points (like faces).

    Smart Collections: Organizes the FCP project into folders like "AI Picks," "Action," and "High Smiles" based on the metadata.
    
