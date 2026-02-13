"""Video rendering module for video montage skill.

Renders a watchable MP4 preview using FFmpeg.
This complements the FCPXML output by providing an immediate preview
while the editable project file is still available for fine-tuning.

Uses FFmpeg filters:
- concat demuxer for clip assembly
- xfade for crossfade transitions  
- amix/overlay for audio mixing
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class VideoRenderError(RuntimeError):
    """Raised when video rendering fails."""


_FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")


@dataclass(frozen=True)
class RenderSpec:
    """Specification for rendering a video montage.
    
    Attributes:
        clips: List of clip segments with timing info
        music_path: Path to background music file
        output_path: Where to save the rendered MP4
        width: Output width (default 1920)
        height: Output height (default 1080)
        fps: Output frame rate (default 30)
        transition_duration: Crossfade duration in seconds (default 0.5)
        music_volume: Music volume multiplier (default 0.3 = 30%)
    """
    clips: list  # List of ClipSegment from timeline_builder
    music_path: Path
    output_path: Path
    width: int = 1920
    height: int = 1080
    fps: int = 30
    transition_duration: float = 0.5
    music_volume: float = 0.3


def _build_ffmpeg_complex_filter(spec: RenderSpec) -> str:
    """Build FFmpeg complex filter graph for clip assembly with transitions.
    
    Creates a filtergraph that:
    1. Loads all clips
    2. Trims each to the specified segment
    3. Applies xfade transitions between clips
    4. Mixes in background music
    
    Args:
        spec: Render specification
        
    Returns:
        FFmpeg complex filter string
    """
    if len(spec.clips) == 0:
        raise VideoRenderError("No clips to render")
    
    if len(spec.clips) == 1:
        # Single clip - simple trim, no transitions
        clip = spec.clips[0]
        return (
            f"[0:v]trim=start={clip.source_start}:duration={clip.source_duration},"
            f"setpts=PTS-STARTPTS[vout];"
            f"[0:a]atrim=start={clip.source_start}:duration={clip.source_duration},"
            f"asetpts=PTS-STARTPTS[aout]"
        )
    
    # Multiple clips - build xfade chain
    filter_parts = []
    
    # First, trim all clips
    for i, clip in enumerate(spec.clips):
        filter_parts.append(
            f"[{i}:v]trim=start={clip.source_start}:duration={clip.source_duration},"
            f"setpts=PTS-STARTPTS[v{i}];"
        )
        filter_parts.append(
            f"[{i}:a]atrim=start={clip.source_start}:duration={clip.source_duration},"
            f"asetpts=PTS-STARTPTS[a{i}];"
        )
    
    # Build xfade chain for video
    # Pattern: v0 + v1 -> v01, v01 + v2 -> v012, etc.
    current_v = "v0"
    current_a = "a0"
    
    for i in range(1, len(spec.clips)):
        next_v = f"v{i}"
        next_a = f"a{i}"
        output_v = f"v_blend{i}" if i < len(spec.clips) - 1 else "vout"
        output_a = f"a_blend{i}" if i < len(spec.clips) - 1 else "aout"
        
        # Calculate offset (when the transition starts)
        # It's the duration of all previous clips minus overlap
        prev_duration = sum(spec.clips[j].source_duration for j in range(i))
        offset = prev_duration - (spec.transition_duration if i > 0 else 0)
        
        # Xfade for video
        filter_parts.append(
            f"[{current_v}][{next_v}]xfade=transition=fade:duration={spec.transition_duration}:offset={offset}[{output_v}];"
        )
        
        # Acrossfade for audio
        filter_parts.append(
            f"[{current_a}][{next_a}]acrossfade=d={spec.transition_duration}[{output_a}];"
        )
        
        current_v = output_v
        current_a = output_a
    
    # Add music mix at the end
    music_idx = len(spec.clips)
    filter_parts.append(
        f"[{current_a}][{music_idx}:a]amix=inputs=2:duration=first:weights=1 {spec.music_volume}[aout]"
    )
    
    return "".join(filter_parts)


def _create_concat_file_list(clips: list, temp_dir: Path) -> Path:
    """Create a concat demuxer file list for FFmpeg.
    
    This is an alternative approach that works well when we don't need
    complex transitions, just simple concatenation with crossfades.
    
    Args:
        clips: List of ClipSegment
        temp_dir: Temporary directory for file
        
    Returns:
        Path to the concat file list
    """
    concat_file = temp_dir / "concat_list.txt"
    
    with open(concat_file, "w") as f:
        for clip in clips:
            # Escape single quotes in path
            path_str = str(clip.source_path).replace("'", "'\\''")
            f.write(f"file '{path_str}'\n")
            f.write(f"inpoint {clip.source_start}\n")
            f.write(f"outpoint {clip.source_start + clip.source_duration}\n")
    
    return concat_file


def render_montage_simple(spec: RenderSpec) -> Path:
    """Render montage using concat demuxer (simpler, no transitions).
    
    This method uses FFmpeg's concat demuxer which is faster and more
    reliable than complex filter graphs, but doesn't support smooth
    transitions between clips.
    
    Args:
        spec: Render specification
        
    Returns:
        Path to rendered MP4
    """
    if len(spec.clips) == 0:
        raise VideoRenderError("No clips to render")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create concat file list
        concat_file = _create_concat_file_list(spec.clips, temp_path)
        
        # Build FFmpeg command
        cmd = [
            _FFMPEG,
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-i", str(spec.music_path),
            "-c:v", "libx264",
            "-preset", "fast",  # Balance of speed/quality
            "-crf", "23",  # Quality (lower = better, 23 is default)
            "-pix_fmt", "yuv420p",  # Compatibility
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # End when shortest input ends
            "-movflags", "+faststart",  # Web optimization
            str(spec.output_path),
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise VideoRenderError(
                f"FFmpeg failed with code {result.returncode}\n"
                f"stderr: {result.stderr[:500]}"
            )
        
        if not spec.output_path.exists():
            raise VideoRenderError("Output file was not created")
        
        return spec.output_path


def render_montage_with_transitions(spec: RenderSpec) -> Path:
    """Render montage with crossfade transitions (complex filter).
    
    This method uses FFmpeg's complex filter graph to create smooth
    crossfade transitions between clips. It's slower but looks better.
    
    Args:
        spec: Render specification
        
    Returns:
        Path to rendered MP4
    """
    if len(spec.clips) == 0:
        raise VideoRenderError("No clips to render")
    
    # Build filter complex
    filter_complex = _build_ffmpeg_complex_filter(spec)
    
    # Build input arguments
    cmd = [
        _FFMPEG,
        "-y",  # Overwrite output
    ]
    
    # Add video clip inputs
    for clip in spec.clips:
        cmd.extend(["-i", str(clip.source_path)])
    
    # Add music input
    cmd.extend(["-i", str(spec.music_path)])
    
    # Add filter complex
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(spec.output_path),
    ])
    
    # Run FFmpeg
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise VideoRenderError(
            f"FFmpeg failed with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:1000]}"
        )
    
    if not spec.output_path.exists():
        raise VideoRenderError("Output file was not created")
    
    return spec.output_path


def render_from_timeline(
    timeline,
    music_path: Path,
    output_path: Path,
    with_transitions: bool = True,
) -> Path:
    """Convenience function to render from a TimelineSpec.
    
    Args:
        timeline: TimelineSpec from timeline_builder
        music_path: Path to music file
        output_path: Where to save MP4
        with_transitions: Use transitions (slower but nicer) or simple concat
        
    Returns:
        Path to rendered MP4
    """
    spec = RenderSpec(
        clips=timeline.clips,
        music_path=music_path,
        output_path=output_path,
        transition_duration=0.5,  # 0.5s crossfade
        music_volume=0.3,  # 30% music volume
    )
    
    if with_transitions and len(spec.clips) > 1:
        return render_montage_with_transitions(spec)
    else:
        return render_montage_simple(spec)


# Aliases for cleaner imports
render = render_from_timeline
render_simple = render_montage_simple
render_with_transitions = render_montage_with_transitions


if __name__ == "__main__":
    # Smoke test - verifies FFmpeg is available
    result = subprocess.run(
        [_FFMPEG, "-version"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        version_line = result.stdout.split("\n")[0]
        print(f"✓ FFmpeg found: {version_line}")
    else:
        print("✗ FFmpeg not found on PATH")
        exit(1)
