"""video-montage-skill: fcpxml_generator

Generate FCPXML (Final Cut Pro XML) timelines that DaVinci Resolve can import.

This module focuses on a pragmatic, Resolve-friendly subset of FCPXML:
- External media references via <asset> + <media-rep kind="original-media">
- A single project/sequence with a primary storyline (<spine>)
- Sequential asset-clips placed at provided offsets/durations (typically beat markers)
- Optional cross-dissolve transitions between adjacent clips

The resulting file is intended for *round-tripping/editability* rather than perfect
fidelity with Final Cut Pro.

Python: 3.10+
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET


class FCPXMLError(RuntimeError):
    """Raised for FCPXML generation errors."""


@dataclass(frozen=True)
class VideoFormat:
    """Timeline video format."""

    width: int = 1920
    height: int = 1080
    fps: float = 30.0


@dataclass(frozen=True)
class TimelineClip:
    """A clip placement on the timeline.

    Args:
        src_path: Path to the media file.
        name: Display name in the editor.
        offset_sec: Timeline offset (seconds from start).
        duration_sec: Clip duration on the timeline.
        start_sec: In-point in the source media.
        has_audio: Whether to flag audio presence; Resolve can infer, but we include it.
    """

    src_path: Path
    name: str
    offset_sec: float
    duration_sec: float
    start_sec: float = 0.0
    has_audio: bool = True


@dataclass(frozen=True)
class Transition:
    """A transition between two adjacent clips in the primary storyline."""

    name: str = "Cross Dissolve"
    duration_sec: float = 1.0


def _pretty_xml(xml_bytes: bytes) -> str:
    dom = minidom.parseString(xml_bytes)
    return dom.toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")


def _as_file_uri(path: Path) -> str:
    # Resolve seems to prefer file:// URIs.
    # Path.as_uri() requires an absolute path.
    return path.expanduser().resolve().as_uri()


def _sec_to_fcpx_time(seconds: float, *, denom: int = 24000) -> str:
    """Convert seconds to an FCPXML rational time string (e.g., '24024/24000s')."""

    if seconds <= 0:
        return "0s"
    fr = Fraction.from_float(seconds).limit_denominator(denom)
    if fr.denominator == 1:
        return f"{fr.numerator}s"
    return f"{fr.numerator}/{fr.denominator}s"


def _fps_to_frame_duration(fps: float) -> Fraction:
    """Return frameDuration as a Fraction of seconds per frame.

    Common NTSC rates are represented precisely.
    """

    if abs(fps - 29.97) < 0.01:
        return Fraction(1001, 30000)
    if abs(fps - 59.94) < 0.01:
        return Fraction(1001, 60000)
    if abs(fps - 23.976) < 0.01 or abs(fps - 23.98) < 0.02:
        return Fraction(1001, 24000)
    if abs(fps - 24.0) < 1e-6:
        return Fraction(1, 24)
    if abs(fps - 25.0) < 1e-6:
        return Fraction(1, 25)
    if abs(fps - 30.0) < 1e-6:
        return Fraction(1, 30)

    # Fallback: best rational approx with a reasonable denominator.
    return Fraction.from_float(1.0 / max(fps, 1e-9)).limit_denominator(60000)


def _fraction_to_fcpx_time(fr: Fraction) -> str:
    if fr <= 0:
        return "0s"
    if fr.denominator == 1:
        return f"{fr.numerator}s"
    return f"{fr.numerator}/{fr.denominator}s"


def build_fcpxml(
    clips: Iterable[TimelineClip],
    *,
    project_name: str = "Montage",
    event_name: str = "Event",
    library_location: str = "",
    video_format: VideoFormat = VideoFormat(),
    transitions: Optional[Transition] = None,
) -> str:
    """Build an FCPXML document.

    Args:
        clips: TimelineClip placements. For beat-synced edits, offset_sec should align to beats.
        project_name: Name shown in Resolve/FCP.
        event_name: Event/bin name.
        library_location: Optional library location attribute.
        video_format: Timeline format.
        transitions: If provided, inserts a Cross Dissolve transition between adjacent clips.

    Returns:
        XML string (UTF-8) ready to write to disk.

    Raises:
        FCPXMLError
    """

    clip_list = list(clips)
    if not clip_list:
        raise FCPXMLError("No clips provided")

    # Validate ordering by offset.
    clip_list.sort(key=lambda c: c.offset_sec)
    for c in clip_list:
        if c.duration_sec <= 0:
            raise FCPXMLError(f"Clip duration must be > 0 for {c.src_path}")
        if c.offset_sec < 0:
            raise FCPXMLError(f"Clip offset must be >= 0 for {c.src_path}")

    root = ET.Element("fcpxml", {"version": "1.10"})

    resources = ET.SubElement(root, "resources")

    fmt_id = "r1"
    fd = _fps_to_frame_duration(video_format.fps)
    fmt_attrs = {
        "id": fmt_id,
        "name": f"FFVideoFormat{video_format.height}p{int(round(video_format.fps))}",
        "width": str(video_format.width),
        "height": str(video_format.height),
        "frameDuration": _fraction_to_fcpx_time(fd),
    }
    ET.SubElement(resources, "format", fmt_attrs)

    # Optional effects referenced by transitions.
    video_effect_id = None
    audio_effect_id = None
    if transitions is not None and transitions.duration_sec > 0:
        video_effect_id = "r_effect_video"
        audio_effect_id = "r_effect_audio"
        ET.SubElement(resources, "effect", {"id": video_effect_id, "name": transitions.name})
        ET.SubElement(resources, "effect", {"id": audio_effect_id, "name": "Audio Crossfade"})

    # Create asset resources.
    asset_ids: dict[Path, str] = {}
    for idx, c in enumerate(clip_list, start=1):
        src = c.src_path.expanduser().resolve()
        if not src.exists():
            # Resolve may still import, but failing early prevents confusing errors later.
            raise FCPXMLError(f"Media file not found: {src}")

        if src not in asset_ids:
            aid = f"r{idx + 1}"  # r1 reserved for format
            asset_ids[src] = aid

            asset = ET.SubElement(
                resources,
                "asset",
                {
                    "id": aid,
                    "name": src.name,
                    "uid": str(uuid.uuid4()),
                    "start": "0s",
                    # duration is optional for assets, but providing it tends to help importers.
                    # We don't know full source duration here; use at least the used duration.
                    "duration": _sec_to_fcpx_time(max(c.start_sec + c.duration_sec, c.duration_sec)),
                    "hasVideo": "1",
                    "hasAudio": "1" if c.has_audio else "0",
                    "format": fmt_id,
                },
            )
            ET.SubElement(
                asset,
                "media-rep",
                {
                    "kind": "original-media",
                    "src": _as_file_uri(src),
                },
            )

    library = ET.SubElement(root, "library", {"location": library_location})
    event = ET.SubElement(library, "event", {"name": event_name, "uid": str(uuid.uuid4())})

    project = ET.SubElement(event, "project", {"name": project_name, "uid": str(uuid.uuid4())})

    # Sequence duration: end of last clip.
    last = max(clip_list, key=lambda c: c.offset_sec + c.duration_sec)
    seq_duration = last.offset_sec + last.duration_sec

    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "format": fmt_id,
            "duration": _sec_to_fcpx_time(seq_duration),
            "tcStart": "0s",
            "tcFormat": "NDF",
        },
    )

    spine = ET.SubElement(sequence, "spine")

    # Build spine: clip, transition, clip, transition...
    for i, c in enumerate(clip_list):
        ref = asset_ids[c.src_path.expanduser().resolve()]
        ET.SubElement(
            spine,
            "asset-clip",
            {
                "name": c.name,
                "ref": ref,
                "offset": _sec_to_fcpx_time(c.offset_sec),
                "start": _sec_to_fcpx_time(c.start_sec),
                "duration": _sec_to_fcpx_time(c.duration_sec),
                "format": fmt_id,
                "tcFormat": "NDF",
            },
        )

        # Insert a transition after this clip, except after the last.
        if transitions is not None and transitions.duration_sec > 0 and i < len(clip_list) - 1:
            # Convention: transition starts at end of clip - half transition.
            # Importers vary in how they interpret this, but this is widely used.
            trans_offset = c.offset_sec + c.duration_sec - (transitions.duration_sec / 2.0)
            trans_el = ET.SubElement(
                spine,
                "transition",
                {
                    "name": transitions.name,
                    "offset": _sec_to_fcpx_time(max(trans_offset, 0.0)),
                    "duration": _sec_to_fcpx_time(transitions.duration_sec),
                },
            )
            if video_effect_id is not None:
                ET.SubElement(trans_el, "filter-video", {"ref": video_effect_id, "name": transitions.name})
            if audio_effect_id is not None:
                ET.SubElement(trans_el, "filter-audio", {"ref": audio_effect_id, "name": "Audio Crossfade"})

    xml_bytes = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    return _pretty_xml(xml_bytes)


def write_fcpxml(
    out_path: str | Path,
    clips: Iterable[TimelineClip],
    *,
    project_name: str = "Montage",
    event_name: str = "Event",
    library_location: str = "",
    video_format: VideoFormat = VideoFormat(),
    transitions: Optional[Transition] = None,
) -> Path:
    """Generate and write an FCPXML file to disk."""

    xml_text = build_fcpxml(
        clips,
        project_name=project_name,
        event_name=event_name,
        library_location=library_location,
        video_format=video_format,
        transitions=transitions,
    )
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(xml_text, encoding="utf-8")
    return outp


def build_sequential_timeline_on_beats(
    media_paths: Iterable[str | Path],
    beat_times_sec: list[float],
    *,
    clip_name_fn=None,
    min_clip_duration_sec: float = 0.5,
    has_audio: bool = True,
) -> list[TimelineClip]:
    """Utility: build clip placements where each clip spans beat[i]..beat[i+1].

    This is a small helper for MVP tests. In a full pipeline, a dedicated
    timeline_builder module will likely handle handles, transitions, and trimming.

    Args:
        media_paths: Clips in order.
        beat_times_sec: Monotonic beat timestamps.

    Returns:
        TimelineClip list with offsets and durations based on beat spacing.
    """

    paths = [Path(p) for p in media_paths]
    if len(beat_times_sec) < 2:
        raise FCPXMLError("Need at least 2 beat times to compute durations")

    beats = sorted(float(b) for b in beat_times_sec)
    if any(b2 <= b1 for b1, b2 in zip(beats, beats[1:])):
        raise FCPXMLError("beat_times_sec must be strictly increasing")

    out: list[TimelineClip] = []
    for i, src in enumerate(paths):
        if i >= len(beats) - 1:
            break
        offset = beats[i]
        duration = max(beats[i + 1] - beats[i], 0.0)
        if duration < min_clip_duration_sec:
            continue
        name = clip_name_fn(src) if clip_name_fn else src.stem
        out.append(
            TimelineClip(
                src_path=src,
                name=name,
                offset_sec=offset,
                duration_sec=duration,
                start_sec=0.0,
                has_audio=has_audio,
            )
        )
    return out


if __name__ == "__main__":
    # Smoke test that builds a minimal FCPXML string.
    # To actually write a file:
    #   python fcpxml_generator.py /abs/path/to/a.mp4 /abs/path/to/b.mp4
    import sys

    if len(sys.argv) >= 3:
        clips = [
            TimelineClip(Path(sys.argv[1]), Path(sys.argv[1]).stem, 0.0, 2.0),
            TimelineClip(Path(sys.argv[2]), Path(sys.argv[2]).stem, 2.0, 2.0),
        ]
        xml_text = build_fcpxml(clips, transitions=Transition(duration_sec=1.0))
        print(xml_text)
    else:
        print("fcpxml_generator.py loaded. Provide 2 media paths to print a sample XML.")
