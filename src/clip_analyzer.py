"""video-montage-skill: clip_analyzer

Phase 1 MVP video clip analysis using local FFmpeg/FFprobe.

This module intentionally avoids cloud APIs and heavyweight dependencies.
It shells out to ffprobe/ffmpeg via subprocess and returns structured results.

Capabilities
- Metadata extraction via ffprobe (duration, resolution, fps, codecs)
- Scene change detection via ffmpeg's scene score (select=gt(scene,THRESH))
- Simple blur scoring via Laplacian-variance on sampled grayscale frames
- Optional face detection via ffmpeg's facedetect filter (if available)
- Thumbnail generation

Notes
- Scene detection uses frame metadata printed by the `metadata=print` filter.
- Blur scoring is computed in Python from PGM frames piped out of ffmpeg.

Python: 3.10+
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


class ClipAnalyzerError(RuntimeError):
    """Base exception for clip analyzer failures."""


class FFmpegNotFoundError(ClipAnalyzerError):
    """Raised when ffmpeg/ffprobe is not available on PATH."""


class FFmpegExecutionError(ClipAnalyzerError):
    """Raised when an ffmpeg/ffprobe command fails."""


_FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
_FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")


@dataclass(frozen=True)
class ClipMetadata:
    """Basic metadata extracted from ffprobe."""

    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False


@dataclass(frozen=True)
class SceneCut:
    """A detected scene cut point."""

    time_sec: float
    scene_score: float


@dataclass(frozen=True)
class BlurSample:
    """Blur measurement for a sampled frame."""

    time_sec: float
    laplacian_variance: float


@dataclass
class ClipAnalysis:
    """Full analysis output for a clip."""

    path: Path
    metadata: ClipMetadata
    scene_cuts: list[SceneCut] = field(default_factory=list)
    blur_samples: list[BlurSample] = field(default_factory=list)
    blur_score: float | None = None
    has_faces: bool | None = None
    thumbnail_path: Path | None = None

    def to_dict(self) -> dict:
        """Convert analysis to a JSON-serializable dict."""

        return {
            "path": str(self.path),
            "metadata": {
                "duration_sec": self.metadata.duration_sec,
                "width": self.metadata.width,
                "height": self.metadata.height,
                "fps": self.metadata.fps,
                "video_codec": self.metadata.video_codec,
                "audio_codec": self.metadata.audio_codec,
                "has_audio": self.metadata.has_audio,
            },
            "scene_cuts": [
                {"time_sec": sc.time_sec, "scene_score": sc.scene_score}
                for sc in self.scene_cuts
            ],
            "blur_samples": [
                {"time_sec": bs.time_sec, "laplacian_variance": bs.laplacian_variance}
                for bs in self.blur_samples
            ],
            "blur_score": self.blur_score,
            "has_faces": self.has_faces,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
        }


def _ensure_binaries() -> None:
    if shutil.which(_FFPROBE) is None:
        raise FFmpegNotFoundError(
            f"ffprobe not found (looked for '{_FFPROBE}'). Install FFmpeg or set FFPROBE_BIN."
        )
    if shutil.which(_FFMPEG) is None:
        raise FFmpegNotFoundError(
            f"ffmpeg not found (looked for '{_FFMPEG}'). Install FFmpeg or set FFMPEG_BIN."
        )


def _run(cmd: list[str], *, timeout_sec: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing stdout/stderr.

    Raises:
        FFmpegExecutionError: if returncode != 0
    """

    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as e:
        raise FFmpegNotFoundError(str(e)) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegExecutionError(f"Command timed out: {cmd}") from e

    if p.returncode != 0:
        msg = (
            "Command failed\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stderr: {p.stderr[-2000:]}"
        )
        raise FFmpegExecutionError(msg)
    return p


def _parse_fraction(s: str) -> float:
    """Parse ffprobe fraction-like strings such as '30000/1001'."""

    if not s:
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def probe_metadata(path: str | Path) -> ClipMetadata:
    """Extract clip metadata with ffprobe.

    Args:
        path: Video file path.

    Returns:
        ClipMetadata

    Raises:
        ClipAnalyzerError
    """

    _ensure_binaries()
    p = Path(path)
    cmd = [
        _FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    out = _run(cmd).stdout
    data = json.loads(out)

    duration = 0.0
    fmt = data.get("format") or {}
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = 0.0

    width = height = 0
    fps = 0.0
    vcodec: str | None = None
    acodec: str | None = None
    has_audio = False

    for s in data.get("streams") or []:
        stype = s.get("codec_type")
        if stype == "video" and width == 0:
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)
            vcodec = s.get("codec_name")
            fps = _parse_fraction(str(s.get("avg_frame_rate") or ""))
            if fps <= 0:
                fps = _parse_fraction(str(s.get("r_frame_rate") or ""))
        elif stype == "audio" and not has_audio:
            has_audio = True
            acodec = s.get("codec_name")

    return ClipMetadata(
        path=p,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=vcodec,
        audio_codec=acodec,
        has_audio=has_audio,
    )


_SCENE_PTS_RE = re.compile(r"pts_time:(?P<t>\d+(?:\.\d+)?)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(?P<s>\d+(?:\.\d+)?)")


def detect_scenes(
    path: str | Path,
    *,
    threshold: float = 0.3,
    max_duration_sec: float | None = None,
    timeout_sec: float | None = 120.0,
) -> list[SceneCut]:
    """Detect scene change timestamps using FFmpeg's scene score.

    Uses: -vf "select='gt(scene,THRESH)',metadata=print"

    Args:
        path: Video file path.
        threshold: Scene score threshold in [0,1]. Typical starting point: 0.3.
        max_duration_sec: If provided, analyze only the first N seconds.
        timeout_sec: Subprocess timeout.

    Returns:
        List of SceneCut entries (time_sec, scene_score).
    """

    _ensure_binaries()
    p = Path(path)

    # We intentionally discard the output video stream; we only want the log lines.
    vf = f"select='gt(scene,{threshold})',metadata=print"
    cmd = [_FFMPEG, "-hide_banner", "-nostdin"]
    if max_duration_sec is not None:
        cmd += ["-t", str(max_duration_sec)]
    cmd += ["-i", str(p), "-an", "-vf", vf, "-f", "null", "-"]

    proc = _run(cmd, timeout_sec=timeout_sec)
    text = proc.stderr + "\n" + proc.stdout

    cuts: list[SceneCut] = []
    # metadata=print output comes in blocks; we just scan for pts_time and scene_score.
    last_time: float | None = None
    for line in text.splitlines():
        m_t = _SCENE_PTS_RE.search(line)
        if m_t:
            last_time = float(m_t.group("t"))
        m_s = _SCENE_SCORE_RE.search(line)
        if m_s and last_time is not None:
            cuts.append(SceneCut(time_sec=last_time, scene_score=float(m_s.group("s"))))
            last_time = None

    # De-dupe nearly identical cut times (ffmpeg sometimes emits repeats).
    cuts.sort(key=lambda c: c.time_sec)
    deduped: list[SceneCut] = []
    for c in cuts:
        if not deduped or abs(deduped[-1].time_sec - c.time_sec) > 1e-3:
            deduped.append(c)
        else:
            # Keep the higher score if the time matches.
            prev = deduped[-1]
            if c.scene_score > prev.scene_score:
                deduped[-1] = c
    return deduped


def _pgm_frames_from_stream(payload: bytes) -> list[tuple[int, int, bytes]]:
    """Parse one or more binary PGM (P5) images from a byte stream.

    Returns a list of (width, height, pixel_bytes).

    This parser is tolerant of comments and whitespace.
    """

    frames: list[tuple[int, int, bytes]] = []
    i = 0
    n = len(payload)

    def _read_token() -> bytes:
        nonlocal i
        while i < n and payload[i] in b" \t\r\n":
            i += 1
        # Comments
        if i < n and payload[i] == ord("#"):
            while i < n and payload[i] not in b"\r\n":
                i += 1
            return _read_token()
        start = i
        while i < n and payload[i] not in b" \t\r\n":
            i += 1
        return payload[start:i]

    while i < n:
        # Seek to a header
        if payload[i : i + 2] != b"P5":
            i += 1
            continue
        magic = payload[i : i + 2]
        i += 2
        # After magic, parse tokens: width height maxval
        w_tok = _read_token()
        if not w_tok:
            break
        h_tok = _read_token()
        if not h_tok:
            break
        max_tok = _read_token()
        if not max_tok:
            break
        try:
            w = int(w_tok)
            h = int(h_tok)
            maxv = int(max_tok)
        except ValueError:
            break
        if magic != b"P5" or w <= 0 or h <= 0 or maxv <= 0:
            break
        # Consume one whitespace char after maxval if present.
        if i < n and payload[i] in b" \t\r\n":
            i += 1
        # Pixel bytes: for maxv < 256, 1 byte per pixel.
        pixel_count = w * h
        if maxv >= 256:
            # 2 bytes/pixel; uncommon for our use; skip.
            pixel_count *= 2
        if i + pixel_count > n:
            break
        pixels = payload[i : i + pixel_count]
        i += pixel_count
        if maxv >= 256:
            # Convert to 8-bit by taking the high byte.
            pixels = pixels[0::2]
        frames.append((w, h, pixels))

    return frames


def _laplacian_variance_gray8(width: int, height: int, pixels: bytes) -> float:
    """Compute Laplacian variance for an 8-bit grayscale image.

    Uses 4-neighborhood Laplacian: L = 4*c - n - s - e - w

    Returns:
        Variance of L over interior pixels.
    """

    if width < 3 or height < 3:
        return 0.0

    # Compute mean and mean of squares in one pass without storing all lap values.
    count = 0
    s1 = 0.0
    s2 = 0.0

    # Access pixels by index; bytes returns int values.
    for y in range(1, height - 1):
        row = y * width
        row_n = (y - 1) * width
        row_s = (y + 1) * width
        for x in range(1, width - 1):
            idx = row + x
            c = pixels[idx]
            lap = 4 * c - pixels[row_n + x] - pixels[row_s + x] - pixels[idx - 1] - pixels[idx + 1]
            count += 1
            s1 += lap
            s2 += lap * lap

    if count == 0:
        return 0.0
    mean = s1 / count
    var = (s2 / count) - (mean * mean)
    return float(var)


def _extract_gray_frames_pgm(
    path: Path,
    *,
    sample_times_sec: list[float],
    scale_width: int = 320,
    timeout_sec: float | None = 60.0,
) -> list[BlurSample]:
    """Extract grayscale PGM frames at specified timestamps and compute blur metric."""

    samples: list[BlurSample] = []

    for t in sample_times_sec:
        # -ss before -i gives faster seek (less accurate on some codecs).
        # For analysis/scoring this is fine.
        cmd = [
            _FFMPEG,
            "-hide_banner",
            "-nostdin",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={scale_width}:-1,format=gray",
            "-f",
            "image2pipe",
            "-vcodec",
            "pgm",
            "-",
        ]

        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(str(e)) from e
        except subprocess.TimeoutExpired as e:
            raise FFmpegExecutionError(f"Frame extract timed out at t={t:.3f}s") from e

        if p.returncode != 0:
            # Skip a bad sample rather than failing the entire clip.
            continue

        frames = _pgm_frames_from_stream(p.stdout)
        if not frames:
            continue
        w, h, pix = frames[0]
        var = _laplacian_variance_gray8(w, h, pix)
        samples.append(BlurSample(time_sec=float(t), laplacian_variance=var))

    return samples


def score_blur(
    path: str | Path,
    *,
    duration_sec: float | None = None,
    samples: int = 10,
    min_spacing_sec: float = 0.5,
    scale_width: int = 320,
) -> tuple[list[BlurSample], float | None]:
    """Compute a blur score by sampling frames and computing Laplacian variance.

    Higher variance generally indicates a sharper image.

    Args:
        path: Video file path.
        duration_sec: Duration; if None will probe.
        samples: Number of samples across the clip.
        min_spacing_sec: Minimum spacing between sample timestamps.
        scale_width: Scale width used for analysis.

    Returns:
        (blur_samples, blur_score) where blur_score is the mean Laplacian variance.
    """

    _ensure_binaries()
    p = Path(path)

    if duration_sec is None:
        duration_sec = probe_metadata(p).duration_sec

    if duration_sec <= 0:
        return ([], None)

    # Sample times from 10%..90% to avoid fade-ins/outs.
    start = 0.1 * duration_sec
    end = 0.9 * duration_sec
    if end <= start:
        start = 0.0
        end = max(duration_sec - 0.01, 0.0)

    if samples <= 1:
        times = [max(0.0, min(duration_sec * 0.5, duration_sec - 0.001))]
    else:
        span = max(end - start, 0.0)
        step = span / (samples - 1) if samples > 1 else span
        step = max(step, min_spacing_sec)
        times = [start + i * step for i in range(samples)]
        times = [max(0.0, min(t, max(duration_sec - 0.001, 0.0))) for t in times]

    blur_samples = _extract_gray_frames_pgm(p, sample_times_sec=times, scale_width=scale_width)
    if not blur_samples:
        return ([], None)

    blur_score = statistics.fmean(s.laplacian_variance for s in blur_samples)
    return (blur_samples, float(blur_score))


def _ffmpeg_has_filter(filter_name: str) -> bool:
    """Return True if ffmpeg reports the given filter is available."""

    try:
        p = _run([_FFMPEG, "-hide_banner", "-filters"], timeout_sec=10.0)
    except ClipAnalyzerError:
        return False
    return bool(re.search(rf"\b{re.escape(filter_name)}\b", p.stdout + p.stderr))


def detect_faces(
    path: str | Path,
    *,
    max_duration_sec: float = 10.0,
    scale_width: int = 640,
    timeout_sec: float | None = 120.0,
) -> bool | None:
    """Detect whether the clip likely contains faces using ffmpeg's facedetect filter.

    Returns:
        - True/False if facedetect filter is available.
        - None if facedetect is unavailable.

    Notes:
        This is a heuristic. For MVP we treat any detection log line as "has faces".
    """

    _ensure_binaries()
    if not _ffmpeg_has_filter("facedetect"):
        return None

    p = Path(path)
    vf = f"scale={scale_width}:-1,facedetect=mode=1:threshold=0.5"
    cmd = [
        _FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-t",
        str(max_duration_sec),
        "-i",
        str(p),
        "-an",
        "-vf",
        vf,
        "-f",
        "null",
        "-",
    ]

    proc = _run(cmd, timeout_sec=timeout_sec)
    text = proc.stderr + "\n" + proc.stdout

    # Common log patterns vary by build.
    if re.search(r"Detected\s+face", text, re.IGNORECASE):
        return True
    if re.search(r"facedetect.*(face|faces)", text, re.IGNORECASE):
        return True
    if re.search(r"\bface\b", text, re.IGNORECASE) and re.search(r"score", text, re.IGNORECASE):
        return True
    return False


def generate_thumbnail(
    path: str | Path,
    *,
    out_dir: str | Path,
    timestamp_sec: float | None = None,
    width: int = 640,
    timeout_sec: float | None = 60.0,
) -> Path:
    """Generate a thumbnail JPEG for the clip.

    Args:
        path: Video file path.
        out_dir: Output directory.
        timestamp_sec: If None, uses mid-clip based on probed duration.
        width: Thumbnail width.

    Returns:
        Path to generated thumbnail.
    """

    _ensure_binaries()
    p = Path(path)
    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    if timestamp_sec is None:
        md = probe_metadata(p)
        timestamp_sec = max(0.0, md.duration_sec * 0.5)

    thumb_path = outd / f"{p.stem}_thumb_{uuid.uuid4().hex[:8]}.jpg"

    cmd = [
        _FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(p),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-1",
        "-q:v",
        "2",
        str(thumb_path),
    ]

    _run(cmd, timeout_sec=timeout_sec)
    return thumb_path


def analyze_clip(
    path: str | Path,
    *,
    work_dir: str | Path | None = None,
    scene_threshold: float = 0.3,
    scene_max_duration_sec: float | None = None,
    blur_samples: int = 10,
    detect_faces_max_duration_sec: float = 10.0,
    make_thumbnail: bool = True,
) -> ClipAnalysis:
    """Run MVP analysis pipeline on a single clip.

    Args:
        path: Video file path.
        work_dir: Directory for generated artifacts (thumbnail). If None, uses sibling `.analysis/`.
        scene_threshold: FFmpeg scene threshold.
        scene_max_duration_sec: Optional cap for faster scene detection.
        blur_samples: Number of blur samples.
        detect_faces_max_duration_sec: Seconds to scan for faces.
        make_thumbnail: Whether to generate a thumbnail.

    Returns:
        ClipAnalysis
    """

    _ensure_binaries()
    p = Path(path)
    if not p.exists():
        raise ClipAnalyzerError(f"Clip not found: {p}")

    md = probe_metadata(p)

    scenes = detect_scenes(
        p,
        threshold=scene_threshold,
        max_duration_sec=scene_max_duration_sec,
    )

    blur_samps, blur_score = score_blur(
        p,
        duration_sec=md.duration_sec,
        samples=blur_samples,
    )

    faces = detect_faces(p, max_duration_sec=detect_faces_max_duration_sec)

    thumb: Path | None = None
    if make_thumbnail:
        out_dir = Path(work_dir) if work_dir is not None else (p.parent / ".analysis")
        thumb = generate_thumbnail(p, out_dir=out_dir)

    return ClipAnalysis(
        path=p,
        metadata=md,
        scene_cuts=scenes,
        blur_samples=blur_samps,
        blur_score=blur_score,
        has_faces=faces,
        thumbnail_path=thumb,
    )


def analyze_clips(
    paths: Iterable[str | Path],
    *,
    work_dir: str | Path | None = None,
    scene_threshold: float = 0.3,
    blur_samples: int = 10,
) -> list[ClipAnalysis]:
    """Analyze multiple clips (serially).

    This is deliberately serial and simple for Phase 1; you can parallelize later.
    """

    results: list[ClipAnalysis] = []
    for p in paths:
        results.append(
            analyze_clip(
                p,
                work_dir=work_dir,
                scene_threshold=scene_threshold,
                blur_samples=blur_samples,
            )
        )
    return results


if __name__ == "__main__":
    # Simple smoke test that exercises parsing/formatting without requiring real media.
    # To use on a real file:
    #   python clip_analyzer.py /path/to/video.mp4
    import sys

    if len(sys.argv) > 1:
        analysis = analyze_clip(sys.argv[1])
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print("clip_analyzer.py loaded. Provide a video path to run analysis.")
