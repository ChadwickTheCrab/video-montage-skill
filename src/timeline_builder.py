"""Timeline builder module for video montage skill.

Coordinates clip analysis, music selection, and cut placement
to build a complete video timeline specification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .music_selector import MusicLibrary, Track, ProjectProfile, calculate_cut_timing


@dataclass
class ClipSegment:
    """Represents a segment of a clip on the timeline."""
    source_path: Path
    source_start: float  # Start time in source clip (seconds)
    source_duration: float  # Duration from source
    timeline_start: float  # Start time on timeline (seconds)
    transition_in: Optional[str] = None  # "cross-dissolve" or None
    transition_out: Optional[str] = None
    notes: str = ""


@dataclass
class TimelineSpec:
    """Complete specification for a video timeline."""
    music_track: Track
    total_duration: float
    clips: list = field(default_factory=list)
    beat_markers: list = field(default_factory=list)
    fps: int = 30
    
    def add_clip(self, segment: ClipSegment):
        """Add a clip segment to the timeline."""
        self.clips.append(segment)
    
    @property
    def clip_count(self) -> int:
        """Number of clips in the timeline."""
        return len(self.clips)
    
    def to_fcpxml_data(self) -> dict:
        """Convert timeline to data structure for FCPXML generation."""
        return {
            "music_track": self.music_track,
            "total_duration": self.total_duration,
            "clips": self.clips,
            "fps": self.fps,
            "beat_markers": self.beat_markers,
        }


class TimelineBuilder:
    """Builds video timelines from analyzed clips and music."""
    
    DEFAULT_TRANSITION_DURATION = 0.5  # seconds
    MIN_CLIP_DURATION = 1.0  # seconds
    MAX_CLIP_DURATION = 8.0  # seconds (for montage pacing)
    
    def __init__(
        self,
        music_library: Optional[MusicLibrary] = None,
        transition_duration: float = DEFAULT_TRANSITION_DURATION
    ):
        """Initialize timeline builder.
        
        Args:
            music_library: MusicLibrary instance. Creates default if None.
            transition_duration: Duration of cross-dissolve transitions.
        """
        self.music_library = music_library or MusicLibrary()
        self.transition_duration = transition_duration
    
    def build_timeline(
        self,
        analyzed_clips: list,
        target_duration: Optional[float] = None,
        music_track_id: Optional[str] = None,
        has_slow_motion: bool = False
    ) -> TimelineSpec:
        """Build a complete timeline from analyzed clips.
        
        Args:
            analyzed_clips: List of ClipAnalysis objects from clip_analyzer
            target_duration: Target total duration. Auto-calculated if None.
            music_track_id: Specific track to use. Auto-selected if None.
            has_slow_motion: Whether clips include slow-motion footage.
            
        Returns:
            TimelineSpec ready for FCPXML generation
        """
        # Calculate total content duration
        total_content_duration = sum(c.duration for c in analyzed_clips)
        
        # Determine target duration
        if target_duration is None:
            # Default: pack into ~80% of a typical 3-minute track
            target_duration = min(180, total_content_duration * 0.6)
        
        # Select or use specified music track
        if music_track_id:
            music_track = self.music_library.get_track_by_id(music_track_id)
            if not music_track:
                raise ValueError(f"Track {music_track_id} not found")
        else:
            profile = ProjectProfile(
                clip_count=len(analyzed_clips),
                total_duration_sec=target_duration,
                avg_clip_duration=total_content_duration / len(analyzed_clips),
                has_slow_motion=has_slow_motion
            )
            music_track = self.music_library.select_track(profile)
        
        # Calculate beat markers for cuts
        beat_markers = calculate_cut_timing(
            music_track,
            len(analyzed_clips),
            target_duration,
            cuts_per_beat=1
        )
        
        # Build timeline spec
        timeline = TimelineSpec(
            music_track=music_track,
            total_duration=target_duration,
            beat_markers=beat_markers
        )
        
        # Distribute clips across beat markers
        self._distribute_clips(timeline, analyzed_clips, beat_markers)
        
        return timeline
    
    def _distribute_clips(
        self,
        timeline: TimelineSpec,
        clips: list,
        beat_markers: list
    ):
        """Distribute clips across beat markers with transitions."""
        for i, (clip, cut_time) in enumerate(zip(clips, beat_markers)):
            # Determine clip segment duration
            if i < len(beat_markers) - 1:
                # Time until next cut
                next_cut = beat_markers[i + 1]
                available_duration = next_cut - cut_time
            else:
                # Last clip - use remaining time
                available_duration = timeline.total_duration - cut_time
            
            # Clip duration with transition overlap
            segment_duration = available_duration
            if i > 0:  # Not first clip - overlap with previous transition
                segment_duration += self.transition_duration / 2
            if i < len(clips) - 1:  # Not last clip - overlap with next
                segment_duration -= self.transition_duration / 2
            
            # Clamp to reasonable range
            segment_duration = max(
                self.MIN_CLIP_DURATION,
                min(self.MAX_CLIP_DURATION, segment_duration)
            )
            
            # Don't exceed source clip duration
            segment_duration = min(segment_duration, clip.duration)
            
            # Pick best segment from source (highest quality middle portion)
            source_start = self._select_best_segment_start(clip, segment_duration)
            
            # Determine transitions
            transition_in = "cross-dissolve" if i > 0 else None
            transition_out = "cross-dissolve" if i < len(clips) - 1 else None
            
            segment = ClipSegment(
                source_path=clip.file_path,
                source_start=source_start,
                source_duration=segment_duration,
                timeline_start=cut_time,
                transition_in=transition_in,
                transition_out=transition_out,
                notes=f"Quality score: {clip.quality_score:.2f}"
            )
            
            timeline.add_clip(segment)
    
    def _select_best_segment_start(
        self,
        clip,
        target_duration: float
    ) -> float:
        """Select the best starting point in a clip for extraction.
        
        Prefers:
        1. Segments with motion (avoid static shots)
        2. Middle of clip (often most stable)
        3. Avoids very beginning/end (often shaky)
        
        Args:
            clip: ClipAnalysis object
            target_duration: How long of a segment we need
            
        Returns:
            Start time in source clip (seconds)
        """
        if clip.duration <= target_duration + 2:
            # Clip is short, use from beginning
            return 0.0
        
        # Default: start 20% into the clip (skip intro shake)
        intro_skip = clip.duration * 0.2
        
        # Use scene changes if available
        if hasattr(clip, 'scene_changes') and clip.scene_changes:
            # Find a scene change that's at least intro_skip in
            for sc_time in clip.scene_changes:
                if sc_time >= intro_skip:
                    # Check if we have enough duration after this point
                    if sc_time + target_duration <= clip.duration * 0.9:
                        return sc_time
        
        # Fallback: middle of clip
        mid_point = clip.duration / 2
        if mid_point - (target_duration / 2) >= 0:
            return max(intro_skip, mid_point - (target_duration / 2))
        
        return intro_skip


def quick_build(
    clip_paths: list,
    output_duration: float = 120.0,
    has_slow_motion: bool = False
) -> TimelineSpec:
    """Quick timeline build from raw clip paths.
    
    Args:
        clip_paths: List of Path objects to video files
        output_duration: Target output duration in seconds
        has_slow_motion: Whether clips include slow-mo
        
    Returns:
        TimelineSpec ready for FCPXML generation
    """
    # Import here to avoid circular dependency
    from .clip_analyzer import analyze_clip
    
    # Analyze all clips
    analyzed = [analyze_clip(p) for p in clip_paths]
    
    # Build timeline
    builder = TimelineBuilder()
    return builder.build_timeline(
        analyzed,
        target_duration=output_duration,
        has_slow_motion=has_slow_motion
    )


if __name__ == "__main__":
    # Test with mock data
    from dataclasses import dataclass
    
    @dataclass
    class MockClip:
        file_path: Path
        duration: float
        quality_score: float
        scene_changes: list = None
    
    clips = [
        MockClip(Path("/tmp/clip1.mp4"), 15.0, 0.8, [2.0, 5.0, 10.0]),
        MockClip(Path("/tmp/clip2.mp4"), 20.0, 0.7, [3.0, 8.0]),
        MockClip(Path("/tmp/clip3.mp4"), 12.0, 0.9, [1.0, 6.0]),
    ]
    
    builder = TimelineBuilder()
    timeline = builder.build_timeline(clips, target_duration=60.0)
    
    print(f"Timeline: {timeline.clip_count} clips")
    print(f"Music: {timeline.music_track.title} ({timeline.music_track.bpm} BPM)")
    print(f"Total duration: {timeline.total_duration:.1f}s")
    print(f"Beat markers: {len(timeline.beat_markers)}")
    for i, clip in enumerate(timeline.clips):
        print(f"  Clip {i+1}: {clip.source_duration:.1f}s at {clip.timeline_start:.1f}s")
