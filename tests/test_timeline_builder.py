"""Tests for timeline_builder module."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from timeline_builder import (
    ClipSegment,
    TimelineSpec,
    TimelineBuilder,
)
from music_selector import Track, ProjectProfile, MusicLibrary


# Mock clip for testing (simulates clip_analyzer.ClipAnalysis)
@dataclass
class MockClip:
    """Mock clip analysis result."""
    file_path: Path
    duration: float
    quality_score: float
    scene_changes: list = None
    
    def __post_init__(self):
        if self.scene_changes is None:
            self.scene_changes = []


class TestClipSegment:
    """Test ClipSegment dataclass."""
    
    def test_segment_creation(self):
        """Verify ClipSegment can be created with all fields."""
        segment = ClipSegment(
            source_path=Path("/tmp/test.mp4"),
            source_start=5.0,
            source_duration=10.0,
            timeline_start=0.0,
            transition_in="cross-dissolve",
            transition_out=None,
            notes="Test segment",
        )
        assert segment.source_path == Path("/tmp/test.mp4")
        assert segment.source_start == 5.0
        assert segment.transition_in == "cross-dissolve"


class TestTimelineSpec:
    """Test TimelineSpec dataclass."""
    
    def test_timeline_creation(self):
        """Verify TimelineSpec creation and clip counting."""
        track = Track(
            id="test",
            title="Test",
            artist="Artist",
            bpm=120,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="medium",
            file="test.mp3",
            tags=[],
        )
        timeline = TimelineSpec(
            music_track=track,
            total_duration=30.0,
            beat_markers=[0.0, 0.5, 1.0],
        )
        assert timeline.clip_count == 0
        
        # Add a clip
        segment = ClipSegment(
            source_path=Path("/tmp/test.mp4"),
            source_start=0.0,
            source_duration=5.0,
            timeline_start=0.0,
        )
        timeline.add_clip(segment)
        assert timeline.clip_count == 1


class TestTimelineBuilder:
    """Test TimelineBuilder functionality."""
    
    def test_builder_initialization(self):
        """Verify TimelineBuilder can be initialized."""
        builder = TimelineBuilder()
        assert builder.music_library is not None
        assert builder.transition_duration == 0.5
    
    def test_build_timeline_with_mock_clips(self):
        """Verify timeline building with mock clips."""
        clips = [
            MockClip(Path("/tmp/clip1.mp4"), 15.0, 0.8),
            MockClip(Path("/tmp/clip2.mp4"), 20.0, 0.7),
            MockClip(Path("/tmp/clip3.mp4"), 12.0, 0.9),
        ]
        
        builder = TimelineBuilder()
        timeline = builder.build_timeline(
            clips,
            target_duration=60.0,
            has_slow_motion=False,
        )
        
        assert timeline.clip_count == 3
        assert timeline.music_track is not None
        assert len(timeline.beat_markers) == 3
    
    def test_clip_distribution(self):
        """Verify clips are distributed across timeline."""
        clips = [
            MockClip(Path("/tmp/clip1.mp4"), 30.0, 0.8),
            MockClip(Path("/tmp/clip2.mp4"), 30.0, 0.7),
        ]
        
        builder = TimelineBuilder()
        timeline = builder.build_timeline(clips, target_duration=30.0)
        
        # Clips should have different timeline positions
        assert timeline.clips[0].timeline_start != timeline.clips[1].timeline_start
    
    def test_transitions_between_clips(self):
        """Verify transitions are added between clips."""
        clips = [
            MockClip(Path("/tmp/clip1.mp4"), 15.0, 0.8),
            MockClip(Path("/tmp/clip2.mp4"), 15.0, 0.7),
        ]
        
        builder = TimelineBuilder()
        timeline = builder.build_timeline(clips, target_duration=30.0)
        
        # First clip should have transition_out
        assert timeline.clips[0].transition_out == "cross-dissolve"
        # Second clip should have transition_in
        assert timeline.clips[1].transition_in == "cross-dissolve"
        # Second clip should NOT have transition_out (it's the last one)
        assert timeline.clips[1].transition_out is None
    
    def test_segment_duration_respects_limits(self):
        """Verify segment durations are within min/max bounds."""
        clips = [
            MockClip(Path("/tmp/clip1.mp4"), 100.0, 0.8),  # Very long clip
        ]
        
        builder = TimelineBuilder()
        timeline = builder.build_timeline(clips, target_duration=30.0)
        
        # Should clamp to MAX_CLIP_DURATION (8 seconds)
        assert timeline.clips[0].source_duration <= 8.0
        assert timeline.clips[0].source_duration >= builder.MIN_CLIP_DURATION


class TestBestSegmentSelection:
    """Test the segment start selection logic."""
    
    def test_skips_intro_for_long_clips(self):
        """Verify 20% intro skip for long clips."""
        clip = MockClip(Path("/tmp/long.mp4"), 100.0, 0.8)
        
        builder = TimelineBuilder()
        start = builder._select_best_segment_start(clip, target_duration=5.0)
        
        # Should start at ~20 seconds (20% of 100)
        assert start >= 20.0
    
    def test_uses_start_for_short_clips(self):
        """Verify short clips use from beginning."""
        clip = MockClip(Path("/tmp/short.mp4"), 5.0, 0.8)
        
        builder = TimelineBuilder()
        start = builder._select_best_segment_start(clip, target_duration=5.0)
        
        # Should start from beginning
        assert start == 0.0
    
    def test_prefers_scene_changes(self):
        """Verify scene changes are preferred as start points."""
        clip = MockClip(
            Path("/tmp/scene.mp4"),
            60.0,
            0.8,
            scene_changes=[5.0, 15.0, 25.0]
        )
        
        builder = TimelineBuilder()
        start = builder._select_best_segment_start(clip, target_duration=10.0)
        
        # Should pick a scene change after intro skip (12 seconds = 20% of 60)
        # So it should pick 15.0 (the first scene change after 12s)
        assert start == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
