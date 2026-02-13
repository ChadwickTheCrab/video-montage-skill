"""Tests for music_selector module."""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from music_selector import (
    Track,
    ProjectProfile,
    MusicLibrary,
    calculate_cut_timing,
    quick_select,
)


class TestTrack:
    """Test Track dataclass and properties."""
    
    def test_beat_duration_calculation(self):
        """Verify beat duration is calculated correctly from BPM."""
        track = Track(
            id="test-120",
            title="Test Track",
            artist="Test Artist",
            bpm=120,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="medium",
            file="test.mp3",
            tags=[],
        )
        # 120 BPM = 0.5 seconds per beat
        assert track.beat_duration == 0.5
    
    def test_beat_duration_60bpm(self):
        """Verify 60 BPM = 1 second per beat."""
        track = Track(
            id="test-60",
            title="Slow",
            artist="Artist",
            bpm=60,
            duration_sec=60,
            genre="Ambient",
            mood="Calm",
            energy="low",
            file="slow.mp3",
            tags=[],
        )
        assert track.beat_duration == 1.0
    
    def test_total_beats_calculation(self):
        """Verify total beats calculation."""
        track = Track(
            id="test-60",
            title="One Minute",
            artist="Artist",
            bpm=60,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="low",
            file="test.mp3",
            tags=[],
        )
        # 60 seconds at 60 BPM = 60 beats
        assert track.total_beats == 60


class TestMusicLibrary:
    """Test MusicLibrary loading and selection."""
    
    def test_library_loads_from_default_path(self):
        """Verify library loads from default config path."""
        lib = MusicLibrary()
        assert len(lib.tracks) > 0
        assert lib.version is not None
    
    def test_library_loads_specific_track(self):
        """Verify get_track_by_id returns correct track."""
        lib = MusicLibrary()
        track = lib.get_track_by_id("upbeat-pop-01")
        assert track is not None
        assert track.title == "Sunny Days"
        assert track.bpm == 128
    
    def test_get_track_by_id_not_found(self):
        """Verify None returned for non-existent track."""
        lib = MusicLibrary()
        track = lib.get_track_by_id("nonexistent")
        assert track is None
    
    def test_list_genres(self):
        """Verify genre listing returns unique genres."""
        lib = MusicLibrary()
        genres = lib.list_genres()
        assert len(genres) > 0
        assert "Pop" in genres
        assert len(genres) == len(set(genres))  # No duplicates
    
    def test_list_by_energy(self):
        """Verify filtering by energy level works."""
        lib = MusicLibrary()
        high_energy = lib.list_by_energy("high")
        assert len(high_energy) > 0
        for track in high_energy:
            assert track.energy == "high"


class TestTrackSelection:
    """Test music selection logic."""
    
    def test_selects_high_bpm_for_many_clips(self):
        """Verify high clip count selects high BPM track."""
        lib = MusicLibrary()
        profile = ProjectProfile(
            clip_count=30,
            total_duration_sec=120,
            avg_clip_duration=4,
            has_slow_motion=False,
        )
        track = lib.select_track(profile)
        # Should pick higher BPM for many clips
        assert track.bpm >= 120
    
    def test_selects_low_bpm_for_few_clips(self):
        """Verify low clip count selects lower BPM track."""
        lib = MusicLibrary()
        profile = ProjectProfile(
            clip_count=3,
            total_duration_sec=60,
            avg_clip_duration=20,
            has_slow_motion=False,
        )
        track = lib.select_track(profile)
        # Should pick lower BPM for few clips
        assert track.bpm <= 110
    
    def test_excludes_used_tracks(self):
        """Verify exclude_ids parameter works."""
        lib = MusicLibrary()
        profile = ProjectProfile(
            clip_count=15,
            total_duration_sec=120,
            avg_clip_duration=8,
        )
        # Get all track IDs
        all_ids = [t.id for t in lib.tracks]
        # Exclude all but one
        exclude = all_ids[:-1]
        track = lib.select_track(profile, exclude_ids=exclude)
        assert track.id == all_ids[-1]
    
    def test_raises_error_if_no_suitable_track(self):
        """Verify error raised when no track fits duration."""
        lib = MusicLibrary()
        profile = ProjectProfile(
            clip_count=5,
            total_duration_sec=1000,  # Way too long for any track
            avg_clip_duration=200,
        )
        with pytest.raises(ValueError):
            lib.select_track(profile)


class TestCalculateCutTiming:
    """Test cut timing calculations."""
    
    def test_single_clip_single_cut(self):
        """Verify single clip returns single cut time."""
        track = Track(
            id="test",
            title="Test",
            artist="Artist",
            bpm=60,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="low",
            file="test.mp3",
            tags=[],
        )
        cuts = calculate_cut_timing(track, clip_count=1, total_duration=30)
        assert len(cuts) == 1
        assert cuts[0] == 1.0  # First beat at 1 second
    
    def test_multiple_clips_on_beats(self):
        """Verify cuts fall on beat markers."""
        track = Track(
            id="test",
            title="Test",
            artist="Artist",
            bpm=60,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="low",
            file="test.mp3",
            tags=[],
        )
        cuts = calculate_cut_timing(track, clip_count=5, total_duration=30)
        assert len(cuts) == 5
        # All cuts should be at integer seconds (60 BPM = 1s per beat)
        for cut in cuts:
            assert cut == int(cut)  # Whole number
    
    def test_cuts_per_beat_parameter(self):
        """Verify cuts_per_beat affects timing."""
        track = Track(
            id="test",
            title="Test",
            artist="Artist",
            bpm=120,
            duration_sec=60,
            genre="Test",
            mood="Test",
            energy="low",
            file="test.mp3",
            tags=[],
        )
        # 120 BPM = 0.5s per beat
        # cuts_per_beat=2 means every 0.25s
        cuts = calculate_cut_timing(track, clip_count=3, total_duration=2, cuts_per_beat=2)
        # Should snap to 0.25s intervals
        assert cuts[1] - cuts[0] == 0.5  # Half beat interval


class TestQuickSelect:
    """Test quick_select convenience function."""
    
    def test_quick_select_returns_track(self):
        """Verify quick_select returns a valid Track."""
        track = quick_select(clip_count=10, total_duration=60)
        assert isinstance(track, Track)
        assert track.bpm > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
