"""Shared fixtures and configuration for video-montage-skill tests."""

import sys
from pathlib import Path

# Ensure src is in path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture
def sample_track_data():
    """Sample track data for testing."""
    return {
        "id": "test-track-01",
        "title": "Test Track",
        "artist": "Test Artist",
        "bpm": 120,
        "duration_sec": 180,
        "genre": "Electronic",
        "mood": "Upbeat",
        "energy": "high",
        "file": "music/test.mp3",
        "tags": ["electronic", "upbeat"],
    }


@pytest.fixture
def mock_project_profile():
    """Mock project profile for testing."""
    from music_selector import ProjectProfile
    return ProjectProfile(
        clip_count=10,
        total_duration_sec=120,
        avg_clip_duration=12,
        has_slow_motion=False,
    )


@pytest.fixture
def temp_music_library(tmp_path):
    """Create a temporary music library for testing."""
    from music_selector import MusicLibrary
    
    # Create a minimal test config
    config = {
        "library_version": "test-1.0",
        "source": "Test",
        "license": "Test",
        "tracks": [
            {
                "id": "test-1",
                "title": "Fast Track",
                "artist": "Artist",
                "bpm": 140,
                "duration_sec": 180,
                "genre": "Electronic",
                "mood": "Energetic",
                "energy": "high",
                "file": "test1.mp3",
                "tags": [],
            },
            {
                "id": "test-2",
                "title": "Slow Track",
                "artist": "Artist",
                "bpm": 70,
                "duration_sec": 300,
                "genre": "Ambient",
                "mood": "Calm",
                "energy": "low",
                "file": "test2.mp3",
                "tags": [],
            },
        ],
    }
    
    config_path = tmp_path / "test_music_library.json"
    config_path.write_text(__import__("json").dumps(config))
    
    return MusicLibrary(config_path)


@pytest.fixture
def sample_video_format():
    """Sample video format for testing."""
    from fcpxml_generator import VideoFormat
    return VideoFormat(width=1920, height=1080, fps=30.0)
