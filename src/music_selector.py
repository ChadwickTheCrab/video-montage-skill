"""Music selector module for video montage skill.

Matches video projects to appropriate music tracks based on
clip count, duration, energy level, and BPM requirements.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Track:
    """Represents a music track from the library."""
    id: str
    title: str
    artist: str
    bpm: int
    duration_sec: int
    duration_display: str = ""  # Human-readable duration like "3:00"
    genre: str = ""
    mood: str = ""
    energy: str = "medium"
    file: str = ""
    tags: list = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def beat_duration(self) -> float:
        """Duration of one beat in seconds."""
        return 60.0 / self.bpm
    
    @property
    def total_beats(self) -> int:
        """Total number of beats in the track."""
        return int(self.duration_sec / self.beat_duration)


@dataclass
class ProjectProfile:
    """Profile of a video project for music matching."""
    clip_count: int
    total_duration_sec: float
    avg_clip_duration: float
    has_slow_motion: bool = False
    mood_preference: Optional[str] = None
    genre_preference: Optional[str] = None


class MusicLibrary:
    """Manages the music track library and selection logic."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Load music library from JSON config.
        
        Args:
            config_path: Path to music_library.json. Defaults to skill config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "music_library.json"
        
        with open(config_path) as f:
            data = json.load(f)
        
        self.tracks = [Track(**t) for t in data["tracks"]]
        self.selection_rules = data["selection_rules"]
        self.version = data["library_version"]
    
    def select_track(
        self,
        profile: ProjectProfile,
        exclude_ids: Optional[list] = None
    ) -> Track:
        """Select the best music track for a project.
        
        Args:
            profile: Project characteristics
            exclude_ids: Track IDs to exclude (already used)
            
        Returns:
            Best matching Track
            
        Raises:
            ValueError: If no suitable track found
        """
        exclude_ids = exclude_ids or []
        candidates = [t for t in self.tracks if t.id not in exclude_ids]
        
        # Filter by duration - track must be long enough
        candidates = [
            t for t in candidates 
            if t.duration_sec >= profile.total_duration_sec * 0.8
        ]
        
        if not candidates:
            raise ValueError(
                f"No track long enough for {profile.total_duration_sec}s project"
            )
        
        # Score each candidate
        scored = [(t, self._score_track(t, profile)) for t in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match, or random from top 3 if tied
        top_score = scored[0][1]
        top_candidates = [t for t, s in scored if s == top_score]
        return random.choice(top_candidates) if len(top_candidates) > 1 else top_candidates[0]
    
    def _score_track(self, track: Track, profile: ProjectProfile) -> float:
        """Score a track match (0-100). Higher is better."""
        score = 50.0  # Base score
        
        # BPM matching based on clip density
        optimal_bpm = self._estimate_optimal_bpm(profile)
        bpm_diff = abs(track.bpm - optimal_bpm)
        bpm_score = max(0, 30 - bpm_diff * 0.5)  # Penalty for BPM mismatch
        score += bpm_score
        
        # Energy matching
        if profile.clip_count >= 20 or profile.has_slow_motion:
            if track.energy == "high":
                score += 10
            elif track.energy == "low":
                score -= 10
        elif profile.clip_count <= 10:
            if track.energy == "low":
                score += 10
            elif track.energy == "high":
                score -= 5
        
        # Genre preference
        if profile.genre_preference and profile.genre_preference.lower() in track.genre.lower():
            score += 10
        
        # Mood preference
        if profile.mood_preference and profile.mood_preference.lower() in track.mood.lower():
            score += 10
        
        # Duration efficiency - prefer tracks that fit without excessive trimming
        duration_ratio = profile.total_duration_sec / track.duration_sec
        if 0.7 <= duration_ratio <= 1.0:
            score += 5  # Good fit
        elif duration_ratio < 0.5:
            score -= 5  # Will waste most of the track
        
        return score
    
    def _estimate_optimal_bpm(self, profile: ProjectProfile) -> int:
        """Estimate optimal BPM based on project characteristics."""
        # More clips = faster cuts = higher BPM
        if profile.clip_count >= 30:
            return 140
        elif profile.clip_count >= 20:
            return 128
        elif profile.clip_count >= 10:
            return 110
        elif profile.clip_count >= 5:
            return 95
        else:
            return 85
    
    def get_track_by_id(self, track_id: str) -> Optional[Track]:
        """Get a specific track by ID."""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
    
    def list_genres(self) -> list:
        """Return list of available genres."""
        return sorted(set(t.genre for t in self.tracks))
    
    def list_by_energy(self, energy: str) -> list:
        """Return tracks matching energy level."""
        return [t for t in self.tracks if t.energy == energy]


def calculate_cut_timing(
    track: Track,
    clip_count: int,
    total_duration: float,
    cuts_per_beat: int = 1
) -> list:
    """Calculate optimal cut times based on track BPM.
    
    Args:
        track: Selected music track
        clip_count: Number of clips to place
        total_duration: Target total duration
        cuts_per_beat: How many cuts per beat (1=on beat, 2=eighth notes, etc.)
        
    Returns:
        List of cut times in seconds
    """
    beat_duration = track.beat_duration
    cut_interval = beat_duration / cuts_per_beat
    
    # Distribute cuts evenly across the duration
    # Start with a small offset (first beat) for musical feel
    first_cut = beat_duration  # Start on first downbeat
    
    if clip_count == 1:
        return [first_cut]
    
    # Space remaining cuts evenly, leaving room for the last clip
    MIN_LAST_CLIP_DURATION = 4.0  # seconds — ensure the final clip has room
    usable_duration = total_duration - first_cut - MIN_LAST_CLIP_DURATION
    cut_times = [first_cut]
    
    for i in range(1, clip_count):
        # Snap to nearest beat subdivision
        ideal_time = first_cut + (i * (usable_duration / (clip_count - 1)))
        beat_number = ideal_time / cut_interval
        snapped_beat = round(beat_number)
        snapped_time = snapped_beat * cut_interval
        
        # Ensure we don't exceed track duration and leave room for last clip
        if snapped_time < track.duration_sec - 0.5 and snapped_time < total_duration - MIN_LAST_CLIP_DURATION:
            cut_times.append(snapped_time)
    
    return cut_times


# Convenience function for quick selection
def quick_select(
    clip_count: int,
    total_duration: float,
    has_slow_motion: bool = False
) -> Track:
    """Quick music selection without full ProjectProfile."""
    library = MusicLibrary()
    profile = ProjectProfile(
        clip_count=clip_count,
        total_duration_sec=total_duration,
        avg_clip_duration=total_duration / clip_count if clip_count > 0 else 0,
        has_slow_motion=has_slow_motion
    )
    return library.select_track(profile)


if __name__ == "__main__":
    # Quick test
    lib = MusicLibrary()
    print(f"Loaded {len(lib.tracks)} tracks")
    print(f"Genres: {lib.list_genres()}")
    
    # Test selection
    profile = ProjectProfile(
        clip_count=15,
        total_duration_sec=120,
        avg_clip_duration=8,
        has_slow_motion=True
    )
    
    track = lib.select_track(profile)
    print(f"\nSelected: {track.title} ({track.bpm} BPM)")
    print(f"Beat duration: {track.beat_duration:.3f}s")
    
    cuts = calculate_cut_timing(track, 15, 120)
    print(f"Cut times: {[f'{c:.2f}' for c in cuts[:5]]}... ({len(cuts)} total)")
