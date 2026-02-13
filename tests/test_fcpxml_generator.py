"""Tests for fcpxml_generator module."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fcpxml_generator import (
    VideoFormat,
    TimelineClip,
    Transition,
    build_fcpxml,
    build_sequential_timeline_on_beats,
    write_fcpxml,
    FCPXMLError,
)


class TestVideoFormat:
    """Test VideoFormat dataclass."""
    
    def test_default_format(self):
        """Verify default video format is 1080p30."""
        fmt = VideoFormat()
        assert fmt.width == 1920
        assert fmt.height == 1080
        assert fmt.fps == 30.0
    
    def test_custom_format(self):
        """Verify custom format can be created."""
        fmt = VideoFormat(width=3840, height=2160, fps=24.0)
        assert fmt.width == 3840
        assert fmt.height == 2160
        assert fmt.fps == 24.0


class TestTimelineClip:
    """Test TimelineClip dataclass."""
    
    def test_clip_creation(self):
        """Verify TimelineClip can be created."""
        clip = TimelineClip(
            src_path=Path("/tmp/test.mp4"),
            name="Test Clip",
            offset_sec=5.0,
            duration_sec=10.0,
        )
        assert clip.src_path == Path("/tmp/test.mp4")
        assert clip.name == "Test Clip"
        assert clip.offset_sec == 5.0
        assert clip.start_sec == 0.0  # Default


class TestTransition:
    """Test Transition dataclass."""
    
    def test_default_transition(self):
        """Verify default transition is Cross Dissolve."""
        trans = Transition()
        assert trans.name == "Cross Dissolve"
        assert trans.duration_sec == 1.0
    
    def test_custom_transition(self):
        """Verify custom transition can be created."""
        trans = Transition(name="Fade", duration_sec=0.5)
        assert trans.name == "Fade"
        assert trans.duration_sec == 0.5


class TestBuildFCPXML:
    """Test FCPXML generation."""
    
    def test_xml_output_structure(self):
        """Verify generated XML has required structure."""
        fmt = VideoFormat(width=1920, height=1080, fps=30.0)
        
        # Add a clip
        clip = TimelineClip(
            src_path=Path("/tmp/test.mp4"),
            name="Test",
            offset_sec=0.0,
            duration_sec=5.0,
        )
        
        # Generate XML
        xml_string = build_fcpxml(
            timeline_clips=[clip],
            video_format=fmt,
            project_name="Test Project",
        )
        
        # Parse and verify structure
        root = ET.fromstring(xml_string)
        assert root.tag == "fcpxml"
        
        # Check for required elements
        resources = root.find(".//resources")
        assert resources is not None
        
        library = root.find(".//library")
        assert library is not None
        
        event = root.find(".//event")
        assert event is not None
        
        project = root.find(".//project")
        assert project is not None
    
    def test_asset_registration(self):
        """Verify clips are registered as assets."""
        clip1 = TimelineClip(
            src_path=Path("/tmp/clip1.mp4"),
            name="Clip 1",
            offset_sec=0.0,
            duration_sec=5.0,
        )
        clip2 = TimelineClip(
            src_path=Path("/tmp/clip2.mp4"),
            name="Clip 2",
            offset_sec=5.0,
            duration_sec=5.0,
        )
        
        xml_string = build_fcpxml(
            timeline_clips=[clip1, clip2],
            project_name="Test",
        )
        
        # Both clips should be in the XML
        assert "/tmp/clip1.mp4" in xml_string or "clip1.mp4" in xml_string
        assert "/tmp/clip2.mp4" in xml_string or "clip2.mp4" in xml_string
    
    def test_pretty_printing(self):
        """Verify XML is properly formatted."""
        clip = TimelineClip(
            src_path=Path("/tmp/test.mp4"),
            name="Test",
            offset_sec=0.0,
            duration_sec=5.0,
        )
        
        xml_string = build_fcpxml(
            timeline_clips=[clip],
            project_name="Test",
        )
        
        # Should have indentation (pretty printed)
        assert "  " in xml_string or "\t" in xml_string
    
    def test_empty_timeline_raises_error(self):
        """Verify empty timeline raises error."""
        with pytest.raises(FCPXMLError):
            build_fcpxml(timeline_clips=[], project_name="Empty")


class TestWriteFCPXML:
    """Test writing FCPXML to file."""
    
    def test_file_output(self, tmp_path):
        """Verify XML can be written to file."""
        clip = TimelineClip(
            src_path=Path("/tmp/test.mp4"),
            name="Test",
            offset_sec=0.0,
            duration_sec=5.0,
        )
        
        output_path = tmp_path / "test.fcpxml"
        write_fcpxml(
            output_path=output_path,
            timeline_clips=[clip],
            project_name="Test",
        )
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "fcpxml" in content


class TestSequentialTimelineBuilder:
    """Test the sequential timeline builder helper."""
    
    def test_builds_correct_number_of_clips(self):
        """Verify builder creates correct number of clips."""
        # Mock source files
        sources = [Path(f"/tmp/clip{i}.mp4") for i in range(5)]
        
        clips, total_dur = build_sequential_timeline_on_beats(
            source_paths=sources,
            target_duration_sec=30.0,
            bpm=120,
            fps=30.0,
        )
        
        assert len(clips) == 5
        assert total_dur > 0
    
    def test_clips_are_sequential(self):
        """Verify clips are placed sequentially without gaps."""
        sources = [Path("/tmp/a.mp4"), Path("/tmp/b.mp4")]
        
        clips, _ = build_sequential_timeline_on_beats(
            source_paths=sources,
            target_duration_sec=10.0,
            bpm=60,  # 1 beat per second
            fps=30.0,
        )
        
        # Second clip should start after first
        assert clips[1].offset_sec > clips[0].offset_sec


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_output(self, tmp_path):
        """Test complete pipeline from timeline to FCPXML file."""
        # This simulates what the main workflow would do
        from timeline_builder import TimelineBuilder
        from dataclasses import dataclass
        
        @dataclass
        class MockClip:
            file_path: Path
            duration: float
            quality_score: float
            scene_changes: list = None
            
            def __post_init__(self):
                if self.scene_changes is None:
                    self.scene_changes = []
        
        # Mock clips
        clips = [
            MockClip(Path("/tmp/clip1.mp4"), 15.0, 0.8),
            MockClip(Path("/tmp/clip2.mp4"), 15.0, 0.7),
        ]
        
        # Build timeline
        timeline_builder = TimelineBuilder()
        timeline = timeline_builder.build_timeline(clips, target_duration=30.0)
        
        # Convert to FCPXML clips
        fcpxml_clips = []
        for clip in timeline.clips:
            tc = TimelineClip(
                src_path=clip.source_path,
                name=clip.source_path.stem,
                offset_sec=clip.timeline_start,
                duration_sec=clip.source_duration,
            )
            fcpxml_clips.append(tc)
        
        # Write output
        output_path = tmp_path / "project.fcpxml"
        write_fcpxml(
            output_path=output_path,
            timeline_clips=fcpxml_clips,
            project_name="Test Montage",
        )
        
        # Verify file
        assert output_path.exists()
        content = output_path.read_text()
        assert "fcpxml" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
