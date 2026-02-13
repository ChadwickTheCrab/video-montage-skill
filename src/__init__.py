"""Video Montage Skill - AI-powered video editing automation.

This skill analyzes video clips, selects appropriate music, and generates
editable project files for professional video editors.
"""

__version__ = "0.1.0"

# Submodules
from . import clip_analyzer
from . import fcpxml_generator
from . import music_selector
from . import timeline_builder

__all__ = [
    "clip_analyzer",
    "fcpxml_generator", 
    "music_selector",
    "timeline_builder",
]
