# Video Montage Skill - Project Plan

## Overview
AI-powered rough-cut video editor that analyzes clips, syncs cuts to music beats, and generates editable project files (FCPXML for DaVinci Resolve/Final Cut Pro/Adobe Premiere).

## Phase 1 MVP Scope
- Local processing only (no cloud APIs)
- Static music library with known BPM metadata
- FFmpeg for video analysis (scene detection, quality scoring)
- FCPXML generation for DaVinci Resolve
- Dropbox webhook integration

## Architecture

```
Dropbox webhook → video-montage-skill
                    ↓
            clip_analyzer.py (FFmpeg)
                    ↓
            music_selector.py (BPM matching)
                    ↓
            timeline_builder.py (cut placement)
                    ↓
            fcpxml_generator.py
                    ↓
            Upload to Dropbox
```

## Model Strategy

| Task | Model | Reasoning |
|------|-------|-----------|
| Architecture, planning, coordination | Kimi K2.5 (current) | Complex reasoning, context awareness |
| FCPXML generation, FFmpeg integration | GPT-5.2 (openrouter/gpt-5.2) | Senior coding, complex file formats |
| Utility scripts, tests, simple helpers | Qwen3-14B (lmstudio) | Cost-efficient, 30k context for simple tasks |
| Decision-making, edge cases | Kimi K2.5 | Requires judgment calls |

## Key Files to Build

1. `config/music_library.json` - Music tracks with BPM/duration metadata
2. `src/clip_analyzer.py` - FFmpeg scene detection + quality scoring
3. `src/music_selector.py` - BPM matching algorithm
4. `src/timeline_builder.py` - Cut placement logic
5. `src/fcpxml_generator.py` - DaVinci Resolve project file generator
6. `src/dropbox_handler.py` - Webhook + file upload
7. `test-data/sample-clips/` - Sample videos for testing

## First Task: Create Music Library Config
Build a JSON catalog of 10-20 royalty-free tracks with:
- File path
- BPM
- Duration
- Genre/Mood
- Energy level

Use tracks from Epidemic Sound/Artlist with known BPM metadata.
