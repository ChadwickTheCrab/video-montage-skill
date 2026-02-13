# Video Montage Skill 🎬

AI-powered rough-cut video editor for professional montages. Analyzes clips, syncs cuts to music beats, and generates editable DaVinci Resolve project files.

## The Problem

Your friend spends hours manually:
1. Reviewing footage to find best moments
2. Selecting music and matching cuts to beats
3. Placing clips on timeline with transitions
4. Creating the rough cut before final polish

## The Solution

**Drop clips + music in Dropbox → Receive editable Resolve project**

AI analyzes, AI selects, AI places cuts on beat markers. Human does creative final polish.

## Phase 1 MVP Features

- ✅ **Local processing** — No cloud APIs, runs on your machine
- ✅ **Static music library** — 12 curated tracks with BPM metadata
- ✅ **FFmpeg analysis** — Scene detection, blur scoring, metadata extraction
- ✅ **Beat-matched cuts** — Cuts land on downbeats automatically
- ✅ **FCPXML output** — Opens in DaVinci Resolve for final editing
- ✅ **Cross-dissolve transitions** — Smooth transitions between clips

## Architecture

```
Dropbox folder/
├── raw-clips/
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
└── music/
    └── (optional: specify mood/genre)

↓ Webhook triggers ↓

video-montage-skill/
├── clip_analyzer.py      ← FFmpeg scene detection + quality scoring
├── music_selector.py     ← BPM matching algorithm
├── timeline_builder.py   ← Cut placement on beat markers
└── fcpxml_generator.py   ← Resolve project file generator

↓ Outputs ↓

Dropbox folder/
└── output/
    ├── project.fcpxml    ← Open in Resolve
    ├── selected-music.mp3
    └── clip-notes.json   ← Quality scores, suggested segments
```

## Usage

```bash
# Install dependencies
pip install -e .

# Ensure FFmpeg is installed
ffmpeg -version

# Configure music library
cp config/music_library.example.json config/music_library.json
# Add your royalty-free tracks with BPM data

# Run analysis
python -m video_montage_skill \
    --input ~/Dropbox/montage-project/raw-clips/ \
    --output ~/Dropbox/montage-project/output/ \
    --target-duration 120 \
    --mood upbeat

# Open output/project.fcpxml in DaVinci Resolve
```

## Model Strategy

| Task | Model | Reasoning |
|------|-------|-----------|
| Planning & architecture | Kimi K2.5 | Complex reasoning, context awareness |
| FFmpeg/FCPXML coding | GPT-5.2 (Codex) | Senior-level systems programming |
| Utilities & tests | Qwen3-14B (LMStudio) | Cost-efficient for simple code |
| Decision-making | Kimi K2.5 | Requires judgment calls |

## Music Library

12 tracks spanning:
- **BPM:** 60-150 (slow ambient to fast trap)
- **Genres:** Pop, Electronic, Hip Hop, Rock, Jazz, Latin, Lo-Fi, Ambient, Funk, Corporate
- **Energy:** Low to High

All tracks must be royalty-free (Epidemic Sound, Artlist, or AI-generated via Mubert API for Phase 2).

## Cost Analysis

**Phase 1 (Local processing):**
- Music: $15/month (Epidemic Sound)
- Compute: $0 (local FFmpeg)
- Tokens: ~$5/customer/month
- **Total:** ~$5/customer/month

**Phase 2 (Cloud APIs):**
- Video Intelligence: $0.05-0.15/minute (after free tier)
- AI Music Generation: $30-50/month base
- **Total:** ~$10-20/customer/month

**Revenue potential:**
- Charge: $300-500/customer/month
- Margin: $280-490/customer
- **10 customers:** $2,800-4,900/month profit

## Roadmap

**Phase 1 (Now):**
- [x] Project structure
- [x] Music library config
- [x] Music selector module
- [x] Timeline builder module
- [ ] FFmpeg clip analyzer (GPT-5.2 coding)
- [ ] FCPXML generator (GPT-5.2 coding)
- [ ] Dropbox webhook handler
- [ ] End-to-end test

**Phase 2 (Later):**
- AI music generation (Mubert API)
- Face detection & focus scoring
- Slow-motion detection
- Multi-format export (Premiere, Final Cut)
- Web dashboard for customers

## License

MIT — This is a template for your own video editing automation business.

## Credits

Built by Pinch 🦀 (AI) + Chad (human partner)
Architecture by Kimi K2.5
Core modules by GPT-5.2
