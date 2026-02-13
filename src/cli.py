"""CLI entry point for video montage skill."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from clip_analyzer import analyze_clip
from music_selector import MusicLibrary, ProjectProfile
from timeline_builder import TimelineBuilder
from fcpxml_generator import write_fcpxml, TimelineClip as FCPClip
from video_renderer import render_from_timeline


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered video montage generator"
    )
    parser.add_argument(
        "clips",
        nargs="+",
        help="Input video clip paths",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for FCPXML and MP4",
    )
    parser.add_argument(
        "-m", "--music",
        type=Path,
        help="Path to music file (auto-select if not provided)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Target duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also render MP4 preview (requires FFmpeg)",
    )
    parser.add_argument(
        "--slow-mo",
        action="store_true",
        help="Input clips contain slow-motion footage",
    )
    
    args = parser.parse_args()
    
    print("🎬 Video Montage Skill")
    print(f"   Processing {len(args.clips)} clips...")
    
    # Step 1: Analyze clips
    print("\n📊 Analyzing clips...")
    analyzed = []
    for clip_path in args.clips:
        print(f"   {clip_path}...", end=" ")
        try:
            result = analyze_clip(Path(clip_path))
            analyzed.append(result)
            print(f"✓ ({result.metadata.duration_sec:.1f}s, quality: {result.blur_score:.2f})")
        except Exception as e:
            print(f"✗ {e}")
            continue
    
    if len(analyzed) == 0:
        print("❌ No clips could be analyzed")
        sys.exit(1)
    
    # Step 2: Build timeline
    print("\n🎵 Building timeline...")
    builder = TimelineBuilder()
    timeline = builder.build_timeline(
        analyzed,
        target_duration=args.duration,
        has_slow_motion=args.slow_mo,
    )
    print(f"   Selected: {timeline.music_track.title} ({timeline.music_track.bpm} BPM)")
    print(f"   {timeline.clip_count} clips, {timeline.total_duration:.1f}s total")
    
    # Step 3: Generate FCPXML
    print("\n📝 Generating FCPXML...")
    args.output.mkdir(parents=True, exist_ok=True)
    
    fcpxml_path = args.output / "montage.fcpxml"
    fcpxml_clips = [
        FCPClip(
            src_path=c.source_path,
            name=c.source_path.stem,
            offset_sec=c.timeline_start,
            duration_sec=c.source_duration,
        )
        for c in timeline.clips
    ]
    
    write_fcpxml(
        out_path=fcpxml_path,
        clips=fcpxml_clips,
        project_name="Montage",
    )
    print(f"   ✓ {fcpxml_path}")
    
    # Step 4: Render MP4 (if requested)
    if args.mp4:
        print("\n🎞️  Rendering MP4 preview...")
        if args.music:
            music_path = args.music
        else:
            # Use the selected track from library
            music_path = Path(__file__).parent.parent / "music-library" / timeline.music_track.file
            if not music_path.exists():
                print(f"   ✗ Music file not found: {music_path}")
                print("   (Provide --music path or add music files to music-library/)")
                sys.exit(1)
        
        mp4_path = args.output / "montage_preview.mp4"
        try:
            render_from_timeline(
                timeline=timeline,
                music_path=music_path,
                output_path=mp4_path,
                with_transitions=True,
            )
            print(f"   ✓ {mp4_path}")
        except Exception as e:
            print(f"   ✗ Render failed: {e}")
            print("   (FCPXML was still created - open in Resolve)")
    
    print("\n✅ Done!")
    print(f"   Output: {args.output}")
    

if __name__ == "__main__":
    main()
