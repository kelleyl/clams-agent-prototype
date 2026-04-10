"""
FFmpeg tools for video preprocessing.
Provides the agent with capabilities to analyze and manipulate video files.
"""

import subprocess
import json
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegTools:
    """FFmpeg-based video preprocessing tools for the CLAMS agent."""

    def __init__(self, output_dir: str = "data/ffmpeg_output"):
        """Initialize FFmpeg tools.

        Args:
            output_dir: Directory for storing output files (frames, clips, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify ffmpeg is available
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg and ffprobe are available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"FFmpeg not available: {e}")
            return False

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata from a video file using ffprobe.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video metadata (duration, resolution, codec, etc.)
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Extract key information
            format_info = data.get("format", {})
            streams = data.get("streams", [])

            # Find video and audio streams
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

            metadata = {
                "filename": os.path.basename(video_path),
                "path": video_path,
                "duration_seconds": float(format_info.get("duration", 0)),
                "duration_formatted": self._format_duration(float(format_info.get("duration", 0))),
                "size_bytes": int(format_info.get("size", 0)),
                "size_mb": round(int(format_info.get("size", 0)) / (1024 * 1024), 2),
                "format": format_info.get("format_name", "unknown"),
                "bitrate_kbps": round(int(format_info.get("bit_rate", 0)) / 1000, 2),
                "video": {
                    "codec": video_stream.get("codec_name", "unknown"),
                    "width": video_stream.get("width", 0),
                    "height": video_stream.get("height", 0),
                    "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                    "total_frames": int(video_stream.get("nb_frames", 0)),
                },
                "audio": {
                    "codec": audio_stream.get("codec_name", "unknown"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": audio_stream.get("channels", 0),
                } if audio_stream else None
            }

            return metadata

        except subprocess.CalledProcessError as e:
            return {"error": f"FFprobe failed: {e.stderr}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse ffprobe output: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def extract_frames(
        self,
        video_path: str,
        output_pattern: Optional[str] = None,
        fps: float = 1.0,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        max_frames: int = 100
    ) -> Dict[str, Any]:
        """Extract frames from a video at specified intervals.

        Args:
            video_path: Path to the video file
            output_pattern: Output filename pattern (default: {video_name}_frame_%04d.jpg)
            fps: Frames per second to extract (default: 1 frame per second)
            start_time: Start time in seconds (optional)
            duration: Duration in seconds to extract (optional)
            max_frames: Maximum number of frames to extract (default: 100)

        Returns:
            Dictionary with extraction results and list of frame paths
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            video_name = Path(video_path).stem
            frame_dir = self.output_dir / f"{video_name}_frames"
            frame_dir.mkdir(exist_ok=True)

            if output_pattern is None:
                output_pattern = str(frame_dir / f"{video_name}_frame_%04d.jpg")

            cmd = ["ffmpeg", "-y", "-i", video_path]

            if start_time is not None:
                cmd.extend(["-ss", str(start_time)])

            if duration is not None:
                cmd.extend(["-t", str(duration)])

            cmd.extend([
                "-vf", f"fps={fps}",
                "-frames:v", str(max_frames),
                "-q:v", "2",  # High quality JPEG
                output_pattern
            ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Get list of extracted frames
            frames = sorted(frame_dir.glob("*.jpg"))

            return {
                "success": True,
                "frame_count": len(frames),
                "frame_directory": str(frame_dir),
                "frames": [str(f) for f in frames[:20]],  # Return first 20 paths
                "fps_extracted": fps,
                "message": f"Extracted {len(frames)} frames at {fps} fps"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def trim_video(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trim a video to a specific segment.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path (optional, auto-generated if not provided)

        Returns:
            Dictionary with trimming results
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            video_name = Path(video_path).stem
            video_ext = Path(video_path).suffix

            if output_path is None:
                output_path = str(
                    self.output_dir / f"{video_name}_trim_{start_time}-{end_time}{video_ext}"
                )

            duration = end_time - start_time

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",  # Copy without re-encoding for speed
                output_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Get metadata of trimmed video
            trimmed_metadata = self.get_video_metadata(output_path)

            return {
                "success": True,
                "output_path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "size_mb": trimmed_metadata.get("size_mb", 0),
                "message": f"Trimmed video from {start_time}s to {end_time}s"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        format: str = "wav"
    ) -> Dict[str, Any]:
        """Extract audio track from a video file.

        Args:
            video_path: Path to the video file
            output_path: Output audio file path (optional)
            format: Audio format (wav, mp3, aac)

        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            video_name = Path(video_path).stem

            if output_path is None:
                output_path = str(self.output_dir / f"{video_name}_audio.{format}")

            codec_map = {
                "wav": ["-acodec", "pcm_s16le"],
                "mp3": ["-acodec", "libmp3lame", "-q:a", "2"],
                "aac": ["-acodec", "aac", "-b:a", "192k"],
            }

            codec_args = codec_map.get(format, codec_map["wav"])

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                *codec_args,
                output_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "success": True,
                "output_path": output_path,
                "format": format,
                "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
                "message": f"Extracted audio to {format} format"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def convert_video(
        self,
        video_path: str,
        output_format: str = "mp4",
        resolution: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert video to a different format or resolution.

        Args:
            video_path: Path to the video file
            output_format: Target format (mp4, webm, avi, mov)
            resolution: Target resolution (e.g., "1280x720", "1920x1080")
            output_path: Output file path (optional)

        Returns:
            Dictionary with conversion results
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            video_name = Path(video_path).stem

            if output_path is None:
                suffix = f"_{resolution}" if resolution else ""
                output_path = str(self.output_dir / f"{video_name}{suffix}.{output_format}")

            cmd = ["ffmpeg", "-y", "-i", video_path]

            if resolution:
                cmd.extend(["-vf", f"scale={resolution.replace('x', ':')}"])

            # Format-specific encoding settings
            if output_format == "mp4":
                cmd.extend(["-c:v", "libx264", "-preset", "medium", "-c:a", "aac"])
            elif output_format == "webm":
                cmd.extend(["-c:v", "libvpx-vp9", "-c:a", "libopus"])
            else:
                cmd.extend(["-c:v", "copy", "-c:a", "copy"])

            cmd.append(output_path)

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            output_metadata = self.get_video_metadata(output_path)

            return {
                "success": True,
                "output_path": output_path,
                "format": output_format,
                "resolution": resolution or "original",
                "size_mb": output_metadata.get("size_mb", 0),
                "message": f"Converted to {output_format}" + (f" at {resolution}" if resolution else "")
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def extract_frame_at_timestamp(
        self,
        video_path: str,
        timestamp: float,
        output_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract a single frame at a specific timestamp.

        Args:
            video_path: Path to the video file
            timestamp: Time in seconds to extract the frame
            output_path: Output file path (optional, auto-generated if not provided)
            width: Optional width to scale the frame
            height: Optional height to scale the frame

        Returns:
            Dictionary with frame path and metadata
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            import hashlib
            video_name = Path(video_path).stem

            # Generate unique frame ID based on video path and timestamp
            frame_id = hashlib.md5(f"{video_path}_{timestamp}".encode()).hexdigest()[:12]

            if output_path is None:
                frames_dir = self.output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                output_path = str(frames_dir / f"{video_name}_{frame_id}.jpg")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",  # High quality JPEG
            ]

            # Add scaling if specified
            if width and height:
                cmd.extend(["-vf", f"scale={width}:{height}"])
            elif width:
                cmd.extend(["-vf", f"scale={width}:-1"])
            elif height:
                cmd.extend(["-vf", f"scale=-1:{height}"])

            cmd.append(output_path)

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "success": True,
                "frame_id": frame_id,
                "frame_path": output_path,
                "timestamp": timestamp,
                "timestamp_formatted": self._format_duration(timestamp),
                "video_path": video_path,
                "message": f"Extracted frame at {self._format_duration(timestamp)}"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def generate_thumbnail_sprite(
        self,
        video_path: str,
        interval: float = 5.0,
        thumb_width: int = 160,
        thumb_height: int = 90,
        columns: int = 10,
        max_thumbnails: int = 100
    ) -> Dict[str, Any]:
        """Generate a sprite image containing thumbnails at regular intervals.

        Creates a single image with multiple thumbnails arranged in a grid,
        useful for timeline scrubbing previews.

        Args:
            video_path: Path to the video file
            interval: Seconds between thumbnails (default: 5)
            thumb_width: Width of each thumbnail (default: 160)
            thumb_height: Height of each thumbnail (default: 90)
            columns: Number of columns in the sprite grid (default: 10)
            max_thumbnails: Maximum number of thumbnails to generate (default: 100)

        Returns:
            Dictionary with sprite path and VTT data for thumbnail mapping
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            import hashlib
            video_name = Path(video_path).stem

            # Get video duration
            metadata = self.get_video_metadata(video_path)
            if "error" in metadata:
                return metadata

            duration = metadata.get("duration_seconds", 0)
            num_thumbs = min(int(duration / interval), max_thumbnails)

            if num_thumbs == 0:
                return {"error": "Video too short for thumbnail generation"}

            # Calculate grid dimensions
            rows = (num_thumbs + columns - 1) // columns
            sprite_width = thumb_width * columns
            sprite_height = thumb_height * rows

            # Generate unique sprite ID
            sprite_id = hashlib.md5(f"{video_path}_{interval}".encode()).hexdigest()[:12]

            sprites_dir = self.output_dir / "sprites"
            sprites_dir.mkdir(exist_ok=True)
            sprite_path = str(sprites_dir / f"{video_name}_{sprite_id}_sprite.jpg")
            vtt_path = str(sprites_dir / f"{video_name}_{sprite_id}.vtt")

            # FFmpeg command to generate sprite
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"fps=1/{interval},scale={thumb_width}:{thumb_height},tile={columns}x{rows}",
                "-frames:v", "1",
                "-q:v", "5",
                sprite_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Generate VTT file for thumbnail mapping
            vtt_content = ["WEBVTT", ""]
            for i in range(num_thumbs):
                start_time = i * interval
                end_time = min((i + 1) * interval, duration)

                # Calculate position in sprite
                col = i % columns
                row = i // columns
                x = col * thumb_width
                y = row * thumb_height

                vtt_content.append(
                    f"{self._format_vtt_time(start_time)} --> {self._format_vtt_time(end_time)}"
                )
                vtt_content.append(
                    f"{Path(sprite_path).name}#xywh={x},{y},{thumb_width},{thumb_height}"
                )
                vtt_content.append("")

            with open(vtt_path, 'w') as f:
                f.write("\n".join(vtt_content))

            return {
                "success": True,
                "sprite_id": sprite_id,
                "sprite_path": sprite_path,
                "vtt_path": vtt_path,
                "thumbnail_count": num_thumbs,
                "interval": interval,
                "sprite_dimensions": {"width": sprite_width, "height": sprite_height},
                "thumbnail_dimensions": {"width": thumb_width, "height": thumb_height},
                "video_duration": duration,
                "message": f"Generated {num_thumbs} thumbnails in sprite"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None,
        max_width: int = 640
    ) -> Dict[str, Any]:
        """Extract a short video clip for preview/review.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path (optional)
            max_width: Maximum width for the clip (default: 640)

        Returns:
            Dictionary with clip path and metadata
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        try:
            import hashlib
            video_name = Path(video_path).stem

            # Generate unique clip ID
            clip_id = hashlib.md5(f"{video_path}_{start_time}_{end_time}".encode()).hexdigest()[:12]

            clips_dir = self.output_dir / "clips"
            clips_dir.mkdir(exist_ok=True)

            if output_path is None:
                output_path = str(clips_dir / f"{video_name}_{clip_id}.mp4")

            duration = end_time - start_time

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-vf", f"scale='min({max_width},iw)':-2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                output_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "success": True,
                "clip_id": clip_id,
                "clip_path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
                "message": f"Extracted {duration:.1f}s clip from {self._format_duration(start_time)} to {self._format_duration(end_time)}"
            }

        except subprocess.CalledProcessError as e:
            return {"error": f"FFmpeg failed: {e.stderr}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT file (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS from fraction string (e.g., '30000/1001')."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return round(float(num) / float(den), 2)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 0.0


def create_ffmpeg_langchain_tools(ffmpeg_tools: FFmpegTools):
    """Create LangChain tools from FFmpegTools instance.

    Args:
        ffmpeg_tools: An initialized FFmpegTools instance

    Returns:
        List of LangChain tool functions
    """
    from langchain_core.tools import tool

    @tool("get_video_info")
    def get_video_info(video_path: str) -> str:
        """Get metadata and information about a video file.

        Use this to learn about a video's duration, resolution, format,
        codec, and other properties before processing it.

        Args:
            video_path: Path to the video file

        Returns:
            JSON string with video metadata
        """
        result = ffmpeg_tools.get_video_metadata(video_path)
        return json.dumps(result, indent=2)

    @tool("extract_video_frames")
    def extract_video_frames(
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 50
    ) -> str:
        """Extract frames from a video at regular intervals.

        Useful for preparing video content for image-based analysis tools
        like OCR, object detection, or scene classification.

        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (default: 1 = one frame per second)
            max_frames: Maximum number of frames to extract (default: 50)

        Returns:
            JSON string with extraction results and frame paths
        """
        result = ffmpeg_tools.extract_frames(
            video_path,
            fps=fps,
            max_frames=max_frames
        )
        return json.dumps(result, indent=2)

    @tool("trim_video_segment")
    def trim_video_segment(
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> str:
        """Trim a video to extract a specific time segment.

        Use this to isolate a portion of a video for focused analysis,
        such as extracting a scene or removing irrelevant content.

        Args:
            video_path: Path to the video file
            start_seconds: Start time in seconds
            end_seconds: End time in seconds

        Returns:
            JSON string with trimmed video path and metadata
        """
        result = ffmpeg_tools.trim_video(video_path, start_seconds, end_seconds)
        return json.dumps(result, indent=2)

    @tool("extract_audio_track")
    def extract_audio_track(video_path: str, format: str = "wav") -> str:
        """Extract the audio track from a video file.

        Useful for preparing audio for speech transcription tools like
        whisper-wrapper without the video overhead.

        Args:
            video_path: Path to the video file
            format: Audio format - 'wav' (best for ASR), 'mp3', or 'aac'

        Returns:
            JSON string with audio file path and metadata
        """
        result = ffmpeg_tools.extract_audio(video_path, format=format)
        return json.dumps(result, indent=2)

    return [
        get_video_info,
        extract_video_frames,
        trim_video_segment,
        extract_audio_track
    ]
