"""
Frame visualization and human-in-the-loop review system.
Provides APIs for frame extraction, serving, and annotation collection.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .ffmpeg_tools import FFmpegTools

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a frame review."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    SKIPPED = "skipped"


@dataclass
class FrameData:
    """Data for a single video frame."""
    frame_id: str
    video_path: str
    timestamp_ms: int
    frame_path: str
    timestamp_formatted: str = ""
    detected_text: Optional[str] = None
    detected_entities: List[str] = field(default_factory=list)
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp_formatted:
            secs = self.timestamp_ms / 1000
            mins = int(secs // 60)
            secs = int(secs % 60)
            self.timestamp_formatted = f"{mins:02d}:{secs:02d}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameReview:
    """A human review of a frame."""
    frame_id: str
    status: ReviewStatus
    reviewer_comment: str = ""
    corrected_text: Optional[str] = None
    corrected_entities: Optional[List[str]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class FrameReviewSession:
    """A session for reviewing multiple frames."""
    session_id: str
    video_path: str
    frames: List[FrameData] = field(default_factory=list)
    reviews: Dict[str, FrameReview] = field(default_factory=dict)
    current_index: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed: bool = False
    source_mmif: Optional[Dict[str, Any]] = None

    def get_current_frame(self) -> Optional[FrameData]:
        if 0 <= self.current_index < len(self.frames):
            return self.frames[self.current_index]
        return None

    def get_pending_frames(self) -> List[FrameData]:
        return [f for f in self.frames if f.frame_id not in self.reviews]

    def get_review_summary(self) -> Dict[str, Any]:
        status_counts = {}
        for review in self.reviews.values():
            status = review.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_frames": len(self.frames),
            "reviewed": len(self.reviews),
            "pending": len(self.frames) - len(self.reviews),
            "by_status": status_counts
        }


class FrameReviewManager:
    """
    Manages frame extraction, serving, and review collection.
    """

    def __init__(self, output_dir: str = "data/frame_reviews"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = FFmpegTools(output_dir=str(self.output_dir / "frames"))
        self.sessions: Dict[str, FrameReviewSession] = {}

    def create_review_session(
        self,
        session_id: str,
        video_path: str,
        timestamps_ms: Optional[List[int]] = None,
        interval_seconds: float = 5.0,
        max_frames: int = 50,
        detected_content: Optional[Dict[str, Any]] = None
    ) -> FrameReviewSession:
        """
        Create a new frame review session.

        Args:
            session_id: Unique session identifier
            video_path: Path to the video file
            timestamps_ms: Specific timestamps to extract (optional)
            interval_seconds: Interval for automatic extraction if no timestamps
            max_frames: Maximum number of frames to extract
            detected_content: Optional dict with detected text/entities per timestamp

        Returns:
            FrameReviewSession with extracted frames
        """
        frames = []
        detected_content = detected_content or {}

        if timestamps_ms:
            # Extract frames at specific timestamps
            for ts in timestamps_ms[:max_frames]:
                result = self.ffmpeg.extract_frame_at_timestamp(
                    video_path,
                    timestamp=ts / 1000.0
                )
                if result.get('success'):
                    content = detected_content.get(str(ts), {})
                    frames.append(FrameData(
                        frame_id=result['frame_id'],
                        video_path=video_path,
                        timestamp_ms=ts,
                        frame_path=result['frame_path'],
                        detected_text=content.get('text'),
                        detected_entities=content.get('entities', []),
                        context=content.get('context', ''),
                    ))
        else:
            # Extract frames at regular intervals
            metadata = self.ffmpeg.get_video_metadata(video_path)
            if 'error' not in metadata:
                duration = metadata.get('duration_seconds', 0)
                num_frames = min(int(duration / interval_seconds), max_frames)

                for i in range(num_frames):
                    ts = i * interval_seconds
                    ts_ms = int(ts * 1000)
                    result = self.ffmpeg.extract_frame_at_timestamp(
                        video_path,
                        timestamp=ts
                    )
                    if result.get('success'):
                        content = detected_content.get(str(ts_ms), {})
                        frames.append(FrameData(
                            frame_id=result['frame_id'],
                            video_path=video_path,
                            timestamp_ms=ts_ms,
                            frame_path=result['frame_path'],
                            detected_text=content.get('text'),
                            detected_entities=content.get('entities', []),
                            context=content.get('context', ''),
                        ))

        session = FrameReviewSession(
            session_id=session_id,
            video_path=video_path,
            frames=frames
        )
        self.sessions[session_id] = session
        logger.info(f"Created review session {session_id} with {len(frames)} frames")

        return session

    def create_review_session_from_mmif(
        self,
        session_id: str,
        video_path: str,
        mmif_output: Dict[str, Any],
        max_frames: int = 50
    ) -> FrameReviewSession:
        """
        Create a review session from CLAMS MMIF output.

        Extracts frames at detected timeframes and associates detected content.

        Args:
            session_id: Unique session identifier
            video_path: Path to the video file
            mmif_output: MMIF output from CLAMS apps
            max_frames: Maximum frames to extract

        Returns:
            FrameReviewSession with frames at detected locations
        """
        timestamps_ms = []
        detected_content = {}
        # Map timestamp_ms -> source annotation metadata for MMIF provenance
        source_metadata = {}

        # Extract timeframes and associated content from MMIF
        for view in mmif_output.get('views', []):
            view_id = view.get('id', '')
            view_app = view.get('metadata', {}).get('app', '')
            for ann in view.get('annotations', []):
                ann_type = ann.get('@type', '')
                ann_id = ann.get('id', '')
                props = ann.get('properties', {})

                # Get timestamp from TimeFrame annotations
                if 'TimeFrame' in ann_type:
                    start_ms = props.get('start', 0)
                    end_ms = props.get('end', start_ms)
                    mid_ms = (start_ms + end_ms) // 2

                    if mid_ms not in timestamps_ms:
                        timestamps_ms.append(mid_ms)
                        detected_content[str(mid_ms)] = {
                            'context': f"Detected: {props.get('label', 'unknown')}",
                            'label': props.get('label'),
                        }
                        source_metadata[str(mid_ms)] = {
                            'source_annotation_id': ann_id,
                            'source_view_id': view_id,
                            'source_label': props.get('label'),
                            'source_app': view_app,
                        }

                # Associate text with closest timestamp
                if 'TextDocument' in ann_type or 'Alignment' in ann_type:
                    text = props.get('text', '')
                    ts = props.get('start', props.get('timestamp', 0))
                    ts_key = str(ts)
                    if ts_key in detected_content:
                        detected_content[ts_key]['text'] = text
                    elif text:
                        # Find closest timestamp
                        closest = min(timestamps_ms, key=lambda x: abs(x - ts), default=ts)
                        if str(closest) in detected_content:
                            detected_content[str(closest)]['text'] = text

        # Sort and limit timestamps
        timestamps_ms = sorted(set(timestamps_ms))[:max_frames]

        session = self.create_review_session(
            session_id=session_id,
            video_path=video_path,
            timestamps_ms=timestamps_ms,
            detected_content=detected_content
        )

        # Attach source annotation metadata to each frame
        for frame in session.frames:
            ts_key = str(frame.timestamp_ms)
            if ts_key in source_metadata:
                frame.metadata.update(source_metadata[ts_key])

        # Store the original MMIF for later view injection
        session.source_mmif = mmif_output

        return session

    def get_session(self, session_id: str) -> Optional[FrameReviewSession]:
        """Get an existing review session."""
        return self.sessions.get(session_id)

    def get_frame(self, session_id: str, frame_id: str) -> Optional[FrameData]:
        """Get a specific frame from a session."""
        session = self.sessions.get(session_id)
        if session:
            for frame in session.frames:
                if frame.frame_id == frame_id:
                    return frame
        return None

    def get_frame_image_path(self, session_id: str, frame_id: str) -> Optional[str]:
        """Get the file path for a frame image."""
        frame = self.get_frame(session_id, frame_id)
        if frame and os.path.exists(frame.frame_path):
            return frame.frame_path
        return None

    def submit_review(
        self,
        session_id: str,
        frame_id: str,
        status: str,
        comment: str = "",
        corrected_text: Optional[str] = None,
        corrected_entities: Optional[List[str]] = None
    ) -> Optional[FrameReview]:
        """
        Submit a review for a frame.

        Args:
            session_id: Session identifier
            frame_id: Frame identifier
            status: Review status (approved, rejected, corrected, skipped)
            comment: Optional reviewer comment
            corrected_text: Corrected text if status is 'corrected'
            corrected_entities: Corrected entities if status is 'corrected'

        Returns:
            FrameReview if successful, None otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Validate frame exists
        frame = None
        for f in session.frames:
            if f.frame_id == frame_id:
                frame = f
                break

        if not frame:
            logger.warning(f"Frame not found: {frame_id}")
            return None

        # Create review
        try:
            review_status = ReviewStatus(status)
        except ValueError:
            review_status = ReviewStatus.SKIPPED

        review = FrameReview(
            frame_id=frame_id,
            status=review_status,
            reviewer_comment=comment,
            corrected_text=corrected_text,
            corrected_entities=corrected_entities
        )

        session.reviews[frame_id] = review

        # Update session index
        session.current_index = min(
            session.current_index + 1,
            len(session.frames) - 1
        )

        # Check if session is complete
        if len(session.reviews) == len(session.frames):
            session.completed = True

        logger.info(f"Review submitted for frame {frame_id}: {status}")
        return review

    def get_next_frame_for_review(
        self,
        session_id: str
    ) -> Optional[FrameData]:
        """Get the next frame that needs review."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        pending = session.get_pending_frames()
        return pending[0] if pending else None

    def export_reviews(self, session_id: str) -> Dict[str, Any]:
        """Export all reviews for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        return {
            "session_id": session_id,
            "video_path": session.video_path,
            "created_at": session.created_at,
            "summary": session.get_review_summary(),
            "frames": [f.to_dict() for f in session.frames],
            "reviews": {k: v.to_dict() for k, v in session.reviews.items()}
        }

    def get_corrections(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all corrections made during review."""
        session = self.sessions.get(session_id)
        if not session:
            return []

        corrections = []
        for frame in session.frames:
            review = session.reviews.get(frame.frame_id)
            if review and review.status == ReviewStatus.CORRECTED:
                corrections.append({
                    "frame_id": frame.frame_id,
                    "timestamp_ms": frame.timestamp_ms,
                    "original_text": frame.detected_text,
                    "corrected_text": review.corrected_text,
                    "original_entities": frame.detected_entities,
                    "corrected_entities": review.corrected_entities,
                    "comment": review.reviewer_comment
                })

        return corrections

    def reviews_to_mmif_view(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Build a CLAMS-style MMIF view from completed reviews.

        Produces both review status annotations (provenance) and corrected
        TextDocument annotations (for downstream consumption).

        Returns:
            MMIF view dict, or None if session not found
        """
        session = self.sessions.get(session_id)
        if not session or not session.reviews:
            return None

        # Determine view ID: increment past existing views in source MMIF
        existing_views = (session.source_mmif or {}).get('views', [])
        view_id = f"v_{len(existing_views)}"

        annotations = []
        ann_counter = 0

        for frame in session.frames:
            review = session.reviews.get(frame.frame_id)
            if not review:
                continue

            source_view_id = frame.metadata.get('source_view_id', '')
            source_ann_id = frame.metadata.get('source_annotation_id', '')
            source_ref = f"{source_view_id}:{source_ann_id}" if source_view_id and source_ann_id else None

            # 1. Review status annotation (always)
            review_ann = {
                "id": f"hr_{ann_counter}",
                "@type": "http://mmif.clams.ai/vocabulary/Annotation/v1",
                "properties": {
                    "review_status": review.status.value,
                    "timestamp_ms": frame.timestamp_ms,
                }
            }
            if source_ref:
                review_ann["properties"]["source_annotation"] = source_ref
            if review.reviewer_comment:
                review_ann["properties"]["reviewer_comment"] = review.reviewer_comment
            annotations.append(review_ann)
            ann_counter += 1

            # 2. Corrected TextDocument (only for corrections with text)
            if review.status == ReviewStatus.CORRECTED and review.corrected_text:
                text_ann = {
                    "id": f"hr_{ann_counter}",
                    "@type": "http://mmif.clams.ai/vocabulary/TextDocument/v1",
                    "properties": {
                        "text": {"@value": review.corrected_text},
                        "timestamp_ms": frame.timestamp_ms,
                    }
                }
                if source_ref:
                    text_ann["properties"]["source_annotation"] = source_ref
                annotations.append(text_ann)
                ann_counter += 1

        view = {
            "id": view_id,
            "metadata": {
                "app": "http://apps.clams.ai/human-review/v1",
                "timestamp": datetime.utcnow().isoformat(),
                "contains": {
                    "http://mmif.clams.ai/vocabulary/Annotation/v1": {
                        "description": "Human review status for source annotations"
                    },
                    "http://mmif.clams.ai/vocabulary/TextDocument/v1": {
                        "description": "Human-corrected text from reviewed frames"
                    }
                },
                "parameters": {
                    "reviewer": "human",
                    "session_id": session_id
                }
            },
            "annotations": annotations
        }

        return view

    def export_reviewed_mmif(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export the full MMIF document with a human review view appended.

        Args:
            session_id: The review session ID

        Returns:
            Complete MMIF dict with the human review view, or None if
            session not found or no source MMIF stored.
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
        if not session.source_mmif:
            logger.warning(f"No source MMIF for session: {session_id}")
            return None

        view = self.reviews_to_mmif_view(session_id)
        if not view:
            return None

        import copy
        mmif = copy.deepcopy(session.source_mmif)
        mmif.setdefault('views', []).append(view)
        return mmif


class HumanInTheLoopAgent:
    """
    Integrates human review into the agent workflow.
    Provides methods for the agent to pause and request human input.
    """

    def __init__(self, review_manager: FrameReviewManager):
        self.review_manager = review_manager
        self.pending_reviews: Dict[str, asyncio.Event] = {}
        self.review_results: Dict[str, Dict[str, Any]] = {}

    async def request_frame_review(
        self,
        session_id: str,
        video_path: str,
        mmif_output: Dict[str, Any],
        timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """
        Request human review of frames and wait for completion.

        This is an async method that the agent can await to pause
        execution until human review is complete.

        Args:
            session_id: Session identifier
            video_path: Path to the video
            mmif_output: MMIF output with detected content
            timeout_seconds: How long to wait for review

        Returns:
            Dict with review results and corrections
        """
        # Create review session
        review_session = self.review_manager.create_review_session_from_mmif(
            session_id=session_id,
            video_path=video_path,
            mmif_output=mmif_output
        )

        if not review_session.frames:
            return {
                "status": "no_frames",
                "message": "No frames to review"
            }

        # Create event for this review
        event = asyncio.Event()
        self.pending_reviews[session_id] = event

        # Wait for review to complete or timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Review timeout for session {session_id}")
            return {
                "status": "timeout",
                "message": f"Review timed out after {timeout_seconds} seconds",
                "partial_results": self.review_manager.export_reviews(session_id)
            }
        finally:
            del self.pending_reviews[session_id]

        # Return results
        return {
            "status": "completed",
            "summary": review_session.get_review_summary(),
            "corrections": self.review_manager.get_corrections(session_id)
        }

    def complete_review(self, session_id: str):
        """Signal that review is complete for a session."""
        if session_id in self.pending_reviews:
            self.pending_reviews[session_id].set()



# Singleton instances
_frame_review_manager = None
_hitl_agent = None


def get_frame_review_manager() -> FrameReviewManager:
    """Get or create the FrameReviewManager singleton."""
    global _frame_review_manager
    if _frame_review_manager is None:
        _frame_review_manager = FrameReviewManager()
    return _frame_review_manager


def get_hitl_agent() -> HumanInTheLoopAgent:
    """Get or create the HumanInTheLoopAgent singleton."""
    global _hitl_agent
    if _hitl_agent is None:
        _hitl_agent = HumanInTheLoopAgent(get_frame_review_manager())
    return _hitl_agent
