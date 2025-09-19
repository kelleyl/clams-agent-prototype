"""
AG-UI integration for CLAMS Agent Prototype.
Provides event-driven communication between frontend and LangGraph agent using the official AG-UI protocol.
"""

import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Official AG-UI Python SDK imports
from ag_ui.core import (
    RunAgentInput, Message, Context, Tool, State,
    # Event types (these should be available based on the TypeScript version)
)

# Define AG-UI event types based on the protocol
class AGUIEventType(Enum):
    # Lifecycle Events
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    
    # Text Message Events
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"  
    TEXT_MESSAGE_CHUNK = "text_message_chunk"
    TEXT_MESSAGE_END = "text_message_end"
    
    # Tool Call Events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_CALL_CHUNK = "tool_call_chunk"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_RESULT = "tool_call_result"
    
    # Thinking Events
    THINKING_START = "thinking_start"
    THINKING_TEXT_MESSAGE_START = "thinking_text_message_start"
    THINKING_TEXT_MESSAGE_CONTENT = "thinking_text_message_content"
    THINKING_TEXT_MESSAGE_END = "thinking_text_message_end"
    THINKING_END = "thinking_end"
    
    # State Events
    STATE_DELTA = "state_delta"
    STATE_SNAPSHOT = "state_snapshot"
    
    # Message Events
    MESSAGES_SNAPSHOT = "messages_snapshot"
    
    # Custom Events
    RAW_EVENT = "raw_event"
    CUSTOM_EVENT = "custom_event"

@dataclass
class AGUIEvent:
    """AG-UI Protocol Event"""
    type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None
    session_id: str = "default"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class AGUIEventEncoder:
    """Encoder for AG-UI events following the protocol specification"""
    
    @staticmethod
    def encode_event(event: AGUIEvent) -> str:
        """Encode AG-UI event to JSON string"""
        return json.dumps(asdict(event))
    
    @staticmethod
    def decode_event(event_data: str) -> AGUIEvent:
        """Decode JSON string to AG-UI event"""
        data = json.loads(event_data)
        return AGUIEvent(**data)
    
    @staticmethod
    def encode_sse(event: AGUIEvent) -> str:
        """Encode event as Server-Sent Event"""
        return f"data: {AGUIEventEncoder.encode_event(event)}\n\n"

from .langgraph_agent import CLAMSAgent, StreamingUpdate
from .pipeline_model import PipelineModel

# Configure logging
logger = logging.getLogger(__name__)


class AGUIEventHandler:
    """
    Handles AG-UI events for CLAMS pipeline generation.
    Bridges between AG-UI protocol and LangGraph agent.
    """
    
    def __init__(self, agent: CLAMSAgent):
        """
        Initialize the AG-UI event handler.
        
        Args:
            agent: LangGraph CLAMS agent instance
        """
        self.agent = agent
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.encoder = AGUIEventEncoder()
    
    async def handle_event(self, event: AGUIEvent) -> AsyncGenerator[AGUIEvent, None]:
        """
        Handle incoming AG-UI events and generate response events.
        
        Args:
            event: Incoming AG-UI event
            
        Yields:
            AG-UI response events
        """
        try:
            session_id = event.session_id
            
            # Initialize session if needed
            if session_id not in self.active_sessions:
                logger.info(f"[{session_id}] NEW_SESSION: Initializing new chat session")
                self.active_sessions[session_id] = {
                    "task_description": "",
                    "conversation_history": [],
                    "pipeline": PipelineModel(name=f"Pipeline-{session_id}"),
                    "created_at": datetime.utcnow().isoformat(),
                    "welcome_sent": False
                }
                logger.info(f"[{session_id}] SESSION_COUNT: Total active sessions: {len(self.active_sessions)}")
            
            session = self.active_sessions[session_id]
            
            # Route event based on AG-UI protocol types
            if event.type == "user_message":
                async for response_event in self._handle_user_message(event, session):
                    yield response_event
                    
            elif event.type == "session_start":
                # Session start - just acknowledge, welcome message will be sent with first user message
                yield AGUIEvent(
                    type=AGUIEventType.CUSTOM_EVENT.value,
                    data={"message": "Session started successfully"},
                    session_id=session_id
                )
                    
            elif event.type == "validation_request":
                async for response_event in self._handle_validation_request(event, session):
                    yield response_event
                    
            elif event.type == "human_feedback":
                async for response_event in self._handle_human_feedback(event, session):
                    yield response_event
                    
            else:
                logger.warning(f"Unhandled event type: {event.type}")
                yield AGUIEvent(
                    type=AGUIEventType.RAW_EVENT.value,
                    data={"error": f"Unhandled event type: {event.type}"},
                    session_id=session_id
                )
                
        except Exception as e:
            logger.error(f"Error handling AG-UI event: {e}")
            yield AGUIEvent(
                type=AGUIEventType.RUN_ERROR.value,
                data={"error": str(e), "event_type": event.type},
                session_id=event.session_id
            )
    
    async def _handle_user_message(self, event: AGUIEvent, session: Dict[str, Any]) -> AsyncGenerator[AGUIEvent, None]:
        """Handle user message events."""
        user_message = event.data.get("message", "")
        task_description = event.data.get("task_description", session.get("task_description", ""))
        
        # Log user message for debugging
        logger.info(f"[{event.session_id}] USER_MESSAGE: '{user_message}' (task: '{task_description}')")
        
        # Send welcome message if this is the first user message
        if not session.get("welcome_sent", False):
            welcome_message = (
                "👋 Hello! I'm your CLAMS pipeline assistant. I can help you create multimedia analysis pipelines using CLAMS tools.\n\n"
                "**What would you like to do?**\n"
                "• Analyze video content (chyrons, scenes, objects)\n"
                "• Process audio (transcription, speech detection)\n"
                "• Extract text from images (OCR)\n"
                "• Create custom analysis pipelines\n\n"
                "Just describe what you'd like to analyze and I'll suggest the right tools!"
            )
            
            yield AGUIEvent(
                type=AGUIEventType.TEXT_MESSAGE_CONTENT.value,
                data={"content": welcome_message},
                session_id=event.session_id
            )
            
            session["welcome_sent"] = True
            session["conversation_history"].append({
                "role": "assistant",
                "content": welcome_message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"[{event.session_id}] WELCOME_SENT: Initial greeting message delivered")
        
        # Update session
        session["task_description"] = task_description
        session["conversation_history"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Signal run start
        yield AGUIEvent(
            type=AGUIEventType.RUN_STARTED.value,
            data={"message": "Processing your request..."},
            session_id=event.session_id
        )
        
        try:
            # Stream agent response
            logger.info(f"[{event.session_id}] AGENT_CALL: Starting stream_response with user_input='{user_message}'")
            
            async for update in self.agent.stream_response(
                user_input=user_message,
                task_description=task_description,
                thread_id=event.session_id
            ):
                # Log agent streaming updates
                logger.info(f"[{event.session_id}] AGENT_UPDATE: {update.type} - {str(update.content)[:200]}...")
                
                # Convert StreamingUpdate to AG-UI event
                agui_event = self._streaming_update_to_agui_event(update, event.session_id)
                
                # Log what we're sending to frontend
                logger.info(f"[{event.session_id}] FRONTEND_EVENT: {agui_event.type} - {str(agui_event.data)[:200]}...")
                
                yield agui_event
                
                # Update session state based on event
                self._update_session_from_streaming_update(session, update)
        
        except Exception as e:
            logger.error(f"Error in user message handling: {e}")
            yield AGUIEvent(
                type=AGUIEventType.RUN_ERROR.value,
                data={"error": str(e)},
                session_id=event.session_id
            )
        
        finally:
            # Signal run finished
            yield AGUIEvent(
                type=AGUIEventType.RUN_FINISHED.value,
                data={"message": "Response complete"},
                session_id=event.session_id
            )
    
    async def _handle_validation_request(self, event: AGUIEvent, session: Dict[str, Any]) -> AsyncGenerator[AGUIEvent, None]:
        """Handle validation request events."""
        validation_data = event.data.get("validation", {})
        
        # Process validation request
        yield AGUIEvent(
            type=AGUIEventType.CUSTOM_EVENT.value,
            data={
                "message": "Processing validation request",
                "validation": validation_data
            },
            session_id=event.session_id
        )
        
        # Here you would implement actual validation logic
        # For now, just acknowledge the request
        yield AGUIEvent(
            type=AGUIEventType.TEXT_MESSAGE_CONTENT.value,
            data={
                "content": f"I understand you want to validate: {validation_data.get('item', 'unknown')}. Please confirm if this looks correct.",
                "requires_human_input": True
            },
            session_id=event.session_id
        )
    
    async def _handle_human_feedback(self, event: AGUIEvent, session: Dict[str, Any]) -> AsyncGenerator[AGUIEvent, None]:
        """Handle human feedback events."""
        feedback = event.data.get("feedback", {})
        approved = feedback.get("approved", False)
        comments = feedback.get("comments", "")
        
        session["conversation_history"].append({
            "role": "human_feedback",
            "approved": approved,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if approved:
            yield AGUIEvent(
                type=AGUIEventType.TEXT_MESSAGE_CONTENT.value,
                data={
                    "content": "Great! I'll proceed with your approved pipeline. " + 
                              (f"Note: {comments}" if comments else "")
                },
                session_id=event.session_id
            )
            
            # Continue with approved pipeline
            yield AGUIEvent(
                type=AGUIEventType.STATE_DELTA.value,
                data={
                    "status": "approved",
                    "pipeline": session["pipeline"].to_dict() if session.get("pipeline") else {}
                },
                session_id=event.session_id
            )
        else:
            yield AGUIEvent(
                type=AGUIEventType.TEXT_MESSAGE_CONTENT.value,
                data={
                    "content": f"I understand. Let me revise the approach. {comments if comments else 'What would you like me to change?'}"
                },
                session_id=event.session_id
            )
    
    def _streaming_update_to_agui_event(self, update: StreamingUpdate, session_id: str) -> AGUIEvent:
        """Convert StreamingUpdate to AG-UI event."""
        event_type_map = {
            "assistant_message": AGUIEventType.TEXT_MESSAGE_CONTENT,
            "tool_selected": AGUIEventType.TOOL_CALL_START,
            "tool_result": AGUIEventType.TOOL_CALL_RESULT,
            "pipeline_updated": AGUIEventType.STATE_DELTA,
            "conversation_complete": AGUIEventType.RUN_FINISHED,
            "error": AGUIEventType.RUN_ERROR
        }
        
        event_type = event_type_map.get(update.type, AGUIEventType.RAW_EVENT)
        
        return AGUIEvent(
            type=event_type.value,
            data=update.content,
            session_id=session_id
        )
    
    def _update_session_from_streaming_update(self, session: Dict[str, Any], update: StreamingUpdate):
        """Update session state based on streaming update."""
        if update.type == "assistant_message":
            session["conversation_history"].append({
                "role": "assistant",
                "content": update.content.get("content", ""),
                "timestamp": update.timestamp
            })
        
        elif update.type == "tool_selected":
            tool_name = update.content.get("tool_name", "")
            if tool_name and session.get("pipeline"):
                # Add tool to pipeline (simplified)
                session["pipeline"].add_node(
                    tool_id=tool_name,
                    tool_data={"name": tool_name, "selected_at": update.timestamp}
                )
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state."""
        return self.active_sessions.get(session_id)
    
    def export_session_pipeline(self, session_id: str, name: Optional[str] = None) -> str:
        """Export session pipeline to YAML."""
        session = self.active_sessions.get(session_id)
        if not session or not session.get("pipeline"):
            return ""
        
        pipeline = session["pipeline"]
        if name:
            pipeline.name = name
        
        return pipeline.to_yaml()
    
    def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


class AGUIServer:
    """
    AG-UI server for CLAMS agent communication.
    Provides Server-Sent Events endpoint for real-time updates.
    """
    
    def __init__(self, agent: CLAMSAgent):
        """
        Initialize AG-UI server.
        
        Args:
            agent: LangGraph CLAMS agent
        """
        self.agent = agent
        self.event_handler = AGUIEventHandler(agent)
        self.active_connections: Dict[str, asyncio.Queue] = {}
    
    async def handle_sse_connection(self, session_id: str) -> AsyncGenerator[str, None]:
        """
        Handle Server-Sent Events connection for real-time updates.
        
        Args:
            session_id: Session identifier
            
        Yields:
            SSE-formatted event strings
        """
        # Create event queue for this connection
        event_queue = asyncio.Queue()
        self.active_connections[session_id] = event_queue
        
        try:
            # Send initial connection event
            initial_event = AGUIEvent(
                type=AGUIEventType.CUSTOM_EVENT.value,
                data={"message": "Connected to CLAMS agent", "session_id": session_id},
                session_id=session_id
            )
            
            yield f"data: {self.event_handler.encoder.encode_event(initial_event)}\n\n"
            
            # Process queued events
            while True:
                try:
                    # Wait for events with timeout to allow heartbeat
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    yield f"data: {self.event_handler.encoder.encode_event(event)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = AGUIEvent(
                        type=AGUIEventType.RAW_EVENT.value,
                        data={"message": "heartbeat"},
                        session_id=session_id
                    )
                    yield f"data: {self.event_handler.encoder.encode_event(heartbeat)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for session: {session_id}")
        finally:
            # Clean up connection
            if session_id in self.active_connections:
                del self.active_connections[session_id]
    
    async def send_event_to_session(self, event: AGUIEvent):
        """Send event to specific session."""
        session_id = event.session_id
        if session_id in self.active_connections:
            await self.active_connections[session_id].put(event)
    
    async def broadcast_event(self, event: AGUIEvent):
        """Broadcast event to all active sessions."""
        for queue in self.active_connections.values():
            await queue.put(event)
    
    async def process_user_event(self, event_data: str) -> List[AGUIEvent]:
        """
        Process incoming user event and return response events.
        
        Args:
            event_data: JSON-encoded AG-UI event
            
        Returns:
            List of response events
        """
        try:
            event = self.event_handler.encoder.decode_event(event_data)
            response_events = []
            
            async for response_event in self.event_handler.handle_event(event):
                response_events.append(response_event)
                # Also send to active SSE connections
                await self.send_event_to_session(response_event)
            
            return response_events
            
        except Exception as e:
            logger.error(f"Error processing user event: {e}")
            error_event = AGUIEvent(
                type=AGUIEventType.RUN_ERROR.value,
                data={"error": str(e)},
                session_id="unknown"
            )
            return [error_event]