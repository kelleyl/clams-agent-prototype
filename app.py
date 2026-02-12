"""
Flask application with AG-UI integration for CLAMS Agent Prototype.
Provides streaming, event-driven communication for real-time pipeline generation.
"""

import os
import json
import asyncio
import logging
import time
from typing import AsyncGenerator
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_template, send_from_directory
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Configure LangSmith tracing if enabled
def configure_langsmith():
    """Configure LangSmith observability if API key is provided."""
    langsmith_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

    if langsmith_enabled and langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "clams-agent")
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        return True
    return False

langsmith_active = configure_langsmith()

from utils.langgraph_agent import CLAMSAgent
from utils.agui_integration import AGUIServer, AGUIEvent, AGUIEventType
from utils.pipeline_model import PipelineStore
from utils.clams_tools import CLAMSToolbox
from utils.config import ConfigManager
from utils.frame_review import get_frame_review_manager, FrameReviewManager
from utils.clams_executor import get_clams_executor, CLAMSExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create dedicated conversation logger
conversation_logger = logging.getLogger('conversation')
conversation_handler = logging.FileHandler('conversation.log')
conversation_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
conversation_logger.addHandler(conversation_handler)
conversation_logger.setLevel(logging.INFO)

def log_conversation(session_id: str, event_type: str, data: dict):
    """Log conversation events for debugging"""
    conversation_logger.info(f"[{session_id}] {event_type}: {json.dumps(data, indent=2)}")

# Initialize Flask app
app = Flask(__name__, 
            static_folder='visualization/dist/assets', 
            template_folder='visualization/dist')

# Enable CORS for development
CORS(app)

# Initialize core components
try:
    config_manager = ConfigManager()
    agent = CLAMSAgent(config_manager=config_manager)
    agui_server = AGUIServer(agent)
    pipeline_store = PipelineStore(storage_dir="data/pipelines")
    toolbox = CLAMSToolbox()
    
    logger.info("Successfully initialized CLAMS agent and AG-UI server")
    logger.info(f"Using LLM provider: {config_manager.get_config().llm.provider}")
    logger.info(f"Model: {config_manager.get_config().llm.model_name}")
    logger.info(f"LangSmith tracing: {'enabled' if langsmith_active else 'disabled'}")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    # Create dummy components for fallback
    agent = None
    agui_server = None
    config_manager = None


@app.route('/')
def index():
    """Serve the main page with navigation."""
    return render_template('index.html')


@app.route('/visualizer')
def visualizer():
    """Serve the pipeline visualizer page."""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """Serve the pipeline chat page."""
    return render_template('index.html')


@app.route('/tools')
def tools():
    """Serve the tool browser page."""
    return render_template('index.html')


@app.route('/assets/<path:filename>')
def serve_static(filename):
    """Serve static files from the assets directory."""
    return send_from_directory('visualization/dist/assets', filename)


# Legacy API endpoints for backward compatibility
@app.route('/api/tools')
def get_tools():
    """Get all available CLAMS tools."""
    try:
        if toolbox:
            # Get tools and serialize them properly
            tools = toolbox.get_tools()
            serialized = {}
            for tool_id, tool in tools.items():
                serialized[tool_id] = {
                    'name': tool.name,
                    'description': tool.description,
                    'app_metadata': tool.app_metadata,
                    'latest_version': tool.app_metadata.get('latest_version', 'unknown')
                }
            return jsonify(serialized)
        return jsonify({})
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tools/metadata')
def get_tools_metadata():
    """Get tool metadata in format suitable for Tool Browser UI."""
    try:
        if toolbox:
            tools = toolbox.get_tools()
            # Transform to match ToolBrowser expected format
            metadata = {}
            for tool_id, tool in tools.items():
                app_metadata = tool.app_metadata
                tool_metadata = app_metadata.get('metadata', {})

                metadata[tool_id] = {
                    'app_metadata': {
                        'metadata': {
                            'name': tool_metadata.get('name', tool_id),
                            'description': tool_metadata.get('description', ''),
                            'input': tool_metadata.get('input', []),
                            'output': tool_metadata.get('output', []),
                            'parameters': tool_metadata.get('parameters', [])
                        }
                    },
                    'latest_version': app_metadata.get('latest_version', 'unknown')
                }
            return jsonify(metadata)
        return jsonify({})
    except Exception as e:
        logger.error(f"Error getting tools metadata: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipelines')
def get_pipelines():
    """Get all available pipelines."""
    try:
        if pipeline_store:
            pipelines = pipeline_store.list_pipelines()
            return jsonify(pipelines)
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting pipelines: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/<name>', methods=['GET'])
def get_pipeline(name):
    """Get a specific pipeline by name."""
    try:
        if pipeline_store:
            pipeline = pipeline_store.load_pipeline(name)
            return jsonify(pipeline.to_dict())
        return jsonify({"error": "Pipeline store not available"}), 500
    except FileNotFoundError:
        return jsonify({"error": f"Pipeline '{name}' not found"}), 404
    except Exception as e:
        logger.error(f"Error getting pipeline {name}: {e}")
        return jsonify({"error": str(e)}), 500


# Create a global event loop for handling async operations
import threading
import queue

# Global thread for async operations
async_thread = None
async_queue = queue.Queue()
async_results = {}
result_counter = 0

def start_async_thread():
    """Start the dedicated async thread for handling events."""
    global async_thread
    if async_thread is None:
        async_thread = threading.Thread(target=async_worker, daemon=True)
        async_thread.start()
        logger.info("Started dedicated async worker thread")

def async_worker():
    """Worker function for the async thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            try:
                # Get task from queue
                task_id, coro = async_queue.get(timeout=1.0)
                
                # Execute coroutine with timeout protection
                try:
                    result = loop.run_until_complete(asyncio.wait_for(coro, timeout=300.0))
                    async_results[task_id] = {"result": result, "error": None}
                except asyncio.TimeoutError:
                    async_results[task_id] = {"result": None, "error": "Task timed out after 300 seconds"}
                    logger.error(f"Async task {task_id} timed out")
                except Exception as e:
                    async_results[task_id] = {"result": None, "error": str(e)}
                    logger.error(f"Async task {task_id} failed: {e}")
                
            except queue.Empty:
                continue
                
    except Exception as e:
        logger.error(f"Async worker thread failed: {e}")
    finally:
        loop.close()

def run_async_task(coro, timeout=300.0):
    """Run an async task in the dedicated thread and wait for result."""
    global result_counter
    
    # Ensure async thread is running
    start_async_thread()
    
    # Generate unique task ID
    task_id = f"task_{result_counter}"
    result_counter += 1
    
    # Submit task
    async_queue.put((task_id, coro))
    
    # Wait for result
    import time
    start_time = time.time()
    while task_id not in async_results:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Async task {task_id} timed out after {timeout}s")
        time.sleep(0.1)
    
    # Get result
    result_data = async_results.pop(task_id)
    if result_data["error"]:
        raise Exception(result_data["error"])
    
    return result_data["result"]

# Modern AG-UI endpoints
@app.route('/api/agui/session/<session_id>/welcome', methods=['POST'])
def send_welcome_message(session_id):
    """Send welcome message for a new session."""
    try:
        if not agui_server:
            return jsonify({"error": "AG-UI server not available"}), 500
        
        # Create a welcome event
        welcome_event_data = {
            "type": "session_start",
            "data": {"message": "Welcome to CLAMS Pipeline Assistant"},
            "session_id": session_id
        }
        
        # Log welcome event
        log_conversation(session_id, "WELCOME_REQUEST", welcome_event_data)
        
        # Process welcome event
        async def process_welcome():
            try:
                event_json = json.dumps(welcome_event_data)
                response_events = await agui_server.process_user_event(event_json)
                return response_events
            except Exception as e:
                logger.error(f"Error processing welcome event: {e}")
                return []
        
        # Run async processing
        try:
            response_events = run_async_task(process_welcome())
            return jsonify({
                "status": "welcome_sent",
                "events_generated": len(response_events),
                "message": "Welcome message delivered"
            })
        except Exception as e:
            logger.error(f"Error running welcome async task: {e}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error handling welcome message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/agui/events', methods=['POST'])
def handle_agui_event():
    """Handle incoming AG-UI events."""
    try:
        if not agui_server:
            return jsonify({"error": "AG-UI server not available"}), 500
        
        event_data = request.get_json()
        session_id = event_data.get('session_id', 'unknown')
        
        # Log incoming user event
        log_conversation(session_id, "INCOMING_EVENT", event_data)
        
        # Process event synchronously with simpler approach
        try:
            event_json = json.dumps(event_data)
            
            # Use run_async_task() with persistent event loop to avoid "Event loop is closed" errors
            response_events = run_async_task(agui_server.process_user_event(event_json))
            
            # Convert events to JSON-serializable format
            events_data = []
            for event in response_events:
                events_data.append({
                    "type": event.type,
                    "data": event.data,
                    "session_id": event.session_id,
                    "timestamp": getattr(event, 'timestamp', None)
                })
            
            return jsonify({
                "status": "processed",
                "events_generated": len(response_events),
                "message": "Event processed successfully",
                "events": events_data  # Include actual response events
            })
        except Exception as e:
            logger.error(f"Error processing AG-UI event: {e}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error handling AG-UI event: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agui/stream/<session_id>')
def stream_agui_events(session_id):
    """Stream AG-UI events via Server-Sent Events."""
    if not agui_server:
        return jsonify({"error": "AG-UI server not available"}), 503
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in agui_server.handle_sse_connection(session_id):
                yield event
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
    
    def sync_generator():
        """Synchronous wrapper for async generator."""
        try:
            # Use asyncio.run() like the working POST endpoint
            async def run_generator():
                events = []
                try:
                    async for event in agui_server.handle_sse_connection(session_id):
                        events.append(event)
                        if len(events) >= 50:  # Prevent memory issues with too many events
                            break
                except Exception as e:
                    logger.error(f"Error in SSE async generator: {e}")
                    error_event = f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\\n\\n"
                    events.append(error_event)
                return events
            
            # Use run_async_task() with persistent event loop
            events = run_async_task(run_generator())
            for event in events:
                yield event
                
        except Exception as e:
            logger.error(f"Error in sync generator: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\\n\\n"
    
    return Response(sync_generator(), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'Access-Control-Allow-Origin': '*',
                       'Access-Control-Allow-Headers': 'Cache-Control'
                   })


# Modern chat API with streaming
@app.route('/api/chat/stream', methods=['POST'])
def stream_chat_response():
    """Stream chat responses in real-time."""
    try:
        if not agent:
            return jsonify({"error": "Agent not available"}), 500
        
        data = request.get_json()
        user_message = data.get('message', '')
        task_description = data.get('task_description', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        async def chat_generator():
            """Generate streaming chat responses."""
            try:
                async for update in agent.stream_response(
                    user_input=user_message,
                    task_description=task_description,
                    thread_id=session_id
                ):
                    # Convert to SSE format
                    event_data = {
                        'type': update.type,
                        'content': update.content,
                        'timestamp': update.timestamp
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
            except Exception as e:
                logger.error(f"Error in chat streaming: {e}")
                error_data = {
                    'type': 'error',
                    'content': {'error': str(e)},
                    'timestamp': str(asyncio.get_event_loop().time())
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        def sync_chat_generator():
            """Synchronous wrapper for async chat generator."""
            # Use the dedicated async thread instead of creating new loops
            try:
                # Collect all chat updates using the dedicated thread
                async def run_chat():
                    updates = []
                    async for update in agent.stream_response(
                        user_input=user_message,
                        task_description=task_description,
                        thread_id=session_id
                    ):
                        # Convert to SSE format
                        event_data = {
                            'type': update.type,
                            'content': update.content,
                            'timestamp': update.timestamp
                        }
                        updates.append(f"data: {json.dumps(event_data)}\\n\\n")
                        if len(updates) >= 100:  # Prevent memory issues
                            break
                    return updates
                
                # Get all updates and yield them with longer timeout
                updates = run_async_task(run_chat(), timeout=180.0)
                for update in updates:
                    yield update
                    
            except Exception as e:
                logger.error(f"Error in sync chat generator: {e}")
                error_data = {
                    'type': 'error',
                    'content': {'error': str(e)},
                    'timestamp': str(time.time())
                }
                yield f"data: {json.dumps(error_data)}\\n\\n"
        
        return Response(sync_chat_generator(),
                       mimetype='text/event-stream',
                       headers={
                           'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*',
                           'Access-Control-Allow-Headers': 'Cache-Control'
                       })
                       
    except Exception as e:
        logger.error(f"Error in stream chat: {e}")
        return jsonify({"error": str(e)}), 500


# Non-streaming chat endpoint for backward compatibility
@app.route('/api/chat/message', methods=['POST'])
def send_chat_message():
    """Send a chat message and get complete response."""
    try:
        if not agent:
            return jsonify({"error": "Agent not available"}), 500
        
        data = request.get_json()
        user_message = data.get('message', '')
        task_description = data.get('task_description', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        async def get_response():
            """Get complete chat response."""
            return await agent.get_response(
                user_input=user_message,
                task_description=task_description,
                thread_id=session_id
            )
        
        # Run async operation with persistent event loop
        try:
            response = run_async_task(get_response())
            return jsonify({
                "response": response.get("content", ""),
                "tool_calls": response.get("tool_calls", []),
                "session_id": session_id,
                "tool_added": len(response.get("tool_calls", [])) > 0
            })
        except Exception as e:
            logger.error(f"Error running async operation: {e}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error in chat message: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/pipeline/<session_id>', methods=['GET'])
def get_session_pipeline(session_id):
    """Get pipeline for a specific session."""
    try:
        if not agui_server:
            return jsonify({"error": "AG-UI server not available"}), 500
        
        session_state = agui_server.event_handler.get_session_state(session_id)
        if not session_state:
            return jsonify({"error": "Session not found"}), 404
        
        pipeline = session_state.get("pipeline")
        if pipeline:
            return jsonify(pipeline.to_dict())
        else:
            return jsonify({"error": "No pipeline found for session"}), 404
            
    except Exception as e:
        logger.error(f"Error getting session pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/export/<session_id>', methods=['GET'])
def export_session_pipeline(session_id):
    """Export session pipeline to YAML."""
    try:
        if not agui_server:
            return jsonify({"error": "AG-UI server not available"}), 500
        
        name = request.args.get('name', f'Pipeline-{session_id}')
        yaml_content = agui_server.event_handler.export_session_pipeline(session_id, name)
        
        if yaml_content:
            return jsonify({
                "yaml": yaml_content,
                "name": name,
                "session_id": session_id
            })
        else:
            return jsonify({"error": "No pipeline found for session"}), 404
            
    except Exception as e:
        logger.error(f"Error exporting session pipeline: {e}")
        return jsonify({"error": str(e)}), 500


# Video metadata endpoint
@app.route('/api/video/metadata', methods=['POST'])
def get_video_metadata():
    """Get metadata for a video file using FFmpeg."""
    try:
        from utils.ffmpeg_tools import FFmpegTools

        data = request.get_json()
        video_path = data.get('video_path', '')

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        # Initialize FFmpeg tools
        ffmpeg = FFmpegTools()
        metadata = ffmpeg.get_video_metadata(video_path)

        if "error" in metadata:
            return jsonify(metadata), 400

        return jsonify(metadata)

    except Exception as e:
        logger.error(f"Error getting video metadata: {e}")
        return jsonify({"error": str(e)}), 500


# Frame extraction endpoint
@app.route('/api/video/frame', methods=['POST'])
def extract_video_frame():
    """Extract a single frame from a video at a specific timestamp."""
    try:
        from utils.ffmpeg_tools import FFmpegTools

        data = request.get_json()
        video_path = data.get('video_path', '')
        timestamp = data.get('timestamp', 0)
        width = data.get('width')
        height = data.get('height')

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        ffmpeg = FFmpegTools()
        result = ffmpeg.extract_frame_at_timestamp(
            video_path,
            timestamp,
            width=width,
            height=height
        )

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error extracting frame: {e}")
        return jsonify({"error": str(e)}), 500


# Serve extracted frames
@app.route('/api/video/frame/<frame_id>')
def serve_frame(frame_id):
    """Serve an extracted frame image."""
    try:
        import os
        from pathlib import Path

        # Look for frame in output directory
        frames_dir = Path("data/ffmpeg_output/frames")

        # Find frame file matching the ID
        for frame_file in frames_dir.glob(f"*{frame_id}*.jpg"):
            return send_from_directory(
                str(frame_file.parent),
                frame_file.name,
                mimetype='image/jpeg'
            )

        return jsonify({"error": "Frame not found"}), 404

    except Exception as e:
        logger.error(f"Error serving frame: {e}")
        return jsonify({"error": str(e)}), 500


# Thumbnail sprite generation endpoint
@app.route('/api/video/sprite', methods=['POST'])
def generate_thumbnail_sprite():
    """Generate a thumbnail sprite for timeline scrubbing."""
    try:
        from utils.ffmpeg_tools import FFmpegTools

        data = request.get_json()
        video_path = data.get('video_path', '')
        interval = data.get('interval', 5.0)
        thumb_width = data.get('thumb_width', 160)
        thumb_height = data.get('thumb_height', 90)

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        ffmpeg = FFmpegTools()
        result = ffmpeg.generate_thumbnail_sprite(
            video_path,
            interval=interval,
            thumb_width=thumb_width,
            thumb_height=thumb_height
        )

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error generating sprite: {e}")
        return jsonify({"error": str(e)}), 500


# Serve sprite images
@app.route('/api/video/sprite/<sprite_id>')
def serve_sprite(sprite_id):
    """Serve a thumbnail sprite image."""
    try:
        from pathlib import Path

        sprites_dir = Path("data/ffmpeg_output/sprites")

        for sprite_file in sprites_dir.glob(f"*{sprite_id}*.jpg"):
            return send_from_directory(
                str(sprite_file.parent),
                sprite_file.name,
                mimetype='image/jpeg'
            )

        return jsonify({"error": "Sprite not found"}), 404

    except Exception as e:
        logger.error(f"Error serving sprite: {e}")
        return jsonify({"error": str(e)}), 500


# Serve VTT files for thumbnails
@app.route('/api/video/vtt/<sprite_id>')
def serve_vtt(sprite_id):
    """Serve a VTT file for thumbnail mapping."""
    try:
        from pathlib import Path

        sprites_dir = Path("data/ffmpeg_output/sprites")

        for vtt_file in sprites_dir.glob(f"*{sprite_id}*.vtt"):
            return send_from_directory(
                str(vtt_file.parent),
                vtt_file.name,
                mimetype='text/vtt'
            )

        return jsonify({"error": "VTT file not found"}), 404

    except Exception as e:
        logger.error(f"Error serving VTT: {e}")
        return jsonify({"error": str(e)}), 500


# Video clip extraction endpoint
@app.route('/api/video/clip', methods=['POST'])
def extract_video_clip():
    """Extract a short video clip for review."""
    try:
        from utils.ffmpeg_tools import FFmpegTools

        data = request.get_json()
        video_path = data.get('video_path', '')
        start_time = data.get('start_time', 0)
        end_time = data.get('end_time', 10)
        max_width = data.get('max_width', 640)

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        ffmpeg = FFmpegTools()
        result = ffmpeg.extract_clip(
            video_path,
            start_time,
            end_time,
            max_width=max_width
        )

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error extracting clip: {e}")
        return jsonify({"error": str(e)}), 500


# Serve video clips
@app.route('/api/video/clip/<clip_id>')
def serve_clip(clip_id):
    """Serve an extracted video clip."""
    try:
        from pathlib import Path

        clips_dir = Path("data/ffmpeg_output/clips")

        for clip_file in clips_dir.glob(f"*{clip_id}*.mp4"):
            return send_from_directory(
                str(clip_file.parent),
                clip_file.name,
                mimetype='video/mp4'
            )

        return jsonify({"error": "Clip not found"}), 404

    except Exception as e:
        logger.error(f"Error serving clip: {e}")
        return jsonify({"error": str(e)}), 500


# Test frame review endpoint (for development)
@app.route('/api/test/frame-review', methods=['POST'])
def test_frame_review():
    """Trigger a frame review request for testing the HITL flow.

    Returns the frame_review_request event directly, which the frontend
    can process just like it does with SSE events.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', f'test-session-{int(time.time())}')
        frame_id = data.get('frame_id', '6c1a8627195e')

        # Create a frame review request event
        frame_data = {
            "frame_id": frame_id,
            "frame_url": f"/api/video/frame/{frame_id}",
            "timestamp": data.get('timestamp', 30.0),
            "timestamp_formatted": data.get('timestamp_formatted', '00:00:30'),
            "video_path": data.get('video_path', '/test/video.mp4'),
            "context": data.get('context', 'Test frame review request'),
            "detected_text": data.get('detected_text', 'SAMPLE TEXT'),
            "detected_entities": data.get('detected_entities', [
                {"name": "Test Entity", "type": "ORG", "confidence": 0.95}
            ]),
            "review_type": data.get('review_type', 'approval')
        }

        # Return the event directly for frontend processing
        return jsonify({
            "success": True,
            "message": "Frame review request created",
            "session_id": session_id,
            "events": [
                {
                    "type": AGUIEventType.FRAME_REVIEW_REQUEST.value,
                    "data": frame_data,
                    "session_id": session_id
                }
            ]
        })

    except Exception as e:
        logger.error(f"Error triggering test frame review: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Frame Review API Endpoints (Human-in-the-Loop)
# ============================================================================

@app.route('/api/frames/<session_id>/<frame_id>')
def get_frame_image(session_id: str, frame_id: str):
    """Serve a frame image for review.

    Returns the actual image file for display in the frontend.
    """
    try:
        frame_manager = get_frame_review_manager()
        frame_path = frame_manager.get_frame_image_path(session_id, frame_id)

        if frame_path and os.path.exists(frame_path):
            directory = os.path.dirname(frame_path)
            filename = os.path.basename(frame_path)
            return send_from_directory(directory, filename, mimetype='image/jpeg')
        else:
            return jsonify({"error": "Frame not found"}), 404

    except Exception as e:
        logger.error(f"Error serving frame {frame_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/sessions', methods=['GET'])
def list_review_sessions():
    """List all active review sessions."""
    try:
        frame_manager = get_frame_review_manager()
        sessions = []

        for session_id, session in frame_manager.sessions.items():
            sessions.append({
                "session_id": session_id,
                "video_path": session.video_path,
                "frame_count": len(session.frames),
                "reviewed_count": len(session.reviews),
                "completed": session.completed,
                "created_at": session.created_at
            })

        return jsonify({
            "success": True,
            "sessions": sessions
        })

    except Exception as e:
        logger.error(f"Error listing review sessions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/<session_id>', methods=['GET'])
def get_review_session(session_id: str):
    """Get details of a specific review session."""
    try:
        frame_manager = get_frame_review_manager()
        session = frame_manager.get_session(session_id)

        if not session:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        frames = [
            {
                "frame_id": f.frame_id,
                "timestamp_ms": f.timestamp_ms,
                "timestamp_formatted": f.timestamp_formatted,
                "frame_url": f"/api/frames/{session_id}/{f.frame_id}",
                "detected_text": f.detected_text,
                "detected_entities": f.detected_entities,
                "context": f.context,
                "reviewed": f.frame_id in session.reviews
            }
            for f in session.frames
        ]

        return jsonify({
            "success": True,
            "session_id": session_id,
            "video_path": session.video_path,
            "frames": frames,
            "summary": session.get_review_summary(),
            "completed": session.completed
        })

    except Exception as e:
        logger.error(f"Error getting review session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/<session_id>/submit', methods=['POST'])
def submit_frame_review(session_id: str):
    """Submit a review for a frame."""
    try:
        data = request.get_json() or {}
        frame_id = data.get('frame_id')
        status = data.get('status', 'skipped')
        comment = data.get('comment', '')
        corrected_text = data.get('corrected_text')
        corrected_entities = data.get('corrected_entities')

        if not frame_id:
            return jsonify({"error": "frame_id is required"}), 400

        frame_manager = get_frame_review_manager()
        review = frame_manager.submit_review(
            session_id=session_id,
            frame_id=frame_id,
            status=status,
            comment=comment,
            corrected_text=corrected_text,
            corrected_entities=corrected_entities
        )

        if not review:
            return jsonify({"error": "Failed to submit review"}), 400

        # Get updated session state
        session = frame_manager.get_session(session_id)
        next_frame = frame_manager.get_next_frame_for_review(session_id)

        return jsonify({
            "success": True,
            "review": review.to_dict(),
            "session_summary": session.get_review_summary() if session else None,
            "next_frame": {
                "frame_id": next_frame.frame_id,
                "frame_url": f"/api/frames/{session_id}/{next_frame.frame_id}",
                "detected_text": next_frame.detected_text
            } if next_frame else None,
            "session_completed": session.completed if session else False
        })

    except Exception as e:
        logger.error(f"Error submitting review: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/<session_id>/export', methods=['GET'])
def export_reviews(session_id: str):
    """Export all reviews for a session."""
    try:
        frame_manager = get_frame_review_manager()
        export_data = frame_manager.export_reviews(session_id)

        if not export_data:
            return jsonify({"error": f"Session not found: {session_id}"}), 404

        return jsonify({
            "success": True,
            "export": export_data
        })

    except Exception as e:
        logger.error(f"Error exporting reviews: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/<session_id>/corrections', methods=['GET'])
def get_corrections(session_id: str):
    """Get all corrections made during a review session."""
    try:
        frame_manager = get_frame_review_manager()
        corrections = frame_manager.get_corrections(session_id)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "corrections": corrections
        })

    except Exception as e:
        logger.error(f"Error getting corrections: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/review/create', methods=['POST'])
def create_review_session():
    """Create a new frame review session from a video."""
    try:
        data = request.get_json() or {}
        video_path = data.get('video_path')
        interval_seconds = data.get('interval_seconds', 5.0)
        max_frames = data.get('max_frames', 30)

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 404

        import uuid
        session_id = str(uuid.uuid4())[:8]

        frame_manager = get_frame_review_manager()
        session = frame_manager.create_review_session(
            session_id=session_id,
            video_path=video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames
        )

        return jsonify({
            "success": True,
            "session_id": session_id,
            "frames_extracted": len(session.frames),
            "video_path": video_path,
            "review_url": f"/api/review/{session_id}"
        })

    except Exception as e:
        logger.error(f"Error creating review session: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# CLAMS Execution API Endpoints
# ============================================================================

@app.route('/api/clams/apps', methods=['GET'])
def list_clams_apps():
    """List available CLAMS apps that can be executed."""
    try:
        executor = get_clams_executor()
        apps = executor.get_available_apps()

        app_info = []
        for app_name in apps:
            metadata = executor.get_app_info(app_name)
            app_info.append({
                "name": app_name,
                "available": True,
                "description": metadata.get('description', 'No description') if metadata else 'No metadata'
            })

        return jsonify({
            "success": True,
            "apps": app_info
        })

    except Exception as e:
        logger.error(f"Error listing CLAMS apps: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/clams/execute', methods=['POST'])
def execute_clams_app():
    """Execute a CLAMS app on a video or MMIF input."""
    try:
        data = request.get_json() or {}
        app_name = data.get('app_name')
        video_path = data.get('video_path')
        input_mmif = data.get('input_mmif')
        parameters = data.get('parameters', {})

        if not app_name:
            return jsonify({"error": "app_name is required"}), 400

        executor = get_clams_executor()

        if app_name not in executor.get_available_apps():
            return jsonify({"error": f"App not available: {app_name}"}), 400

        # Create or use input MMIF
        if input_mmif:
            mmif = input_mmif
        elif video_path:
            mmif = executor.create_initial_mmif(video_path)
        else:
            return jsonify({"error": "Either video_path or input_mmif is required"}), 400

        # Execute the app
        result = executor.execute_app(app_name, mmif, parameters)

        if result.success:
            return jsonify({
                "success": True,
                "app_name": app_name,
                "execution_time": result.execution_time_seconds,
                "annotations_created": result.annotations_created,
                "output_mmif": result.output_mmif
            })
        else:
            return jsonify({
                "success": False,
                "app_name": app_name,
                "error": result.error
            }), 500

    except Exception as e:
        logger.error(f"Error executing CLAMS app: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/clams/pipeline', methods=['POST'])
def execute_clams_pipeline():
    """Execute a pipeline of CLAMS apps on a video."""
    try:
        data = request.get_json() or {}
        apps = data.get('apps', [])
        video_path = data.get('video_path')
        parameters = data.get('parameters', {})

        if not apps:
            return jsonify({"error": "apps list is required"}), 400

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 404

        executor = get_clams_executor()
        results = executor.execute_pipeline(apps, video_path, parameters)

        summary = []
        for r in results:
            summary.append({
                "app": r.app_name,
                "success": r.success,
                "annotations_created": r.annotations_created,
                "execution_time": r.execution_time_seconds,
                "error": r.error if not r.success else None
            })

        # Get final MMIF if all succeeded
        final_mmif = None
        if all(r.success for r in results):
            final_mmif = results[-1].output_mmif

        return jsonify({
            "success": all(r.success for r in results),
            "pipeline": apps,
            "results": summary,
            "total_annotations": sum(r.annotations_created for r in results if r.success),
            "final_mmif": final_mmif
        })

    except Exception as e:
        logger.error(f"Error executing CLAMS pipeline: {e}")
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy",
            "agent_available": agent is not None,
            "agui_server_available": agui_server is not None,
            "toolbox_available": toolbox is not None,
            "pipeline_store_available": pipeline_store is not None,
            "langsmith_tracing": langsmith_active
        }
        
        if agent:
            # Test agent connectivity
            tools = toolbox.get_tools() if toolbox else {}
            status["tools_count"] = len(tools)
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Check if running in development mode
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Starting CLAMS Agent Prototype server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Agent available: {agent is not None}")
    logger.info(f"AG-UI server available: {agui_server is not None}")
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')