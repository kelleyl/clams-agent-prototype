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
from flask import Flask, render_template, request, jsonify, Response, stream_template, send_from_directory
from flask_cors import CORS

from utils.langgraph_agent import CLAMSAgent
from utils.agui_integration import AGUIServer, AGUIEvent, AGUIEventType
from utils.pipeline_model import PipelineStore
from utils.clams_tools import CLAMSToolbox
from utils.config import ConfigManager

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
            return jsonify(toolbox.get_tools())
        return jsonify({})
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
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

def run_async_task(coro, timeout=120.0):
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
            
            # Use asyncio.run() for simpler execution
            response_events = asyncio.run(agui_server.process_user_event(event_json))
            
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
            
            # Use asyncio.run() instead of the complex worker thread system
            events = asyncio.run(run_generator())
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
        
        # Run async operation with simpler approach
        try:
            response = asyncio.run(get_response())
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
            "pipeline_store_available": pipeline_store is not None
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