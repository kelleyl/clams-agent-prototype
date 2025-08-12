"""
Flask application with AG-UI integration for CLAMS Agent Prototype.
Provides streaming, event-driven communication for real-time pipeline generation.
"""

import os
import json
import asyncio
import logging
from typing import AsyncGenerator
from flask import Flask, render_template, request, jsonify, Response, stream_template
from flask_cors import CORS

from utils.langgraph_agent import CLAMSAgent
from utils.agui_integration import AGUIServer, AGUIEvent, EventType
from utils.pipeline_model import PipelineStore
from utils.clams_tools import CLAMSToolbox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='visualization/dist/assets', 
            template_folder='visualization/dist')

# Enable CORS for development
CORS(app)

# Initialize core components
try:
    agent = CLAMSAgent()
    agui_server = AGUIServer(agent)
    pipeline_store = PipelineStore(storage_dir="data/pipelines")
    toolbox = CLAMSToolbox()
    
    logger.info("Successfully initialized CLAMS agent and AG-UI server")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    # Create dummy components for fallback
    agent = None
    agui_server = None


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


# Modern AG-UI endpoints
@app.route('/api/agui/events', methods=['POST'])
def handle_agui_event():
    """Handle incoming AG-UI events."""
    try:
        if not agui_server:
            return jsonify({"error": "AG-UI server not available"}), 500
        
        event_data = request.get_json()
        
        # Process event asynchronously and return immediate response
        async def process_event():
            try:
                event_json = json.dumps(event_data)
                response_events = await agui_server.process_user_event(event_json)
                return response_events
            except Exception as e:
                logger.error(f"Error processing AG-UI event: {e}")
                return []
        
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_events = loop.run_until_complete(process_event())
            return jsonify({
                "status": "processed",
                "events_generated": len(response_events),
                "message": "Event processed successfully"
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error handling AG-UI event: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agui/stream/<session_id>')
def stream_agui_events(session_id):
    """Stream AG-UI events via Server-Sent Events."""
    if not agui_server:
        return jsonify({"error": "AG-UI server not available"}), 500
    
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = event_generator()
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
    
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = chat_generator()
                while True:
                    try:
                        yield loop.run_until_complete(async_gen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()
        
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
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(get_response())
            return jsonify({
                "response": response.get("content", ""),
                "tool_calls": response.get("tool_calls", []),
                "session_id": session_id,
                "tool_added": len(response.get("tool_calls", [])) > 0
            })
        finally:
            loop.close()
            
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