import os
import json
import yaml
import re
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from utils.pipeline_model import PipelineModel, PipelineStore
from utils.langgraph_agent import LangGraphPipelineAgent, ChatContext
from utils.clams_tools import CLAMSToolbox
from utils.download_app_directory import get_app_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize app
app = Flask(__name__, 
            static_folder='visualization/dist/assets', 
            template_folder='visualization/dist')

# Initialize pipeline storage
pipeline_store = PipelineStore(storage_dir="data/pipelines")

# Initialize CLAMS toolbox and chat agent
toolbox = CLAMSToolbox()
chat_agent = LangGraphPipelineAgent()

# Global state for current chat context
current_chat_context = None

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

# Add a catch-all route for static files
@app.route('/assets/<path:filename>')
def serve_static(filename):
    """Serve static files from the assets directory."""
    return send_from_directory('visualization/dist/assets', filename)

@app.route('/api/tools')
def get_tools():
    """Get all available CLAMS tools."""
    return jsonify(toolbox.get_tools())

@app.route('/api/pipelines')
def get_pipelines():
    """Get all available pipelines."""
    pipelines = pipeline_store.list_pipelines()
    return jsonify(pipelines)

@app.route('/api/pipeline/<name>', methods=['GET'])
def get_pipeline(name):
    """Get a specific pipeline by name."""
    try:
        pipeline = pipeline_store.load_pipeline(name)
        return jsonify(pipeline.to_dict())
    except FileNotFoundError:
        return jsonify({"error": f"Pipeline '{name}' not found"}), 404

@app.route('/api/pipeline', methods=['POST'])
def save_pipeline():
    """Save a pipeline."""
    try:
        data = request.json
        pipeline = PipelineModel.from_dict(data)
        path = pipeline_store.save_pipeline(pipeline)
        return jsonify({"message": f"Pipeline saved to {path}", "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/export/<name>', methods=['GET'])
def export_pipeline(name):
    """Export a pipeline to YAML."""
    try:
        pipeline = pipeline_store.load_pipeline(name)
        yaml_content = pipeline.to_yaml()
        return jsonify({"yaml": yaml_content, "name": name})
    except FileNotFoundError:
        return jsonify({"error": f"Pipeline '{name}' not found"}), 404

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    """Start a new chat session."""
    global current_chat_context
    
    try:
        data = request.json
        task = data.get('task', 'Create a CLAMS pipeline')
        
        # Initialize a new chat context
        current_chat_context = ChatContext(task_description=task)
        
        # Generate initial response that directly addresses the task
        initial_message = f"Help me create a CLAMS pipeline to {task}. Please suggest appropriate tools and explain the pipeline structure."
        response = chat_agent.chat_response(
            current_chat_context,
            initial_message
        )
        
        # Add the response to the context
        current_chat_context.conversation_history.append({
            "role": "Assistant",
            "content": response
        })
        
        return jsonify({
            "message": f"Chat session started for task: {task}",
            "response": response
        })
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/message', methods=['POST'])
def send_message():
    """Send a message to the chat agent."""
    global current_chat_context
    
    try:
        data = request.json
        message = data.get('message', '')
        
        if not current_chat_context:
            return jsonify({"error": "No active chat session. Please start a new chat."}), 400
        
        # Add user message to context
        current_chat_context.conversation_history.append({
            "role": "User",
            "content": message
        })
        
        # Generate response
        response = chat_agent.chat_response(current_chat_context, message)
        
        # Add assistant response to context
        current_chat_context.conversation_history.append({
            "role": "Assistant",
            "content": response
        })
        
        # Check for tool suggestion
        tool_name = None
        tool_match = re.search(r'I suggest using the "?([a-zA-Z0-9_-]+)"? tool', response)
        if tool_match and tool_match.group(1) in chat_agent.tool_metadata:
            tool_name = tool_match.group(1)
            current_chat_context.add_selected_tool(tool_name, chat_agent.tool_metadata[tool_name])
        
        return jsonify({
            "response": response,
            "tool_added": tool_name is not None,
            "tool_name": tool_name,
            "pipeline_state": current_chat_context.get_pipeline_state()
        })
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/generate', methods=['POST'])
def generate_pipeline():
    """Generate a pipeline from the current chat."""
    global current_chat_context
    
    try:
        if not current_chat_context:
            return jsonify({"error": "No active chat session. Please start a new chat."}), 400
        
        data = request.json
        name = data.get('name', 'Chat Generated Pipeline')
        
        # Update pipeline name
        current_chat_context.pipeline.name = name
        
        # Save pipeline
        path = pipeline_store.save_pipeline(current_chat_context.pipeline)
        
        # Get pipeline data
        pipeline_data = current_chat_context.pipeline.to_dict()
        
        return jsonify({
            "message": f"Pipeline '{name}' generated and saved to {path}",
            "name": name,
            "path": path,
            "pipeline": pipeline_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/export', methods=['GET'])
def export_chat_pipeline():
    """Export the current chat pipeline to YAML."""
    global current_chat_context
    
    try:
        if not current_chat_context:
            return jsonify({"error": "No active chat session. Please start a new chat."}), 400
        
        yaml_content = current_chat_context.export_pipeline()
        return jsonify({"yaml": yaml_content, "name": current_chat_context.pipeline.name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/pipeline', methods=['GET'])
def get_chat_pipeline():
    """Get the current pipeline from the chat."""
    global current_chat_context
    
    try:
        if not current_chat_context:
            return jsonify({"error": "No active chat session. Please start a new chat."}), 400
        
        pipeline_data = current_chat_context.pipeline.to_dict()
        return jsonify(pipeline_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)