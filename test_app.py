#!/usr/bin/env python3
"""
Test version of the CLAMS Agent app without heavy dependencies.
This helps verify the basic structure works before installing all packages.
"""

import os
import json
from flask import Flask, render_template, jsonify

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='visualization/dist/assets', 
            template_folder='visualization/dist')

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Basic health check."""
    return jsonify({
        "status": "healthy",
        "message": "Test app is running",
        "components": {
            "flask": "✅ Working",
            "templates": "✅ Available" if os.path.exists('visualization/dist/index.html') else "❌ Missing",
            "static": "✅ Available" if os.path.exists('visualization/dist/assets') else "❌ Missing"
        }
    })

@app.route('/api/tools')
def get_tools():
    """Mock tools endpoint."""
    return jsonify({
        "mock-tool-1": {"description": "Mock tool for testing"},
        "mock-tool-2": {"description": "Another mock tool"}
    })

@app.route('/api/test')
def test_endpoint():
    """Test endpoint to verify API is working."""
    return jsonify({
        "message": "API is working!",
        "app_structure": {
            "routes": ["GET /", "GET /api/health", "GET /api/tools", "GET /api/test"],
            "templates_dir": app.template_folder,
            "static_dir": app.static_folder
        }
    })

if __name__ == '__main__':
    logger.info("Starting test CLAMS Agent app...")
    logger.info("This is a test version without LLM dependencies")
    
    # Check for frontend build
    if not os.path.exists('visualization/dist/index.html'):
        logger.warning("Frontend not built. Run: cd visualization && npm run build")
        print("⚠️  Frontend not built. To build:")
        print("   cd visualization")
        print("   npm install")
        print("   npm run build")
        print()
    
    print("🧪 Test app starting...")
    print("📍 URL: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/api/health")
    print("🧪 Test API: http://localhost:5000/api/test")
    
    app.run(debug=True, port=5000)