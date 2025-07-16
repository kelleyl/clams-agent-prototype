# CLAMS Agent Prototype

## Overview

The CLAMS Agent Prototype leverages Large Language Models (LLMs) to automate the generation of pipelines of CLAMS tools based on task descriptions and available tool metadata. This prototype enables users to analyze video content through natural language queries and visualize the results through an interactive interface.

## Project Goals

- Automate the construction of CLAMS tool pipelines using LLMs
- Enable natural language interaction for video content analysis tasks
- Provide intuitive visualization of computational analysis results
- Streamline multimedia processing workflows

## Key Features

### Chat Interface
- Natural language interaction for requesting information about video content
- LLM-powered interpretation of user requests
- Automatic generation of appropriate CLAMS tool pipelines
- Parameter optimization for efficient video processing

### Visualization Tool
- Interactive exploration of pipeline outputs in MMIF format
- Integrated video player for synchronized content viewing
- Dynamic presentation of computational analysis results
- User-friendly interface for exploring video annotations

### Pipeline Generation
- Intelligent selection of appropriate CLAMS tools based on user queries
- Automatic configuration of tool parameters
- Optimization of processing workflows for efficiency
- Support for diverse multimedia analysis tasks

## Technical Overview

### MMIF (Multimedia Interchange Format)
The system uses MMIF as its core data format, enabling standardized exchange of multimedia annotations between different components of the processing pipeline.

### CLAMS Platform Integration
This prototype integrates with the CLAMS (Computational Linguistics Applications for Multimedia Services) platform, leveraging its ecosystem of multimedia analysis tools.

### Architecture
The system consists of:
- A chat interface for query input and results display
- An LLM-powered pipeline generation system
- A visualization interface for exploring MMIF data
- A video player component for content viewing

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 16.x or higher (for frontend visualization)
- npm or yarn package manager

### 1. Environment Setup
```bash
# Clone the repository (if not already done)
cd clams-agent-prototype

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Build (Required for Web Interface)
```bash
# Navigate to visualization directory
cd visualization

# Install Node.js dependencies
npm install

# Build the frontend
npm run build

# Return to project root
cd ..
```

### 3. Running the Application

#### Option A: Web Interface (Recommended)
```bash
# Ensure you're in the project root with virtual environment activated
python app.py
```
The web interface will be available at `http://localhost:5000`

#### Option B: Command Line Interface
```bash
# Ensure you're in the project root with virtual environment activated
python pipeline_chat.py
```
This provides an interactive command-line chat interface.

### 4. Configuration (Optional)
The application uses default configuration settings. To customize:
- LLM model parameters can be configured in `utils/config.py`
- Default settings work for most use cases
- Configuration is automatically saved to `config.json` when modified

### 5. Troubleshooting
- **Port 5000 in use**: Change the port in `app.py` (line 228): `app.run(debug=True, port=5001)`
- **Frontend build errors**: Ensure Node.js 16+ is installed, delete `node_modules` and `package-lock.json`, then run `npm install` again
- **Python dependency errors**: Ensure you're using the correct virtual environment and all dependencies are installed

## Usage Examples

### Sample Queries
- "Identify all speaking segments in this news broadcast"
- "Find all scenes containing cars in this movie"
- "Detect and transcribe all text visible in this documentary"

### Workflow
1. Load a video or select a collection of videos
2. Enter a natural language query about the content
3. The system generates and executes an appropriate CLAMS tool pipeline
4. Results are displayed in the visualization interface
5. Explore the results interactively alongside the video

## Related
- CLAMS Project (https://clams.ai/)
