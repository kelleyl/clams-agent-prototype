# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLAMS Agent Prototype is an LLM-powered system that automates the generation of CLAMS (Computational Linguistics Applications for Multimedia Services) tool pipelines based on natural language queries. The system uses modern LangGraph patterns with AG-UI integration for real-time communication and streaming responses.

## Architecture

### Core Components

**Flask Application (`app.py`)**:
- AG-UI integrated web server with streaming capabilities
- Server-Sent Events endpoints for real-time communication (`/api/agui/stream/*`)
- Event-driven architecture with standardized AG-UI events
- REST API endpoints for tool discovery and pipeline management

**LangGraph Agent (`utils/langgraph_agent.py`)**:
- Uses `create_react_agent()` for modern LangGraph patterns
- Proper state management with `CLAMSAgentState` TypedDict
- Streaming response capabilities via async generators
- Built-in memory management and conversation persistence

**AG-UI Integration (`utils/agui_integration.py`)**:
- Standardized event handling between frontend and LangGraph agent
- Real-time streaming updates with Server-Sent Events
- Human-in-the-loop validation workflows
- Session-based conversation management

**CLAMS Integration (`utils/clams_tools.py`)**:
- `CLAMSToolbox` class interfaces with CLAMS app directory
- Fetches metadata from `https://apps.clams.ai/` for available tools
- Provides tool discovery and metadata parsing capabilities

**Pipeline Model (`utils/pipeline_model.py`)**:
- `PipelineModel` class represents pipeline structure with nodes and edges
- `PipelineStore` handles persistence to `data/pipelines/` directory
- Supports YAML export/import for pipeline configurations

**React Frontend (`visualization/`)**:
- TypeScript/React application built with Vite
- Interactive pipeline visualization using ReactFlow
- AG-UI integrated chat interface for natural language pipeline construction
- Material-UI components for consistent styling

### Data Flow

1. User submits query through `Chat` component with AG-UI integration
2. Frontend sends AG-UI event to `/api/agui/events` endpoint
3. `AGUIEventHandler` processes event and streams to `CLAMSAgent`
4. Agent generates streaming responses via async generators
5. Updates sent to frontend via Server-Sent Events (`/api/agui/stream/*`)
6. Real-time pipeline construction with live tool selection feedback
7. Human-in-the-loop validation points for complex decisions

## Development Commands

### Python Backend

**Environment setup**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Easy launcher** (recommended):
```bash
python run.py
# Interactive launcher with application banner and features
```

**Direct application launch**:
```bash
python app.py
# Direct server start on http://localhost:5000
```

**Run tests**:
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_app_directory.py

# Run with coverage
python -m pytest tests/ --cov=utils --cov-report=html
```

### Frontend Development

**Install dependencies** (includes AG-UI):
```bash
cd visualization
npm install  # Installs @ag-ui/core, @ag-ui/client, @ag-ui/encoder
```

**Build frontend** (required for web interface):
```bash
cd visualization
npm run build
cd ..
```

**Development mode**:
```bash
cd visualization
npm start  # Development server with hot reload
```

**Frontend linting and testing**:
```bash
cd visualization
npm run lint
npm run test
```

## Configuration

**LLM Configuration (`utils/config.py`)**:
- `LLMConfig` class manages model settings (temperature, max_length, etc.)
- `ConfigManager` handles loading/saving configuration to `config.json`
- Default model: "gpt-4o-mini" with temperature 0.7

**App Metadata (`utils/download_app_directory.py`)**:
- Fetches CLAMS tool metadata from `https://apps.clams.ai/appdir.json`
- Caches metadata in `data/app_directory.json`
- Handles version resolution and error recovery

## Key Patterns

### Event-Driven Architecture

**AG-UI Events**:
- Standardized AG-UI events for all agent-frontend communication
- Real-time streaming via Server-Sent Events
- Event types: `user_message`, `assistant_message`, `tool_selected`, `pipeline_updated`, etc.

**Event Flow**:
```python
# Frontend sends event
event = AGUIEvent(type="user_message", data={"message": "..."}, session_id="...")

# Backend processes via AGUIEventHandler
async for response in handler.handle_event(event):
    # Stream responses back to frontend
    yield response
```

### Streaming State Management

**LangGraph State**:
- `CLAMSAgentState` TypedDict with proper LangGraph annotations
- Async generators for real-time response streaming
- Session-based conversation persistence with memory checkpointing

**Streaming Responses**:
```python
async def stream_response(self, user_input: str, thread_id: str):
    # Generate streaming updates
    async for update in self.agent.astream(state, config):
        yield StreamingUpdate(type="...", content={...})
```

### Tool Integration

**Modern Tool Patterns**:
- Tools converted to proper LangChain tool format
- Automatic tool compatibility analysis based on MMIF types
- Streaming tool selection and execution feedback

**Tool Compatibility**:
```python
def _types_compatible(self, output_type: str, input_type: str) -> bool:
    # Check MMIF type compatibility for pipeline chaining
    return self._check_mmif_compatibility(output_type, input_type)
```

### Human-in-the-Loop Integration

**Validation Events**:
- `validation_request` and `human_feedback` events
- Interactive pipeline validation and modification
- Real-time collaboration between human and AI

**Session Management**:
- Per-session conversation and pipeline state
- Persistent storage of conversation history
- Session-based tool selection and pipeline building

### MMIF Integration

**Data Exchange**:
- System uses MMIF (Multimedia Interchange Format) for data exchange
- Tool metadata includes input/output specifications in MMIF format
- Pipeline execution processes MMIF documents through tool chain

**Type System**:
- Automatic extraction of input/output types from MMIF URIs
- Type compatibility checking for pipeline construction
- Support for video, audio, text, and annotation types

### Error Handling

**Robust Processing**:
- Network errors in app metadata fetching are logged and handled gracefully
- Streaming errors are captured and sent as error events
- Session recovery and continuation capabilities

**User Experience**:
- Graceful degradation when services are unavailable
- Clear error messages with actionable guidance
- Health check endpoints for system monitoring

## Testing Strategy

**Unit Tests (`tests/test_app_directory.py`)**:
- Mock-based testing for external API dependencies
- Tests app metadata fetching, error handling, and data transformation
- Uses `unittest.mock.patch` for HTTP request mocking

**Test Coverage**:
- Focus on external integrations (CLAMS app directory API)
- Error scenarios and edge cases
- Data structure validation and transformation logic
- AG-UI event processing and streaming responses

## Deployment

**Local Development**:
```bash
python run.py --direct  # Direct launch without interactive banner
```

**Health Monitoring**:
- `/api/health` endpoint provides system status
- Component availability checking
- Tool count and connectivity verification

**Environment Variables**:
- `FLASK_ENV=development` for debug mode
- `PORT=5000` to change default port
- `OPENAI_API_KEY` for LLM access (required)