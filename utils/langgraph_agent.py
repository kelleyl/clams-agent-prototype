"""
Hybrid CLAMS Pipeline Agent with separated planning and execution.
Uses conversational planning + direct tool execution for reliability.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated, AsyncGenerator
import logging
import json
import asyncio
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .pipeline_model import PipelineModel, PipelineStore
from .clams_tools import CLAMSToolbox
from .config import ConfigManager
from .planning_agent import CLAMSPlanningAgent
from .pipeline_execution import CLAMSExecutionEngine, PipelinePlan, ExecutionProgress

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CLAMSAgentState(TypedDict):
    """Modern LangGraph state with proper annotations."""
    messages: Annotated[List[AnyMessage], add_messages]
    task_description: str
    pipeline_dict: Dict[str, Any]  # Serializable pipeline representation
    selected_tools: List[str]
    execution_context: Dict[str, Any]
    streaming_updates: List[Dict[str, Any]]
    human_feedback_requested: bool
    current_step: str


@dataclass
class StreamingUpdate:
    """Represents a streaming update event."""
    type: str  # 'tool_selected', 'pipeline_updated', 'validation_requested', etc.
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: str(asyncio.get_event_loop().time()))


class CLAMSAgent:
    """
    Hybrid CLAMS pipeline agent with separated planning and execution.
    Combines conversational planning with direct tool execution for reliability.
    """
    
    def __init__(self, config_manager: ConfigManager = None):
        """
        Initialize the CLAMS agent.
        
        Args:
            config_manager: Configuration manager instance
        """
        # Initialize configuration
        self.config_manager = config_manager or ConfigManager()
        self.llm_config = self.config_manager.get_config().llm
        
        # Initialize LLM for backward compatibility
        if self.llm_config.provider == "ollama":
            self.llm = ChatOllama(
                model=self.llm_config.model_name,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                top_p=self.llm_config.top_p
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.llm_config.model_name, 
                streaming=True,
                temperature=self.llm_config.temperature
            )
        
        # Initialize core components
        self.planner = CLAMSPlanningAgent(config_manager)
        self.executor = CLAMSExecutionEngine()
        self.pipeline_store = PipelineStore()
        
        # Legacy support - initialize tools for backward compatibility
        self.toolbox = CLAMSToolbox()
        self.tools = self._initialize_clams_tools()
        self.tool_metadata = self._initialize_tool_metadata()
        
        # Memory for conversation persistence
        self.memory = MemorySaver()
        
        # Create the agent using modern patterns (for legacy support)
        self.app = self._create_modern_agent()
    
    def _initialize_clams_tools(self) -> List[BaseTool]:
        """Initialize CLAMS tools as proper LangChain tools."""
        tools = []
        
        for tool_name, clams_tool in self.toolbox.get_tools().items():
            # CLAMS tools are already LangChain BaseTool instances
            tools.append(clams_tool)
        
        return tools
    
    def _create_langgraph_tool(self, tool_name: str, tool_data: Dict[str, Any]) -> BaseTool:
        """Create a LangChain tool from CLAMS tool metadata."""
        from langchain_core.tools import tool
        
        app_metadata = tool_data.get('app_metadata', {})
        metadata = app_metadata.get('metadata', {})
        description = metadata.get('description', f'CLAMS tool: {tool_name}')
        
        @tool(name=tool_name, description=description)
        def clams_tool_wrapper(query: str) -> str:
            """Execute CLAMS tool and return results."""
            # For now, simulate tool execution
            # In a real implementation, this would call the actual CLAMS tool
            return f"Executed {tool_name} with query: {query}. This tool {description}"
        
        # Add metadata for pipeline construction
        clams_tool_wrapper.tool_metadata = metadata
        return clams_tool_wrapper
    
    def _initialize_tool_metadata(self) -> Dict[str, Any]:
        """Initialize tool metadata for pipeline construction."""
        metadata = {}
        
        for tool_name, clams_tool in self.toolbox.get_tools().items():
            app_info = clams_tool.app_metadata
            tool_metadata = app_info.get('metadata', {})
            
            # Extract and clean input/output types
            input_types = self._extract_types(tool_metadata.get('input', []))
            output_types = self._extract_types(tool_metadata.get('output', []))
            
            metadata[tool_name] = {
                'description': tool_metadata.get('description', ''),
                'input_types': input_types,
                'output_types': output_types,
                'parameters': tool_metadata.get('parameters', []),
                'app_version': tool_metadata.get('app_version', 'unknown')
            }
        
        return metadata
    
    def _extract_types(self, type_list: List[Dict[str, Any]]) -> List[str]:
        """Extract clean type names from MMIF type URIs."""
        types = []
        for type_info in type_list:
            if isinstance(type_info, dict) and '@type' in type_info:
                type_uri = type_info['@type']
                # Extract clean type name
                type_name = type_uri.split('/')[-1]
                # Remove version numbers
                clean_name = type_name.split('v')[0] if 'v' in type_name else type_name
                if clean_name:
                    types.append(clean_name)
        return types
    
    def _create_modern_agent(self):
        """Create the agent using modern LangGraph patterns."""
        # Create system message for CLAMS pipeline assistance
        system_message = self._create_system_message()
        
        # Bind tools directly to the LLM first
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Use create_react_agent for modern pattern with checkpointer
        agent = create_react_agent(
            llm_with_tools,
            self.tools,
            prompt=system_message,
            checkpointer=self.memory
        )
        
        return agent
    
    def _create_system_message(self) -> SystemMessage:
        """Create a comprehensive system message for the agent."""
        tool_descriptions = self._get_tool_descriptions()
        
        return SystemMessage(content=f"""You are a CLAMS (Computational Language and Audiovisual Multimedia Systems) pipeline expert.

Your role is to help users create effective multimedia analysis pipelines by:

1. Understanding user requirements for video/audio analysis
2. Selecting appropriate CLAMS tools based on input/output compatibility
3. Constructing logical tool sequences (pipelines)
4. Explaining tool functionality and pipeline reasoning

Available CLAMS Tools:
{tool_descriptions}

Pipeline Construction Rules:
- Consider tool input/output type compatibility
- Video processing typically starts with VideoDocument input
- OCR tools need video frames or images
- Speech recognition tools need audio input
- Text analysis tools need transcribed text
- Always explain your tool selection reasoning

IMPORTANT: When users ask you to use specific tools or demonstrate pipelines, you MUST call the actual tool functions. Do not just describe them. Use the tool calling mechanism to execute the tools with appropriate parameters.

When suggesting tools:
1. First understand what the user wants to accomplish
2. Identify the required input type (video, audio, image, text)
3. Select tools that produce the needed analysis
4. Chain tools logically (output of one becomes input of next)
5. CALL the actual tools using function calls to demonstrate the pipeline

Be conversational but precise. Ask clarifying questions if the user's requirements are unclear.
""")
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        descriptions = []
        
        for tool_name, metadata in self.tool_metadata.items():
            input_types = ", ".join(metadata['input_types']) if metadata['input_types'] else "Any"
            output_types = ", ".join(metadata['output_types']) if metadata['output_types'] else "Various"
            
            descriptions.append(
                f"- {tool_name}: {metadata['description']}\n"
                f"  Input: {input_types} | Output: {output_types}"
            )
        
        return "\n".join(descriptions)
    
    async def stream_response(self, 
                             user_input: str, 
                             task_description: str = "",
                             thread_id: str = "default") -> AsyncGenerator[StreamingUpdate, None]:
        """
        Generate streaming responses for real-time UI updates.
        
        Args:
            user_input: User's message
            task_description: Overall task description
            thread_id: Conversation thread identifier
            
        Yields:
            StreamingUpdate objects for real-time UI updates
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "task_description": task_description,
                "pipeline_dict": {"name": "Streaming Pipeline", "nodes": [], "edges": []},
                "selected_tools": [],
                "execution_context": {},
                "streaming_updates": [],
                "human_feedback_requested": False,
                "current_step": "processing"
            }
            
            # Configuration for persistent conversation
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream the agent execution
            async for chunk in self.app.astream(initial_state, config=config):
                for node_name, node_state in chunk.items():
                    if "messages" in node_state:
                        messages = node_state["messages"]
                        
                        # Process each message for streaming updates
                        for msg in messages:
                            if isinstance(msg, AIMessage):
                                # Stream assistant message
                                yield StreamingUpdate(
                                    type="assistant_message",
                                    content={
                                        "content": msg.content,
                                        "node": node_name
                                    }
                                )
                                
                                # Check for tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        yield StreamingUpdate(
                                            type="tool_selected",
                                            content={
                                                "tool_name": tool_call.get('name', ''),
                                                "args": tool_call.get('args', {}),
                                                "reasoning": "Selected for pipeline construction"
                                            }
                                        )
                            
                            elif isinstance(msg, ToolMessage):
                                # Stream tool results
                                yield StreamingUpdate(
                                    type="tool_result",
                                    content={
                                        "tool_name": msg.name,
                                        "result": msg.content,
                                        "node": node_name
                                    }
                                )
                    
                    # Stream pipeline updates
                    if "pipeline_dict" in node_state:
                        pipeline = node_state["pipeline_dict"]
                        yield StreamingUpdate(
                            type="pipeline_updated",
                            content={
                                "nodes": len(pipeline.get("nodes", [])),
                                "edges": len(pipeline.get("edges", [])),
                                "status": "updated"
                            }
                        )
            
            # Final completion update
            yield StreamingUpdate(
                type="conversation_complete",
                content={"status": "completed"}
            )
            
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield StreamingUpdate(
                type="error",
                content={"error": str(e)}
            )
    
    async def get_response(self, 
                          user_input: str, 
                          task_description: str = "",
                          thread_id: str = "default") -> Dict[str, Any]:
        """
        Get a complete response (non-streaming) for backward compatibility.
        
        Args:
            user_input: User's message
            task_description: Overall task description  
            thread_id: Conversation thread identifier
            
        Returns:
            Response dictionary with content and metadata
        """
        try:
            # Collect all streaming updates
            updates = []
            assistant_content = ""
            tool_calls = []
            
            async for update in self.stream_response(user_input, task_description, thread_id):
                updates.append(update)
                
                if update.type == "assistant_message":
                    assistant_content = update.content.get("content", "")
                elif update.type == "tool_selected":
                    tool_calls.append(update.content)
            
            return {
                "content": assistant_content,
                "tool_calls": tool_calls,
                "updates": updates,
                "thread_id": thread_id
            }
            
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return {
                "content": f"Sorry, I encountered an error: {str(e)}",
                "tool_calls": [],
                "updates": [],
                "thread_id": thread_id
            }
    
    # New hybrid approach methods
    async def plan_pipeline(self, user_query: str) -> PipelinePlan:
        """
        Generate a structured pipeline plan for the user's request.
        
        Args:
            user_query: User's description of what they want to accomplish
            
        Returns:
            PipelinePlan: Structured pipeline plan with tools and reasoning
        """
        return await self.planner.suggest_pipeline(user_query)
    
    async def explain_pipeline(self, user_query: str) -> str:
        """
        Generate a conversational explanation of the suggested pipeline.
        
        Args:
            user_query: User's request
            
        Returns:
            Conversational explanation of the pipeline plan
        """
        return await self.planner.conversational_planning(user_query)
    
    async def execute_pipeline(self, 
                              plan: PipelinePlan, 
                              input_mmif: str) -> AsyncGenerator[ExecutionProgress, None]:
        """
        Execute a pipeline plan with progress streaming.
        
        Args:
            plan: The pipeline plan to execute
            input_mmif: Input MMIF data or file path
            
        Yields:
            ExecutionProgress: Real-time progress updates
        """
        async for progress in self.executor.execute_plan(plan, input_mmif):
            yield progress
    
    async def process_request(self, 
                             user_query: str, 
                             input_mmif: Optional[str] = None,
                             auto_execute: bool = False) -> Dict[str, Any]:
        """
        Complete hybrid processing: planning + optional execution.
        
        Args:
            user_query: User's request description
            input_mmif: Optional MMIF input for execution
            auto_execute: If True, automatically execute the plan
            
        Returns:
            Dictionary with plan, explanation, and execution results
        """
        try:
            # Phase 1: Planning
            plan = await self.plan_pipeline(user_query)
            explanation = await self.explain_pipeline(user_query)
            
            result = {
                "plan": plan,
                "explanation": explanation,
                "execution_results": None,
                "status": "planned"
            }
            
            # Phase 2: Optional execution
            if auto_execute and input_mmif:
                # Validate plan first
                issues = self.executor.validate_plan(plan)
                if issues:
                    result["status"] = "validation_failed"
                    result["validation_issues"] = issues
                    return result
                
                # Execute with progress collection
                execution_updates = []
                async for progress in self.execute_pipeline(plan, input_mmif):
                    execution_updates.append(progress)
                
                result["execution_results"] = execution_updates
                final_progress = execution_updates[-1] if execution_updates else None
                if final_progress and final_progress.status == "completed":
                    result["status"] = "completed"
                else:
                    result["status"] = "execution_failed"
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                "plan": None,
                "explanation": f"Sorry, I encountered an error: {str(e)}",
                "execution_results": None,
                "status": "error",
                "error": str(e)
            }
    
    def validate_pipeline(self, plan: PipelinePlan) -> List[str]:
        """Validate a pipeline plan and return any issues."""
        return self.executor.validate_plan(plan)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.executor.get_available_tools()
    
    def suggest_compatible_tools(self, last_tool_name: str) -> List[str]:
        """Suggest tools compatible with the last tool in pipeline."""
        if not last_tool_name or last_tool_name not in self.tool_metadata:
            # Return common starting tools
            return ['transnet-wrapper', 'whisper-wrapper', 'easyocr-wrapper']
        
        compatible_tools = []
        last_tool = self.tool_metadata[last_tool_name]
        
        for tool_name, metadata in self.tool_metadata.items():
            if tool_name == last_tool_name:
                continue
            
            # Check output -> input compatibility
            for output_type in last_tool.get('output_types', []):
                for input_type in metadata.get('input_types', []):
                    if self._types_compatible(output_type, input_type):
                        compatible_tools.append(tool_name)
                        break
        
        return compatible_tools[:5]  # Return top 5 suggestions
    
    def _types_compatible(self, output_type: str, input_type: str) -> bool:
        """Check if output type is compatible with input type."""
        # Normalize for comparison
        output_lower = output_type.lower()
        input_lower = input_type.lower()
        
        # Direct match
        if output_lower == input_lower:
            return True
        
        # Common compatibility patterns
        compatibility_map = {
            'videodocument': ['timeframe', 'boundingbox'],
            'timeframe': ['alignment', 'textdocument'],
            'alignment': ['textdocument'],
            'textdocument': ['namedentity', 'entity']
        }
        
        return input_lower in compatibility_map.get(output_lower, [])
    
    async def create_pipeline_from_conversation(self, thread_id: str = "default") -> PipelineModel:
        """Extract pipeline from conversation history."""
        # This would analyze the conversation and extract selected tools
        # For now, return a basic pipeline
        pipeline = PipelineModel(name="Generated Pipeline")
        
        # Add tools based on conversation analysis
        # This is a simplified implementation
        pipeline.add_node("example-tool", {"description": "Example tool"})
        
        return pipeline


# Aliases for convenience
LangGraphPipelineAgent = CLAMSAgent
ModernCLAMSAgent = CLAMSAgent