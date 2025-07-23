from typing import List, Dict, Any, Optional, TypedDict, Annotated
import logging
import re
import json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .pipeline_model import PipelineModel, PipelineStore
from .clams_tools import CLAMSToolbox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Simplified state following the ReAct pattern from the example."""
    messages: Annotated[list[AnyMessage], add_messages]
    task_description: Optional[str]
    selected_tools: List[str]  # List of tool names used in pipeline
    pipeline: PipelineModel

@dataclass
class ChatContext:
    """Stores context for pipeline construction."""
    task_description: str
    selected_tools: List[str] = None
    conversation_history: List[Dict[str, str]] = None
    pipeline: Optional[PipelineModel] = None
    
    def __post_init__(self):
        self.selected_tools = self.selected_tools or []
        self.conversation_history = self.conversation_history or []
        self.pipeline = self.pipeline or PipelineModel(name="Chat Generated Pipeline")
    
    def add_selected_tool(self, tool_name: str, tool_metadata: Dict[str, Any]):
        """Add a selected tool to the pipeline."""
        if tool_name not in self.selected_tools:
            self.selected_tools.append(tool_name)
            
            # Add to the pipeline model
            node_id = self.pipeline.add_node(tool_name, tool_metadata)
            
            # If this isn't the first tool, connect it to the previous one
            if len(self.selected_tools) > 1:
                prev_tool = self.selected_tools[-2]
                prev_node_id = f"{prev_tool}-{len(self.selected_tools) - 2}"
                self.pipeline.add_edge(prev_node_id, node_id)
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state as a formatted string."""
        if not self.selected_tools:
            return "No tools selected yet."
            
        state = "Current pipeline:\n"
        for i, tool_name in enumerate(self.selected_tools, 1):
            state += f"Step {i}: {tool_name}\n"
        
        return state
    
    def export_pipeline(self, name: Optional[str] = None) -> str:
        """Export the current pipeline to YAML."""
        if name:
            self.pipeline.name = name
        return self.pipeline.to_yaml()

class LangGraphPipelineAgent:
    """
    LangGraph-based pipeline agent for CLAMS pipeline generation using ReAct pattern.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the LangGraph pipeline agent.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        
        # Initialize OpenAI model (following the example)
        self.llm = ChatOpenAI(model=model_name)
        
        # Initialize CLAMS toolbox
        self.toolbox = CLAMSToolbox()
        self.tools = list(self.toolbox.get_tools().values())
        
        # Tool metadata cache for pipeline generation
        self.tool_metadata = self._initialize_tool_metadata()
        
        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        
        # Pipeline storage
        self.pipeline_store = PipelineStore()
        
        # Create the ReAct graph
        self.graph = self._create_react_graph()
        
        # Create memory checkpoint
        self.checkpointer = MemorySaver()
        
        # Compile the graph
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        
    def _initialize_tool_metadata(self) -> Dict[str, Any]:
        """Initialize tool metadata from the toolbox."""
        metadata = {}
        
        for tool_name, tool in self.toolbox.get_tools().items():
            app_info = tool.app_metadata
            tool_metadata = app_info.get('metadata', {})
            
            # Extract input types
            input_types = []
            for input_type in tool_metadata.get('input', []):
                if isinstance(input_type, dict) and '@type' in input_type:
                    type_uri = input_type['@type']
                    type_name = type_uri.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '').replace('v4', '').replace('v5', '')
                    if type_name:
                        input_types.append(type_name)
            
            # Extract output types
            output_types = []
            for output_type in tool_metadata.get('output', []):
                if isinstance(output_type, dict) and '@type' in output_type:
                    type_uri = output_type['@type']
                    type_name = type_uri.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '').replace('v4', '').replace('v5', '')
                    if type_name:
                        output_types.append(type_name)
            
            metadata[tool_name] = {
                'description': tool_metadata.get('description', ''),
                'input_types': input_types,
                'output_types': output_types,
                'parameters': tool_metadata.get('parameters', [])
            }
        
        return metadata
    
    def _create_react_graph(self) -> StateGraph:
        """Create the ReAct graph following the example pattern."""
        # Create the graph
        builder = StateGraph(AgentState)
        
        # Define the assistant node
        def assistant(state: AgentState):
            # Create system message for CLAMS pipeline assistance
            textual_description_of_tools = self._get_textual_tool_descriptions()
            
            task = state.get("task_description", "Create a CLAMS pipeline")
            selected_tools = state.get("selected_tools", [])
            
            sys_msg = SystemMessage(content=f"""You are a helpful assistant that helps users create CLAMS (Computational Language and Multimodal Analytics Studio) pipelines.

You have access to these CLAMS tools:
{textual_description_of_tools}

Current task: {task}
Selected tools so far: {', '.join(selected_tools) if selected_tools else 'None'}

Your job is to:
1. Understand what the user wants to accomplish
2. Suggest appropriate CLAMS tools for their task
3. Use the tools to demonstrate pipeline creation
4. Explain how tools work together in a pipeline

When suggesting tools, explain why each tool is suitable and how it fits into the pipeline. Always consider tool compatibility - the output types of one tool should match the input types of the next tool.

If a user asks you to use a specific tool or create a pipeline, go ahead and call the appropriate tools to demonstrate the pipeline.""")

            return {
                "messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]
            }
        
        # Define nodes: these do the work
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message requires a tool, route to tools
            # Otherwise, provide a direct response
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        return builder
    
    def _get_textual_tool_descriptions(self) -> str:
        """Get textual descriptions of all tools."""
        descriptions = []
        
        for tool in self.tools:
            # Get the first few lines of the description for brevity
            desc_lines = tool.description.split('\n')
            short_desc = desc_lines[0] if desc_lines else f"CLAMS tool: {tool.name}"
            descriptions.append(f"{tool.name}: {short_desc}")
            
        return "\n".join(descriptions)
    
    def chat_response(self, context: ChatContext, user_input: str) -> str:
        """
        Generate a response using the ReAct pattern.
        
        Args:
            context: Current chat context
            user_input: User's latest message
            
        Returns:
            Agent's response string
        """
        # Create initial state
        state = {
            "messages": [HumanMessage(content=user_input)],
            "task_description": context.task_description,
            "selected_tools": context.selected_tools,
            "pipeline": context.pipeline
        }
        
        # Execute the graph
        try:
            # Create configuration for checkpointer
            config = {"configurable": {"thread_id": "default"}}
            result = self.app.invoke(state, config=config)
            
            # Extract the final assistant message
            messages = result.get("messages", [])
            if messages:
                # Get the last assistant message
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        # Update context with any tools that were used
                        self._update_context_from_messages(context, messages)
                        return msg.content
                        
                # Fallback to last message content
                return messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            else:
                return "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error in chat_response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _update_context_from_messages(self, context: ChatContext, messages: List[AnyMessage]):
        """Update the chat context based on tool calls in the message history."""
        for msg in messages:
            # Check for tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call['name']
                    if tool_name in self.tool_metadata:
                        context.add_selected_tool(tool_name, self.tool_metadata[tool_name])
    
    def suggest_compatible_tools(self, last_tool_name: str) -> List[str]:
        """Suggest tools that are compatible with the last tool in the pipeline."""
        if not last_tool_name or last_tool_name not in self.tool_metadata:
            return []
            
        compatible_tools = []
        last_tool = self.tool_metadata[last_tool_name]
        
        for name, metadata in self.tool_metadata.items():
            if name == last_tool_name:
                continue
                
            # Check if any output type matches any input type
            for output_type in last_tool.get('output_types', []):
                for input_type in metadata.get('input_types', []):
                    if output_type.lower() in input_type.lower() or input_type.lower() in output_type.lower():
                        compatible_tools.append(name)
                        break
        
        return compatible_tools
    
    def export_pipeline(self, context: ChatContext, name: Optional[str] = None) -> str:
        """Export the current pipeline to YAML."""
        return context.export_pipeline(name)
    
    def save_pipeline(self, context: ChatContext, name: Optional[str] = None) -> str:
        """Save the current pipeline to a file."""
        store = PipelineStore()
        if name:
            context.pipeline.name = name
        return store.save_pipeline(context.pipeline)

PipelineAgentChat = LangGraphPipelineAgent 