from typing import List, Dict, Any, Optional, TypedDict, Annotated
import logging
import re
import operator
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .pipeline_model import PipelineModel, PipelineStore
from .clams_tools import CLAMSToolbox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State of the LangGraph agent."""
    messages: Annotated[List[dict], operator.add]
    task_description: str
    current_step: int
    selected_tools: List[Dict[str, Any]]
    pipeline_state: str
    tool_metadata: Dict[str, Any]
    pipeline: PipelineModel
    conversation_history: List[Dict[str, str]]

@dataclass
class ChatContext:
    """Stores context for pipeline construction."""
    task_description: str
    current_step: int = 1
    selected_tools: List[Dict[str, Any]] = None
    pipeline_state: Dict[str, Any] = None
    conversation_history: List[Dict[str, str]] = None
    pipeline: Optional[PipelineModel] = None
    
    def __post_init__(self):
        self.selected_tools = self.selected_tools or []
        self.pipeline_state = self.pipeline_state or {}
        self.conversation_history = self.conversation_history or []
        self.pipeline = self.pipeline or PipelineModel(name="Chat Generated Pipeline")
    
    def add_selected_tool(self, tool_name: str, tool_info: Dict[str, Any]):
        """Add a selected tool to the pipeline."""
        self.selected_tools.append({
            "step": self.current_step,
            "name": tool_name,
            "info": tool_info
        })
        
        # Add to the pipeline model as well
        node_id = self.pipeline.add_node(tool_name, tool_info)
        
        # If this isn't the first tool, connect it to the previous one
        if len(self.selected_tools) > 1:
            prev_tool = self.selected_tools[-2]
            prev_node_id = f"{prev_tool['name']}-{self.current_step - 2}"
            self.pipeline.add_edge(prev_node_id, node_id)
            
        self.current_step += 1
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state as a formatted string."""
        if not self.selected_tools:
            return "No tools selected yet."
            
        state = "Current pipeline:\n"
        for tool in self.selected_tools:
            state += f"Step {tool['step']}: {tool['name']}\n"
            if 'input_types' in tool['info']:
                state += f"  - Inputs: {', '.join(tool['info']['input_types'])}\n"
            if 'output_types' in tool['info']:
                state += f"  - Outputs: {', '.join(tool['info']['output_types'])}\n"
        return state
    
    def export_pipeline(self, name: Optional[str] = None) -> str:
        """Export the current pipeline to YAML."""
        if name:
            self.pipeline.name = name
        return self.pipeline.to_yaml()
    
    def save_pipeline(self, storage_dir: str = "data/pipelines", name: Optional[str] = None) -> str:
        """Save the current pipeline to a file."""
        if name:
            self.pipeline.name = name
            
        store = PipelineStore(storage_dir)
        return store.save_pipeline(self.pipeline)

class LangGraphPipelineAgent:
    """
    LangGraph-based pipeline agent for CLAMS pipeline generation.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LangGraph pipeline agent.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        
        # Initialize CLAMS toolbox
        self.toolbox = CLAMSToolbox()
        self.tools = list(self.toolbox.get_tools().values())
        
        # Tool metadata cache
        self.tool_metadata = self._initialize_tool_metadata()
        
        # Pipeline storage
        self.pipeline_store = PipelineStore()
        
        # Create the LangGraph
        self.graph = self._create_graph()
        
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
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create a workflow
        workflow = StateGraph(AgentState)
        
        # Create system prompt
        system_prompt = """You are a helpful assistant for designing CLAMS pipelines.
        You have access to various CLAMS tools for video analysis.

        Your task is to help users create pipelines of interoperable CLAMS tools. A pipeline is interoperable when:
        1. Each tool's OUTPUT type must be compatible with the INPUT type required by the next tool
        2. The first tool in the pipeline must accept the input type that the user has available
        3. The last tool in the pipeline must produce the output type that the user needs

        Current task: {task_description}

        Current pipeline state:
        {pipeline_state}

        Available tools (Format: Tool: <Name> | Inputs: <Inputs> | Outputs: <Outputs>):
        {tool_details}

        IMPORTANT: Always check that consecutive tools are compatible by ensuring the output types of one tool match the input types required by the next.

        When suggesting tools, explain why each tool is suitable and how it fits into the pipeline."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
        
        # Define nodes
        def agent_node(state: AgentState) -> AgentState:
            """Agent node that processes user input and generates responses."""
            logger.info(f"Agent node processing: {state}")
            
            # Get the latest message
            messages = state["messages"]
            if not messages:
                return state
                
            # Format the prompt with current state
            tool_details = self.get_tool_details()
            
            # Create context message
            context_msg = f"""
            Current task: {state.get('task_description', 'Create a CLAMS pipeline')}
            
            Current pipeline state:
            {state.get('pipeline_state', 'No tools selected yet.')}
            
            Available tools:
            {tool_details}
            """
            
            # Execute the agent
            try:
                result = agent_executor.invoke({
                    "messages": [
                        SystemMessage(content=context_msg),
                        HumanMessage(content=messages[-1]["content"])
                    ]
                })
                
                # Add the response to messages
                new_messages = [{"role": "assistant", "content": result["output"]}]
                
                # Update state
                updated_state = {
                    **state,
                    "messages": new_messages
                }
                
                return updated_state
                
            except Exception as e:
                logger.error(f"Error in agent node: {e}")
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                return {
                    **state,
                    "messages": [{"role": "assistant", "content": error_msg}]
                }
        
        def tool_selection_node(state: AgentState) -> AgentState:
            """Node that processes tool selection and updates pipeline state."""
            logger.info("Tool selection node processing")
            
            # Check if the latest message contains a tool suggestion
            messages = state["messages"]
            if not messages:
                return state
                
            latest_message = messages[-1]["content"]
            
            # Look for tool suggestions in the message
            tool_match = re.search(r'I suggest using the "?([a-zA-Z0-9_-]+)"? tool', latest_message)
            if tool_match:
                tool_name = tool_match.group(1)
                if tool_name in self.tool_metadata:
                    # Add the tool to selected tools
                    selected_tools = state.get("selected_tools", [])
                    current_step = state.get("current_step", 1)
                    
                    selected_tools.append({
                        "step": current_step,
                        "name": tool_name,
                        "info": self.tool_metadata[tool_name]
                    })
                    
                    # Update pipeline state
                    pipeline_state = self._format_pipeline_state(selected_tools)
                    
                    return {
                        **state,
                        "selected_tools": selected_tools,
                        "current_step": current_step + 1,
                        "pipeline_state": pipeline_state
                    }
            
            return state
        
        # Add nodes to the workflow
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool_selection", tool_selection_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_edge("agent", "tool_selection")
        workflow.add_edge("tool_selection", END)
        
        return workflow
    
    def _format_pipeline_state(self, selected_tools: List[Dict[str, Any]]) -> str:
        """Format pipeline state for display."""
        if not selected_tools:
            return "No tools selected yet."
            
        state = "Current pipeline:\n"
        for tool in selected_tools:
            state += f"Step {tool['step']}: {tool['name']}\n"
            if 'input_types' in tool['info']:
                state += f"  - Inputs: {', '.join(tool['info']['input_types'])}\n"
            if 'output_types' in tool['info']:
                state += f"  - Outputs: {', '.join(tool['info']['output_types'])}\n"
        return state
    
    def get_tool_details(self) -> str:
        """Get detailed information about available tools."""
        tool_details = []
        
        for name, tool_info in self.tool_metadata.items():
            inputs = ', '.join(tool_info.get('input_types', [])) or 'None'
            outputs = ', '.join(tool_info.get('output_types', [])) or 'None'
            detail = f"Tool: {name} | Inputs: {inputs} | Outputs: {outputs}"
            tool_details.append(detail)
            
        return "\n".join(tool_details)
    
    def chat_response(self, context: ChatContext, user_input: str) -> str:
        """
        Generate a response using the LangGraph agent.
        
        Args:
            context: Current chat context
            user_input: User's latest message
            
        Returns:
            Agent's response string
        """
        # Create initial state
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "task_description": context.task_description,
            "current_step": context.current_step,
            "selected_tools": context.selected_tools,
            "pipeline_state": context.get_pipeline_state(),
            "tool_metadata": self.tool_metadata,
            "conversation_history": context.conversation_history
        }
        
        # Execute the graph
        try:
            result = self.app.invoke(state)
            
            # Extract the response
            messages = result.get("messages", [])
            if messages:
                response = messages[-1]["content"]
                return response
            else:
                return "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error in chat_response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
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
        return context.save_pipeline(name=name)
    
    def interactive_pipeline_design(self, task_description: str) -> Optional[PipelineModel]:
        """
        Start an interactive chat session to design a pipeline.
        
        Args:
            task_description: Initial task description
            
        Returns:
            The generated pipeline model if completed successfully, None otherwise
        """
        context = ChatContext(
            task_description=task_description,
            pipeline_state={},
            conversation_history=[]
        )
        
        print("\nWelcome to the CLAMS Pipeline Designer!")
        print("Describe your task and I'll help you create an optimal pipeline.")
        print("Type 'done' when you're ready to generate the pipeline.")
        print("Type 'export' to export the pipeline to YAML.")
        print("Type 'save' to save the pipeline to a file.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("User: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                    
                if user_input.lower() == 'done':
                    print("\nGenerating pipeline...")
                    final_pipeline = context.get_pipeline_state()
                    print(f"\nFinal Pipeline:\n{final_pipeline}")
                    return context.pipeline
                
                if user_input.lower() == 'export':
                    name = input("Enter pipeline name (or press Enter to use default): ").strip()
                    yaml_str = self.export_pipeline(context, name or None)
                    print(f"\nPipeline YAML:\n{yaml_str}")
                    continue
                
                if user_input.lower() == 'save':
                    name = input("Enter pipeline name (or press Enter to use default): ").strip()
                    path = self.save_pipeline(context, name or None)
                    print(f"\nPipeline saved to: {path}")
                    continue
                
                # Add user message to history
                context.conversation_history.append({
                    "role": "User",
                    "content": user_input
                })
                
                # Generate and print response
                response = self.chat_response(context, user_input)
                print(f"\nAssistant: {response}\n")
                
                # Add assistant response to history
                context.conversation_history.append({
                    "role": "Assistant",
                    "content": response
                })
                
                # Check if response contains a tool suggestion
                tool_match = re.search(r'I suggest using the "?([a-zA-Z0-9_-]+)"? tool', response)
                if tool_match and tool_match.group(1) in self.tool_metadata:
                    tool_name = tool_match.group(1)
                    context.add_selected_tool(tool_name, self.tool_metadata[tool_name])
                    
                    compatible_tools = self.suggest_compatible_tools(tool_name)
                    if compatible_tools:
                        print(f"\nTools compatible with {tool_name}: {', '.join(compatible_tools)}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
        return None

# Alias for backward compatibility
PipelineAgentChat = LangGraphPipelineAgent 