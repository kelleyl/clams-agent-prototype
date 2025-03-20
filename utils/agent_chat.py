from typing import List, Dict, Any, Optional
import logging
import re
from dataclasses import dataclass
from transformers import ReactCodeAgent, HfApiEngine
import os

from .pipeline_model import PipelineModel, PipelineStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable for faster downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
        """
        Export the current pipeline to YAML.
        
        Args:
            name: Optional name for the pipeline
            
        Returns:
            YAML string of the pipeline
        """
        if name:
            self.pipeline.name = name
        return self.pipeline.to_yaml()
    
    def save_pipeline(self, storage_dir: str = "data/pipelines", name: Optional[str] = None) -> str:
        """
        Save the current pipeline to a file.
        
        Args:
            storage_dir: Directory to save the pipeline
            name: Optional name for the pipeline
            
        Returns:
            Path to the saved pipeline
        """
        if name:
            self.pipeline.name = name
            
        store = PipelineStore(storage_dir)
        return store.save_pipeline(self.pipeline)

class PipelineAgentChat:
    """
    Agent-based chat interface for CLAMS pipeline generation using Hugging Face's Transformers Agents.
    """
    
    def __init__(self, model_name: str = "unsloth/DeepSeek-R1-GGUF"):
        """
        Initialize the pipeline agent chat.
        
        Args:
            model_name: Name of the model to use for the agent
        """
        # Initialize the LLM engine with specific configuration for DeepSeek-R1
        self.llm_engine = HfApiEngine(
            model=model_name,
            max_tokens=4096,  # Maximum number of tokens in the response
            timeout=120  # API request timeout in seconds
        )
        
        # Create the agent with custom system prompt
        system_prompt = """You are a helpful assistant for designing CLAMS pipelines.
You have access to various CLAMS tools for video analysis.

<<authorized_imports>>
import json
import re
from typing import Dict, Any, List

Your task is to help users create pipelines of interoperable CLAMS tools. A pipeline is interoperable when:
1. Each tool's OUTPUT type must be compatible with the INPUT type required by the next tool
2. The first tool in the pipeline must accept the input type that the user has available
3. The last tool in the pipeline must produce the output type that the user needs

Current task: {task_description}

Current pipeline state:
{pipeline_state}

Tool compatibility information:
{compatibility_info}

Previous conversation:
{conversation_history}

Available tools:
{tool_details}

IMPORTANT: Always check that consecutive tools are compatible. A tool is compatible with the next one if its output types match or include the input types required by the next tool.

User: {user_input}
Assistant:"""

        # Initialize with empty tools list
        self.agent = ReactCodeAgent(
            tools=[],  # Empty list for now, will be populated later
            llm_engine=self.llm_engine,
            system_prompt=system_prompt
        )
        
        # Tool metadata cache
        self.tool_metadata = {}
        
        # Pipeline storage
        self.pipeline_store = PipelineStore()
        
        # Initialize tools from app directory
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize tools from the app directory JSON file."""
        import json
        import os
        
        # Load app directory
        app_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'app_directory.json')
        if not os.path.exists(app_dir_path):
            raise FileNotFoundError(f"App directory not found at {app_dir_path}")
            
        with open(app_dir_path, 'r') as f:
            app_directory = json.load(f)
        
        # Process each app in the directory
        for app_name, app_info in app_directory.items():
            metadata = app_info.get('metadata', {})
            
            # Extract input types
            input_types = []
            for input_type in metadata.get('input', []):
                if isinstance(input_type, dict) and '@type' in input_type:
                    # Extract the type name from the URI
                    type_uri = input_type['@type']
                    type_name = type_uri.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '').replace('v4', '').replace('v5', '')
                    input_types.append(type_name)
            
            # Extract output types
            output_types = []
            for output_type in metadata.get('output', []):
                if isinstance(output_type, dict) and '@type' in output_type:
                    # Extract the type name from the URI
                    type_uri = output_type['@type']
                    type_name = type_uri.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '').replace('v4', '').replace('v5', '')
                    output_types.append(type_name)
            
            # Create tool metadata
            self.tool_metadata[app_name] = {
                'description': metadata.get('description', ''),
                'input_types': input_types,
                'output_types': output_types,
                'parameters': metadata.get('parameters', [])
            }
            
        if not self.tool_metadata:
            raise ValueError("No tools were loaded from the app directory")
            
        logger.info(f"Loaded {len(self.tool_metadata)} tools from app directory")
    
    def get_tool_details(self) -> str:
        """Get detailed information about available tools."""
        tool_details = []
        
        for name, tool_info in self.tool_metadata.items():
            # Create tool detail string
            detail = f"Tool: {name}\n"
            detail += f"Description: {tool_info['description']}\n"
            detail += f"Input Types: {', '.join(tool_info['input_types'])}\n"
            detail += f"Output Types: {', '.join(tool_info['output_types'])}\n"
            detail += "---\n"
            
            tool_details.append(detail)
            
        return "\n".join(tool_details)
    
    def get_compatibility_info(self) -> str:
        """Generate information about which tools are compatible with each other."""
        compatibility = []
        
        # For each tool, find compatible next tools
        for name, metadata in self.tool_metadata.items():
            compatible_tools = []
            
            # A tool is compatible if its output matches another tool's input
            for next_name, next_metadata in self.tool_metadata.items():
                if name == next_name:
                    continue
                    
                # Check if any output type matches any input type
                for output_type in metadata.get('output_types', []):
                    for input_type in next_metadata.get('input_types', []):
                        # Simple string matching (can be improved with semantic matching)
                        if output_type.lower() in input_type.lower() or input_type.lower() in output_type.lower():
                            compatible_tools.append(next_name)
                            break
            
            if compatible_tools:
                compatibility.append(f"{name} can be followed by: {', '.join(compatible_tools)}")
        
        return "\n".join(compatibility)
        
    def chat_response(self, context: ChatContext, user_input: str) -> str:
        """
        Generate a response in the chat conversation using the agent.
        
        Args:
            context: Current chat context
            user_input: User's message
            
        Returns:
            Assistant's response
        """
        try:
            # Format conversation history
            history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context.conversation_history
            ])
            
            # Get pipeline state
            pipeline_state = context.get_pipeline_state()
            
            # Get tool details and compatibility info
            tool_details = self.get_tool_details()
            compatibility_info = self.get_compatibility_info()
            
            # Generate response
            response = self.agent.run(
                task=context.task_description,
                pipeline_state=pipeline_state,
                compatibility_info=compatibility_info,
                conversation_history=history,
                tool_details=tool_details,
                user_input=user_input
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return f"Error generating response: {str(e)}"
            
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
        """
        Export the current pipeline to YAML.
        
        Args:
            context: Current chat context
            name: Optional name for the pipeline
            
        Returns:
            YAML string of the pipeline
        """
        return context.export_pipeline(name)
    
    def save_pipeline(self, context: ChatContext, name: Optional[str] = None) -> str:
        """
        Save the current pipeline to a file.
        
        Args:
            context: Current chat context
            name: Optional name for the pipeline
            
        Returns:
            Path to the saved pipeline
        """
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
                    # Generate a final report of the pipeline
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
                # This is a simple heuristic and could be improved
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