from typing import Dict, List, Any, Optional, Set, Tuple
import json
import logging
import re
from dataclasses import dataclass
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_backend import LLMBackend
from utils.download_app_directory import get_app_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    """Stores context for pipeline construction."""
    task_description: str
    current_step: int = 1
    selected_tools: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.selected_tools = self.selected_tools or []
    
    def add_selected_tool(self, tool_name: str, tool_info: Dict[str, Any]):
        """Add a selected tool to the pipeline."""
        self.selected_tools.append({
            "step": self.current_step,
            "name": tool_name,
            "info": tool_info
        })
        self.current_step += 1
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state as a formatted string."""
        if not self.selected_tools:
            return "No tools selected yet."
            
        state = "Current pipeline:\n"
        for tool in self.selected_tools:
            state += f"Step {tool['step']}: {tool['name']}\n"
        return state

class ValidationError(Exception):
    """Custom exception for pipeline validation errors."""
    pass

class PipelineGenerator:
    """
    Generates valid CLAMS tool pipelines using LLM and tool metadata.
    Ensures tool dependencies are met by validating input/output types.
    """
    
    def __init__(self, llm: LLMBackend):
        """
        Initialize the pipeline generator.
        
        Args:
            llm: LLMBackend instance for generating pipelines
        """
        self.llm = llm
        self.app_metadata = get_app_metadata()
        logger.info(f"Loaded app metadata type: {type(self.app_metadata)}")
        if isinstance(self.app_metadata, list):
            logger.error("App metadata is a list instead of expected dictionary!")
            logger.debug(f"App metadata content: {self.app_metadata}")
            # Convert list to dictionary if possible
            try:
                self.app_metadata = {app.get('name', f'app_{i}'): app for i, app in enumerate(self.app_metadata)}
                logger.info("Successfully converted app metadata list to dictionary")
            except Exception as e:
                logger.error(f"Failed to convert app metadata list to dictionary: {e}")
                raise
        
    def _format_tool_metadata(self) -> str:
        """
        Format tool metadata into a minimal string for the LLM prompt.
        Includes only essential information: name, version, brief description, and simplified I/O types.
        """
        metadata_str = "Available CLAMS Tools:\n\n"
        
        logger.info(f"Formatting metadata for {len(self.app_metadata)} tools")
        
        def simplify_type(type_str: str) -> str:
            """Extract the base annotation type from the full URI."""
            if not isinstance(type_str, str):
                return str(type_str)
            # Extract the last part after the last '/'
            parts = type_str.split('/')
            if len(parts) > 1:
                # Remove version number if present
                base_type = parts[-1].split('v')[0]
                return base_type
            return type_str
        
        for app_name, app_info in self.app_metadata.items():
            try:
                if isinstance(app_info, dict):
                    metadata = app_info.get("metadata", {})
                    version = app_info.get("latest_version", "unknown")
                elif isinstance(app_info, list):
                    metadata = next((item.get("metadata", {}) for item in app_info if isinstance(item, dict)), {})
                    version = "unknown"
                else:
                    continue
                
                if not metadata:
                    continue
                
                # Get brief description (first 100 characters)
                description = metadata.get('description', 'No description available')
                brief_description = description[:100] + ('...' if len(description) > 100 else '')
                
                # Format basic info
                metadata_str += f"Tool: {app_name} (v{version})\n"
                metadata_str += f"Description: {brief_description}\n"
                
                # Format simplified input types
                inputs = metadata.get("input", [])
                if inputs:
                    input_types = set()
                    for input_type in inputs:
                        if isinstance(input_type, dict):
                            type_str = input_type.get('@type', '')
                            if type_str:
                                input_types.add(simplify_type(type_str))
                    if input_types:
                        metadata_str += f"Inputs: {', '.join(sorted(input_types))}\n"
                
                # Format simplified output types
                outputs = metadata.get("output", [])
                if outputs:
                    output_types = set()
                    for output_type in outputs:
                        if isinstance(output_type, dict):
                            type_str = output_type.get('@type', '')
                            if type_str:
                                output_types.add(simplify_type(type_str))
                    if output_types:
                        metadata_str += f"Outputs: {', '.join(sorted(output_types))}\n"
                
                metadata_str += "\n"
                
            except Exception as e:
                logger.error(f"Error processing app {app_name}: {e}")
                continue
            
        return metadata_str
        
    def _create_pipeline_prompt(self, task_description: str) -> str:
        """
        Create a prompt for the LLM that includes tool metadata and task description.
        
        Args:
            task_description: Description of the video analysis task
            
        Returns:
            Complete prompt for the LLM
        """
        base_prompt = """You are a CLAMS pipeline generator. Your task is to create a valid pipeline of CLAMS tools 
that can accomplish the given task. Each tool in the pipeline must have its input requirements satisfied by either:
1. The initial video input (VideoDocument)
2. The output of a previous tool in the pipeline

Rules for pipeline generation:
1. Tools must be ordered so that each tool's input requirements are met
2. Each tool should serve a clear purpose in achieving the task
3. The pipeline should be efficient (avoid unnecessary tools)
4. Specify any relevant tool parameters that should be set

Below is the metadata for available CLAMS tools, including their input and output types:

{tool_metadata}

Task Description:
{task_description}

Generate a pipeline that accomplishes this task. For each tool in the pipeline:
1. Explain why it's needed
2. Specify what inputs it requires and where they come from
3. Describe what outputs it produces
4. List any important parameters that should be set

Format your response as:
PIPELINE:
1. First tool name
   - Purpose: Why this tool is needed
   - Inputs: What inputs it uses and their source
   - Outputs: What outputs it produces
   - Parameters: Any specific parameters to set
2. Second tool name
   ...etc.

EXPLANATION:
A brief explanation of how the pipeline accomplishes the task and why this order is necessary.
"""
        
        return base_prompt.format(
            tool_metadata=self._format_tool_metadata(),
            task_description=task_description
        )
        
    def chat_response(self, context: ChatContext, user_input: str) -> str:
        """
        Generate a response in the chat conversation.
        
        Args:
            context: Current chat context
            user_input: User's message
            
        Returns:
            Assistant's response
        """
        # Create prompt with current context
        prompt = self._create_chat_prompt(context)
        
        try:
            # Generate response
            response = self.llm.generate_response(prompt=prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"Error generating response: {str(e)}"
            
    def interactive_pipeline_design(self, task_description: str) -> Optional[str]:
        """
        Start an interactive chat session to design a pipeline.
        
        Args:
            task_description: Initial task description
            
        Returns:
            Generated pipeline description if successful, None otherwise
        """
        context = ChatContext(task_description=task_description)
        
        print("\nWelcome to the CLAMS Pipeline Designer!")
        print("Describe your task and I'll help you create an optimal pipeline.")
        print("Type 'done' when you're ready to generate the pipeline.")
        print("Type 'quit' to exit.\n")
        
        # Initial response
        initial_response = self.chat_response(
            context,
            f"I need to create a pipeline for this task: {task_description}"
        )
        print(f"\nAssistant: {initial_response}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    return None
                    
                if user_input.lower() == 'done':
                    # Generate final pipeline
                    print("\nGenerating pipeline based on our discussion...")
                    pipeline = self.generate_pipeline(
                        task_description,
                        context=context
                    )
                    
                    # Validate the pipeline
                    validation = self.validate_pipeline(pipeline)
                    if validation["is_valid"]:
                        return pipeline
                    else:
                        print("\nError: Generated pipeline is not valid:")
                        for error in validation["errors"]:
                            print(f"- {error}")
                        print("\nLet's continue our discussion to resolve these issues.")
                        continue
                
                # Get assistant's response
                response = self.chat_response(context, user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
    def generate_pipeline(
        self,
        task_description: str,
        max_new_tokens: Optional[int] = None,
        context: Optional[ChatContext] = None
    ) -> str:
        """
        Generate a valid CLAMS pipeline for the given task.
        
        Args:
            task_description: Description of what needs to be accomplished
            max_new_tokens: Optional maximum number of tokens to generate
            context: Optional chat context with additional information
            
        Returns:
            Generated pipeline description
        """
        # If we have chat context, include it in the prompt
        if context:
            prompt = f"""Based on our discussion about the task:
{task_description}

Current pipeline state:
{context.get_pipeline_state()}

Now, generate a complete pipeline description following this format:

PIPELINE:
1. First tool name
   - Purpose: Why this tool is needed
   - Inputs: What inputs it uses and their source
   - Outputs: What outputs it produces
   - Parameters: Any specific parameters to set
2. Second tool name
   ...etc.

EXPLANATION:
A brief explanation of how the pipeline accomplishes the task and why this order is necessary.
"""
        else:
            prompt = self._create_pipeline_prompt(task_description)
        
        try:
            response = self.llm.generate_response(
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating pipeline: {e}")
            return f"Error generating pipeline: {str(e)}"

    def _parse_pipeline_description(self, pipeline_description: str) -> List[Dict[str, Any]]:
        """
        Parse the pipeline description into a structured format.
        
        Args:
            pipeline_description: The pipeline description from the LLM
            
        Returns:
            List of dictionaries containing parsed tool information
        """
        # Split into pipeline and explanation sections
        sections = pipeline_description.split("EXPLANATION:")
        pipeline_section = sections[0].split("PIPELINE:")[1].strip()
        
        tools = []
        current_tool = {}
        
        # Parse each tool entry
        for line in pipeline_section.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for new tool entry (starts with number)
            if re.match(r'^\d+\.', line):
                if current_tool:
                    tools.append(current_tool)
                current_tool = {"name": line.split('.')[1].strip().lower()}
                continue
                
            # Parse tool properties
            if line.startswith('-'):
                prop_match = re.match(r'-\s*(\w+):\s*(.+)', line)
                if prop_match:
                    key, value = prop_match.groups()
                    current_tool[key.lower()] = value.strip()
                    
        # Add last tool
        if current_tool:
            tools.append(current_tool)
            
        return tools

    def _get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool metadata dictionary
        """
        # Try exact match first
        if tool_name in self.app_metadata:
            return self.app_metadata[tool_name].get("metadata", {})
            
        # Try case-insensitive match
        tool_name_lower = tool_name.lower()
        for name, info in self.app_metadata.items():
            if name.lower() == tool_name_lower:
                return info.get("metadata", {})
                
        return {}

    def _get_available_types(self, tools_so_far: List[Dict[str, Any]]) -> Set[str]:
        """
        Get all available output types from tools used so far.
        
        Args:
            tools_so_far: List of tools that come before in the pipeline
            
        Returns:
            Set of available output types
        """
        available_types = {"http://mmif.clams.ai/vocabulary/VideoDocument/v1"}
        
        for tool in tools_so_far:
            tool_metadata = self._get_tool_metadata(tool["name"])
            for output_type in tool_metadata.get("output", []):
                available_types.add(output_type.get("@type", ""))
                
        return available_types

    def _validate_tool_parameters(self, tool_name: str, parameters: str) -> Tuple[bool, str]:
        """
        Validate that specified parameters exist and have valid values.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameter string from pipeline description
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool_metadata = self._get_tool_metadata(tool_name)
        valid_params = {
            param["name"]: param 
            for param in tool_metadata.get("parameters", [])
        }
        
        if not parameters or parameters.lower() == "none" or parameters.lower() == "default":
            return True, ""
            
        # Parse parameters string
        try:
            param_pairs = [p.strip() for p in parameters.split(',')]
            for pair in param_pairs:
                name, value = pair.split('=')
                name = name.strip()
                value = value.strip()
                
                if name not in valid_params:
                    return False, f"Invalid parameter '{name}' for tool '{tool_name}'"
                    
                param_info = valid_params[name]
                
                # Check parameter type
                if param_info.get("type") == "boolean":
                    if value.lower() not in ["true", "false"]:
                        return False, f"Parameter '{name}' must be boolean"
                elif param_info.get("type") == "integer":
                    try:
                        int(value)
                    except ValueError:
                        return False, f"Parameter '{name}' must be integer"
                elif param_info.get("type") == "number":
                    try:
                        float(value)
                    except ValueError:
                        return False, f"Parameter '{name}' must be number"
                        
                # Check choices if specified
                choices = param_info.get("choices", [])
                if choices and value not in choices:
                    return False, f"Value '{value}' not in allowed choices for parameter '{name}'"
                    
        except Exception as e:
            return False, f"Error parsing parameters: {str(e)}"
            
        return True, ""
            
    def validate_pipeline(self, pipeline_description: str) -> Dict[str, Any]:
        """
        Validate a generated pipeline to ensure all tool dependencies are met.
        
        Args:
            pipeline_description: The pipeline description from the LLM
            
        Returns:
            Dictionary containing validation results:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "tools": List[Dict] # Parsed tool information
            }
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "tools": []
        }
        
        try:
            # Parse pipeline description
            tools = self._parse_pipeline_description(pipeline_description)
            result["tools"] = tools
            
            # Validate each tool in sequence
            available_types = {"http://mmif.clams.ai/vocabulary/VideoDocument/v1"}
            
            for i, tool in enumerate(tools):
                tool_name = tool["name"]
                tool_metadata = self._get_tool_metadata(tool_name)
                
                # Check if tool exists
                if not tool_metadata:
                    result["errors"].append(f"Tool '{tool_name}' not found in available tools")
                    continue
                    
                # Validate inputs
                required_inputs = {
                    input_type["@type"]
                    for input_type in tool_metadata.get("input", [])
                    if input_type.get("required", False)
                }
                
                missing_inputs = required_inputs - available_types
                if missing_inputs:
                    result["errors"].append(
                        f"Tool '{tool_name}' requires inputs {missing_inputs} "
                        f"which are not available at step {i+1}"
                    )
                    
                # Validate parameters
                if "parameters" in tool:
                    is_valid, error = self._validate_tool_parameters(
                        tool_name,
                        tool["parameters"]
                    )
                    if not is_valid:
                        result["errors"].append(error)
                        
                # Add tool outputs to available types
                for output_type in tool_metadata.get("output", []):
                    available_types.add(output_type.get("@type", ""))
                    
            result["is_valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Error validating pipeline: {str(e)}")
            
        return result 

    def _get_tools_by_type(self, type_name: str) -> List[str]:
        """Get all tools that output a specific type."""
        tools = []
        for app_name, app_info in self.app_metadata.items():
            metadata = app_info.get("metadata", {})
            outputs = metadata.get("output", [])
            if any(output.get("@type", "").lower().endswith(type_name.lower()) 
                  for output in outputs):
                tools.append(app_name)
        return tools
        
    def _create_chat_prompt(self, context: ChatContext) -> str:
        """
        Create a prompt for the chat interaction.
        
        Args:
            context: Current chat context
            
        Returns:
            Prompt for the LLM
        """
        base_prompt = """You are a CLAMS pipeline assistant helping users create optimal pipelines for video analysis tasks.
You have access to these CLAMS tools and their capabilities:

{tool_metadata}

Current task: {task_description}

Current pipeline state:
{pipeline_state}

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. DO NOT generate or simulate user messages or choices
2. DO NOT invent tools that don't exist in the tool metadata
3. DO NOT add unnecessary steps to the pipeline
4. ONLY present tools that are directly relevant to the current step
5. For video text analysis tasks:
   - Only suggest tools that process video or images
   - Do not suggest audio processing tools
   - Do not suggest tools unrelated to the current step

Current Step {current_step}:
For this step, list ONLY tools that:
1. Can process the outputs from previous steps (if any)
2. Produce outputs needed for the task
3. Are specifically relevant to {task_description}

Format your response like this:
"Step {current_step}: [Current step description]

Available tools for this step:
1. [Tool Name] (v[version])
   - Inputs: [input types]
   - Outputs: [output types]
   - Capabilities: [key features]
   - Parameters: [important parameters]
   - Advantages: [pros]
   - Disadvantages: [cons]

2. [Next tool option...]

Please choose a tool for this step."

"""
        return base_prompt.format(
            tool_metadata=self._format_tool_metadata(),
            task_description=context.task_description,
            pipeline_state=context.get_pipeline_state(),
            current_step=context.current_step
        )

def main():
    """Command-line interface for the pipeline generator."""
    from utils.llm_backend import LLMBackend
    
    # Initialize LLM and pipeline generator
    llm = LLMBackend()
    pipeline_gen = PipelineGenerator(llm)
    
    print("CLAMS Pipeline Generator")
    print("=======================")
    
    while True:
        try:
            task = input("\nEnter your task (or 'quit' to exit): ").strip()
            
            if task.lower() == 'quit':
                break
                
            if 'pipeline' in task.lower():
                # Start interactive session
                pipeline = pipeline_gen.interactive_pipeline_design(task)
                if pipeline:
                    print("\nGenerated Pipeline:")
                    print("==================")
                    print(pipeline)
            else:
                # Regular chat response
                context = ChatContext(task_description=task)
                response = pipeline_gen.chat_response(context, task)
                print(f"\nAssistant: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main() 