from typing import Dict, Any, Optional
import json
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from .download_app_directory import get_app_metadata

class CLAMSTool(BaseTool):
    """LangChain tool for CLAMS applications."""
    
    def __init__(self, name: str, description: str, app_metadata: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.description = description
        self.app_metadata = app_metadata
    
    def _run(
        self,
        video: str,
        parameters: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the CLAMS tool on the given video with optional parameters.
        
        Args:
            video: Path to the input video file
            parameters: JSON string containing optional parameters
            run_manager: Callback manager for tool execution
            
        Returns:
            JSON string containing MMIF annotations
        """
        # Parse parameters if provided
        params_dict = {}
        if parameters:
            try:
                params_dict = json.loads(parameters)
            except json.JSONDecodeError:
                raise ValueError("Parameters must be a valid JSON string")
                
        # This is a placeholder - in real implementation this would call the CLAMS app
        result = {"annotations": f"Processed {video} with {self.name}"}
        if params_dict:
            result["parameters_used"] = params_dict
            
        return json.dumps(result)
    
    async def _arun(
        self,
        video: str,
        parameters: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run."""
        return self._run(video, parameters, run_manager)

class CLAMSToolbox:
    """Collection of CLAMS tools for use with LangChain agents."""
    
    def __init__(self):
        """Initialize the CLAMS toolbox."""
        self.app_metadata = get_app_metadata()
        self.tools = self._create_tools()
        
    def _create_tools(self) -> Dict[str, BaseTool]:
        """Create BaseTool instances for each CLAMS app."""
        tools = {}
        
        for app_name, app_info in self.app_metadata.items():
            metadata = app_info.get("metadata", {})
            
            # Extract and format input types
            input_types = []
            for input_type in metadata.get('input', []):
                if isinstance(input_type, dict):
                    type_str = input_type.get('@type', 'unknown')
                    if input_type.get('required', False):
                        type_str += ' (required)'
                    input_types.append(type_str)
                elif isinstance(input_type, list):
                    # Handle nested lists of input types (alternatives)
                    alt_types = []
                    for alt in input_type:
                        if isinstance(alt, dict):
                            alt_types.append(alt.get('@type', 'unknown'))
                    if alt_types:
                        input_types.append(f"one of [{', '.join(alt_types)}]")
            
            # Extract and format output types
            output_types = []
            for output_type in metadata.get('output', []):
                if isinstance(output_type, dict):
                    type_str = output_type.get('@type', 'unknown')
                    if 'properties' in output_type:
                        props = []
                        if 'timeUnit' in output_type['properties']:
                            props.append(f"timeUnit={output_type['properties']['timeUnit']}")
                        if 'labelset' in output_type['properties']:
                            props.append(f"labelset={output_type['properties']['labelset']}")
                        if props:
                            type_str += f" ({', '.join(props)})"
                    output_types.append(type_str)
            
            # Extract and format parameters
            parameters = []
            for param in metadata.get('parameters', []):
                if isinstance(param, dict):
                    param_str = param.get('name', 'unknown')
                    param_type = param.get('type', '')
                    param_desc = param.get('description', '')
                    param_default = param.get('default', None)
                    
                    if param_type:
                        param_str += f" ({param_type})"
                    if param_default is not None:
                        param_str += f" = {param_default}"
                    if param_desc:
                        param_str += f": {param_desc}"
                        
                    parameters.append(param_str)
            
            # Create tool description
            description = f"""CLAMS tool for {metadata.get('description', 'video analysis')}.

Inputs:
{chr(10).join('- ' + inp for inp in input_types)}

Outputs:
{chr(10).join('- ' + out for out in output_types)}

Parameters:
{chr(10).join('- ' + param for param in parameters)}

Version: {app_info.get('latest_version', 'unknown')}"""

            # Create the tool
            tool = CLAMSTool(name=app_name, description=description, app_metadata=app_info)
            tools[app_name] = tool
            
        return tools
    
    def get_tools(self) -> Dict[str, BaseTool]:
        """Get all available CLAMS tools."""
        return self.tools
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a specific CLAMS tool by name."""
        return self.tools.get(name) 