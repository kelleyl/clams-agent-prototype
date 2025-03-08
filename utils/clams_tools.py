from typing import Dict, Any
from transformers.tools import Tool
from .download_app_directory import get_app_metadata

class CLAMSToolbox:
    """Collection of CLAMS tools for use with Transformers Agents."""
    
    def __init__(self):
        """Initialize the CLAMS toolbox."""
        self.app_metadata = get_app_metadata()
        self.tools = self._create_tools()
        
    def _create_tools(self) -> Dict[str, Tool]:
        """Create Tool instances for each CLAMS app."""
        tools = {}
        
        for app_name, app_info in self.app_metadata.items():
            metadata = app_info.get("metadata", {})
            
            # Create tool description
            description = f"""CLAMS tool for {metadata.get('description', 'video analysis')}.
            
Inputs: {[input_type.get('@type') for input_type in metadata.get('input', [])]}
Outputs: {[output_type.get('@type') for output_type in metadata.get('output', [])]}
Parameters: {[param.get('name') for param in metadata.get('parameters', [])]}

Version: {app_info.get('latest_version', 'unknown')}"""

            # Create the tool
            tool = Tool(
                name=app_name,
                description=description,
                inputs=[
                    ("video", "path to the input video file"),
                    ("parameters", "optional parameters for the tool")
                ],
                outputs=[
                    ("annotations", "MMIF annotations produced by the tool")
                ]
            )
            
            tools[app_name] = tool
            
        return tools
    
    def get_tools(self) -> Dict[str, Tool]:
        """Get all available CLAMS tools."""
        return self.tools
    
    def get_tool(self, name: str) -> Tool:
        """Get a specific CLAMS tool by name."""
        return self.tools.get(name) 