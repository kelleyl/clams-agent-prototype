from typing import Dict, Any, Optional
import json
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from .download_app_directory import get_app_metadata

logger = logging.getLogger(__name__)

class CLAMSTool(BaseTool):
    """LangChain tool for CLAMS applications."""
    
    app_metadata: Dict[str, Any]
    
    def __init__(self, name: str, description: str, app_metadata: Dict[str, Any]):
        super().__init__(name=name, description=description, app_metadata=app_metadata)
    
    def _run(
        self,
        input_mmif: str,
        config: str = None,
        parameters: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the CLAMS tool on the given MMIF input with optional parameters.
        
        Args:
            input_mmif: Path to input MMIF file or MMIF content as string
            config: Configuration name (e.g., 'default.yaml')
            parameters: JSON string containing optional parameters
            run_manager: Callback manager for tool execution
            
        Returns:
            MMIF output as string
        """
        try:
            # Determine the app directory path based on the tool name
            app_dir = self._find_app_directory()
            if not app_dir:
                return json.dumps({"error": f"Could not find app directory for {self.name}"})
            
            # Check if input is a file path or MMIF content
            if os.path.isfile(input_mmif):
                input_file = input_mmif
            else:
                # Create temporary input file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.mmif', delete=False) as tmp:
                    tmp.write(input_mmif)
                    input_file = tmp.name
            
            # Set up command - prefer cli.py if it exists, otherwise use app.py
            cli_script = os.path.join(app_dir, 'cli.py')
            app_script = os.path.join(app_dir, 'app.py')
            
            if os.path.exists(cli_script):
                script_path = cli_script
            elif os.path.exists(app_script):
                script_path = app_script
            else:
                return json.dumps({"error": f"No executable script found in {app_dir}"})
            
            venv_python = os.path.join(app_dir, '.venv', 'bin', 'python')
            
            # Use system python if venv doesn't exist
            if not os.path.exists(venv_python):
                venv_python = 'python'
            
            # Build command
            cmd = [venv_python, script_path]
            
            # Handle different script types
            if script_path.endswith('cli.py'):
                # For cli.py, add config and parameters first, then input file
                if config:
                    config_path = os.path.join(app_dir, 'config', config)
                    if os.path.exists(config_path):
                        cmd.extend(['--config', f'config/{config}'])
                
                # Add additional parameters
                if parameters:
                    try:
                        params_dict = json.loads(parameters)
                        for key, value in params_dict.items():
                            cmd.extend([f'--{key}', str(value)])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid parameters JSON: {parameters}")
                
                # Add input file for cli.py
                cmd.append(input_file)
                
            else:
                # For app.py, create a temporary CLI wrapper
                temp_cli_wrapper = self._create_temp_cli_wrapper(app_dir, input_file, config, parameters)
                if temp_cli_wrapper:
                    cmd = [venv_python, temp_cli_wrapper]
                else:
                    return json.dumps({"error": f"Failed to create CLI wrapper for {self.name}"})
            
            logger.info(f"Executing CLAMS tool {self.name}: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up temporary files
            if not os.path.isfile(input_mmif):
                try:
                    os.unlink(input_file)
                except:
                    pass
            
            # Clean up temporary CLI wrapper if we created one
            if not script_path.endswith('cli.py') and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except:
                    pass
            
            if result.returncode == 0:
                return result.stdout
            else:
                error_msg = f"Tool execution failed with return code {result.returncode}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})
                
        except subprocess.TimeoutExpired:
            error_msg = f"Tool {self.name} execution timed out"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Tool {self.name} execution failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
    
    def _find_app_directory(self) -> Optional[str]:
        """Find the app directory for this tool."""
        # Common app directory patterns
        clams_apps_root = "/home/kmlynch/clams_apps"
        
        # Try different naming patterns
        patterns = [
            f"app-{self.name}",
            f"app-{self.name}-wrapper", 
            f"app-{self.name.replace('-wrapper', '')}",
            self.name
        ]
        
        for pattern in patterns:
            app_path = os.path.join(clams_apps_root, pattern)
            if os.path.isdir(app_path) and os.path.exists(os.path.join(app_path, 'app.py')):
                return app_path
        
        logger.warning(f"Could not find app directory for {self.name}")
        return None
    
    def _create_temp_cli_wrapper(self, app_dir: str, input_file: str, config: str = None, parameters: str = None) -> Optional[str]:
        """Create a temporary CLI wrapper for apps that only have app.py."""
        try:
            # Create temporary Python script
            wrapper_script = f"""#!/usr/bin/env python3
import sys
import os
import json

# Add app directory to path
sys.path.insert(0, '{app_dir}')

# Import the app
import app

def main():
    # Get the app instance
    clamsapp = app.get_app() if hasattr(app, 'get_app') else None
    
    if clamsapp is None:
        # Try to instantiate the app class directly
        for name, obj in vars(app).items():
            if hasattr(obj, '__bases__') and any('ClamsApp' in str(base) for base in obj.__bases__):
                clamsapp = obj()
                break
    
    if clamsapp is None:
        print(json.dumps({{"error": "Could not instantiate CLAMS app"}}))
        return
    
    # Read input MMIF
    with open('{input_file}', 'r') as f:
        input_mmif = f.read()
    
    # Set up parameters
    params = {{}}
    {f'''
    # Add config parameters if specified
    # TODO: Parse config file {config}
    ''' if config else ''}
    
    {f'''
    # Add additional parameters
    try:
        param_dict = {parameters}
        params.update(param_dict)
    except:
        pass
    ''' if parameters else ''}
    
    try:
        # Run the app
        output_mmif = clamsapp.annotate(input_mmif, **params)
        print(output_mmif)
    except Exception as e:
        print(json.dumps({{"error": f"App execution failed: {{str(e)}}"}}))

if __name__ == "__main__":
    main()
"""
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(wrapper_script)
                return tmp.name
                
        except Exception as e:
            logger.error(f"Failed to create CLI wrapper: {e}")
            return None
    
    async def _arun(
        self,
        input_mmif: str,
        config: str = None,
        parameters: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run."""
        return self._run(input_mmif, config, parameters, run_manager)

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