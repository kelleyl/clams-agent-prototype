#!/usr/bin/env python3
"""
Command-line interface for the CLAMS Pipeline Generator.
This script provides an interactive chat interface for creating CLAMS pipelines.
"""

import os
import sys
import logging
from typing import Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.agent_chat import PipelineAgentChat
from utils.clams_tools import CLAMSToolbox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Command-line interface for the pipeline generator."""
    # Initialize the toolbox and chat agent
    toolbox = CLAMSToolbox()
    chat = PipelineAgentChat()
    
    # Add CLAMS tools to the agent
    chat.agent.toolbox.add_tools(toolbox.get_tools().values())
    
    print("CLAMS Pipeline Generator")
    print("=======================")
    print("\nAvailable tools:")
    for name, tool in toolbox.get_tools().items():
        print(f"- {name}")
    
    while True:
        try:
            task = input("\nEnter your task (or 'quit' to exit): ").strip()
            
            if task.lower() == 'quit':
                break
                
            # Start interactive session
            pipeline = chat.interactive_pipeline_design(task)
            if pipeline:
                print("\nGenerated Pipeline:")
                print("==================")
                print(pipeline)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main() 