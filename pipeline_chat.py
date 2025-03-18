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
    try:
        # Initialize the toolbox
        logger.info("Initializing CLAMS toolbox...")
        toolbox = CLAMSToolbox()
        
        # Get all CLAMS tools
        tools = toolbox.get_tools()
        if not tools:
            logger.warning("No CLAMS tools were found. Please check your app metadata.")
            print("No CLAMS tools were found. Please check your app metadata.")
            return
        
        logger.info(f"Found {len(tools)} CLAMS tools")
        
        # Initialize the chat agent
        logger.info("Initializing the pipeline agent chat...")
        chat = PipelineAgentChat()
        
        # Add CLAMS tools to the agent one by one
        logger.info("Adding tools to the agent...")
        for name, tool in tools.items():
            try:
                chat.agent.toolbox.add_tool(tool)
                logger.info(f"Added tool: {name}")
            except Exception as e:
                logger.error(f"Error adding tool {name}: {str(e)}")
        
        print("CLAMS Pipeline Generator")
        print("=======================")
        print(f"\nAvailable tools ({len(tools)}):")
        for name in tools.keys():
            print(f"- {name}")
        
        while True:
            try:
                task = input("\nEnter your task (or 'quit' to exit): ").strip()
                
                if not task:
                    print("Task description cannot be empty. Please try again.")
                    continue
                    
                if task.lower() == 'quit':
                    break
                    
                # Start interactive session
                logger.info(f"Starting interactive session for task: {task}")
                chat.interactive_pipeline_design(task)
                    
            except KeyboardInterrupt:
                logger.info("User interrupted the session")
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                print(f"\nError: {str(e)}")
                continue
    
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}")
        print(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 