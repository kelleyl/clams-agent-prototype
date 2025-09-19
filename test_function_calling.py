#!/usr/bin/env python3
"""
Test function calling capability with the current LLM setup.
"""

import asyncio
from utils.langgraph_agent import CLAMSAgent
from utils.config import ConfigManager

async def test_function_calling():
    """Test if the LLM can make function calls."""
    print("=== Testing Function Calling ===")
    
    config_manager = ConfigManager()
    agent = CLAMSAgent(config_manager)
    
    # Test tool recognition
    print(f"Agent has {len(agent.tools)} tools available")
    print("Tools:", [tool.name for tool in agent.tools[:5]])
    
    # Test with explicit function calling request
    response = await agent.get_response(
        "Please use the easyocr-wrapper tool to analyze this text. Call the function with any sample MMIF input."
    )
    
    print(f"Response type: {type(response)}")
    
    if isinstance(response, dict):
        print(f"Response content: {response['content'][:200]}...")
        print(f"Tool calls made: {len(response['tool_calls'])}")
        
        if response['tool_calls']:
            print("Tool calls:")
            for call in response['tool_calls']:
                print(f"  - {call}")
        else:
            print("No tool calls detected")
        
        tool_call_count = len(response['tool_calls'])
    else:
        print(f"Response is not a dict, it's: {type(response)}")
        print(f"Response content: {str(response)[:200]}...")
        tool_call_count = 0
    
    # Test the underlying LLM model capabilities
    print("\n=== Testing Direct LLM Function Binding ===")
    try:
        # Test if the LLM supports tools by binding them directly
        llm_with_tools = agent.llm.bind_tools(agent.tools[:2])  # Test with first 2 tools
        
        messages = [{"role": "user", "content": "Use the easyocr-wrapper tool on some test data"}]
        direct_response = await llm_with_tools.ainvoke(messages)
        
        print(f"Direct LLM response type: {type(direct_response)}")
        print(f"Has tool_calls attribute: {hasattr(direct_response, 'tool_calls')}")
        if hasattr(direct_response, 'tool_calls'):
            print(f"Tool calls: {direct_response.tool_calls}")
        
    except Exception as e:
        print(f"Direct tool binding failed: {e}")
    
    return tool_call_count > 0

async def main():
    success = await test_function_calling()
    print(f"\nFunction calling test: {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)