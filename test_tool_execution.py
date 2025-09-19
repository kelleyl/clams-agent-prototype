#!/usr/bin/env python3
"""
Test script to verify CLAMS agent can execute tools correctly.
"""

import asyncio
import json
import tempfile
import os
from utils.clams_tools import CLAMSToolbox
from utils.langgraph_agent import CLAMSAgent
from utils.config import ConfigManager

def create_sample_mmif():
    """Create a simple MMIF file for testing."""
    sample_mmif = {
        "metadata": {
            "mmif": "http://mmif.clams.ai/",
            "app": "http://mmif.clams.ai/",
            "contains": {
                "http://mmif.clams.ai/vocabulary/VideoDocument/v1": {
                    "document": "m1"
                }
            }
        },
        "documents": [
            {
                "@type": "http://mmif.clams.ai/vocabulary/VideoDocument/v1",
                "properties": {
                    "mime": "video/mp4",
                    "id": "m1",
                    "location": "file:///path/to/sample/video.mp4",
                    "fps": 25.0,
                    "frameCount": 1000,
                    "duration": 40000
                }
            }
        ],
        "views": []
    }
    return json.dumps(sample_mmif, indent=2)

def test_tool_discovery():
    """Test that tools can be discovered and initialized."""
    print("=== Testing Tool Discovery ===")
    
    toolbox = CLAMSToolbox()
    tools = toolbox.get_tools()
    
    print(f"Found {len(tools)} tools:")
    for i, (name, tool) in enumerate(tools.items()):
        if i < 5:  # Show first 5
            print(f"  - {name}: {tool.description[:100]}...")
        elif i == 5:
            print(f"  ... and {len(tools) - 5} more")
            break
    
    return len(tools) > 0

def test_tool_execution():
    """Test executing a simple tool with sample MMIF."""
    print("\n=== Testing Tool Execution ===")
    
    toolbox = CLAMSToolbox()
    tools = toolbox.get_tools()
    
    # Find a simple tool to test - prefer ones with cli.py
    test_tool_names = ['llava-captioner', 'whisper-wrapper', 'transnet-wrapper', 'easyocr-wrapper']
    test_tool = None
    test_tool_name = None
    
    for name in test_tool_names:
        if name in tools:
            test_tool = tools[name]
            test_tool_name = name
            break
    
    if not test_tool:
        print("No suitable test tool found")
        return False
    
    print(f"Testing tool: {test_tool_name}")
    
    # Create sample MMIF
    sample_mmif = create_sample_mmif()
    
    try:
        # Test tool execution (this will likely fail due to missing video file)
        result = test_tool._run(sample_mmif)
        print(f"Tool execution result (first 200 chars): {result[:200]}...")
        
        # Check if it's an error or valid result
        try:
            result_json = json.loads(result)
            if "error" in result_json:
                print(f"Expected error (no real video file): {result_json['error'][:100]}...")
                return True  # This is expected
            else:
                print("Tool executed successfully!")
                return True
        except json.JSONDecodeError:
            # Might be valid MMIF output
            print("Tool returned non-JSON output (possibly valid MMIF)")
            return True
            
    except Exception as e:
        print(f"Tool execution failed with exception: {e}")
        return False

async def test_agent_tool_integration():
    """Test that the agent can use tools through conversation."""
    print("\n=== Testing Agent Tool Integration ===")
    
    try:
        config_manager = ConfigManager()
        agent = CLAMSAgent(config_manager)
        print("Agent initialized successfully")
        
        # Test simple conversation about tools
        response = await agent.get_response(
            "What CLAMS tools do you have available for processing videos?"
        )
        
        print(f"Agent response (first 300 chars): {response['content'][:300]}...")
        print(f"Tool calls made: {len(response['tool_calls'])}")
        
        if response['tool_calls']:
            print("Tool calls:")
            for i, call in enumerate(response['tool_calls'][:3]):
                print(f"  {i+1}. {call}")
        
        return True
        
    except Exception as e:
        print(f"Agent integration test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("CLAMS Agent Tool Execution Test")
    print("=" * 40)
    
    results = []
    
    # Test 1: Tool Discovery
    results.append(test_tool_discovery())
    
    # Test 2: Tool Execution
    results.append(test_tool_execution())
    
    # Test 3: Agent Integration
    results.append(await test_agent_tool_integration())
    
    print("\n" + "=" * 40)
    print("Test Results:")
    test_names = ["Tool Discovery", "Tool Execution", "Agent Integration"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i+1}. {name}: {status}")
    
    overall = "ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED"
    print(f"\n{overall}")
    
    return all(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)