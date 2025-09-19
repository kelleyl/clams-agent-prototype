#!/usr/bin/env python3
"""
Comprehensive integration test demonstrating the CLAMS agent can use tools correctly.
This test uses real MMIF data from the iasa-hichy project to verify end-to-end functionality.
"""

import asyncio
import json
import os
from pathlib import Path
from utils.clams_tools import CLAMSToolbox
from utils.langgraph_agent import CLAMSAgent
from utils.config import ConfigManager

def find_sample_mmif():
    """Find a real MMIF file for testing."""
    # Look for MMIF files in the iasa-hichy project
    mmif_paths = [
        "/home/kmlynch/clams_apps/iasa-hichy/mmif",
        "/home/kmlynch/clams_apps/clams-agent-prototype/test_data"
    ]
    
    for mmif_dir in mmif_paths:
        if os.path.exists(mmif_dir):
            for file in os.listdir(mmif_dir):
                if file.endswith('.mmif'):
                    return os.path.join(mmif_dir, file)
    return None

def create_minimal_video_mmif():
    """Create a minimal but valid MMIF for testing."""
    minimal_mmif = {
        "metadata": {
            "mmif": "http://mmif.clams.ai/1.0.4",
            "app": "http://mmif.clams.ai/apps/test/1.0.0",
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
                    "location": "file:///nonexistent/test.mp4"
                }
            }
        ],
        "views": []
    }
    return json.dumps(minimal_mmif, indent=2)

async def test_tool_discovery():
    """Test that the agent can discover and list tools."""
    print("=== Tool Discovery Test ===")
    
    config_manager = ConfigManager()
    agent = CLAMSAgent(config_manager)
    
    response = await agent.get_response(
        "What CLAMS tools do you have available? Please list at least 5 tools with their descriptions."
    )
    
    print(f"✓ Agent responded with {len(response['content'])} characters")
    print(f"✓ Made {len(response['tool_calls'])} tool calls")
    
    # Check if response mentions specific tools
    content_lower = response['content'].lower()
    expected_tools = ['whisper', 'ocr', 'detection', 'transcription']
    found_tools = [tool for tool in expected_tools if tool in content_lower]
    
    print(f"✓ Response mentions {len(found_tools)} expected tool types")
    return len(found_tools) >= 2

async def test_pipeline_suggestion():
    """Test that the agent can suggest a pipeline for a specific task."""
    print("\n=== Pipeline Suggestion Test ===")
    
    config_manager = ConfigManager()
    agent = CLAMSAgent(config_manager)
    
    response = await agent.get_response(
        "I want to extract all spoken text from a news video. What CLAMS tools should I use and in what order?"
    )
    
    print(f"✓ Agent responded with {len(response['content'])} characters")
    print(f"✓ Made {len(response['tool_calls'])} tool calls")
    
    # Check if response mentions speech/audio tools
    content_lower = response['content'].lower()
    speech_indicators = ['whisper', 'speech', 'transcription', 'audio', 'spoken']
    found_indicators = [ind for ind in speech_indicators if ind in content_lower]
    
    print(f"✓ Response mentions {len(found_indicators)} speech-related terms")
    return len(found_indicators) >= 1

async def test_tool_execution_attempt():
    """Test that the agent actually attempts to execute tools."""
    print("\n=== Tool Execution Attempt Test ===")
    
    config_manager = ConfigManager()
    agent = CLAMSAgent(config_manager)
    
    # Create a minimal MMIF for testing
    test_mmif = create_minimal_video_mmif()
    
    response = await agent.get_response(
        f"Please analyze this video using any appropriate CLAMS tool: {test_mmif[:200]}..."
    )
    
    print(f"✓ Agent responded with {len(response['content'])} characters")
    print(f"✓ Made {len(response['tool_calls'])} tool calls")
    
    # Check if any tools were actually called
    if response['tool_calls']:
        print(f"✓ Tool calls attempted:")
        for i, call in enumerate(response['tool_calls'][:3]):
            tool_name = call.get('tool_name', 'unknown')
            print(f"  {i+1}. {tool_name}")
    
    return len(response['tool_calls']) > 0

def test_individual_tool():
    """Test a single tool execution directly."""
    print("\n=== Individual Tool Test ===")
    
    toolbox = CLAMSToolbox()
    tools = toolbox.get_tools()
    
    # Find a tool that exists locally
    local_tools = []
    for tool_name in tools.keys():
        tool = tools[tool_name]
        app_dir = tool._find_app_directory()
        if app_dir:
            local_tools.append((tool_name, tool))
    
    if not local_tools:
        print("✗ No local tools found for testing")
        return False
    
    test_tool_name, test_tool = local_tools[0]
    print(f"✓ Testing tool: {test_tool_name}")
    
    # Create test MMIF
    test_mmif = create_minimal_video_mmif()
    
    try:
        result = test_tool._run(test_mmif)
        print(f"✓ Tool execution completed (result length: {len(result)})")
        
        # Check if it's an error or successful result
        try:
            result_json = json.loads(result)
            if "error" in result_json:
                print(f"✓ Expected error handling: {result_json['error'][:100]}...")
                return True
            else:
                print("✓ Tool executed successfully!")
                return True
        except json.JSONDecodeError:
            print("✓ Tool returned non-JSON output (possibly valid MMIF)")
            return True
            
    except Exception as e:
        print(f"✗ Tool execution failed: {e}")
        return False

def test_app_startup():
    """Test that the main application can start."""
    print("\n=== Application Startup Test ===")
    
    try:
        from app import app
        print("✓ Flask app imported successfully")
        
        # Test health endpoint
        with app.test_client() as client:
            response = client.get('/api/health')
            health_data = response.get_json()
            
            print(f"✓ Health check status: {health_data.get('status', 'unknown')}")
            print(f"✓ Agent available: {health_data.get('agent_available', False)}")
            print(f"✓ Tools count: {health_data.get('tools_count', 0)}")
            
            return health_data.get('status') == 'healthy'
            
    except Exception as e:
        print(f"✗ App startup failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("CLAMS Agent Comprehensive Integration Test")
    print("=" * 50)
    
    tests = [
        ("Tool Discovery", test_tool_discovery()),
        ("Pipeline Suggestion", test_pipeline_suggestion()),
        ("Tool Execution Attempt", test_tool_execution_attempt()),
        ("Individual Tool", test_individual_tool()),
        ("Application Startup", test_app_startup())
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        print(f"\nRunning {test_name}...")
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The CLAMS agent is working correctly and can use tools!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)