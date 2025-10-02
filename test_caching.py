#!/usr/bin/env python3
"""
Test script to verify app directory caching is working properly.
"""

import time
import logging
from utils.clams_tools import CLAMSToolbox

# Configure logging to see cache behavior
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_caching():
    """Test that multiple CLAMSToolbox instances use cached data."""
    print("=== Testing CLAMS Agent Caching Improvements ===\n")
    
    print("1. Creating first CLAMSToolbox instance...")
    start_time = time.time()
    toolbox1 = CLAMSToolbox()
    time1 = time.time() - start_time
    print(f"   First instance created in {time1:.2f} seconds")
    print(f"   Number of tools loaded: {len(toolbox1.get_tools())}")
    
    print("\n2. Creating second CLAMSToolbox instance...")
    start_time = time.time()
    toolbox2 = CLAMSToolbox()
    time2 = time.time() - start_time
    print(f"   Second instance created in {time2:.2f} seconds")
    print(f"   Number of tools loaded: {len(toolbox2.get_tools())}")
    
    print("\n3. Creating third CLAMSToolbox instance...")
    start_time = time.time()
    toolbox3 = CLAMSToolbox()
    time3 = time.time() - start_time
    print(f"   Third instance created in {time3:.2f} seconds")
    print(f"   Number of tools loaded: {len(toolbox3.get_tools())}")
    
    # Check if instances are actually the same (singleton pattern)
    print(f"\n4. Singleton verification:")
    print(f"   toolbox1 is toolbox2: {toolbox1 is toolbox2}")
    print(f"   toolbox2 is toolbox3: {toolbox2 is toolbox3}")
    
    # Performance improvement
    if time2 < time1 * 0.1 and time3 < time1 * 0.1:
        print(f"\n✅ SUCCESS: Caching is working! Subsequent instances are {time1/time2:.1f}x faster")
    else:
        print(f"\n❌ ISSUE: Caching may not be working properly")
    
    # Test some tools
    print(f"\n5. Available tools sample:")
    tools = list(toolbox1.get_tools().keys())[:5]
    for tool_name in tools:
        tool = toolbox1.get_tool(tool_name)
        print(f"   - {tool_name}: {tool.description[:60]}...")
    
    print(f"\n6. Testing refresh functionality...")
    start_time = time.time()
    toolbox1.refresh_metadata()
    refresh_time = time.time() - start_time
    print(f"   Metadata refresh took {refresh_time:.2f} seconds")

if __name__ == "__main__":
    test_caching()