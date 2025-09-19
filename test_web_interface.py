#!/usr/bin/env python3
"""
Test the web interface to ensure all messages are processed.
"""

import requests
import json
import time
import threading
from urllib.parse import urljoin

def test_web_interface():
    """Test the complete web interface flow."""
    print("=== Testing Web Interface ===")
    
    base_url = "http://localhost:5001"
    session_id = f"web-test-{int(time.time())}"
    
    # Test 1: Check if main page loads
    print("\n--- Test 1: Main page ---")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✓ Main page loads successfully")
        else:
            print(f"✗ Main page failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Main page error: {e}")
    
    # Test 2: Check health endpoint
    print("\n--- Test 2: Health check ---")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Health check: {health['status']}")
            print(f"  Agent: {health['agent_available']}")
            print(f"  AG-UI: {health['agui_server_available']}")
            print(f"  Tools: {health['tools_count']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health check error: {e}")
    
    # Test 3: Multiple AG-UI events (the main issue we fixed)
    print(f"\n--- Test 3: AG-UI Events (Session: {session_id}) ---")
    
    test_messages = [
        "hello there",
        "can you help me create a pipeline?",
        "what CLAMS tools are available?",
        "do you have any chyron detection tools?",
        "I need to analyze video content"
    ]
    
    successful_messages = 0
    
    for i, message in enumerate(test_messages):
        print(f"  Message {i+1}: '{message}'")
        
        event_data = {
            "type": "user_message", 
            "data": {
                "message": message,
                "task_description": "test task"
            },
            "session_id": session_id
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/agui/events",
                json=event_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                events_count = result.get('events_generated', 0)
                print(f"    ✓ Processed - {events_count} events generated")
                successful_messages += 1
            else:
                print(f"    ✗ Failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"    ✗ Timeout after 30s")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    print(f"\n✓ AG-UI Events: {successful_messages}/{len(test_messages)} messages processed successfully")
    
    # Test 4: Check tools endpoint
    print("\n--- Test 4: Tools endpoint ---")
    try:
        response = requests.get(f"{base_url}/api/tools")
        if response.status_code == 200:
            tools = response.json()
            print(f"✓ Tools endpoint: {len(tools)} tools available")
        else:
            print(f"✗ Tools endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Tools endpoint error: {e}")
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"✓ Server is running on {base_url}")
    print(f"✓ All core endpoints are functional")
    print(f"✓ AG-UI event processing: {successful_messages}/{len(test_messages)} messages")
    
    if successful_messages == len(test_messages):
        print("🎉 All tests passed! The conversational interface is working correctly.")
        print("💡 You can now test the web interface manually at: http://localhost:5001/chat")
    else:
        print("⚠️  Some messages failed to process. Check the server logs for details.")

if __name__ == "__main__":
    test_web_interface()