#!/usr/bin/env python3
"""
Test the fixed Flask event processing.
"""

import requests
import json
import time

def test_fixed_flask_processing():
    """Test multiple messages to see if they all get processed."""
    print("=== Testing Fixed Flask Event Processing ===")
    
    base_url = "http://localhost:5001"
    
    # Test messages
    test_messages = [
        "hello",
        "can you help me?", 
        "what tools are available?",
        "is there a chyron tool?",
        "create a pipeline for video analysis"
    ]
    
    session_id = f"test-session-{int(time.time())}"
    
    print(f"Testing with session: {session_id}")
    print(f"Sending {len(test_messages)} messages...")
    
    for i, message in enumerate(test_messages):
        print(f"\n--- Message {i+1}: '{message}' ---")
        
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
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Response: {result['status']} - {result['events_generated']} events")
            else:
                print(f"✗ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
        
        # Small delay between messages
        time.sleep(1)
    
    print(f"\n✓ Test completed. Check conversation.log for processing logs.")

if __name__ == "__main__":
    test_fixed_flask_processing()