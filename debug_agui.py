#!/usr/bin/env python3
"""
Debug script to test AG-UI integration.
"""

import asyncio
import json
from utils.langgraph_agent import CLAMSAgent
from utils.agui_integration import AGUIServer, AGUIEvent
from utils.config import ConfigManager

async def test_agui_integration():
    """Test the AG-UI event processing."""
    print("=== Testing AG-UI Integration ===")
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        agent = CLAMSAgent(config_manager)
        agui_server = AGUIServer(agent)
        print("✓ Components initialized")
        
        # Create a test event (similar to what the frontend sends)
        test_event_data = {
            "type": "user_message",
            "data": {
                "message": "hello",
                "task_description": "test task"
            },
            "session_id": "debug-session-123"
        }
        
        print(f"✓ Test event created: {test_event_data}")
        
        # Test the process_user_event method
        print("\n--- Testing process_user_event ---")
        event_json = json.dumps(test_event_data)
        
        response_events = await agui_server.process_user_event(event_json)
        print(f"✓ Received {len(response_events)} response events")
        
        for i, event in enumerate(response_events):
            print(f"  Event {i+1}: {event.type} - {str(event.data)[:100]}...")
        
        # Test the event handler directly
        print("\n--- Testing event handler directly ---")
        try:
            event = agui_server.event_handler.encoder.decode_event(event_json)
            print(f"✓ Event decoded: {event.type}")
            
            event_count = 0
            async for response_event in agui_server.event_handler.handle_event(event):
                event_count += 1
                print(f"  Direct event {event_count}: {response_event.type} - {str(response_event.data)[:100]}...")
                if event_count >= 5:  # Prevent infinite loops
                    break
            
            if event_count == 0:
                print("✗ No events received from direct handler")
            
        except Exception as e:
            print(f"✗ Direct event handler failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"✗ AG-UI integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agui_integration())