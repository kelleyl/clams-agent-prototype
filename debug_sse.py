#!/usr/bin/env python3
"""
Debug script to test SSE connection issues
"""

import asyncio
import json
from utils.agui_integration import AGUIServer, AGUIEvent
from utils.langgraph_agent import CLAMSAgent
from utils.config import ConfigManager

async def test_sse_generation():
    """Test SSE event generation without Flask"""
    print("Testing SSE event generation...")
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        agent = CLAMSAgent(config_manager)
        agui_server = AGUIServer(agent)
        
        print("✓ Components initialized")
        
        # Test SSE connection generator
        session_id = "debug-session"
        
        print(f"Testing SSE connection for session: {session_id}")
        
        # Create a short-lived SSE connection
        event_count = 0
        async for sse_data in agui_server.handle_sse_connection(session_id):
            print(f"Event {event_count}: {sse_data[:100]}...")
            event_count += 1
            
            if event_count >= 3:  # Just test first few events
                break
        
        print(f"✓ SSE connection generated {event_count} events")
        
        # Test event processing
        test_event_data = json.dumps({
            "type": "user_message",
            "data": {
                "message": "Hello, test message",
                "task_description": "Debug test"
            },
            "session_id": session_id
        })
        
        print("Testing event processing...")
        events = await agui_server.process_user_event(test_event_data)
        print(f"✓ Processed event, generated {len(events)} response events")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("CLAMS Agent SSE Debug Test")
    print("=" * 40)
    
    success = await test_sse_generation()
    
    if success:
        print("\n✅ SSE components are working correctly")
        print("The issue might be in the Flask integration or browser connection")
    else:
        print("\n❌ SSE components have issues")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)