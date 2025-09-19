#!/usr/bin/env python3
"""
Debug script to test the exact Flask event processing path.
This simulates what happens when the web interface sends a message.
"""

import asyncio
import json
from utils.langgraph_agent import CLAMSAgent
from utils.agui_integration import AGUIServer
from utils.config import ConfigManager

def test_flask_event_processing():
    """Test the exact processing path that Flask uses."""
    print("=== Testing Flask Event Processing Path ===")
    
    try:
        # Initialize components (same as Flask app)
        config_manager = ConfigManager()
        agent = CLAMSAgent(config_manager)
        agui_server = AGUIServer(agent)
        print("✓ Components initialized")
        
        # Create event data (exactly like what frontend sends)
        event_data = {
            "type": "user_message",
            "data": {
                "message": "hello",
                "task_description": "test task"
            },
            "session_id": "debug-flask-session"
        }
        
        print(f"✓ Event data created: {event_data}")
        
        # Simulate the Flask async processing function
        async def process_event():
            try:
                event_json = json.dumps(event_data)
                print(f"✓ Event JSON: {event_json}")
                response_events = await agui_server.process_user_event(event_json)
                print(f"✓ Response events received: {len(response_events)}")
                return response_events
            except Exception as e:
                print(f"✗ Error in process_event: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        # Test the problematic new event loop pattern from Flask
        print("\n--- Testing Flask's Event Loop Pattern ---")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_events = loop.run_until_complete(process_event())
            print(f"✓ Flask pattern succeeded: {len(response_events)} events")
            for i, event in enumerate(response_events):
                print(f"  Event {i+1}: {event.type} - {str(event.data)[:100]}...")
        except Exception as e:
            print(f"✗ Flask pattern failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()
        
        # Test with existing event loop (might work better)
        print("\n--- Testing with Existing Event Loop ---")
        try:
            # Get the existing event loop if it exists
            try:
                current_loop = asyncio.get_running_loop()
                print("✓ Found existing event loop")
                # Can't use run_until_complete on running loop
                print("  Note: Cannot test with running loop using run_until_complete")
            except RuntimeError:
                print("✓ No running event loop, can create new one")
                response_events = asyncio.run(process_event())
                print(f"✓ asyncio.run succeeded: {len(response_events)} events")
                for i, event in enumerate(response_events):
                    print(f"  Event {i+1}: {event.type} - {str(event.data)[:100]}...")
        except Exception as e:
            print(f"✗ asyncio.run failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flask_event_processing()