#!/usr/bin/env python3
"""
Debug script to test agent response functionality.
"""

import asyncio
from utils.langgraph_agent import CLAMSAgent
from utils.config import ConfigManager

async def test_agent_response():
    """Test if the agent can respond to a simple query."""
    print("Initializing agent...")
    
    try:
        config_manager = ConfigManager()
        agent = CLAMSAgent(config_manager)
        print("✓ Agent initialized successfully")
        
        # Test the explain_pipeline method (which should work)
        print("\n=== Testing explain_pipeline method ===")
        explanation = await agent.explain_pipeline("hello")
        print(f"✓ Explanation received: {len(explanation)} characters")
        print(f"First 200 chars: {explanation[:200]}...")
        
        # Test the stream_response method (which might be failing)
        print("\n=== Testing stream_response method ===")
        update_count = 0
        try:
            async for update in agent.stream_response("hello", "", "debug-session"):
                update_count += 1
                print(f"✓ Update {update_count}: {update.type} - {str(update.content)[:100]}...")
                if update_count >= 5:  # Limit to avoid infinite loops
                    print("   (stopping after 5 updates)")
                    break
        except Exception as e:
            print(f"✗ stream_response failed: {e}")
            import traceback
            traceback.print_exc()
        
        if update_count == 0:
            print("✗ No updates received from stream_response")
        else:
            print(f"✓ Received {update_count} updates")
            
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_response())