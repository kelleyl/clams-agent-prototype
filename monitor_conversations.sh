#!/bin/bash

echo "🔍 CLAMS Agent Conversation Monitor"
echo "=================================="
echo "Monitoring conversation.log for real-time chat debugging"
echo "Server logs will also appear in the background"
echo ""
echo "Instructions:"
echo "1. Keep this terminal open"
echo "2. Go to http://localhost:5000 in your browser"  
echo "3. Start a chat and send messages"
echo "4. Watch the logs below to see what the agent receives/sends"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Create the log file if it doesn't exist
touch conversation.log

# Monitor the conversation log in real-time
echo "📝 Starting conversation monitoring..."
tail -f conversation.log