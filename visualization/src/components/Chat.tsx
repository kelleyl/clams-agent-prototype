import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  styled,
  CircularProgress,
  Alert,
  Grid,
  IconButton,
  Tooltip,
  Chip,
  LinearProgress,
} from '@mui/material';
import { 
  VideoLibrary, 
  Upload, 
  Send, 
  PlayArrow, 
  Pause, 
  Settings,
  Timeline,
  SmartToy 
} from '@mui/icons-material';

// AG-UI integration (with fallback for development)
interface AGUIEvent {
  type: string;
  data: any;
  timestamp?: string;
  session_id: string;
}

interface StreamingUpdate {
  type: string;
  content: any;
  timestamp: string;
}

interface ChatProps {
  onPipelineGenerated: (pipeline: any) => void;
}

const ChatContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: 'calc(100vh - 200px)',
  display: 'flex',
  flexDirection: 'column',
}));

const VideoContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.grey[900],
}));

const MessageList = styled(List)(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  maxHeight: '60vh',
}));

const MessageItem = styled(ListItem)<{ isuser: string }>(({ theme, isuser }) => ({
  flexDirection: 'column',
  alignItems: isuser === 'true' ? 'flex-end' : 'flex-start',
  padding: theme.spacing(1, 2),
}));

const MessageContent = styled(Paper)<{ isuser: string }>(({ theme, isuser }) => ({
  padding: theme.spacing(1, 2),
  backgroundColor: isuser === 'true' ? theme.palette.primary.light : theme.palette.grey[100],
  color: isuser === 'true' ? theme.palette.primary.contrastText : theme.palette.text.primary,
  borderRadius: theme.shape.borderRadius,
  maxWidth: '80%',
}));

const ToolChip = styled(Chip)(({ theme }) => ({
  margin: theme.spacing(0.5),
  backgroundColor: theme.palette.success.light,
  color: theme.palette.success.contrastText,
}));

const StreamingIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1),
  backgroundColor: theme.palette.info.light,
  borderRadius: theme.shape.borderRadius,
  margin: theme.spacing(1, 0),
}));

const VideoPlayer = styled('video')({
  width: '100%',
  height: '100%',
  objectFit: 'contain',
});

export const Chat: React.FC<ChatProps> = ({ onPipelineGenerated }) => {
  // State management
  const [messages, setMessages] = useState<Array<{ role: string; content: string; timestamp?: string; tools?: string[] }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState('');
  const [chatStarted, setChatStarted] = useState(false);
  const [sessionId] = useState(() => `session-${Date.now()}`);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [pipelineNodes, setPipelineNodes] = useState(0);
  
  // Video state
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // AG-UI SSE connection for real-time updates
  useEffect(() => {
    if (chatStarted && !eventSourceRef.current) {
      const eventSource = new EventSource(`/api/agui/stream/${sessionId}`);
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleStreamingUpdate(data);
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        setError('Connection to agent lost. Please refresh the page.');
      };
      
      return () => {
        eventSource.close();
        eventSourceRef.current = null;
      };
    }
  }, [chatStarted, sessionId]);

  // Handle streaming updates from AG-UI
  const handleStreamingUpdate = useCallback((update: StreamingUpdate | AGUIEvent) => {
    const updateType = update.type;
    const content = 'content' in update ? update.content : update.data;
    
    switch (updateType) {
      // AG-UI Protocol Events
      case 'text_message_content':
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: content.content || content.message || content || '',
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        setStreaming(false);
        break;
        
      case 'run_started':
        setStreaming(true);
        setLoading(true);
        break;
        
      case 'run_finished':
        setStreaming(false);
        setLoading(false);
        break;
        
      case 'run_error':
        setError(content.error || content.message || 'Unknown error');
        setStreaming(false);
        setLoading(false);
        break;
        
      case 'tool_call_start':
        const toolName = content.tool_name || content.name || 'Unknown Tool';
        setSelectedTools(prev => prev.includes(toolName) ? prev : [...prev, toolName]);
        
        setMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `🔧 Selected tool: ${toolName}`,
            tools: [toolName],
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        break;
        
      case 'tool_call_result':
        setMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `✅ Tool executed: ${content.tool_name || 'Unknown'} - ${content.result || 'Success'}`,
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        break;
        
      case 'state_delta':
        if (content.nodes !== undefined) {
          setPipelineNodes(content.nodes || 0);
          if (content.nodes > 0) {
            setMessages(prev => [
              ...prev,
              {
                role: 'system',
                content: `📋 Pipeline updated: ${content.nodes} tools, ${content.edges || 0} connections`,
                timestamp: update.timestamp || new Date().toISOString()
              }
            ]);
          }
        }
        break;
        
      // Legacy events for backward compatibility
      case 'assistant_message':
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: content.content || content.message || '',
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        setStreaming(false);
        break;
        
      case 'tool_selected':
        const legacyToolName = content.tool_name || content.name || 'Unknown Tool';
        setSelectedTools(prev => prev.includes(legacyToolName) ? prev : [...prev, legacyToolName]);
        
        setMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `🔧 Selected tool: ${legacyToolName}`,
            tools: [legacyToolName],
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        break;
        
      case 'tool_result':
        setMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `✅ Tool executed: ${content.tool_name || 'Unknown'} - ${content.result || 'Success'}`,
            timestamp: update.timestamp || new Date().toISOString()
          }
        ]);
        break;
        
      case 'pipeline_updated':
        setPipelineNodes(content.nodes || 0);
        if (content.nodes > 0) {
          setMessages(prev => [
            ...prev,
            {
              role: 'system',
              content: `📋 Pipeline updated: ${content.nodes} tools, ${content.edges || 0} connections`,
              timestamp: update.timestamp || new Date().toISOString()
            }
          ]);
        }
        break;
        
      case 'streaming_start':
        setStreaming(true);
        break;
        
      case 'streaming_end':
        setStreaming(false);
        break;
        
      case 'conversation_complete':
        setStreaming(false);
        setLoading(false);
        break;
        
      case 'error':
        setError(content.error || content.message || 'Unknown error');
        setStreaming(false);
        setLoading(false);
        break;
        
      // Ignore heartbeat and connection events
      case 'custom_event':
        if (content.message === 'heartbeat' || content.message === 'Connected to CLAMS agent') {
          // Ignore these system events
          return;
        }
        // For other custom events, log them
        console.log('Custom event:', updateType, content);
        break;
        
      case 'raw_event':
        // Handle raw events (usually heartbeats or system messages)
        if (content.message === 'heartbeat') {
          // Ignore heartbeat messages
          return;
        }
        console.log('Raw event:', updateType, content);
        break;
        
      default:
        console.log('Unhandled update type:', updateType, content);
    }
  }, []);

  // Video upload handler
  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  };

  // Video play/pause toggle
  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Start chat session
  const startChat = async () => {
    if (!task.trim()) {
      setError('Please enter a task description.');
      return;
    }

    setLoading(true);
    setError(null);
    setChatStarted(true);

    try {
      // Send initial AG-UI event
      const initialEvent: AGUIEvent = {
        type: 'user_message',
        data: {
          message: `I want to create a CLAMS pipeline for: ${task}`,
          task_description: task
        },
        session_id: sessionId
      };

      const response = await fetch('/api/agui/events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(initialEvent),
      });

      if (!response.ok) {
        throw new Error('Failed to start chat session');
      }

      // Handle the response events directly
      const responseData = await response.json();
      
      // Add initial user message to UI
      setMessages([{
        role: 'user',
        content: task,
        timestamp: new Date().toISOString()
      }]);

      // Process response events from POST request
      if (responseData.events && Array.isArray(responseData.events)) {
        responseData.events.forEach((event: AGUIEvent) => {
          handleStreamingUpdate(event);
        });
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setChatStarted(false);
    } finally {
      setLoading(false);
    }
  };

  // Send message
  const sendMessage = async () => {
    if (!input.trim() || !chatStarted || loading) return;

    const userMessage = input;
    setInput('');
    setLoading(true);
    setError(null);

    // Add user message to UI immediately
    setMessages(prev => [
      ...prev,
      {
        role: 'user',
        content: userMessage,
        timestamp: new Date().toISOString()
      }
    ]);

    try {
      // Send AG-UI event
      const messageEvent: AGUIEvent = {
        type: 'user_message',
        data: {
          message: userMessage,
          task_description: task
        },
        session_id: sessionId
      };

      const response = await fetch('/api/agui/events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageEvent),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      // Handle the response events directly
      const responseData = await response.json();
      
      // Process response events from POST request
      if (responseData.events && Array.isArray(responseData.events)) {
        responseData.events.forEach((event: AGUIEvent) => {
          handleStreamingUpdate(event);
        });
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Generate pipeline from conversation
  const generatePipeline = async () => {
    try {
      const response = await fetch(`/api/chat/pipeline/${sessionId}`);
      if (response.ok) {
        const pipelineData = await response.json();
        onPipelineGenerated(pipelineData);
        
        setMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: '🎯 Pipeline generated successfully! You can now view it in the visualizer.',
            timestamp: new Date().toISOString()
          }
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate pipeline');
    }
  };

  // Handle enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatStarted) {
        sendMessage();
      } else {
        startChat();
      }
    }
  };

  return (
    <Grid container spacing={2} sx={{ height: 'calc(100vh - 100px)' }}>
      {/* Video Section */}
      <Grid item xs={12} md={6}>
        <VideoContainer>
          {videoUrl ? (
            <>
              <VideoPlayer
                ref={videoRef}
                src={videoUrl}
                controls
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <Tooltip title={isPlaying ? 'Pause' : 'Play'}>
                  <IconButton onClick={togglePlayPause} color="primary">
                    {isPlaying ? <Pause /> : <PlayArrow />}
                  </IconButton>
                </Tooltip>
              </Box>
            </>
          ) : (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
              }}
            >
              <VideoLibrary sx={{ fontSize: 60, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Upload a Video
              </Typography>
              <input
                accept="video/*"
                style={{ display: 'none' }}
                id="video-upload"
                type="file"
                onChange={handleVideoUpload}
              />
              <label htmlFor="video-upload">
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<Upload />}
                >
                  Choose Video
                </Button>
              </label>
            </Box>
          )}
        </VideoContainer>
      </Grid>

      {/* Chat Section */}
      <Grid item xs={12} md={6}>
        <ChatContainer>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <SmartToy sx={{ mr: 1 }} />
            <Typography variant="h4">
              AI Pipeline Assistant
            </Typography>
            {selectedTools.length > 0 && (
              <Box sx={{ ml: 2, display: 'flex', alignItems: 'center' }}>
                <Timeline sx={{ mr: 1, fontSize: 20 }} />
                <Typography variant="caption">
                  {selectedTools.length} tools selected
                </Typography>
              </Box>
            )}
          </Box>

          {/* Selected Tools */}
          {selectedTools.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Pipeline Tools:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {selectedTools.map((tool, index) => (
                  <ToolChip
                    key={index}
                    label={tool}
                    size="small"
                    icon={<Settings />}
                  />
                ))}
              </Box>
            </Box>
          )}
          
          {!chatStarted ? (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body1" gutterBottom>
                Describe the video analysis pipeline you want to create:
              </Typography>
              <TextField
                fullWidth
                label="Task Description"
                variant="outlined"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
                multiline
                rows={3}
                sx={{ mb: 2 }}
                placeholder="e.g., Extract text from video, transcribe speech, detect objects..."
              />
              <Button
                variant="contained"
                color="primary"
                onClick={startChat}
                disabled={loading || !task.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <SmartToy />}
              >
                {loading ? 'Starting...' : 'Start AI Assistant'}
              </Button>
            </Box>
          ) : (
            <>
              {/* Streaming Indicator */}
              {streaming && (
                <StreamingIndicator>
                  <CircularProgress size={16} sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    AI is thinking...
                  </Typography>
                  <LinearProgress sx={{ ml: 2, flex: 1 }} />
                </StreamingIndicator>
              )}

              {/* Message List */}
              <MessageList>
                {messages.map((message, index) => (
                  <MessageItem key={index} isuser={message.role === 'user' ? 'true' : 'false'}>
                    <MessageContent isuser={message.role === 'user' ? 'true' : 'false'}>
                      <Typography variant="body1">{message.content}</Typography>
                      {message.tools && (
                        <Box sx={{ mt: 1 }}>
                          {message.tools.map((tool, i) => (
                            <ToolChip key={i} label={tool} size="small" />
                          ))}
                        </Box>
                      )}
                    </MessageContent>
                  </MessageItem>
                ))}
                <div ref={messagesEndRef} />
              </MessageList>

              {/* Input Area */}
              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  label="Type your message..."
                  variant="outlined"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading || streaming}
                  multiline
                  maxRows={4}
                />
                <IconButton
                  color="primary"
                  onClick={sendMessage}
                  disabled={loading || streaming || !input.trim()}
                  sx={{ alignSelf: 'flex-end' }}
                >
                  {loading || streaming ? <CircularProgress size={24} /> : <Send />}
                </IconButton>
              </Box>

              {/* Action Buttons */}
              <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  onClick={generatePipeline}
                  disabled={loading || selectedTools.length === 0}
                  startIcon={<Timeline />}
                >
                  Generate Pipeline ({pipelineNodes} nodes)
                </Button>
              </Box>
            </>
          )}

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
        </ChatContainer>
      </Grid>
    </Grid>
  );
};