import React, { useState, useRef, useEffect } from 'react';
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
  Divider,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  IconButton,
  Tooltip,
} from '@mui/material';
import { VideoLibrary, Upload, Send, PlayArrow, Pause } from '@mui/icons-material';

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

const PipelineState = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  backgroundColor: theme.palette.grey[50],
  whiteSpace: 'pre-wrap',
}));

const VideoPlayer = styled('video')({
  width: '100%',
  height: '100%',
  objectFit: 'contain',
});

export const Chat: React.FC<ChatProps> = ({ onPipelineGenerated }) => {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState('');
  const [chatStarted, setChatStarted] = useState(false);
  const [pipelineState, setPipelineState] = useState('No pipeline yet.');
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [pipelineName, setPipelineName] = useState('');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  };

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

  const startChat = async () => {
    if (!task.trim()) {
      setError('Please enter a task description.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ task }),
      });

      if (!response.ok) {
        throw new Error('Failed to start chat session');
      }

      setChatStarted(true);
      // Add system welcome message
      setMessages([
        {
          role: 'Assistant',
          content: `I'll help you create a pipeline for: ${task}. What kind of video analysis do you need to perform?`,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !chatStarted) return;

    // Add user message to UI
    setMessages([...messages, { role: 'User', content: input }]);
    const userMessage = input;
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      // Add assistant's response to the UI
      setMessages(msgs => [...msgs, { role: 'Assistant', content: data.response }]);
      
      // Update pipeline state if provided
      if (data.pipeline_state) {
        setPipelineState(data.pipeline_state);
      }
      
      // If a tool was added, indicate it in the UI
      if (data.tool_added) {
        setMessages(msgs => [
          ...msgs,
          { 
            role: 'System', 
            content: `Added ${data.tool_name} to the pipeline.` 
          }
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const generatePipeline = async () => {
    setSaveDialogOpen(true);
  };
  
  const handleGeneratePipeline = async () => {
    if (!pipelineName.trim()) {
      setError('Please enter a pipeline name.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/chat/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: pipelineName }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate pipeline');
      }
      
      const data = await response.json();
      
      // Call the callback with the generated pipeline
      onPipelineGenerated(data.pipeline);
      
      // Add system message about successful generation
      setMessages(msgs => [
        ...msgs,
        { 
          role: 'System', 
          content: `Pipeline "${data.name}" generated successfully! You can now view it in the visualizer.` 
        }
      ]);
      
      setSaveDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

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
          <Typography variant="h4" gutterBottom>
            Pipeline Chat
          </Typography>
          
          {!chatStarted ? (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body1" gutterBottom>
                Describe the pipeline you want to create:
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
                rows={2}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                color="primary"
                onClick={startChat}
                disabled={loading || !task.trim()}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Start Chat'}
              </Button>
            </Box>
          ) : (
            <>
              <MessageList>
                {messages.map((message, index) => (
                  <MessageItem key={index} isuser={message.role === 'User' ? 'true' : 'false'}>
                    <MessageContent isuser={message.role === 'User' ? 'true' : 'false'}>
                      <Typography variant="body1">{message.content}</Typography>
                    </MessageContent>
                  </MessageItem>
                ))}
                <div ref={messagesEndRef} />
              </MessageList>

              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  label="Type your message..."
                  variant="outlined"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                  multiline
                  maxRows={4}
                />
                <IconButton
                  color="primary"
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                >
                  {loading ? <CircularProgress size={24} /> : <Send />}
                </IconButton>
              </Box>

              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={generatePipeline}
                  disabled={loading}
                >
                  Generate Pipeline
                </Button>
              </Box>
            </>
          )}

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </ChatContainer>
      </Grid>

      {/* Save Pipeline Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)}>
        <DialogTitle>Save Pipeline</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Pipeline Name"
            fullWidth
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            disabled={loading}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleGeneratePipeline} color="primary" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}; 