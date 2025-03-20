import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  List, 
  ListItem,
  Tooltip,
  styled,
  AppBar,
  Toolbar,
  Button,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField
} from '@mui/material';
import { Pipeline } from './Pipeline';
import { AppDirectory, AnnotationType } from '../types/AppTypes';
import { useNavigate, Link, Routes, Route } from 'react-router-dom';
import { Chat } from './Chat';
import appDirectoryData from '../../../data/app_directory.json';

const StyledListItem = styled(ListItem)(({ theme }) => ({
  cursor: 'move',
  padding: theme.spacing(1),
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  borderRadius: theme.shape.borderRadius,
  marginBottom: theme.spacing(0.5),
}));

const StyledTooltip = styled(Tooltip)(({ theme }) => ({
  '& .MuiTooltip-tooltip': {
    maxWidth: 400,
    fontSize: '0.875rem',
    padding: theme.spacing(2),
    backgroundColor: theme.palette.background.paper,
    color: theme.palette.text.primary,
    boxShadow: theme.shadows[4],
    border: `1px solid ${theme.palette.divider}`,
  },
}));

const StyledLink = styled(Link)(({ theme }) => ({
  color: theme.palette.primary.main,
  textDecoration: 'none',
  margin: theme.spacing(0, 1),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

export const App: React.FC = () => {
  const [appDirectory, setAppDirectory] = useState<AppDirectory>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPipeline, setCurrentPipeline] = useState<any>(null);
  const [pipelines, setPipelines] = useState<string[]>([]);
  const [pipelineMenuAnchor, setPipelineMenuAnchor] = useState<null | HTMLElement>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [pipelineName, setPipelineName] = useState('');
  const [yamlContent, setYamlContent] = useState('');
  
  const navigate = useNavigate();

  useEffect(() => {
    try {
      setAppDirectory(appDirectoryData as unknown as AppDirectory);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load app directory');
    } finally {
      setLoading(false);
    }
  }, []);
  
  useEffect(() => {
    // Fetch available pipelines
    fetch('/api/pipelines')
      .then(response => response.json())
      .then(data => {
        setPipelines(data);
      })
      .catch(err => {
        console.error('Failed to fetch pipelines:', err);
      });
  }, []);

  const onDragStart = (event: React.DragEvent<HTMLLIElement>, appId: string) => {
    event.dataTransfer.setData('application/clams-app', appId);
    event.dataTransfer.effectAllowed = 'move';
  };

  const renderAnnotationType = (input: AnnotationType) => (
    <span>
      {input['@type']}
      {input.required && ' (required)'}
      {input.properties?.timeUnit && ` (${input.properties.timeUnit})`}
    </span>
  );

  const renderInput = (input: AnnotationType | AnnotationType[]) => {
    if (Array.isArray(input)) {
      return (
        <span>
          One of: [
          {input.map((i, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 && ', '}
              {renderAnnotationType(i)}
            </React.Fragment>
          ))}
          ]
        </span>
      );
    }
    return renderAnnotationType(input);
  };

  const renderToolMetadata = (app: AppDirectory[string]) => {
    const { metadata } = app;
    return (
      <Box>
        <Typography variant="subtitle1" gutterBottom>
          {metadata.description}
        </Typography>
        <Typography variant="subtitle2" color="primary" gutterBottom>
          Inputs:
        </Typography>
        <Box component="ul" sx={{ mt: 0, mb: 1 }}>
          {metadata.input.map((input, idx) => (
            <li key={idx}>
              {renderInput(input)}
            </li>
          ))}
        </Box>
        <Typography variant="subtitle2" color="primary" gutterBottom>
          Outputs:
        </Typography>
        <Box component="ul" sx={{ mt: 0, mb: 0 }}>
          {metadata.output.map((output, idx) => (
            <li key={idx}>
              {output['@type']}
              {output.properties?.timeUnit && ` (${output.properties.timeUnit})`}
              {output.properties?.labelset && ` [${output.properties.labelset.join(', ')}]`}
            </li>
          ))}
        </Box>
      </Box>
    );
  };
  
  const handlePipelineMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    setPipelineMenuAnchor(event.currentTarget);
  };
  
  const handlePipelineMenuClose = () => {
    setPipelineMenuAnchor(null);
  };
  
  const loadPipeline = (name: string) => {
    fetch(`/api/pipeline/${name}`)
      .then(response => response.json())
      .then(data => {
        setCurrentPipeline(data);
        handlePipelineMenuClose();
      })
      .catch(err => {
        console.error(`Failed to load pipeline ${name}:`, err);
      });
  };
  
  const loadChatPipeline = () => {
    fetch('/api/chat/pipeline')
      .then(response => response.json())
      .then(data => {
        setCurrentPipeline(data);
      })
      .catch(err => {
        console.error('Failed to load chat pipeline:', err);
      });
  };
  
  const exportPipeline = () => {
    if (!currentPipeline) return;
    
    fetch(`/api/pipeline/export/${currentPipeline.name}`)
      .then(response => response.json())
      .then(data => {
        setYamlContent(data.yaml);
        setExportDialogOpen(true);
      })
      .catch(err => {
        console.error('Failed to export pipeline:', err);
      });
  };
  
  const savePipeline = () => {
    if (!currentPipeline) return;
    
    setSaveDialogOpen(true);
    setPipelineName(currentPipeline.name || '');
  };
  
  const handleSavePipeline = () => {
    if (!currentPipeline || !pipelineName) return;
    
    const pipelineToSave = {
      ...currentPipeline,
      name: pipelineName
    };
    
    fetch('/api/pipeline', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(pipelineToSave)
    })
      .then(response => response.json())
      .then(data => {
        setSaveDialogOpen(false);
        
        // Refresh pipeline list
        fetch('/api/pipelines')
          .then(response => response.json())
          .then(pipelinesData => {
            setPipelines(pipelinesData);
          })
          .catch(err => {
            console.error('Failed to fetch pipelines:', err);
          });
      })
      .catch(err => {
        console.error('Failed to save pipeline:', err);
      });
  };

  if (loading) {
    return (
      <Container>
        <Typography>Loading app directory...</Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <Typography color="error">{error}</Typography>
      </Container>
    );
  }

  const getDisplayName = (appId: string, app: AppDirectory[string]) => {
    const name = app.metadata.name || appId.split('/').pop() || appId;
    return `${name} (v${app.latest_version})`;
  };

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            CLAMS Pipeline Designer
          </Typography>
          <StyledLink to="/">Home</StyledLink>
          <StyledLink to="/visualizer">Pipeline Visualizer</StyledLink>
          <StyledLink to="/chat">Pipeline Chat</StyledLink>
          
          <Button 
            color="inherit" 
            onClick={handlePipelineMenuOpen}
          >
            Pipelines
          </Button>
          <Menu
            anchorEl={pipelineMenuAnchor}
            open={Boolean(pipelineMenuAnchor)}
            onClose={handlePipelineMenuClose}
          >
            {pipelines.map(name => (
              <MenuItem key={name} onClick={() => loadPipeline(name)}>
                {name}
              </MenuItem>
            ))}
            <MenuItem onClick={loadChatPipeline}>
              Load from Chat
            </MenuItem>
          </Menu>
          
          <Button 
            color="inherit"
            onClick={exportPipeline}
            disabled={!currentPipeline}
          >
            Export
          </Button>
          
          <Button 
            color="inherit"
            onClick={savePipeline}
            disabled={!currentPipeline}
          >
            Save
          </Button>
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="xl">
        <Box sx={{ flexGrow: 1, mt: 4 }}>
          <Routes>
            <Route path="/" element={
              <Box textAlign="center">
                <Typography variant="h4" gutterBottom>
                  Welcome to the CLAMS Pipeline Designer
                </Typography>
                <Typography variant="body1" paragraph>
                  This tool allows you to create CLAMS pipelines through visualization or chat.
                </Typography>
                <Grid container spacing={4} justifyContent="center">
                  <Grid item xs={12} md={5}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                      <Typography variant="h5" gutterBottom>
                        Pipeline Visualizer
                      </Typography>
                      <Typography variant="body2" paragraph>
                        Drag and drop CLAMS tools to create a pipeline visually.
                      </Typography>
                      <Button 
                        variant="contained" 
                        color="primary"
                        onClick={() => navigate('/visualizer')}
                      >
                        Open Visualizer
                      </Button>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={5}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                      <Typography variant="h5" gutterBottom>
                        Pipeline Chat
                      </Typography>
                      <Typography variant="body2" paragraph>
                        Describe your needs in natural language to generate a pipeline.
                      </Typography>
                      <Button 
                        variant="contained" 
                        color="primary"
                        onClick={() => navigate('/chat')}
                      >
                        Open Chat
                      </Button>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            } />
            <Route path="/visualizer" element={
              <Grid container spacing={2}>
                {/* Tool Palette */}
                <Grid item xs={3}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Available Tools
                    </Typography>
                    <List>
                      {Object.entries(appDirectory)
                        .sort(([, a], [, b]) => {
                          const nameA = a.metadata.name || '';
                          const nameB = b.metadata.name || '';
                          return nameA.localeCompare(nameB);
                        })
                        .map(([appId, app]) => (
                          <StyledTooltip
                            key={appId}
                            title={renderToolMetadata(app)}
                            placement="right"
                            arrow
                          >
                            <StyledListItem
                              draggable
                              onDragStart={(e: React.DragEvent<HTMLLIElement>) => onDragStart(e, appId)}
                            >
                              <Typography>
                                {getDisplayName(appId, app)}
                              </Typography>
                            </StyledListItem>
                          </StyledTooltip>
                        ))}
                    </List>
                  </Paper>
                </Grid>

                {/* Pipeline Canvas */}
                <Grid item xs={9}>
                  <Pipeline 
                    appDirectory={appDirectory} 
                    initialPipeline={currentPipeline}
                    onPipelineChange={setCurrentPipeline}
                  />
                </Grid>
              </Grid>
            } />
            <Route path="/chat" element={
              <Chat onPipelineGenerated={setCurrentPipeline} />
            } />
          </Routes>
        </Box>
      </Container>
      
      {/* Export Dialog */}
      <Dialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Pipeline YAML</DialogTitle>
        <DialogContent>
          <TextField
            multiline
            fullWidth
            rows={20}
            value={yamlContent}
            variant="outlined"
            InputProps={{
              readOnly: true,
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Close</Button>
          <Button 
            onClick={() => {
              navigator.clipboard.writeText(yamlContent);
              alert('YAML copied to clipboard!');
            }}
          >
            Copy to Clipboard
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Save Dialog */}
      <Dialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
      >
        <DialogTitle>Save Pipeline</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Pipeline Name"
            fullWidth
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSavePipeline}>Save</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}; 