import React, { useState, useEffect } from 'react';
import { Box, Container, Grid, Paper, Typography, List, ListItem } from '@mui/material';
import { Pipeline } from './Pipeline';
import { AppDirectory } from '../types/AppTypes';
import appDirectoryData from 'data/app_directory.json';

export const App: React.FC = () => {
  const [appDirectory, setAppDirectory] = useState<AppDirectory>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setAppDirectory(appDirectoryData as AppDirectory);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load app directory');
    } finally {
      setLoading(false);
    }
  }, []);

  const onDragStart = (event: React.DragEvent<HTMLLIElement>, appId: string) => {
    event.dataTransfer.setData('application/clams-app', appId);
    event.dataTransfer.effectAllowed = 'move';
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

  return (
    <Container maxWidth="xl">
      <Box sx={{ flexGrow: 1, mt: 4 }}>
        <Grid container spacing={2}>
          {/* Tool Palette */}
          <Grid item xs={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Available Tools
              </Typography>
              <List>
                {Object.entries(appDirectory).map(([appId, app]) => (
                  <ListItem
                    key={appId}
                    draggable
                    onDragStart={(e: React.DragEvent<HTMLLIElement>) => onDragStart(e, appId)}
                    sx={{
                      cursor: 'move',
                      '&:hover': {
                        bgcolor: 'action.hover',
                      },
                    }}
                  >
                    <Typography>
                      {app.metadata.name} (v{app.latest_version})
                    </Typography>
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>

          {/* Pipeline Canvas */}
          <Grid item xs={9}>
            <Pipeline appDirectory={appDirectory} />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}; 