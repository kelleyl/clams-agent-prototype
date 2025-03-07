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
  styled
} from '@mui/material';
import { Pipeline } from './Pipeline';
import { AppDirectory, AnnotationType } from '../types/AppTypes';
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

export const App: React.FC = () => {
  const [appDirectory, setAppDirectory] = useState<AppDirectory>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setAppDirectory(appDirectoryData as unknown as AppDirectory);
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
            <Pipeline appDirectory={appDirectory} />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}; 