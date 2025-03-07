import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { Card, CardContent, CardHeader, Typography, IconButton, styled } from '@mui/material';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import { AppDirectoryEntry } from '../types/AppTypes';

interface AppNodeProps {
  data: {
    app: AppDirectoryEntry;
    appId: string;
  };
}

const StyledCard = styled(Card)(({ theme }) => ({
  minWidth: 200,
  maxWidth: 300,
  backgroundColor: theme.palette.background.paper,
  '& .drag-handle': {
    cursor: 'grab',
    '&:active': {
      cursor: 'grabbing',
    },
  },
}));

const StyledCardHeader = styled(CardHeader)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(1),
  '& .MuiCardHeader-title': {
    fontSize: '0.9rem',
    fontWeight: 'bold',
  },
  '& .MuiCardHeader-action': {
    margin: 0,
  },
}));

export const AppNode: React.FC<AppNodeProps> = memo(({ data }) => {
  const { app, appId } = data;

  return (
    <StyledCard variant="outlined">
      <Handle type="target" position={Position.Top} />
      <StyledCardHeader
        className="drag-handle"
        title={app.metadata.name || appId.split('/').pop()}
        action={
          <IconButton size="small" className="drag-handle">
            <DragIndicatorIcon />
          </IconButton>
        }
      />
      <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
        <Typography variant="body2" color="text.secondary" noWrap>
          v{app.latest_version}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          {app.metadata.description}
        </Typography>
      </CardContent>
      <Handle type="source" position={Position.Bottom} />
    </StyledCard>
  );
}); 