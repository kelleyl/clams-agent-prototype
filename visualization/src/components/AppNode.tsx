import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { Card, CardContent, Typography } from '@mui/material';
import { AppDirectoryEntry } from '../types/AppTypes';

interface AppNodeProps {
  data: {
    app: AppDirectoryEntry;
    appId: string;
  };
}

export const AppNode: React.FC<AppNodeProps> = memo(({ data }) => {
  const { app, appId } = data;

  return (
    <Card
      sx={{
        minWidth: 200,
        maxWidth: 300,
        bgcolor: 'background.paper',
        boxShadow: 2,
      }}
    >
      <Handle type="target" position={Position.Top} />
      <CardContent>
        <Typography variant="h6" component="div" gutterBottom>
          {app.metadata.name}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Version: {app.latest_version}
        </Typography>
        <Typography variant="body2" color="text.secondary" noWrap>
          {app.metadata.description}
        </Typography>
        <Typography variant="caption" display="block">
          Input: {app.metadata.input.map(i => i['@type']).join(', ')}
        </Typography>
        <Typography variant="caption" display="block">
          Output: {app.metadata.output.map(o => o['@type']).join(', ')}
        </Typography>
      </CardContent>
      <Handle type="source" position={Position.Bottom} />
    </Card>
  );
}); 