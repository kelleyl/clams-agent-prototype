import React, { useCallback, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Connection,
  addEdge,
  Background,
  Controls,
  MiniMap,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { AppDirectory, PipelineNode, PipelineEdge } from '../types/AppTypes';
import { Box, Paper } from '@mui/material';
import { AppNode } from './AppNode';
import { validateConnection } from '../utils/pipelineValidation';

interface PipelineProps {
  appDirectory: AppDirectory;
}

const nodeTypes = {
  app: AppNode,
};

export const Pipeline: React.FC<PipelineProps> = ({ appDirectory }) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  const onConnect = useCallback(
    (params: Connection) => {
      if (!params.source || !params.target) return;

      const sourceNode = nodes.find(n => n.id === params.source) as PipelineNode;
      const targetNode = nodes.find(n => n.id === params.target) as PipelineNode;

      if (!sourceNode || !targetNode) return;

      const isValid = validateConnection(sourceNode.data.app, targetNode.data.app);
      const newEdge: PipelineEdge = {
        ...params,
        id: `${params.source}-${params.target}`,
        type: isValid ? 'default' : 'invalid',
      };

      setEdges(prevEdges => addEdge(newEdge, prevEdges));
    },
    [nodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const appId = event.dataTransfer.getData('application/clams-app');
      if (!appId || !appDirectory[appId]) return;

      const position = {
        x: event.clientX - event.currentTarget.getBoundingClientRect().left,
        y: event.clientY - event.currentTarget.getBoundingClientRect().top,
      };

      const newNode: PipelineNode = {
        id: `${appId}-${nodes.length}`,
        type: 'app',
        position,
        data: {
          app: appDirectory[appId],
          appId,
        },
      };

      setNodes(prevNodes => [...prevNodes, newNode]);
    },
    [appDirectory, nodes]
  );

  return (
    <Paper
      sx={{
        width: '100%',
        height: '80vh',
        border: '1px solid #ccc',
        borderRadius: 2,
      }}
    >
      <Box sx={{ width: '100%', height: '100%' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onConnect={onConnect}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={nodeTypes}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </Box>
    </Paper>
  );
}; 