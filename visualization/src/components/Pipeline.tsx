import React, { useCallback, useState, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Connection,
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  XYPosition,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { AppDirectory, PipelineNode, PipelineEdge } from '../types/AppTypes';
import { Box, Paper } from '@mui/material';
import { AppNode } from './AppNode';
import { validateConnection } from '../utils/pipelineValidation';

interface PipelineProps {
  appDirectory: AppDirectory;
  initialPipeline?: any;
  onPipelineChange?: (pipeline: any) => void;
}

const nodeTypes = {
  app: AppNode,
};

export const Pipeline: React.FC<PipelineProps> = ({ 
  appDirectory, 
  initialPipeline,
  onPipelineChange 
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<PipelineNode['data']>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<PipelineEdge>([]);

  // Load initial pipeline if provided
  useEffect(() => {
    if (initialPipeline && initialPipeline.nodes && initialPipeline.edges) {
      // Convert the pipeline nodes to ReactFlow nodes
      const rfNodes = initialPipeline.nodes.map((node: any) => ({
        id: node.id,
        type: 'app',
        position: node.position,
        data: {
          app: appDirectory[node.tool_id],
          appId: node.tool_id,
        },
        dragHandle: '.drag-handle',
      }));
      
      // Convert the pipeline edges to ReactFlow edges
      const rfEdges = initialPipeline.edges.map((edge: any) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: 'default',
      }));
      
      setNodes(rfNodes);
      setEdges(rfEdges);
    }
  }, [initialPipeline, appDirectory, setNodes, setEdges]);
  
  // Update the pipeline when nodes or edges change
  useEffect(() => {
    if (onPipelineChange) {
      // Convert ReactFlow nodes to pipeline nodes
      const pipelineNodes = nodes.map(node => ({
        id: node.id,
        tool_id: node.data.appId,
        data: node.data.app,
        position: node.position,
      }));
      
      // Convert ReactFlow edges to pipeline edges
      const pipelineEdges = edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      }));
      
      onPipelineChange({
        name: initialPipeline?.name || 'New Pipeline',
        nodes: pipelineNodes,
        edges: pipelineEdges,
      });
    }
  }, [nodes, edges, onPipelineChange, initialPipeline]);

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
        source: params.source,
        target: params.target,
        type: isValid ? 'default' : 'invalid',
      };

      setEdges(prevEdges => addEdge(newEdge, prevEdges));
    },
    [nodes, setEdges]
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

      // Get the drop position relative to the ReactFlow viewport
      const reactFlowBounds = (event.target as Element)
        .closest('.react-flow')
        ?.getBoundingClientRect();

      if (!reactFlowBounds) return;

      const position: XYPosition = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: PipelineNode = {
        id: `${appId}-${nodes.length}`,
        type: 'app',
        position,
        data: {
          app: appDirectory[appId],
          appId,
        },
        dragHandle: '.drag-handle',
      };

      setNodes(prevNodes => [...prevNodes, newNode]);
    },
    [appDirectory, nodes, setNodes]
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
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
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