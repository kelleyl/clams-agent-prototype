"""
Shared pipeline model for CLAMS pipeline tools.
This module provides data structures and utilities for pipeline representation
that can be used by both the chat interface and visual pipeline editor.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union


class PipelineModel:
    """
    Represents a CLAMS pipeline with tools and connections.
    This model can be shared between the chat interface and visual editor.
    """

    def __init__(self, name: str = "New Pipeline"):
        """
        Initialize a new pipeline model.
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.nodes = []  # List of tools in the pipeline
        self.edges = []  # List of connections between tools
    
    def add_node(self, tool_id: str, tool_data: Dict[str, Any], position: Dict[str, float] = None) -> str:
        """
        Add a tool node to the pipeline.
        
        Args:
            tool_id: ID of the tool
            tool_data: Metadata for the tool
            position: Optional position for visual display
            
        Returns:
            Unique node ID
        """
        # Create a unique ID for this node
        node_id = f"{tool_id}-{len(self.nodes)}"
        
        # Create node with position for visual display
        node = {
            "id": node_id,
            "tool_id": tool_id,
            "data": tool_data,
            "position": position or {"x": 100, "y": 100 * len(self.nodes)}
        }
        
        self.nodes.append(node)
        return node_id
    
    def add_edge(self, source_id: str, target_id: str) -> str:
        """
        Add a connection between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            Edge ID
        """
        edge_id = f"{source_id}-{target_id}"
        
        # Check if edge already exists
        for edge in self.edges:
            if edge["source"] == source_id and edge["target"] == target_id:
                return edge["id"]
        
        edge = {
            "id": edge_id,
            "source": source_id,
            "target": target_id
        }
        
        self.edges.append(edge)
        return edge_id
    
    def clear(self):
        """Clear all nodes and edges from the pipeline."""
        self.nodes = []
        self.edges = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline to a dictionary representation.
        
        Returns:
            Dictionary representation of the pipeline
        """
        return {
            "name": self.name,
            "nodes": self.nodes,
            "edges": self.edges
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineModel':
        """
        Create a pipeline from a dictionary representation.
        
        Args:
            data: Dictionary representation of the pipeline
            
        Returns:
            New PipelineModel instance
        """
        pipeline = cls(name=data.get("name", "Imported Pipeline"))
        pipeline.nodes = data.get("nodes", [])
        pipeline.edges = data.get("edges", [])
        return pipeline
    
    def to_yaml(self) -> str:
        """
        Convert the pipeline to YAML format.
        
        Returns:
            YAML string representation of the pipeline
        """
        pipeline_dict = self.to_dict()
        return yaml.dump(pipeline_dict, default_flow_style=False)
    
    def save_yaml(self, filepath: str):
        """
        Save the pipeline to a YAML file.
        
        Args:
            filepath: Path to save the YAML file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_yaml())
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'PipelineModel':
        """
        Create a pipeline from a YAML string.
        
        Args:
            yaml_str: YAML string representation of a pipeline
            
        Returns:
            New PipelineModel instance
        """
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> 'PipelineModel':
        """
        Load a pipeline from a YAML file.
        
        Args:
            filepath: Path to the YAML file
            
        Returns:
            New PipelineModel instance
        """
        with open(filepath, 'r') as f:
            return cls.from_yaml(f.read())


class PipelineStore:
    """
    Manages storage and retrieval of pipelines.
    """

    def __init__(self, storage_dir: str = "data/pipelines"):
        """
        Initialize the pipeline store.
        
        Args:
            storage_dir: Directory to store pipeline files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_pipeline(self, pipeline: PipelineModel, name: Optional[str] = None) -> str:
        """
        Save a pipeline to storage.
        
        Args:
            pipeline: Pipeline to save
            name: Optional name for the pipeline
            
        Returns:
            Filepath of the saved pipeline
        """
        if name:
            pipeline.name = name
            
        # Generate a filename from the pipeline name
        filename = pipeline.name.lower().replace(" ", "_") + ".yaml"
        filepath = os.path.join(self.storage_dir, filename)
        
        pipeline.save_yaml(filepath)
        return filepath
    
    def load_pipeline(self, name_or_path: str) -> PipelineModel:
        """
        Load a pipeline from storage.
        
        Args:
            name_or_path: Name of the pipeline or path to the pipeline file
            
        Returns:
            Loaded pipeline
        """
        if os.path.isfile(name_or_path):
            return PipelineModel.load_yaml(name_or_path)
            
        # Try to find by name
        filename = name_or_path.lower().replace(" ", "_") + ".yaml"
        filepath = os.path.join(self.storage_dir, filename)
        
        if os.path.isfile(filepath):
            return PipelineModel.load_yaml(filepath)
            
        raise FileNotFoundError(f"Pipeline not found: {name_or_path}")
    
    def list_pipelines(self) -> List[str]:
        """
        List all available pipelines.
        
        Returns:
            List of pipeline names
        """
        pipelines = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".yaml"):
                pipeline_name = filename[:-5].replace("_", " ").title()
                pipelines.append(pipeline_name)
        return pipelines 