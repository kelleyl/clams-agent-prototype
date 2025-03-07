import { Node, Edge } from 'reactflow';

export interface AppMetadata {
  name?: string;
  description: string;
  app_version?: string;
  mmif_version?: string;
  input: (AnnotationType | AnnotationType[])[];
  output: AnnotationType[];
  parameters: Parameter[];
}

export interface AnnotationType {
  "@type": string;
  required?: boolean;
  properties?: {
    timeUnit?: string;
    labelset?: string[];
    [key: string]: any;
  };
}

export interface Parameter {
  name: string;
  description: string;
  type: string;
  default?: any;
}

export interface AppDirectoryEntry {
  latest_version: string;
  metadata: AppMetadata;
}

export interface AppDirectory {
  [key: string]: AppDirectoryEntry;
}

export type PipelineNode = Node<{
  app: AppDirectoryEntry;
  appId: string;
}> & {
  type: 'app';
  dragHandle?: string;
};

export type PipelineEdge = Edge & {
  type: 'default' | 'invalid';
}; 