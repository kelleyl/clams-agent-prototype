export interface AppMetadata {
  name: string;
  description: string;
  app_version: string;
  mmif_version: string;
  input: AnnotationType[];
  output: AnnotationType[];
  parameters: Parameter[];
}

export interface AnnotationType {
  "@type": string;
  required?: boolean;
  properties?: Record<string, any>;
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

export interface PipelineNode {
  id: string;
  type: 'app';
  data: {
    app: AppDirectoryEntry;
    appId: string;
  };
  position: { x: number; y: number };
}

export interface PipelineEdge {
  id: string;
  source: string;
  target: string;
  type: 'default' | 'invalid';
} 