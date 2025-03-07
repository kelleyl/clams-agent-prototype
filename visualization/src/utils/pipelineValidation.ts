import { AppDirectoryEntry, AnnotationType } from '../types/AppTypes';

/**
 * Checks if two annotation types are compatible
 */
const areTypesCompatible = (outputType: AnnotationType, inputType: AnnotationType): boolean => {
  // Remove version numbers from types for comparison
  const normalizeType = (type: string): string => {
    return type.split('/').slice(0, -1).join('/');
  };

  const normalizedOutput = normalizeType(outputType['@type']);
  const normalizedInput = normalizeType(inputType['@type']);

  return normalizedOutput === normalizedInput;
};

/**
 * Validates if a connection between two apps is valid based on their input/output types
 */
export const validateConnection = (
  sourceApp: AppDirectoryEntry,
  targetApp: AppDirectoryEntry
): boolean => {
  // Check if any output type from source matches any input type from target
  return sourceApp.metadata.output.some(outputType =>
    targetApp.metadata.input.some(inputType =>
      areTypesCompatible(outputType, inputType)
    )
  );
}; 