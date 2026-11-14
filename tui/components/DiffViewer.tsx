import React from 'react';
import { Box, Text } from 'ink';

interface DiffViewerProps {
  diff: string;
  maxLines?: number;
}

export function DiffViewer({ diff, maxLines = 60 }: DiffViewerProps) {
  const lines = diff.split('\n').slice(0, maxLines);
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="#B39DDB" paddingX={1}>
      <Text bold color="#B39DDB">
        Patch Diff
      </Text>
      {lines.map((line, i) => {
        let color = '#B2BEC3';
        if (line.startsWith('+++') || line.startsWith('---')) {
          color = '#74B9FF';
        } else if (line.startsWith('+')) {
          color = '#00B894';
        } else if (line.startsWith('-')) {
          color = '#D63031';
        } else if (line.startsWith('@@')) {
          color = '#FDCB6E';
        }
        return (
          <Text key={i} color={color}>
            {line}
          </Text>
        );
      })}
      {diff.split('\n').length > maxLines && (
        <Text color="#636E72">… {diff.split('\n').length - maxLines} more lines</Text>
      )}
    </Box>
  );
}

