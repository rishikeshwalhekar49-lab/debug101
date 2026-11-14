import React from 'react';
import { Box, Text } from 'ink';

interface MetricMeterProps {
  label: string;
  value: number; // 0..1
  width?: number;
  color?: string;
}

export function MetricMeter({ label, value, width = 12, color = '#7FFFD4' }: MetricMeterProps) {
  const filled = Math.max(0, Math.min(width, Math.round(value * width)));
  const bar = '█'.repeat(filled) + '░'.repeat(width - filled);
  const pct = `${Math.round(value * 100)}%`;
  return (
    <Box flexDirection="row" gap={1}>
      <Text color="#636E72">{label.padEnd(14)}</Text>
      <Text color={color}>{bar}</Text>
      <Text color="#B2BEC3"> {pct}</Text>
    </Box>
  );
}

interface ValidationMetricsProps {
  pValue?: number;
  effectSize?: number;
  confidence?: number;
}

export function ValidationMetrics({ pValue, effectSize, confidence }: ValidationMetricsProps) {
  return (
    <Box flexDirection="column">
      {confidence !== undefined && (
        <MetricMeter label="Confidence" value={confidence} color="#87CEEB" />
      )}
      {effectSize !== undefined && (
        <MetricMeter label="Effect Size" value={Math.min(1, Math.max(0, effectSize))} color="#7FFFD4" />
      )}
      {pValue !== undefined && (
        <MetricMeter label="Significance" value={Math.min(1, Math.max(0, 1 - pValue))} color="#B39DDB" />
      )}
    </Box>
  );
}

