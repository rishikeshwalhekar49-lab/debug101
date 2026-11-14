import React from 'react';
import { Text } from 'ink';

interface BadgeProps {
  label: string;
  variant?: 'critical' | 'warning' | 'info' | 'success' | 'muted';
}

const VARIANT_COLORS: Record<string, string> = {
  critical: '#FF7675',
  warning: '#FDCB6E',
  info: '#74B9FF',
  success: '#55EFC4',
  muted: '#636E72',
};

export function Badge({ label, variant = 'info' }: BadgeProps) {
  const color = VARIANT_COLORS[variant] ?? VARIANT_COLORS['info'];
  return (
    <Text color={color} bold>
      {`[${label.toUpperCase()}]`}
    </Text>
  );
}

export function SeverityBadge({ severity }: { severity: string }) {
  const variant =
    severity === 'critical'
      ? 'critical'
      : severity === 'warning'
      ? 'warning'
      : 'info';
  return <Badge label={severity} variant={variant} />;
}

