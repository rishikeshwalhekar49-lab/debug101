import chalk from 'chalk';

export const colors = {
  mint: chalk.hex('#7FFFD4'),
  cyan: chalk.hex('#87CEEB'),
  lavender: chalk.hex('#B39DDB'),
  amber: chalk.hex('#FFD54F'),
  rose: chalk.hex('#F48FB1'),
  sage: chalk.hex('#A5D6A7'),

  critical: chalk.hex('#FF7675').bold,
  warning: chalk.hex('#FDCB6E').bold,
  info: chalk.hex('#74B9FF'),

  dim: chalk.grey,
  bold: chalk.bold,
  muted: chalk.hex('#636E72'),
  success: chalk.hex('#55EFC4').bold,
  error: chalk.hex('#FF7675').bold,

  added: chalk.hex('#00B894'),
  removed: chalk.hex('#D63031'),
  hunk: chalk.hex('#74B9FF'),
};

export function severityColor(sev: string) {
  if (sev === 'critical') return colors.critical;
  if (sev === 'warning') return colors.warning;
  return colors.info;
}

export function confidenceBar(value: number, width = 10): string {
  const filled = Math.round(Math.max(0, Math.min(1, value)) * width);
  return colors.mint('█'.repeat(filled)) + colors.muted('░'.repeat(width - filled));
}

