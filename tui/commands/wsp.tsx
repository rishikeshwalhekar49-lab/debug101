import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { resolveTarget, runPython, type WeightReportData, stopRunning } from '../utils/pythonBridge.js';
import { Badge, SeverityBadge } from '../components/Badge.js';

export const description = 'Inspect a model\'s weights for anomalies (read-only)';

export const options = z.object({
  maxSteps: z.number().default(60).describe('Optimizer steps to train before the weights are read'),
});

export const args = z.tuple([
  z.string().describe('Path to a Python training script'),
]);

type Props = {
  options?: z.infer<typeof options>;
  args?: z.infer<typeof args>;
  onBack?: () => void;
  isEmbedded?: boolean;
};

export default function Wsp({
  options: opts = { maxSteps: 60 },
  args: cmdArgs = [''],
  onBack,
}: Props) {
  const shown = cmdArgs[0] ?? ''; // what the user typed, for display
  const script = resolveTarget(shown);
  const { exit } = useApp();

  const [phase, setPhase] = useState<'running' | 'done' | 'error'>('running');
  const [report, setReport] = useState<WeightReportData | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  useInput((input, key) => {
    if (key.escape || input === 'b') {
      if (onBack) onBack();
      else exit();
      return;
    }
    if (input === 'q') exit();
  });

  useEffect(() => () => stopRunning(), []); // leaving the screen stops the command

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba-ui wsp <script>');
      setPhase('error');
      return;
    }
    (async () => {
      let failed = false;
      for await (const ev of runPython(['wsp', script, '--max-steps', String(opts.maxSteps)])) {
        if (ev.event === 'weight_report') setReport(ev.data);
        if (ev.event === 'error') {
          failed = true;
          setErrorMsg(ev.message);
          setPhase('error');
        }
        if (ev.event === 'done' && !failed) setPhase('done');
      }
    })();
  }, [script]);

  const findings = report?.findings ?? [];

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#FFD54F">
          ⬡ Weight inspection
        </Text>
        <Text color="#636E72">{shown}</Text>
        <Badge label="READ-ONLY" variant="muted" />
      </Box>

      {phase === 'running' && (
        <Box gap={2}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">
            Training for {opts.maxSteps} steps in the sandbox, then reading the weights. Nothing is modified…
          </Text>
        </Box>
      )}

      {report && (
        <Box flexDirection="column">
          <Text color="#B2BEC3">Source: {report.source}</Text>
          {report.inspected && (
            <Text color="#B2BEC3">Values inspected: {report.total_parameters.toLocaleString()}</Text>
          )}
        </Box>
      )}

      {findings.map((f, i) => (
        <Box
          key={i}
          flexDirection="column"
          borderStyle="round"
          borderColor={f.severity === 'critical' ? '#FF7675' : f.severity === 'warning' ? '#FDCB6E' : '#74B9FF'}
          paddingX={1}
        >
          <Box gap={2}>
            <SeverityBadge severity={f.severity} />
            <Text bold color="#B2BEC3">
              {f.anomaly_type}
            </Text>
            <Text color="#636E72">
              {f.layer_name} [{f.param_shape.join(' × ')}]
            </Text>
          </Box>
          <Text color="#DFE6E9">{f.message}</Text>
          {f.suggested_action && <Text color="#55EFC4">next: {f.suggested_action}</Text>}
        </Box>
      ))}

      {phase === 'done' && report?.inspected && findings.length === 0 && (
        <Text color="#55EFC4">✔ No weight anomalies found.</Text>
      )}

      {(report?.diagnostics ?? []).map((d, i) => (
        <Text key={i} color="#636E72">
          note: {d}
        </Text>
      ))}

      {phase === 'error' && <Text color="#FF7675">✖ {errorMsg}</Text>}

      {phase !== 'running' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="#636E72">
            This command only reads weights. To repair a bug in the training code, use Fix.
          </Text>
          <Text color="#636E72">Press [Esc/b] to return to menu  ·  [q] to quit</Text>
        </Box>
      )}
    </Box>
  );
}
