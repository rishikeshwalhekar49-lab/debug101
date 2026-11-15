import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { resolveTarget, runPython, type FindingData, type SbaEvent, stopRunning } from '../utils/pythonBridge.js';
import { SeverityBadge } from '../components/Badge.js';

export const options = z.object({
  maxSteps: z.number().default(300).describe('Max optimizer steps to watch'),
  maxSeconds: z.number().default(900).describe('Time limit in seconds'),
  stopOnCritical: z.boolean().default(false).describe('Stop on first critical finding'),
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

export default function Monitor({
  options: opts = { maxSteps: 300, maxSeconds: 900, stopOnCritical: false },
  args: cmdArgs = [''],
  onBack,
  isEmbedded = false,
}: Props) {
  const shown = cmdArgs[0] ?? ''; // what the user typed, for display
  const script = resolveTarget(shown);
  const { exit } = useApp();
  const [alerts, setAlerts] = useState<FindingData[]>([]);
  const [done, setDone] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  useInput((input, key) => {
    if (key.escape || input === 'b') {
      if (onBack) {
        onBack();
      } else {
        exit();
      }
    }
    if (input === 'q') {
      exit();
    }
  });

  useEffect(() => () => stopRunning(), []); // leaving the screen stops the command

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba-ui monitor <script>');
      return;
    }
    const pyArgs = ['monitor', script, '--max-steps', String(opts.maxSteps)];
    if (opts.stopOnCritical) pyArgs.push('--stop-on-critical');

    (async () => {
      for await (const ev of runPython(pyArgs)) {
        if (ev.event === 'runtime_alert') {
          setAlerts((prev) => [
            ...prev,
            {
              bug_id: ev.bug_id,
              severity: ev.severity as FindingData['severity'],
              file: ev.file,
              line: ev.line,
              message: ev.message,
            },
          ]);
        }
        if (ev.event === 'done') {
          setDone(true);
        }
        if (ev.event === 'error') {
          setErrorMsg(ev.message);
          setDone(true);
        }
      }
    })();
  }, [script]);

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#7FFFD4">
          ⬡ Monitor
        </Text>
        <Text color="#636E72">{shown}</Text>
      </Box>

      {!done && (
        <Box gap={2}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">Watching runtime hooks (loss, gradients, activations)…</Text>
        </Box>
      )}
      {errorMsg && <Text color="#FF7675">✖ {errorMsg}</Text>}

      {alerts.length === 0 && done && (
        <Text color="#55EFC4">✔ No anomalies detected in the training loop.</Text>
      )}

      <Box flexDirection="column" gap={1}>
        {alerts.map((a, i) => (
          <Box
            key={i}
            flexDirection="column"
            borderStyle="round"
            borderColor={a.severity === 'critical' ? '#FF7675' : '#FDCB6E'}
            paddingX={1}
          >
            <Box gap={2}>
              <SeverityBadge severity={a.severity} />
              <Text bold color="#B2BEC3">
                {a.bug_id}
              </Text>
            </Box>
            <Text color="#DFE6E9">{a.message}</Text>
            <Text color="#636E72">
              {a.file}:{a.line}
            </Text>
          </Box>
        ))}
      </Box>

      {done && (
        <Box marginTop={1}>
          <Text color="#636E72">
            Press [Esc/b] to return to menu  ·  [q] to quit
          </Text>
        </Box>
      )}
    </Box>
  );
}
