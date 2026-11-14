import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { runPython, type SbaEvent } from '../utils/pythonBridge.js';

export const options = z.object({
  live: z.boolean().default(false).describe('Send a live test request to each model'),
});

type Props = {
  options?: z.infer<typeof options>;
  onBack?: () => void;
  isEmbedded?: boolean;
};

interface Check {
  ok: boolean;
  label: string;
  advice: string;
}

export default function Doctor({ options: opts = { live: false }, onBack, isEmbedded = false }: Props) {
  const { exit } = useApp();
  const [checks, setChecks] = useState<Check[]>([]);
  const [done, setDone] = useState(false);
  const [problems, setProblems] = useState(0);

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

  useEffect(() => {
    const pyArgs = ['doctor'];
    if (opts.live) pyArgs.push('--live');

    (async () => {
      for await (const ev of runPython(pyArgs)) {
        if (ev.event === 'doctor_check') {
          setChecks((prev) => [
            ...prev,
            { ok: ev.ok, label: ev.label, advice: ev.advice },
          ]);
          if (!ev.ok) setProblems((p) => p + 1);
        }
        if (ev.event === 'done') {
          setDone(true);
        }
      }
    })();
  }, []);

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#7FFFD4">
          ⬡ Doctor Check
        </Text>
        {opts.live && <Text color="#636E72">--live</Text>}
      </Box>

      {!done && checks.length === 0 && (
        <Box gap={2}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">Checking environment, models, and dependencies…</Text>
        </Box>
      )}

      <Box flexDirection="column">
        {checks.map((c, i) => (
          <Box key={i} flexDirection="row" gap={2}>
            <Text color={c.ok ? '#55EFC4' : '#FF7675'}>{c.ok ? '✔' : '✖'}</Text>
            <Text color={c.ok ? '#B2BEC3' : '#FF7675'} bold={!c.ok}>
              {c.label}
            </Text>
            {!c.ok && c.advice && (
              <Text color="#636E72">→ {c.advice}</Text>
            )}
          </Box>
        ))}
      </Box>

      {done && (
        <Box flexDirection="column" marginTop={1} gap={1}>
          {problems === 0 ? (
            <Text color="#55EFC4" bold>
              ✔ All checks passed. System ready.
            </Text>
          ) : (
            <Text color="#FF7675" bold>
              ✖ {problems} problem(s) found.
            </Text>
          )}
          <Text color="#636E72">
            Press [Esc/b] to return to menu  ·  [q] to quit
          </Text>
        </Box>
      )}
    </Box>
  );
}
