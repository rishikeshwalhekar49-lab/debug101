import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import z from 'zod';
import { resolveTarget, runPython, type SbaEvent, stopRunning } from '../utils/pythonBridge.js';
import { MultiAgentStepper, type Step } from '../components/MultiAgentStepper.js';
import { MetricMeter } from '../components/MetricMeter.js';

export const options = z.object({
  noLlm: z.boolean().default(false).describe('Use rules and statistics only'),
  maxSteps: z.number().default(150).describe('Max monitor steps'),
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

interface Hypothesis {
  rank: number;
  bug_id: string;
  confidence: number;
  description: string;
  diagnosed_by?: string;
}

export default function Diagnose({
  options: opts = { noLlm: false, maxSteps: 150 },
  args: cmdArgs = [''],
  onBack,
  isEmbedded = false,
}: Props) {
  const shown = cmdArgs[0] ?? ''; // what the user typed, for display
  const script = resolveTarget(shown);
  const { exit } = useApp();

  const [steps, setSteps] = useState<Step[]>([
    { id: 'perception', label: 'Perception (Telemetry & AST)', status: 'pending' },
    { id: 'diagnosis', label: 'Causal Engine Diagnosis', status: 'pending' },
  ]);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
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

  function updateStep(id: string, status: Step['status'], detail?: string) {
    setSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status, detail } : s))
    );
  }

  useEffect(() => () => stopRunning(), []); // leaving the screen stops the command

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba-ui diagnose <script>');
      return;
    }
    const pyArgs = ['diagnose', script, '--max-steps', String(opts.maxSteps)];
    if (opts.noLlm) pyArgs.push('--no-llm');

    (async () => {
      for await (const ev of runPython(pyArgs)) {
        if (ev.event === 'phase') {
          const id =
            ev.phase === 'static_analysis' ||
            ev.phase === 'semantic_review' ||
            ev.phase === 'runtime_monitor'
              ? 'perception'
              : ev.phase;
          if (ev.status === 'start') updateStep(id, 'running');
          if (ev.status === 'done')
            updateStep(
              id,
              'done',
              (ev['findings'] as number | undefined) !== undefined
                ? `${ev['findings']} findings`
                : undefined
            );
          if (ev.status === 'error') updateStep(id, 'error');
        }
        if (ev.event === 'hypothesis') {
          setHypotheses((prev) => [
            ...prev,
            {
              rank: ev.rank,
              bug_id: ev.bug_id,
              confidence: ev.confidence,
              description: ev.description,
              diagnosed_by: ev.diagnosed_by,
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
        <Text bold color="#B39DDB">
          ⬡ Diagnose
        </Text>
        <Text color="#636E72">{shown}</Text>
      </Box>

      <MultiAgentStepper steps={steps} />

      {errorMsg && <Text color="#FF7675">✖ {errorMsg}</Text>}

      {hypotheses.length > 0 && (
        <Box flexDirection="column" gap={1} marginTop={1}>
          <Text bold color="#87CEEB">
            Ranked Root-Cause Hypotheses
          </Text>
          {hypotheses.map((h) => (
            <Box
              key={h.rank}
              flexDirection="column"
              borderStyle="round"
              borderColor="#B39DDB"
              paddingX={1}
            >
              <Box gap={2}>
                <Text color="#FFD54F" bold>
                  #{h.rank}
                </Text>
                <Text bold color="#B2BEC3">
                  {h.bug_id}
                </Text>
                {h.diagnosed_by && (
                  <Text color="#636E72">by {h.diagnosed_by}</Text>
                )}
              </Box>
              <MetricMeter label="Confidence" value={h.confidence} color="#B39DDB" />
              <Text color="#DFE6E9">{h.description}</Text>
            </Box>
          ))}
        </Box>
      )}

      {done && hypotheses.length === 0 && !errorMsg && (
        <Text color="#55EFC4">✔ No bugs diagnosed.</Text>
      )}

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
