import React, { useEffect, useState, useRef } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { runPython, type SbaEvent } from '../utils/pythonBridge.js';
import { DiffViewer } from '../components/DiffViewer.js';
import { Badge } from '../components/Badge.js';

export const options = z.object({
  maxIter: z.number().default(3).describe('Max repair attempts'),
  dryRun: z.boolean().default(false).describe('Show patch without applying'),
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

interface WeightPatchInfo {
  target_param: string;
  surgery_type: string;
  description: string;
  params_affected: number;
  pct_affected: number;
  delta_norm: number;
}

export default function Wsp({
  options: opts = { maxIter: 3, dryRun: false },
  args: cmdArgs = [''],
  onBack,
}: Props) {
  const script = cmdArgs[0] ?? '';
  const { exit } = useApp();

  const [phase, setPhase] = useState<'running' | 'proposal' | 'validating' | 'done' | 'error'>('running');
  const [iteration, setIteration] = useState(1);
  const [origin, setOrigin] = useState('dual');
  const [codeDiff, setCodeDiff] = useState('');
  const [weightPatches, setWeightPatches] = useState<WeightPatchInfo[]>([]);
  const [selectedAction, setSelectedAction] = useState<number>(0);
  const [outcome, setOutcome] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  const ACTIONS = ['Implement Solution & Enter Gate', 'Reject / Skip Solution', 'Stop Pipeline'] as const;

  useInput((input, key) => {
    if (key.escape || input === 'b') {
      if (onBack) onBack();
      else exit();
      return;
    }
    if (input === 'q') {
      exit();
      return;
    }

    if (phase === 'proposal') {
      if (key.upArrow) setSelectedAction((i) => Math.max(0, i - 1));
      if (key.downArrow) setSelectedAction((i) => Math.min(ACTIONS.length - 1, i + 1));
      if (key.return) {
        if (selectedAction === 0) {
          setPhase('validating');
        } else if (selectedAction === 1) {
          setOutcome('Solution skipped by user. Advancing to next iteration...');
          setPhase('done');
        } else {
          setOutcome('Pipeline stopped manually by user.');
          setPhase('done');
        }
      }
    }
  });

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba wsp <script>');
      setPhase('error');
      return;
    }

    const pyArgs = ['wsp', script, '--max-iter', String(opts.maxIter), '--yes'];
    if (opts.dryRun) pyArgs.push('--dry-run');

    (async () => {
      for await (const ev of runPython(pyArgs)) {
        if (ev.event === 'phase') {
          if (ev.phase === 'iteration' && ev.iteration) {
            setIteration(Number(ev.iteration));
          }
          if (ev.phase === 'gate1') {
            setPhase('validating');
          }
          if (ev.phase === 'dual_pipeline' && ev.status === 'done') {
            setOutcome(ev.verdict === 'success' ? 'Validated & Accepted.' : 'Completed.');
            setPhase('done');
          }
        }
        if (ev.event === 'wsp_proposal') {
          setOrigin(String(ev.origin || 'dual'));
          setCodeDiff(String(ev.code_diff || ''));
          setWeightPatches((ev.weight_patches as WeightPatchInfo[]) || []);
          setPhase('proposal');
        }
        if (ev.event === 'error') {
          setErrorMsg(ev.message);
          setPhase('error');
        }
        if (ev.event === 'done') {
          setPhase((p) => (p === 'error' ? 'error' : 'done'));
        }
      }
    })();
  }, [script]);

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#FFD54F">
          ⚡ Weight-Space Programming (WSP)
        </Text>
        <Text color="#636E72">{script}</Text>
        <Badge label={`ITER ${iteration}/${opts.maxIter}`} variant="info" />
        {opts.dryRun && <Badge label="DRY RUN" variant="muted" />}
      </Box>

      {phase === 'running' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">
            Running simultaneous Code-Space (AST/Hooks) & Weight-Space (SVD/Kurtosis) perception…
          </Text>
        </Box>
      )}

      {phase === 'validating' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#87CEEB">
            Evaluating solution across Gate 1 (Overfit Oracle) & Gate 2 (Multi-Seed Sandbox)…
          </Text>
        </Box>
      )}

      {/* Proposal Card */}
      {(phase === 'proposal' || phase === 'validating' || phase === 'done') && (
        <Box flexDirection="column" gap={1} marginTop={1}>
          <Box gap={2}>
            <Text bold color="#55EFC4">
              Proposed Solution
            </Text>
            <Badge label={origin.toUpperCase()} variant="info" />
          </Box>

          {/* Weight Surgery Details */}
          {weightPatches.length > 0 && (
            <Box
              flexDirection="column"
              gap={1}
              borderStyle="round"
              borderColor="#B39DDB"
              paddingX={2}
              paddingY={1}
            >
              <Text bold color="#B39DDB">
                🔬 Weight-Space Surgery ({weightPatches.length} patch(es))
              </Text>
              {weightPatches.map((wp, i) => (
                <Box key={i} flexDirection="column" paddingLeft={1}>
                  <Box gap={2}>
                    <Text bold color="#FFD54F">
                      • {wp.target_param}
                    </Text>
                    <Badge label={wp.surgery_type} variant="info" />
                  </Box>
                  <Text color="#DFE6E9">{wp.description}</Text>
                  <Text color="#636E72">
                    Δ Norm: {wp.delta_norm.toFixed(4)} | Parameters affected:{' '}
                    {wp.pct_affected.toFixed(3)}%
                  </Text>
                </Box>
              ))}
            </Box>
          )}

          {/* Code Diff Details */}
          {codeDiff && <DiffViewer diff={codeDiff} />}
        </Box>
      )}

      {/* Interactive Choice Box */}
      {phase === 'proposal' && (
        <Box
          flexDirection="column"
          gap={1}
          marginTop={1}
          borderStyle="round"
          borderColor="#FFD54F"
          paddingX={2}
          paddingY={1}
        >
          <Text bold color="#FFD54F">
            Would you like to implement this solution and run statistical validation?
          </Text>
          {ACTIONS.map((action, i) => (
            <Box key={action} gap={2}>
              <Text color={i === selectedAction ? '#FFD54F' : '#636E72'}>
                {i === selectedAction ? '▶' : ' '}
              </Text>
              <Text
                color={i === selectedAction ? '#55EFC4' : '#B2BEC3'}
                bold={i === selectedAction}
              >
                {action}
              </Text>
            </Box>
          ))}
          <Text color="#636E72">↑↓ navigate · ↵ confirm choice · [Esc/b] back</Text>
        </Box>
      )}

      {phase === 'done' && (
        <Box flexDirection="column" marginTop={1} gap={1}>
          <Text color="#55EFC4">✔ {outcome || 'Pipeline finished.'}</Text>
          <Text color="#636E72">Press [Esc/b] to return to menu · [q] to quit</Text>
        </Box>
      )}

      {phase === 'error' && (
        <Box flexDirection="column" gap={1}>
          <Text color="#FF7675">✖ {errorMsg}</Text>
          <Text color="#636E72">Press [Esc/b] to return to menu · [q] to quit</Text>
        </Box>
      )}
    </Box>
  );
}
