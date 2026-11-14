import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { runPython, type SbaEvent } from '../utils/pythonBridge.js';
import { MultiAgentStepper, buildFixSteps, type Step } from '../components/MultiAgentStepper.js';
import { DiffViewer } from '../components/DiffViewer.js';
import { ValidationMetrics } from '../components/MetricMeter.js';
import { Badge } from '../components/Badge.js';

export const options = z.object({
  maxIter: z.number().default(3).describe('Max repair attempts per bug'),
  maxRounds: z.number().default(3).describe('Max bug rounds'),
  dryRun: z.boolean().default(false).describe('Show patch without applying'),
  noLlm: z.boolean().default(false).describe('Rules and statistics only'),
  seeds: z.number().default(5).describe('Seeds for statistical gate'),
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

type UIPhase = 'running' | 'approval' | 'applying' | 'done' | 'error';

interface VerdictInfo {
  bug_id: string;
  accepted: boolean;
  validator: string;
  p_value?: number;
  effect_size?: number;
  notes: string;
}

const APPROVAL_CHOICES = ['Apply to File', 'Keep Dry-Run', 'Retry Fix'] as const;
type ApprovalChoice = (typeof APPROVAL_CHOICES)[number];

export default function Fix({
  options: opts = { maxIter: 3, maxRounds: 3, dryRun: false, noLlm: false, seeds: 5 },
  args: cmdArgs = [''],
  onBack,
  isEmbedded = false,
}: Props) {
  const script = cmdArgs[0] ?? '';
  const { exit } = useApp();

  const [uiPhase, setUiPhase] = useState<UIPhase>('running');
  const [steps, setSteps] = useState<Step[]>(buildFixSteps());
  const [pendingDiff, setPendingDiff] = useState('');
  const pendingDiffRef = React.useRef('');
  const [verdicts, setVerdicts] = useState<VerdictInfo[]>([]);
  const [approvalIdx, setApprovalIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [outcome, setOutcome] = useState('');

  function updateStep(id: string, status: Step['status'], detail?: string) {
    setSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status, detail } : s))
    );
  }  useInput((input, key) => {
    if (key.escape || input === 'b') {
      if (onBack) {
        onBack();
        return;
      } else {
        exit();
        return;
      }
    }
    if (input === 'q') {
      exit();
      return;
    }

    if (uiPhase === 'approval') {
      if (key.upArrow) setApprovalIdx((i) => Math.max(0, i - 1));
      if (key.downArrow) setApprovalIdx((i) => Math.min(APPROVAL_CHOICES.length - 1, i + 1));
      if (key.return) {
        const choice: ApprovalChoice = APPROVAL_CHOICES[approvalIdx]!;
        setUiPhase('applying');
        if (choice === 'Apply to File') {
          setOutcome('Patch validated and accepted. (To auto-apply, run: sba fix <script>)');
        } else if (choice === 'Retry Fix') {
          setOutcome('Fix rejected by user. Re-run to start another iteration.');
        } else {
          setOutcome('Patch kept as dry-run. Target file unchanged.');
        }
        setUiPhase('done');
      }
    }
  });

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba-ui fix <script>');
      setUiPhase('error');
      return;
    }
    const pyArgs = [
      'fix',
      script,
      '--max-iter',
      String(opts.maxIter),
      '--max-rounds',
      String(opts.maxRounds),
    ];
    if (opts.dryRun) pyArgs.push('--dry-run');
    if (opts.noLlm) pyArgs.push('--no-llm');

    // Keep a ref so the done-handler can read the latest diff without stale closure
    let latestDiff = '';
    let hadError = false;

    // Map Python supervisor phase names → stepper step IDs
    const phaseMap: Record<string, string> = {
      perception: 'perception',
      diagnosis: 'diagnosis',
      repair: 'repair',
      validation: 'gate1',   // Overfit Oracle fires inside validation
      gate1: 'gate1',
      gate2: 'gate2',
      memory: 'memory',
    };

    function handleEvent(ev: SbaEvent) {
      if (ev.event === 'phase') {
        const stepId = phaseMap[ev.phase] ?? ev.phase;
        if (ev.status === 'start') updateStep(stepId, 'running');
        if (ev.status === 'done') updateStep(stepId, 'done');
        if (ev.status === 'error') updateStep(stepId, 'error');
      }
      if (ev.event === 'patch') {
        latestDiff = ev.diff;
        setPendingDiff(ev.diff);
        pendingDiffRef.current = ev.diff;
      }
      if (ev.event === 'verdict') {
        setVerdicts((prev) => [
          ...prev,
          {
            bug_id: ev.bug_id,
            accepted: ev.accepted,
            validator: ev.validator,
            p_value: ev.p_value,
            effect_size: ev.effect_size,
            notes: ev.notes,
          },
        ]);
        // Multi-seed validator results → gate2 step
        if (ev.validator && ev.validator.toLowerCase().includes('seed')) {
          if (ev.accepted) {
            updateStep('gate2', 'done', `p=${ev.p_value !== undefined ? ev.p_value.toFixed(3) : '—'}`);
          } else {
            updateStep('gate2', 'error', 'rejected');
          }
        } else {
          // Overfit oracle result → gate1 step
          if (ev.accepted) {
            updateStep('gate1', 'done');
          } else {
            updateStep('gate1', 'error', 'overfit');
          }
        }
      }
      if (ev.event === 'error') {
        hadError = true;
        setErrorMsg(ev.message);
        setUiPhase('error');
      }
      if (ev.event === 'done') {
        if (hadError) return;
        if (latestDiff) {
          setUiPhase('approval');
        } else {
          setOutcome(
            ev.exit_code === 0
              ? 'Pipeline finished cleanly. No patches required.'
              : 'Pipeline completed without generating a verified patch.'
          );
          setUiPhase('done');
        }
      }
    }

    (async () => {
      for await (const ev of runPython(pyArgs)) {
        handleEvent(ev);
      }
    })();
  }, [script]);

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#FFD54F">
          ⬡ Fix Loop
        </Text>
        <Text color="#636E72">{script}</Text>
        {opts.dryRun && <Badge label="DRY RUN" variant="muted" />}
      </Box>

      <MultiAgentStepper steps={steps} />

      {verdicts.length > 0 && (
        <Box flexDirection="column" gap={1} marginTop={1}>
          <Text bold color="#87CEEB">
            Validation Verdicts
          </Text>
          {verdicts.map((v, i) => (
            <Box key={i} flexDirection="column" paddingLeft={2}>
              <Box gap={2}>
                <Badge
                  label={v.accepted ? 'ACCEPTED' : 'REJECTED'}
                  variant={v.accepted ? 'success' : 'critical'}
                />
                <Text color="#636E72">{v.validator}</Text>
              </Box>
              <ValidationMetrics pValue={v.p_value} effectSize={v.effect_size} />
              {v.notes && <Text color="#636E72">{v.notes}</Text>}
            </Box>
          ))}
        </Box>
      )}

      {pendingDiff &&
        (uiPhase === 'approval' || uiPhase === 'applying' || uiPhase === 'done') && (
          <DiffViewer diff={pendingDiff} />
        )}

      {uiPhase === 'approval' && (
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
            What would you like to do with this patch?
          </Text>
          {APPROVAL_CHOICES.map((choice, i) => (
            <Box key={choice} gap={2}>
              <Text color={i === approvalIdx ? '#FFD54F' : '#636E72'}>
                {i === approvalIdx ? '▶' : ' '}
              </Text>
              <Text
                color={i === approvalIdx ? '#55EFC4' : '#B2BEC3'}
                bold={i === approvalIdx}
              >
                {choice}
              </Text>
            </Box>
          ))}
          <Text color="#636E72">↑↓ navigate  ·  ↵ confirm choice  ·  [Esc/b] back</Text>
        </Box>
      )}

      {uiPhase === 'running' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">Multi-agent repair pipeline executing…</Text>
        </Box>
      )}

      {uiPhase === 'done' && (
        <Box flexDirection="column" marginTop={1} gap={1}>
          <Text color={outcome.includes('cleanly') || outcome.includes('accepted') ? '#55EFC4' : '#FFD54F'}>
            {outcome || 'Pipeline finished.'}
          </Text>
          <Text color="#636E72">Press [Esc/b] to return to menu  ·  [q] to quit</Text>
        </Box>
      )}

      {uiPhase === 'error' && (
        <Box flexDirection="column" gap={1}>
          <Text color="#FF7675">✖ {errorMsg}</Text>
          <Text color="#636E72">Press [Esc/b] to return to menu  ·  [q] to quit</Text>
        </Box>
      )}
    </Box>
  );
}
