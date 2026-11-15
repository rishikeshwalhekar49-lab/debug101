import React, { useEffect, useRef, useState } from 'react';
import path from 'node:path';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { resolveTarget, runPython, type SbaEvent } from '../utils/pythonBridge.js';
import { MultiAgentStepper, buildFixSteps, type Step } from '../components/MultiAgentStepper.js';
import { DiffViewer } from '../components/DiffViewer.js';
import { Badge } from '../components/Badge.js';

export const options = z.object({
  maxIter: z.number().default(3).describe('Max repair attempts per bug'),
  dryRun: z.boolean().default(false).describe('Show the validated patch and stop, without asking to apply it'),
  noLlm: z.boolean().default(false).describe('Rules and statistics only'),
  seeds: z.number().default(5).describe('Seeds for the statistical gate (3 to 5)'),
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
  p_value?: number;
  effect_size?: number;
  notes: string;
}

interface AcceptedPatch {
  bug_id: string;
  source: string;
  diff: string;
  pending: boolean;
  blocked?: string | null;
}

// The pipeline always runs as a dry run here, so the script is untouched when this menu
// appears. "Apply" then asks the backend to write exactly the patch that was validated.
const APPROVAL_CHOICES = [
  'Apply this patch to the file',
  'Do not apply (the file stays unchanged)',
] as const;

export default function Fix({
  options: opts = { maxIter: 3, dryRun: false, noLlm: false, seeds: 5 },
  args: cmdArgs = [''],
  onBack,
}: Props) {
  const shown = cmdArgs[0] ?? ''; // what the user typed, for display
  const script = resolveTarget(shown);
  const { exit } = useApp();

  const [uiPhase, setUiPhase] = useState<UIPhase>('running');
  const uiPhaseRef = useRef<UIPhase>('running');
  const [steps, setSteps] = useState<Step[]>(buildFixSteps());
  const [accepted, setAccepted] = useState<AcceptedPatch | null>(null);
  const [verdicts, setVerdicts] = useState<VerdictInfo[]>([]);
  const [approvalIdx, setApprovalIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [outcome, setOutcome] = useState('');
  const [changed, setChanged] = useState(false);

  function go(next: UIPhase) {
    uiPhaseRef.current = next;
    setUiPhase(next);
  }

  function updateStep(id: string, status: Step['status'], detail?: string) {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, status, detail } : s)));
  }

  async function applyPatch() {
    go('applying');
    let failed = '';
    let backup = '';
    let wrote = false;
    for await (const ev of runPython(['apply', script])) {
      if (ev.event === 'applied') {
        wrote = true;
        backup = ev.backup ?? '';
      }
      if (ev.event === 'error') failed = ev.message;
    }
    if (wrote) {
      setChanged(true);
      setOutcome(
        `Patch applied to ${shown}.` +
          (backup ? ` The original is saved beside it as ${path.basename(backup)}.` : '') +
          ' Run Fix again to look for further bugs.'
      );
      go('done');
    } else {
      setErrorMsg(`The patch was not applied, and the file is unchanged. ${failed}`.trim());
      go('error');
    }
  }

  useInput((input, key) => {
    if (uiPhaseRef.current === 'applying') return; // never abandon a write half-way
    if (key.escape || input === 'b') {
      if (onBack) onBack();
      else exit();
      return;
    }
    if (input === 'q') {
      exit();
      return;
    }
    if (uiPhaseRef.current === 'approval') {
      if (key.upArrow) setApprovalIdx((i) => Math.max(0, i - 1));
      if (key.downArrow) setApprovalIdx((i) => Math.min(APPROVAL_CHOICES.length - 1, i + 1));
      if (key.return) {
        if (approvalIdx === 0) {
          void applyPatch();
        } else {
          setOutcome(
            'Not applied. The file is unchanged. The validated patch is kept, so you can still write it ' +
              `later with: python -m silent_bug_agent apply ${script}`
          );
          go('done');
        }
      }
    }
  });

  useEffect(() => {
    if (!script) {
      setErrorMsg('Usage: sba-ui fix <script>');
      go('error');
      return;
    }
    const pyArgs = [
      'fix',
      script,
      '--dry-run',
      '--max-iter',
      String(opts.maxIter),
      '--seeds',
      String(opts.seeds),
    ];
    if (opts.noLlm) pyArgs.push('--no-llm');

    let acceptedPatch: AcceptedPatch | null = null;
    let hadError = false;

    function handleEvent(ev: SbaEvent) {
      if (ev.event === 'phase') {
        const known = ['perception', 'diagnosis', 'repair', 'gate1', 'gate2', 'memory'];
        if (!known.includes(ev.phase)) return;
        if (ev.status === 'start') updateStep(ev.phase, 'running');
        if (ev.status === 'error') updateStep(ev.phase, 'error');
        if (ev.status === 'done') {
          const isGate = ev.phase === 'gate1' || ev.phase === 'gate2';
          if (isGate && ev['skipped'] === true) updateStep(ev.phase, 'skipped', 'not run: gate 1 rejected the patch');
          else if (isGate && ev['accepted'] === false) updateStep(ev.phase, 'error', 'rejected');
          else if (ev.phase === 'repair' && ev['candidate'] === false) updateStep(ev.phase, 'error', 'no usable patch');
          else updateStep(ev.phase, 'done');
        }
      }
      if (ev.event === 'verdict') {
        setVerdicts((prev) => [
          ...prev,
          {
            bug_id: ev.bug_id,
            accepted: ev.accepted,
            p_value: ev.p_value ?? undefined,
            effect_size: ev.effect_size ?? undefined,
            notes: ev.notes,
          },
        ]);
      }
      if (ev.event === 'accepted_patch') {
        acceptedPatch = {
          bug_id: ev.bug_id,
          source: ev.source,
          diff: ev.diff,
          pending: ev.pending,
          blocked: ev.blocked,
        };
        setAccepted(acceptedPatch);
      }
      if (ev.event === 'error') {
        hadError = true;
        setErrorMsg(ev.message);
        go('error');
      }
      if (ev.event === 'done') {
        if (hadError) return;
        if (!acceptedPatch) {
          setOutcome(
            ev.exit_code === 0
              ? 'No bug that needs a repair was found. The file is unchanged.'
              : 'No patch passed validation, so nothing was changed.'
          );
          go('done');
        } else if (acceptedPatch.blocked) {
          setOutcome(`A patch passed validation, but the file was not changed. ${acceptedPatch.blocked}`);
          go('done');
        } else if (opts.dryRun || !acceptedPatch.pending) {
          setOutcome(
            'Dry run: a patch passed validation and the file is unchanged. Write it with: ' +
              `python -m silent_bug_agent apply ${script}`
          );
          go('done');
        } else {
          go('approval');
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
        <Text color="#636E72">{shown}</Text>
        {opts.dryRun && <Badge label="DRY RUN" variant="muted" />}
      </Box>

      <MultiAgentStepper steps={steps} />

      {verdicts.length > 0 && (
        <Box flexDirection="column" gap={1} marginTop={1}>
          <Text bold color="#87CEEB">
            Validation verdicts
          </Text>
          {verdicts.map((v, i) => (
            <Box key={i} flexDirection="column" paddingLeft={2}>
              <Box gap={2}>
                <Badge
                  label={v.accepted ? 'ACCEPTED' : 'REJECTED'}
                  variant={v.accepted ? 'success' : 'critical'}
                />
                <Text color="#B2BEC3">{v.bug_id}</Text>
              </Box>
              {(v.p_value !== undefined || v.effect_size !== undefined) && (
                <Text color="#B2BEC3">
                  {v.p_value !== undefined ? `p = ${v.p_value.toPrecision(3)}` : ''}
                  {v.p_value !== undefined && v.effect_size !== undefined ? '   ' : ''}
                  {v.effect_size !== undefined ? `effect size (Cohen's d) = ${v.effect_size.toFixed(2)}` : ''}
                </Text>
              )}
              {v.notes && <Text color="#636E72">{v.notes}</Text>}
            </Box>
          ))}
        </Box>
      )}

      {accepted && uiPhase !== 'running' && (
        <Box flexDirection="column" gap={1}>
          <Box gap={2}>
            <Text bold color="#55EFC4">
              Validated patch for {accepted.bug_id}
            </Text>
            <Badge label={accepted.source} variant="info" />
            <Badge label={changed ? 'WRITTEN TO FILE' : 'FILE NOT CHANGED'} variant={changed ? 'success' : 'muted'} />
          </Box>
          <DiffViewer diff={accepted.diff} />
        </Box>
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
            This patch passed both gates. Nothing has been written yet. Apply it?
          </Text>
          {APPROVAL_CHOICES.map((choice, i) => (
            <Box key={choice} gap={2}>
              <Text color={i === approvalIdx ? '#FFD54F' : '#636E72'}>
                {i === approvalIdx ? '▶' : ' '}
              </Text>
              <Text color={i === approvalIdx ? '#55EFC4' : '#B2BEC3'} bold={i === approvalIdx}>
                {choice}
              </Text>
            </Box>
          ))}
          <Text color="#636E72">↑↓ navigate  ·  ↵ confirm  ·  [Esc/b] back without applying</Text>
        </Box>
      )}

      {uiPhase === 'running' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">Running the repair pipeline as a dry run. Your file is not being changed…</Text>
        </Box>
      )}

      {uiPhase === 'applying' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">Writing the validated patch…</Text>
        </Box>
      )}

      {uiPhase === 'done' && (
        <Box flexDirection="column" marginTop={1} gap={1}>
          <Text color={changed ? '#55EFC4' : '#FFD54F'}>{outcome || 'Finished.'}</Text>
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
