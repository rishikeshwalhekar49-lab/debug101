import React, { useEffect, useRef, useState } from 'react';
import path from 'node:path';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { resolveTarget, runPython, stopRunning, type SbaEvent } from '../utils/pythonBridge.js';
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
  diff: string;
  pending: boolean;
  copyTo?: string | null; // set for a bundled example: Apply writes a fixed copy here instead
}

/** A path as short as it can be shown: relative to where the user is, unless that climbs out. */
function short(file: string): string {
  const rel = path.relative(process.cwd(), file);
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? rel : file;
}

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
  const [note, setNote] = useState(''); // why this patch differs from the one that was turned down
  const [written, setWritten] = useState(''); // '', 'file' or 'copy'
  const [attempt, setAttempt] = useState(1);

  function go(next: UIPhase) {
    uiPhaseRef.current = next;
    setUiPhase(next);
  }

  function updateStep(id: string, status: Step['status'], detail?: string) {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, status, detail } : s)));
  }

  // The pipeline always runs as a dry run, so the script is untouched when the menu appears.
  // `another` turns down the patch validated last time and asks the backend for a different one.
  async function runPipeline(another: boolean) {
    setSteps(buildFixSteps());
    setVerdicts([]);
    setAccepted(null);
    setOutcome('');
    setNote('');
    setApprovalIdx(0);
    go('running');

    const pyArgs = ['fix', script, '--dry-run', '--max-iter', String(opts.maxIter), '--seeds', String(opts.seeds)];
    if (opts.noLlm) pyArgs.push('--no-llm');
    if (another) pyArgs.push('--another');

    let acceptedPatch: AcceptedPatch | null = null;
    let anotherNote = '';
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
        acceptedPatch = { bug_id: ev.bug_id, diff: ev.diff, pending: ev.pending, copyTo: ev.copy_to };
        setAccepted(acceptedPatch);
      }
      if (ev.event === 'result') {
        const runs = (ev.data as { runs?: Array<{ another_note?: string }> })?.runs ?? [];
        anotherNote = runs[runs.length - 1]?.another_note ?? '';
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
            anotherNote ||
              (ev.exit_code === 0
                ? 'No bug that needs a repair was found. The file is unchanged.'
                : 'No patch passed validation, so nothing was changed.')
          );
          go('done');
        } else if (opts.dryRun || !acceptedPatch.pending) {
          setOutcome(
            'Dry run: a patch passed validation and the file is unchanged. Write it with: ' +
              `python -m silent_bug_agent apply ${script}`
          );
          go('done');
        } else {
          setNote(anotherNote);
          go('approval');
        }
      }
    }

    for await (const ev of runPython(pyArgs)) {
      handleEvent(ev);
    }
  }

  async function applyPatch() {
    go('applying');
    let failed = '';
    let target = '';
    let backup = '';
    let isCopy = false;
    for await (const ev of runPython(['apply', script])) {
      if (ev.event === 'applied') {
        target = ev.written;
        backup = ev.backup ?? '';
        isCopy = ev.copy;
      }
      if (ev.event === 'error') failed = ev.message;
    }
    if (!target) {
      setErrorMsg(`The fix was not applied, and the file is unchanged. ${failed}`.trim());
      go('error');
      return;
    }
    setWritten(isCopy ? 'copy' : 'file');
    setOutcome(
      isCopy
        ? `Fix applied. The fixed version was written to ${short(target)}. The bundled example itself was left ` +
            'as it is, because the tests rely on its planted bug. To look for further bugs, run Fix on that copy ' +
            '(choose "Custom file path").'
        : `Fix applied to ${shown}.` +
            (backup ? ` The original is saved beside it as ${path.basename(backup)}.` : '') +
            ' Run Fix again to look for further bugs.'
    );
    go('done');
  }

  const choices = [
    accepted?.copyTo
      ? `Apply the fix  (writes the fixed version to ${short(accepted.copyTo)})`
      : 'Apply the fix to the file',
    'Do not apply  (leave the file as it is)',
    'Try another fix',
  ];

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
      if (key.downArrow) setApprovalIdx((i) => Math.min(choices.length - 1, i + 1));
      if (key.return) {
        if (approvalIdx === 0) {
          void applyPatch();
        } else if (approvalIdx === 1) {
          setOutcome(
            'Not applied. The file is unchanged. The validated patch is kept, so you can still write it later ' +
              `with: python -m silent_bug_agent apply ${script}`
          );
          go('done');
        } else {
          setAttempt((n) => n + 1);
          void runPipeline(true);
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
    void runPipeline(false);
  }, [script]);

  // Leaving the screen stops a dry run in progress. It writes nothing, so stopping it is safe;
  // keys are ignored while a patch is being written, so that step is never cut short.
  useEffect(() => () => stopRunning(), []);

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#FFD54F">
          ⬡ Fix Loop
        </Text>
        <Text color="#636E72">{shown}</Text>
        {opts.dryRun && <Badge label="DRY RUN" variant="muted" />}
        {attempt > 1 && <Badge label={`FIX ${attempt}`} variant="info" />}
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
            <Badge
              label={written === 'file' ? 'WRITTEN TO FILE' : written === 'copy' ? 'FIXED COPY WRITTEN' : 'FILE NOT CHANGED'}
              variant={written ? 'success' : 'muted'}
            />
          </Box>
          {note !== '' && <Text color="#FFD54F">{note}</Text>}
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
            This patch passed both gates. Nothing has been written yet. What would you like to do?
          </Text>
          {choices.map((choice, i) => (
            <Box key={i} gap={2}>
              <Text color={i === approvalIdx ? '#FFD54F' : '#636E72'}>
                {i === approvalIdx ? '▶' : ' '}
              </Text>
              <Text color={i === approvalIdx ? '#55EFC4' : '#B2BEC3'} bold={i === approvalIdx}>
                {choice}
              </Text>
            </Box>
          ))}
          {accepted?.copyTo && (
            <Text color="#636E72">
              This is a bundled example. The tests rely on its planted bug, so the fix goes to a copy.
            </Text>
          )}
          <Text color="#636E72">↑↓ navigate  ·  ↵ confirm  ·  [Esc/b] back without applying</Text>
        </Box>
      )}

      {uiPhase === 'running' && (
        <Box gap={2} marginTop={1}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">
            {attempt > 1
              ? 'Looking for a different fix. Your file is not being changed…'
              : 'Running the repair pipeline as a dry run. Your file is not being changed…'}
          </Text>
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
          <Text color={written ? '#55EFC4' : '#FFD54F'}>{outcome || 'Finished.'}</Text>
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
