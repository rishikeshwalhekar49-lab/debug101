import React, { useEffect, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import z from 'zod';
import { resolveTarget, runPython, type FindingData, type SbaEvent } from '../utils/pythonBridge.js';
import { SeverityBadge } from '../components/Badge.js';

export const options = z.object({
  semantic: z.boolean().default(false).describe('Enable LLM semantic review'),
});

export const args = z.tuple([
  z.string().describe('Path to a Python file or directory'),
]);

type Props = {
  options?: z.infer<typeof options>;
  args?: z.infer<typeof args>;
  onBack?: () => void;
  isEmbedded?: boolean;
};

type Phase = 'idle' | 'static_analysis' | 'semantic_review' | 'done' | 'error';

export default function Scan({ options: opts = { semantic: false }, args: cmdArgs = [''], onBack, isEmbedded = false }: Props) {
  const shown = cmdArgs[0] ?? ''; // what the user typed, for display
  const target = resolveTarget(shown);
  const { exit } = useApp();

  const [phase, setPhase] = useState<Phase>('idle');
  const [findings, setFindings] = useState<FindingData[]>([]);
  const [diagnostics, setDiagnostics] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState('');
  const [summary, setSummary] = useState<Record<string, number>>({});

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
    if (!target) {
      setErrorMsg('Usage: sba-ui scan <path>');
      setPhase('error');
      return;
    }

    const pyArgs = ['scan', target];
    if (opts.semantic) pyArgs.push('--semantic');

    (async () => {
      setPhase('static_analysis');
      for await (const ev of runPython(pyArgs)) {
        handleEvent(ev);
      }
    })();

    function handleEvent(ev: SbaEvent) {
      if (ev.event === 'phase') {
        if (ev.status === 'start') setPhase(ev.phase as Phase);
        if (ev.status === 'done' && ev.phase === 'semantic_review') setPhase('done');
        if (ev.status === 'done' && ev.phase === 'static_analysis' && !opts.semantic) setPhase('done');
      }
      if (ev.event === 'findings_batch') {
        setFindings(ev.findings);
        setDiagnostics(ev.diagnostics);
      }
      if (ev.event === 'result') {
        const data = ev.data as { summary?: { by_severity?: Record<string, number> } };
        setSummary(data?.summary?.by_severity ?? {});
      }
      if (ev.event === 'error') {
        setErrorMsg(ev.message);
        setPhase('error');
      }
      if (ev.event === 'done') {
        setPhase('done');
      }
    }
  }, [target]);

  if (phase === 'error') {
    return (
      <Box flexDirection="column" padding={1} gap={1}>
        <Text color="#FF7675">✖ Error: {errorMsg}</Text>
        <Text color="#636E72">Press [Esc/b] to return to menu  ·  [q] to quit</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      <Box gap={2}>
        <Text bold color="#87CEEB">
          ⬡ Scan
        </Text>
        <Text color="#636E72">{shown}</Text>
      </Box>

      {phase !== 'done' && (
        <Box gap={2}>
          <Spinner type="dots" />
          <Text color="#B2BEC3">
            {phase === 'static_analysis'
              ? 'Running deterministic AST static analysis…'
              : 'Running LLM semantic review…'}
          </Text>
        </Box>
      )}

      {findings.length > 0 && (
        <Box flexDirection="column" gap={1}>
          {findings.map((f, i) => (
            <Box
              key={i}
              flexDirection="column"
              borderStyle="round"
              borderColor={
                f.severity === 'critical'
                  ? '#FF7675'
                  : f.severity === 'warning'
                  ? '#FDCB6E'
                  : '#74B9FF'
              }
              paddingX={1}
            >
              <Box gap={2}>
                <SeverityBadge severity={f.severity} />
                <Text bold color="#B2BEC3">
                  {f.bug_id}
                </Text>
                <Text color="#636E72">
                  {f.file}:{f.line}
                </Text>
              </Box>
              <Text color="#DFE6E9">{f.message}</Text>
              {f.evidence && (
                <Text color="#636E72">
                  &gt; {f.evidence.trim()}
                </Text>
              )}
              {f.suggested_fix && (
                <Text color="#55EFC4">fix: {f.suggested_fix}</Text>
              )}
            </Box>
          ))}
        </Box>
      )}

      {phase === 'done' && (
        <Box flexDirection="column" gap={1} marginTop={1}>
          <Box flexDirection="row" gap={3}>
            <Text bold color="#87CEEB">
              Summary:
            </Text>
            <Text color="#FF7675">critical: {summary['critical'] ?? 0}</Text>
            <Text color="#FDCB6E">warning: {summary['warning'] ?? 0}</Text>
            <Text color="#74B9FF">info: {summary['info'] ?? 0}</Text>
          </Box>
          <Text color="#636E72">
            Press [Esc/b] to return to menu  ·  [q] to quit
          </Text>
        </Box>
      )}

      {diagnostics.map((d, i) => (
        <Text key={i} color="#636E72">
          note: {d}
        </Text>
      ))}
    </Box>
  );
}
