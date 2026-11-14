import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

export type SbaEvent =
  | { event: 'phase'; phase: string; status: 'start' | 'done' | 'error'; [key: string]: unknown }
  | { event: 'finding'; data: FindingData }
  | { event: 'findings_batch'; findings: FindingData[]; diagnostics: string[] }
  | { event: 'hypothesis'; rank: number; bug_id: string; confidence: number; description: string; diagnosed_by?: string }
  | { event: 'patch'; bug_id: string; strategy: string; source: string; diff: string; confidence: number }
  | { event: 'verdict'; bug_id: string; accepted: boolean; validator: string; p_value?: number; effect_size?: number; notes: string }
  | { event: 'result'; data: unknown }
  | { event: 'runtime_alert'; severity: string; bug_id: string; file: string; line: number; message: string }
  | { event: 'doctor_check'; ok: boolean; label: string; advice: string }
  | { event: 'wsp_proposal'; title?: string; origin?: string; code_diff?: string; weight_patches?: unknown[]; iteration?: number }
  | { event: 'error'; message: string; detail: string }
  | { event: 'done'; exit_code: number };

export interface FindingData {
  bug_id: string;
  severity: 'critical' | 'warning' | 'info';
  file: string;
  line: number;
  message: string;
  evidence?: string;
  suggested_fix?: string;
  detector?: string;
  confidence?: number;
}

function findProjectRoot(): string {
  const __filename = fileURLToPath(import.meta.url);
  let cur = path.dirname(__filename);
  for (let i = 0; i < 6; i++) {
    if (
      fs.existsSync(path.join(cur, 'silent_bug_agent')) &&
      fs.existsSync(path.join(cur, 'pyproject.toml'))
    ) {
      return cur;
    }
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return path.resolve(path.dirname(__filename), '..', '..');
}

function resolvePython(projectRoot: string): string {
  const candidates = [
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : null,
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'bin', 'python') : null,
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, '.venv', 'bin', 'python'),
    'python',
    'python3',
  ].filter(Boolean) as string[];

  for (const c of candidates) {
    if (path.isAbsolute(c) && fs.existsSync(c)) {
      return c;
    }
  }
  return 'python';
}

export async function* runPython(command: string[]): AsyncGenerator<SbaEvent> {
  const projectRoot = findProjectRoot();
  const python = resolvePython(projectRoot);

  const args = ['-m', 'silent_bug_agent', ...command, '--events'];

  const proc = spawn(python, args, {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  let stderr = '';
  proc.stderr?.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  const rl = createInterface({ input: proc.stdout! });
  let receivedAnyEvent = false;

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const event = JSON.parse(trimmed) as SbaEvent;
      receivedAnyEvent = true;
      yield event;
    } catch {
      // Non-JSON line from third-party libraries; ignore
    }
  }

  const exitCode = await new Promise<number>((resolve) => proc.on('close', (code) => resolve(code ?? 0)));

  if (!receivedAnyEvent && exitCode !== 0) {
    yield {
      event: 'error',
      message: stderr.trim() || `Process exited with code ${exitCode}`,
      detail: stderr.trim(),
    };
    yield {
      event: 'done',
      exit_code: exitCode,
    };
  }
}
