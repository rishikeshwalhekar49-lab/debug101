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
  | {
      event: 'accepted_patch';
      bug_id: string;
      strategy: string;
      source: string;
      diff: string;
      confidence: number;
      applied: boolean;
      backup?: string | null;
      pending: boolean;
      blocked?: string | null;
    }
  | { event: 'applied'; target: string; backup?: string | null; bug_id: string }
  | { event: 'weight_report'; data: WeightReportData }
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

export interface WeightFindingData {
  layer_name: string;
  param_shape: number[];
  anomaly_type: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  suggested_action?: string;
}

export interface WeightReportData {
  model_name: string;
  source: string;
  inspected: boolean;
  total_parameters: number;
  findings: WeightFindingData[];
  diagnostics: string[];
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
    // set by `python -m silent_bug_agent ui`: the interpreter that has the package installed
    process.env.SBA_PYTHON ?? null,
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : null,
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'bin', 'python') : null,
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, '.venv', 'bin', 'python'),
  ].filter(Boolean) as string[];

  for (const c of candidates) {
    if (path.isAbsolute(c) && fs.existsSync(c)) {
      return c;
    }
  }
  // macOS and most Linux systems have no bare `python` command
  return process.platform === 'win32' ? 'python' : 'python3';
}

/**
 * Turn a script path into an absolute one. A relative path means the folder the user
 * launched from; only if nothing is there is it tried against the project, which is
 * how the bundled sample paths in the menu resolve.
 */
export function resolveTarget(target: string): string {
  if (!target || path.isAbsolute(target)) return target;
  const fromCwd = path.resolve(process.cwd(), target);
  if (fs.existsSync(fromCwd)) return fromCwd;
  const fromProject = path.resolve(findProjectRoot(), target);
  return fs.existsSync(fromProject) ? fromProject : fromCwd;
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

  // Registered before reading, so an early exit or a failed launch is never missed.
  const finished = new Promise<number>((resolve) => {
    proc.on('close', (code) => resolve(code ?? 0));
    proc.on('error', (err) => {
      stderr += `Could not start Python (${python}): ${err.message}`;
      resolve(127);
    });
  });

  const rl = createInterface({ input: proc.stdout! });
  let sawDone = false;

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue; // the training script's own prints share stdout
    try {
      const event = JSON.parse(trimmed) as SbaEvent;
      if (typeof (event as { event?: unknown }).event !== 'string') continue;
      if (event.event === 'done') sawDone = true;
      yield event;
    } catch {
      // not an event line
    }
  }

  const exitCode = await finished;

  // A crash can end the stream with no `done`. Without this the screen would spin forever.
  if (!sawDone) {
    if (exitCode !== 0) {
      const tail = stderr.trim().split('\n').slice(-6).join('\n');
      yield {
        event: 'error',
        message: tail || `The Python process exited with code ${exitCode}`,
        detail: stderr.trim(),
      };
    }
    yield { event: 'done', exit_code: exitCode };
  }
}
