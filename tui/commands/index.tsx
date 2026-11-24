import React, { useState } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import Doctor from './doctor.js';
import Scan from './scan.js';
import Monitor from './monitor.js';
import Diagnose from './diagnose.js';
import Fix from './fix.js';
import Wsp from './wsp.js';

const COMMANDS = [
  { key: '1', label: 'fix', name: 'FIX', description: 'Find a bug, write a patch, test it, then ask before changing your file' },
  { key: '2', label: 'diagnose', name: 'DIAGNOSE', description: 'Rank the likely root causes of what the detectors found' },
  { key: '3', label: 'scan', name: 'SCAN', description: 'Static rules over the code, without running it' },
  { key: '4', label: 'snippet', name: 'SNIPPET', description: 'Check a code fragment saved in a file. Static only: nothing is run or validated' },
  { key: '5', label: 'monitor', name: 'MONITOR', description: 'Run the script under PyTorch hooks (loss, gradients, activations)' },
  { key: '6', label: 'wsp', name: 'WEIGHTS', description: 'Read-only check of the trained weights (NaN, dead biases, collapsed rank)' },
  { key: '7', label: 'doctor', name: 'DOCTOR', description: 'Check the installation, keys and models' },
];

// Bundled examples are never edited: for them, "Apply the fix" writes a fixed copy to fixed_examples/.
const SAMPLE_SCRIPTS = [
  { path: 'examples/sample_scripts/buggy_mnist.py', desc: 'Three planted bugs: softmax before the loss, no zero_grad, no input normalisation' },
  { path: 'examples/sample_scripts/benchmark_01_missing_zero_grad.py', desc: 'Benchmark 01: missing optimizer.zero_grad()' },
  { path: 'examples/sample_scripts/benchmark_02_train_eval_mode.py', desc: 'Benchmark 02: missing model.eval() before validation' },
  { path: 'examples/sample_scripts/benchmark_04_softmax_before_crossentropy.py', desc: 'Benchmark 04: softmax before CrossEntropyLoss' },
  { path: 'examples/sample_scripts/benchmark_06_dead_relu.py', desc: 'Benchmark 06: dead ReLU units from a bias initialisation' },
  { path: 'examples/sample_scripts/clean_mnist.py', desc: 'A healthy script, for comparison' },
  { path: 'examples/snippets/training_step.txt', desc: 'A pasted fragment, not a full script (for SNIPPET)' },
];

type ViewMode = 'menu' | 'target_select' | 'custom_input' | 'executing';

export default function Index() {
  const [selectedCmdIdx, setSelectedCmdIdx] = useState(0);
  const [selectedScriptIdx, setSelectedScriptIdx] = useState(0);
  const [mode, setMode] = useState<ViewMode>('menu');
  const [activeCommand, setActiveCommand] = useState<string>('fix');
  const [targetScript, setTargetScript] = useState<string>(SAMPLE_SCRIPTS[0]!.path);
  const [customPathInput, setCustomPathInput] = useState<string>('examples/sample_scripts/buggy_mnist.py');
  const { exit } = useApp();

  useInput((input, key) => {
    if (mode === 'menu') {
      if (key.upArrow) setSelectedCmdIdx((s) => Math.max(0, s - 1));
      if (key.downArrow) setSelectedCmdIdx((s) => Math.min(COMMANDS.length - 1, s + 1));
      if (key.return) {
        const cmd = COMMANDS[selectedCmdIdx]!;
        setActiveCommand(cmd.label);
        if (cmd.label === 'doctor') {
          setMode('executing');
        } else {
          setMode('target_select');
        }
      }
      if (input === 'q' || key.escape) exit();
    } else if (mode === 'target_select') {
      const totalOptions = SAMPLE_SCRIPTS.length + 1; // +1 for custom input
      if (key.upArrow) setSelectedScriptIdx((s) => Math.max(0, s - 1));
      if (key.downArrow) setSelectedScriptIdx((s) => Math.min(totalOptions - 1, s + 1));
      if (key.return) {
        if (selectedScriptIdx === SAMPLE_SCRIPTS.length) {
          // Custom path option
          setMode('custom_input');
        } else {
          setTargetScript(SAMPLE_SCRIPTS[selectedScriptIdx]!.path);
          setMode('executing');
        }
      }
      if (key.escape || input === 'b') {
        setMode('menu');
      }
      if (input === 'q') exit();
    } else if (mode === 'custom_input') {
      if (key.return) {
        setTargetScript(customPathInput.trim() || SAMPLE_SCRIPTS[0]!.path);
        setMode('executing');
      } else if (key.backspace || key.delete) {
        setCustomPathInput((prev) => prev.slice(0, -1));
      } else if (key.escape) {
        setMode('target_select');
      } else if (input && !key.ctrl && !key.meta) {
        setCustomPathInput((prev) => prev + input);
      }
    }
  });

  // When in executing mode, render the chosen component with an onBack callback
  if (mode === 'executing') {
    if (activeCommand === 'doctor') {
      return <Doctor options={{ live: false }} onBack={() => setMode('menu')} isEmbedded={true} />;
    }
    if (activeCommand === 'scan' || activeCommand === 'snippet') {
      return (
        <Scan
          options={{ semantic: false, snippet: activeCommand === 'snippet' }}
          args={[targetScript]}
          onBack={() => setMode('menu')}
          isEmbedded={true}
        />
      );
    }
    if (activeCommand === 'monitor') {
      return (
        <Monitor
          options={{ maxSteps: 300, maxSeconds: 900, stopOnCritical: false }}
          args={[targetScript]}
          onBack={() => setMode('menu')}
          isEmbedded={true}
        />
      );
    }
    if (activeCommand === 'diagnose') {
      return (
        <Diagnose
          options={{ noLlm: false, maxSteps: 150 }}
          args={[targetScript]}
          onBack={() => setMode('menu')}
          isEmbedded={true}
        />
      );
    }
    if (activeCommand === 'fix') {
      return (
        <Fix
          options={{ maxIter: 3, dryRun: false, noLlm: false, seeds: 5 }}
          args={[targetScript]}
          onBack={() => setMode('menu')}
          isEmbedded={true}
        />
      );
    }
    if (activeCommand === 'wsp') {
      return (
        <Wsp
          options={{ maxSteps: 60 }}
          args={[targetScript]}
          onBack={() => setMode('menu')}
          isEmbedded={true}
        />
      );
    }
  }

  return (
    <Box flexDirection="column" padding={1} gap={1}>
      {/* Prominent Header - DEEP LEARNING BUG DETECTOR (Bigger, No Logo) */}
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color="#87CEEB">
          {'  ____  _____ _____ ____    _     _____    _    ____  _   _ ___ _   _  ____ '}
        </Text>
        <Text bold color="#87CEEB">
          {' |  _ \\| ____| ____|  _ \\  | |   | ____|  / \\  |  _ \\| \\ | |_ _| \\ | |/ ___|'}
        </Text>
        <Text bold color="#7FFFD4">
          {' | | | |  _| |  _| | |_) | | |   |  _|   / _ \\ | |_) |  \\| || ||  \\| | |  _ '}
        </Text>
        <Text bold color="#7FFFD4">
          {' | |_| | |___| |___|  __/  | |___| |___ / ___ \\|  _ <| |\\  || || |\\  | |_| |'}
        </Text>
        <Text bold color="#B39DDB">
          {' |____/|_____|_____|_|     |_____|_____/_/   \\_\\_| \\_\\_| \\_|___|_| \\_|\\____|'}
        </Text>
        <Text bold color="#7FFFD4">
          {'  ____  _   _  ____   ____  _____ _____ _____ ____ _____ ___  ____  '}
        </Text>
        <Text bold color="#7FFFD4">
          {' | __ )| | | |/ ___| |  _ \\| ____|_   _| ____/ ___|_   _/ _ \\|  _ \\ '}
        </Text>
        <Text bold color="#B39DDB">
          {' |  _ \\| | | | |  _  | | | |  _|   | | |  _|| |     | || | | | |_) |'}
        </Text>
        <Text bold color="#B39DDB">
          {' | |_) | |_| | |_| | | |_| | |___  | | | |__| |___  | || |_| |  _ < '}
        </Text>
        <Text bold color="#FFD54F">
          {' |____/ \\___/ \\____| |____/|_____| |_| |_____\\____| |_| \\___/|_| \\_\\'}
        </Text>
        <Box marginTop={1}>
          <Text color="#B2BEC3">
            {'  Perception · Causal Diagnosis · Repair Synthesis · Empirical Validation'}
          </Text>
        </Box>
      </Box>

      {/* Main Menu View */}
      {mode === 'menu' && (
        <Box flexDirection="column" gap={1}>
          <Text color="#87CEEB" bold>
            Select an action:
          </Text>

          <Box flexDirection="column">
            {COMMANDS.map((cmd, i) => (
              <Box key={cmd.key} flexDirection="row" gap={2} paddingY={0}>
                <Text color={i === selectedCmdIdx ? '#FFD54F' : '#636E72'} bold>
                  {i === selectedCmdIdx ? '▶' : ' '}
                </Text>
                <Text bold color={i === selectedCmdIdx ? '#55EFC4' : '#B2BEC3'}>
                  {cmd.name.padEnd(10)}
                </Text>
                <Text color={i === selectedCmdIdx ? '#DFE6E9' : '#636E72'}>
                  {cmd.description}
                </Text>
              </Box>
            ))}
          </Box>

          <Box marginTop={1}>
            <Text color="#636E72">
              ↑↓ navigate  ·  ↵ select action  ·  q quit
            </Text>
          </Box>
        </Box>
      )}

      {/* Script Selection View */}
      {mode === 'target_select' && (
        <Box flexDirection="column" gap={1}>
          <Box gap={1}>
            <Text color="#FFD54F" bold>
              Target Script for:
            </Text>
            <Text color="#7FFFD4" bold>
              {activeCommand.toUpperCase()}
            </Text>
          </Box>

          <Text color="#B2BEC3">
            Choose a training script to inspect/run:
          </Text>

          <Box flexDirection="column">
            {SAMPLE_SCRIPTS.map((s, i) => (
              <Box key={s.path} flexDirection="row" gap={2}>
                <Text color={i === selectedScriptIdx ? '#FFD54F' : '#636E72'} bold>
                  {i === selectedScriptIdx ? '▶' : ' '}
                </Text>
                <Text bold color={i === selectedScriptIdx ? '#55EFC4' : '#B2BEC3'}>
                  {s.path.padEnd(65)}
                </Text>
                <Text color="#636E72">{s.desc}</Text>
              </Box>
            ))}
            <Box flexDirection="row" gap={2}>
              <Text color={selectedScriptIdx === SAMPLE_SCRIPTS.length ? '#FFD54F' : '#636E72'} bold>
                {selectedScriptIdx === SAMPLE_SCRIPTS.length ? '▶' : ' '}
              </Text>
              <Text bold color={selectedScriptIdx === SAMPLE_SCRIPTS.length ? '#55EFC4' : '#B2BEC3'}>
                {'Custom file path... (enter your own script path)'}
              </Text>
            </Box>
          </Box>

          <Box marginTop={1}>
            <Text color="#636E72">
              ↑↓ navigate  ·  ↵ run  ·  [Esc/b] back to menu  ·  q quit
            </Text>
          </Box>
        </Box>
      )}

      {/* Custom Path Text Input View */}
      {mode === 'custom_input' && (
        <Box flexDirection="column" gap={1} borderStyle="round" borderColor="#87CEEB" padding={1}>
          <Text bold color="#87CEEB">
            Enter path to script or directory:
          </Text>
          <Box flexDirection="row" gap={1}>
            <Text color="#7FFFD4" bold>&gt;</Text>
            <Text color="#FFFFFF" bold>{customPathInput}</Text>
            <Text color="#7FFFD4">█</Text>
          </Box>
          <Text color="#636E72">
            Type path  ·  ↵ execute  ·  [Esc] back to selection
          </Text>
        </Box>
      )}
    </Box>
  );
}
