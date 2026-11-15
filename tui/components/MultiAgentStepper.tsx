import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';

export type StepStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped';

export interface Step {
  id: string;
  label: string;
  status: StepStatus;
  detail?: string;
}

interface StepperProps {
  steps: Step[];
}

function StepIcon({ status }: { status: StepStatus }) {
  if (status === 'running') return <Spinner type="dots" />;
  if (status === 'done') return <Text color="#55EFC4">✔</Text>;
  if (status === 'error') return <Text color="#FF7675">✖</Text>;
  if (status === 'skipped') return <Text color="#636E72">–</Text>;
  return <Text color="#636E72">○</Text>;
}

export function MultiAgentStepper({ steps }: StepperProps) {
  return (
    <Box flexDirection="column">
      {steps.map((step) => (
        <Box key={step.id} flexDirection="row" gap={1}>
          <Box width={2}>
            <StepIcon status={step.status} />
          </Box>
          <Text
            color={
              step.status === 'done'
                ? '#55EFC4'
                : step.status === 'running'
                ? '#87CEEB'
                : step.status === 'error'
                ? '#FF7675'
                : step.status === 'skipped'
                ? '#636E72'
                : '#B2BEC3'
            }
            bold={step.status === 'running'}
          >
            {step.label}
          </Text>
          {step.detail && <Text color="#636E72"> {step.detail}</Text>}
        </Box>
      ))}
    </Box>
  );
}

export function buildFixSteps(): Step[] {
  return [
    { id: 'perception', label: 'Perception (static rules + runtime hooks + LLM review)', status: 'pending' },
    { id: 'diagnosis', label: 'Diagnosis (causal engine)', status: 'pending' },
    { id: 'repair', label: 'Fix synthesis (LLM patch, or a rule recipe)', status: 'pending' },
    { id: 'gate1', label: 'Gate 1: one-batch overfit check', status: 'pending' },
    { id: 'gate2', label: 'Gate 2: multi-seed statistical test', status: 'pending' },
    { id: 'memory', label: 'Experience memory', status: 'pending' },
  ];
}

