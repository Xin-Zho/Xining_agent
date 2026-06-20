import api, { API_BASE } from './client';
import storage from '../utils/storage';

export interface AgentTaskSummary {
  id: number;
  title: string;
  status: string;
  total_tokens: number;
  duration_ms: number;
  created_at: string;
}

export interface AgentStep {
  step_number: number;
  status: string;
  step_type: string;
  tool_name: string | null;
  tool_args: string | null;
  tool_result: string | null;
  thought: string | null;
  duration_ms: number | null;
}

export interface AgentTaskDetail {
  id: number;
  title: string;
  description: string;
  status: string;
  plan_json: string | null;
  final_answer: string | null;
  conversation_id: number | null;
  total_tokens: number;
  duration_ms: number;
  created_at: string;
  steps: AgentStep[];
}

export async function createAgentTask(description: string, conversationId?: number) {
  const { data } = await api.post('/api/agent/tasks', {
    description,
    conversation_id: conversationId ?? null,
  });
  return data as { task_id: number; status: string };
}

export async function listAgentTasks(): Promise<AgentTaskSummary[]> {
  const { data } = await api.get('/api/agent/tasks');
  return data;
}

export async function getAgentTask(id: number): Promise<AgentTaskDetail> {
  const { data } = await api.get(`/api/agent/tasks/${id}`);
  return data;
}

export async function deleteAgentTask(id: number) {
  await api.delete(`/api/agent/tasks/${id}`);
}

export interface ClaudeConfigItem {
  value: string | string[];
  source: 'env' | 'file' | 'default';
}

export interface ClaudeConfig {
  model: ClaudeConfigItem;
  allowed_tools: ClaudeConfigItem;
  permission_mode: ClaudeConfigItem;
  append_system_prompt: ClaudeConfigItem;
  extra_dirs: ClaudeConfigItem;
}

export async function getClaudeConfig(): Promise<ClaudeConfig> {
  const { data } = await api.get('/api/claude-config');
  return data;
}

export async function updateClaudeConfig(updates: Partial<{
  model: string;
  allowed_tools: string;
  permission_mode: string;
  append_system_prompt: string;
  extra_dirs: string[];
}>): Promise<ClaudeConfig> {
  const { data } = await api.put('/api/claude-config', updates);
  return data;
}

export async function createAgentSocket(taskId: number): Promise<WebSocket> {
  const wsBase = API_BASE.replace(/^http/, 'ws');
  const token = await storage.getItem('jwt');
  return new WebSocket(`${wsBase}/ws/agent/${taskId}?token=${token ?? ''}`);
}
