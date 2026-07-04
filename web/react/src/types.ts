export interface User {
  id: number
  username: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  steps?: AgentStep[]
}

export interface AgentStep {
  type: 'thought' | 'tool_call' | 'plan' | 'review' | 'thinking'
  tool_name?: string
  tool_args?: string
  observation?: string
  thought?: string
  duration_ms?: number
  status?: string
}

export interface Agent {
  id: string
  name: string
  description: string
  systemPrompt: string
}

export interface ToolExecution {
  task_id: number
  step_num: number
  tool_name: string
  args: Record<string, unknown>
  approved?: boolean
}

export type AgentMode = 'react' | 'normal'
export type FontSize = 'normal' | 'large'
