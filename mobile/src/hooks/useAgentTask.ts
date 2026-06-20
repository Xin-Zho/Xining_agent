import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE } from '../api/client';
import storage from '../utils/storage';

export interface AgentStep {
  step_num: number;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  tool_name?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  content?: string;
  message?: string;
  error?: string;
  reason?: string;
  duration_ms?: number;
}

export type TaskStatus = 'connecting' | 'connected' | 'disconnected' | 'completed' | 'failed' | 'cancelled' | 'cancelling';

// ── 干预系统类型 ──────────────────────────────────────────

export type FeedbackStatus = 'sending' | 'acknowledged' | 'rejected' | 'timeout' | 'applied' | 'digested';

export interface PendingFeedback {
  id: string;
  content: string;
  priority: string;
  status: FeedbackStatus;
  mergedInto?: string;
}

export interface DigestEntry {
  id: string;
  decision: 'adopted' | 'rejected' | 'partial' | 'unknown';
  summary: string;
  userContent?: string;
}

export type StepListItem =
  | { kind: 'step'; data: AgentStep }
  | { kind: 'feedback_banner'; data: DigestEntry }
  | { kind: 'review_banner'; data: Record<string, unknown> };

interface UseAgentTaskReturn {
  status: TaskStatus;
  taskStatus: string;
  steps: StepListItem[];
  finalAnswer: string | null;
  error: string | null;
  totalSteps: number;
  totalTokens: number;
  durationMs: number;
  cancel: () => void;
  intervene: (id: string, content: string, priority?: string) => void;
  retract: (feedbackId: string) => void;
  pendingFeedbacks: Map<string, PendingFeedback>;
  agentDigestions: DigestEntry[];
  inputBarState: 'idle' | 'waiting_ack' | 'waiting_injection' | 'injected';
}

export function useAgentTask(taskId: number | null): UseAgentTaskReturn {
  const [connStatus, setConnStatus] = useState<TaskStatus>('disconnected');
  const [taskStatus, setTaskStatus] = useState<string>('pending');
  const [steps, setSteps] = useState<StepListItem[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [totalSteps, setTotalSteps] = useState(0);
  const [totalTokens, setTotalTokens] = useState(0);
  const [durationMs, setDurationMs] = useState(0);

  // 干预系统状态
  const [pendingFeedbacks, setPendingFeedbacks] = useState<Map<string, PendingFeedback>>(new Map());
  const [agentDigestions, setAgentDigestions] = useState<DigestEntry[]>([]);
  const [inputBarState, setInputBarState] = useState<'idle' | 'waiting_ack' | 'waiting_injection' | 'injected'>('idle');

  const wsRef = useRef<WebSocket | null>(null);
  const terminalStatusRef = useRef<TaskStatus | null>(null);
  const ackTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const pendingFeedbacksRef = useRef(pendingFeedbacks);
  pendingFeedbacksRef.current = pendingFeedbacks;

  const setConnectionStatus = useCallback((status: TaskStatus) => {
    if (status === 'completed' || status === 'failed' || status === 'cancelled') {
      terminalStatusRef.current = status;
    } else if (status === 'connecting' || status === 'connected') {
      terminalStatusRef.current = null;
    }
    setConnStatus(status);
  }, []);

  const intervene = useCallback((id: string, content: string, priority: string = 'normal') => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    setPendingFeedbacks((prev) => {
      const next = new Map(prev);
      next.set(id, { id, content, priority, status: 'sending' });
      return next;
    });
    setInputBarState('waiting_ack');

    wsRef.current.send(JSON.stringify({
      type: 'intervention',
      id,
      content,
      priority,
    }));

    const timer = setTimeout(() => {
      setPendingFeedbacks((prev) => {
        const next = new Map(prev);
        const fb = next.get(id);
        if (fb && fb.status === 'sending') {
          next.set(id, { ...fb, status: 'timeout' });
        }
        return next;
      });
      setInputBarState('idle');
      ackTimersRef.current.delete(id);
    }, 5000);
    ackTimersRef.current.set(id, timer);
  }, []);

  const retract = useCallback((feedbackId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    setPendingFeedbacks((prev) => {
      const next = new Map(prev);
      const fb = next.get(feedbackId);
      if (fb && (fb.status === 'sending' || fb.status === 'acknowledged')) {
        next.set(feedbackId, { ...fb, status: 'digested' });
      }
      return next;
    });

    wsRef.current.send(JSON.stringify({
      type: 'retract',
      feedback_id: feedbackId,
    }));
  }, []);

  const connect = useCallback(async () => {
    if (!taskId) return;

    terminalStatusRef.current = null;
    setTaskStatus('pending');
    setSteps([]);
    setFinalAnswer(null);
    setError(null);
    setTotalSteps(0);
    setTotalTokens(0);
    setDurationMs(0);
    setPendingFeedbacks(new Map());
    setAgentDigestions([]);
    setInputBarState('idle');
    // 清除旧连接的定时器
    ackTimersRef.current.forEach((timer) => clearTimeout(timer));
    ackTimersRef.current.clear();
    setConnectionStatus('connecting');

    const token = await storage.getItem('jwt');
    const wsBase = API_BASE.replace(/^http/, 'ws');
    const ws = new WebSocket(`${wsBase}/ws/agent/${taskId}`);

    ws.onopen = () => {
      ws.send(JSON.stringify({ token: token ?? '', action: 'auth' }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'auth_ok') {
          setConnectionStatus('connected');
          return;
        }

        switch (msg.type) {
          case 'task_started':
            setTaskStatus('executing');
            break;
          case 'task_complete':
            setFinalAnswer(msg.final_answer);
            setTotalSteps(msg.total_steps);
            setTotalTokens(msg.total_tokens);
            setDurationMs(msg.duration_ms);
            setTaskStatus('completed');
            setConnectionStatus('completed');
            ws.close();
            break;
          case 'task_error':
            setError(msg.error);
            setTaskStatus('failed');
            setConnectionStatus('failed');
            ws.close();
            break;
          case 'task_cancelled':
            setTaskStatus('cancelled');
            setConnectionStatus('cancelled');
            ws.close();
            break;
          case 'step_start':
            setSteps((prev) => [
              ...prev,
              {
                kind: 'step',
                data: {
                  step_num: msg.step_num,
                  type: msg.type || 'tool_call',
                  status: 'running',
                  tool_name: msg.tool_name,
                  args: msg.args,
                  message: msg.message,
                },
              },
            ]);
            break;
          case 'step_complete':
            setSteps((prev) =>
              prev.map((s) => {
                if (s.kind === 'step' && s.data.step_num === msg.step_num) {
                  if (msg.type === 'review' && msg.review) {
                    return {
                      kind: 'review_banner' as const,
                      data: msg.review,
                    };
                  }
                  return {
                    kind: 'step' as const,
                    data: {
                      ...s.data,
                      status: 'completed' as const,
                      result: msg.result,
                      content: msg.content,
                      duration_ms: msg.duration_ms,
                    },
                  };
                }
                return s;
              })
            );
            break;
          case 'step_error':
            setSteps((prev) =>
              prev.map((s) =>
                s.kind === 'step' && s.data.step_num === msg.step_num
                  ? { kind: 'step' as const, data: { ...s.data, status: 'failed' as const, error: msg.error } }
                  : s
              )
            );
            break;
          case 'step_skipped':
            setSteps((prev) =>
              prev.map((s) =>
                s.kind === 'step' && s.data.step_num === msg.step_num
                  ? { kind: 'step' as const, data: { ...s.data, status: 'skipped' as const, reason: msg.reason } }
                  : s
              )
            );
            break;
          case 'confirmation_required':
            ws.send(JSON.stringify({ action: 'confirm', step_num: msg.step_num, approved: true }));
            break;

          // ── 干预系统消息 ──────────────────────────
          case 'intervention_ack':
            setPendingFeedbacks((prev) => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'acknowledged' });
              }
              return next;
            });
            setInputBarState('waiting_injection');
            const ackTimer = ackTimersRef.current.get(msg.id);
            if (ackTimer) { clearTimeout(ackTimer); ackTimersRef.current.delete(msg.id); }
            break;

          case 'intervention_rejected':
            setPendingFeedbacks((prev) => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'rejected' });
              }
              return next;
            });
            setInputBarState('idle');
            break;

          case 'intervention_applied':
            if (msg.merged_into) {
              setPendingFeedbacks((prev) => {
                const next = new Map(prev);
                const fb = next.get(msg.id);
                if (fb) {
                  next.set(msg.id, { ...fb, status: 'digested', mergedInto: msg.merged_into });
                }
                return next;
              });
            } else {
              setPendingFeedbacks((prev) => {
                const next = new Map(prev);
                const fb = next.get(msg.id);
                if (fb) {
                  next.set(msg.id, { ...fb, status: 'applied' });
                }
                return next;
              });
              setInputBarState('injected');
            }
            break;

          case 'feedback_digested':
            setPendingFeedbacks((prev) => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'digested' });
              }
              return next;
            });
            const entry: DigestEntry = {
              id: msg.id,
              decision: msg.decision as DigestEntry['decision'],
              summary: msg.summary,
              userContent: pendingFeedbacksRef.current.get(msg.id)?.content,
            };
            setAgentDigestions((prev) => [...prev, entry]);
            setSteps((prev) => [
              ...prev,
              { kind: 'feedback_banner', data: entry },
            ]);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      if (!terminalStatusRef.current) {
        setConnectionStatus('disconnected');
      }
    };
    ws.onclose = () => {
      if (!terminalStatusRef.current) {
        setConnectionStatus('disconnected');
      }
    };

    wsRef.current = ws;
  }, [setConnectionStatus, taskId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const cancel = useCallback(() => {
    setConnStatus('cancelling');
    wsRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  return {
    status: connStatus,
    taskStatus,
    steps,
    finalAnswer,
    error,
    totalSteps,
    totalTokens,
    durationMs,
    cancel,
    intervene,
    retract,
    pendingFeedbacks,
    agentDigestions,
    inputBarState,
  };
}
