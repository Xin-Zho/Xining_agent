import React, { useEffect, useState, useRef } from 'react';
import { View, Text, FlatList, TouchableOpacity, StyleSheet, ActivityIndicator, TextInput } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { useAgentTask, StepListItem, TaskStatus } from '../../src/hooks/useAgentTask';
import { getAgentTask, AgentTaskDetail } from '../../src/api/agent';
import AgentStepCard from '../../src/components/AgentStepCard';
import AgentProgressBar from '../../src/components/AgentProgressBar';
import MarkdownRenderer from '../../src/components/MarkdownRenderer';
import FeedbackDigestBanner, { DigestEntry } from '../../src/components/FeedbackDigestBanner';
import ReviewBanner, { ReviewEntry } from '../../src/components/ReviewBanner';
import { statusLabel, statusColor } from '../../src/utils/format';

function parseJsonField(value: string | null) {
  if (!value) return undefined;

  try {
    return JSON.parse(value) as Record<string, unknown>;
  } catch {
    return { raw: value };
  }
}

export default function AgentTaskScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const taskId = Number(id);
  const [taskDetail, setTaskDetail] = useState<AgentTaskDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [feedbackInput, setFeedbackInput] = useState('');

  useEffect(() => {
    if (!Number.isFinite(taskId) || taskId <= 0) {
      setLoadError('无效的任务 ID');
      setLoadingDetail(false);
      return;
    }

    (async () => {
      try {
        setLoadError(null);
        const detail = await getAgentTask(taskId);
        setTaskDetail(detail);
      } catch (e: any) {
        const message = e?.response?.data?.detail || e?.message || '加载任务详情失败';
        setLoadError(message);
      } finally {
        setLoadingDetail(false);
      }
    })();
  }, [taskId]);

  const {
    status: connStatus,
    taskStatus,
    steps,
    finalAnswer,
    error,
    totalTokens,
    durationMs,
    cancel,
    intervene,
    inputBarState,
  } = useAgentTask(
    taskDetail?.status === 'completed'
    || taskDetail?.status === 'failed'
    || taskDetail?.status === 'cancelled'
      ? null
      : taskId
  );

  const isRunning = connStatus === 'connected' || connStatus === 'connecting';
  const isDone = connStatus === 'completed' || connStatus === 'failed' || connStatus === 'cancelled'
    || taskDetail?.status === 'completed' || taskDetail?.status === 'failed' || taskDetail?.status === 'cancelled';
  const hasLiveTaskState = connStatus !== 'disconnected' || steps.length > 0 || !!finalAnswer || !!error;

  if (loadingDetail) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#4A90D9" />
      </View>
    );
  }

  if (loadError) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorTitle}>加载失败</Text>
        <Text style={styles.errorMsg}>{loadError}</Text>
      </View>
    );
  }

  const displayStatus = hasLiveTaskState ? taskStatus : taskDetail?.status || 'pending';

  // 构建 displaySteps：优先 ws 数据，回退 API
  let displaySteps: StepListItem[] = [];
  if (steps.length > 0) {
    displaySteps = steps;
  } else if (taskDetail?.steps) {
    displaySteps = taskDetail.steps.map((s) => ({
      kind: 'step' as const,
      data: {
        step_num: s.step_number,
        type: s.step_type,
        status: s.status as 'pending' | 'running' | 'completed' | 'failed' | 'skipped',
        tool_name: s.tool_name || undefined,
        args: parseJsonField(s.tool_args),
        result: parseJsonField(s.tool_result),
        content: s.thought || undefined,
        duration_ms: s.duration_ms || undefined,
      },
    }));
  }

  const displayAnswer = finalAnswer || taskDetail?.final_answer;
  const completedCount = displaySteps.filter(
    (s) => s.kind === 'step' && (s.data.status === 'completed' || s.data.status === 'failed' || s.data.status === 'skipped')
  ).length;
  const totalCount = displaySteps.length;

  const handleSendFeedback = () => {
    const content = feedbackInput.trim();
    if (!content) return;
    const fbId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    intervene(fbId, content, 'normal');
    setFeedbackInput('');
  };

  return (
    <View style={styles.container}>
      {/* Status header */}
      <View style={[styles.statusBar, { backgroundColor: statusColor(displayStatus) }]}>
        <Text style={styles.statusText}>
          {statusLabel(displayStatus)}
          {isRunning && <Text> ...</Text>}
        </Text>
        {taskDetail?.title && (
          <Text style={styles.taskTitle} numberOfLines={1}>{taskDetail.title}</Text>
        )}
      </View>

      {/* Progress */}
      {totalCount > 0 && <AgentProgressBar completed={completedCount} total={totalCount} />}

      {/* Steps list */}
      <FlatList
        data={displaySteps}
        keyExtractor={(item) =>
          item.kind === 'feedback_banner' ? `digest-${item.data.id}`
          : item.kind === 'review_banner' ? `review-${(item.data as any).step_num || Date.now()}`
          : `step-${item.data.step_num}`
        }
        renderItem={({ item }) => {
          if (item.kind === 'feedback_banner') {
            return <FeedbackDigestBanner entry={item.data as DigestEntry} />;
          }
          if (item.kind === 'review_banner') {
            return <ReviewBanner entry={item.data as ReviewEntry} />;
          }
          return <AgentStepCard step={item.data} />;
        }}
        style={styles.list}
        ListEmptyComponent={isRunning ? (
          <View style={styles.center}>
            <ActivityIndicator size="large" color="#4A90D9" />
            <Text style={styles.waiting}>Agent 正在规划任务...</Text>
          </View>
        ) : null}
      />

      {/* Intervention input bar */}
      {isRunning && (
        <View style={styles.interventionBar}>
          <TextInput
            style={[
              styles.interventionInput,
              inputBarState !== 'idle' && inputBarState !== 'injected' && styles.interventionInputDisabled,
            ]}
            value={feedbackInput}
            onChangeText={setFeedbackInput}
            placeholder={
              inputBarState === 'waiting_ack' ? '正在发送...'
              : inputBarState === 'waiting_injection' ? 'Agent 下次思考时会处理'
              : '告诉 Agent 调整方向...'
            }
            placeholderTextColor={inputBarState === 'idle' || inputBarState === 'injected' ? '#999' : '#ccc'}
            editable={inputBarState === 'idle' || inputBarState === 'injected'}
            multiline={false}
            returnKeyType="send"
            onSubmitEditing={handleSendFeedback}
          />
          <TouchableOpacity
            style={[
              styles.interventionSendBtn,
              (inputBarState !== 'idle' && inputBarState !== 'injected') && styles.sendBtnDisabled,
            ]}
            onPress={handleSendFeedback}
            disabled={inputBarState !== 'idle' && inputBarState !== 'injected'}
          >
            <Text style={styles.sendBtnText}>发送</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Final answer */}
      {displayAnswer && (
        <View style={styles.answerSection}>
          <Text style={styles.answerTitle}>最终结果</Text>
          <MarkdownRenderer content={displayAnswer} />
          <Text style={styles.stats}>
            Token: {totalTokens || taskDetail?.total_tokens || 0} | 耗时: {durationMs || taskDetail?.duration_ms || 0}ms
          </Text>
        </View>
      )}

      {/* Error */}
      {error && (
        <View style={styles.errorSection}>
          <Text style={styles.errorTitle}>执行失败</Text>
          <Text style={styles.errorMsg}>{error}</Text>
        </View>
      )}

      {/* Cancel button */}
      {isRunning && (
        <TouchableOpacity style={styles.cancelBtn} onPress={cancel}>
          <Text style={styles.cancelText}>取消任务</Text>
        </TouchableOpacity>
      )}

      {/* ID display for WebSocket-less tasks */}
      {isDone && !isRunning && (
        <Text style={styles.doneHint}>
          当前状态：{statusLabel(displayStatus)}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  statusBar: { padding: 16, alignItems: 'center' },
  statusText: { color: '#fff', fontSize: 18, fontWeight: '700' },
  taskTitle: { color: '#fff', fontSize: 13, marginTop: 4, opacity: 0.9 },
  list: { flex: 1 },
  waiting: { color: '#888', marginTop: 12, fontSize: 14 },
  answerSection: { backgroundColor: '#fff', margin: 16, borderRadius: 10, padding: 16, maxHeight: 300 },
  answerTitle: { fontSize: 16, fontWeight: '700', color: '#333', marginBottom: 8 },
  stats: { fontSize: 11, color: '#aaa', marginTop: 8, textAlign: 'center' },
  errorSection: { backgroundColor: '#FFF3F3', margin: 16, borderRadius: 10, padding: 16 },
  errorTitle: { fontSize: 16, fontWeight: '700', color: '#F44336', marginBottom: 4 },
  errorMsg: { fontSize: 14, color: '#D32F2F' },
  cancelBtn: { backgroundColor: '#F44336', margin: 16, borderRadius: 10, padding: 14, alignItems: 'center' },
  cancelText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  doneHint: { textAlign: 'center', color: '#999', fontSize: 13, padding: 16 },

  // Intervention bar styles
  interventionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  interventionInput: {
    flex: 1,
    height: 40,
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 16,
    fontSize: 14,
    color: '#333',
  },
  interventionInputDisabled: {
    backgroundColor: '#efefef',
    color: '#aaa',
  },
  interventionSendBtn: {
    marginLeft: 10,
    backgroundColor: '#4A90D9',
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 8,
  },
  sendBtnDisabled: {
    backgroundColor: '#ccc',
  },
  sendBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
