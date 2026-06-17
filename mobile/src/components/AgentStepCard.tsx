import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import type { AgentStep } from '../hooks/useAgentTask';
import { formatDuration } from '../utils/format';

interface Props {
  step: AgentStep;
}

interface ToolLabel {
  icon: string;
  label: string;
  detail?: string;
}

function toolLabel(step: AgentStep): ToolLabel {
  const name = step.tool_name || step.type || 'tool';
  const args = step.args;

  switch (name) {
    case 'Read': {
      const path = typeof args?.file_path === 'string' ? args.file_path : '';
      const shortPath = path.split(/[/\\]/).slice(-2).join('/');
      return { icon: '\u{1F4D6}', label: 'Read', detail: shortPath || undefined };
    }
    case 'Write': {
      const path = typeof args?.file_path === 'string' ? args.file_path : '';
      const shortPath = path.split(/[/\\]/).slice(-2).join('/');
      return { icon: '\u{270F}\u{FE0F}', label: 'Write', detail: shortPath || undefined };
    }
    case 'Edit': {
      const path = typeof args?.file_path === 'string' ? args.file_path : '';
      const shortPath = path.split(/[/\\]/).slice(-2).join('/');
      return { icon: '\u{270F}\u{FE0F}', label: 'Edit', detail: shortPath || undefined };
    }
    case 'Bash': {
      const cmd = typeof args?.command === 'string' ? args.command : '';
      const shortCmd = cmd.length > 40 ? cmd.slice(0, 40) + '...' : cmd;
      return { icon: '\u{26A1}', label: 'Bash', detail: shortCmd || undefined };
    }
    case 'Grep': {
      const pattern = typeof args?.pattern === 'string' ? args.pattern : '';
      return { icon: '\u{1F50D}', label: 'Grep', detail: pattern ? `"${pattern}"` : undefined };
    }
    case 'Glob': {
      const pattern = typeof args?.pattern === 'string' ? args.pattern : '';
      return { icon: '\u{1F50D}', label: 'Glob', detail: pattern || undefined };
    }
    case 'WebFetch': {
      const url = typeof args?.url === 'string' ? args.url : '';
      let shortUrl = '';
      try { shortUrl = new URL(url).hostname; } catch { shortUrl = url.slice(0, 40); }
      return { icon: '\u{1F310}', label: 'WebFetch', detail: shortUrl || undefined };
    }
    case 'WebSearch': {
      const q = typeof args?.query === 'string' ? args.query : '';
      const shortQ = q.length > 30 ? q.slice(0, 30) + '...' : q;
      return { icon: '\u{1F310}', label: 'WebSearch', detail: shortQ || undefined };
    }
    case 'Task':
      return { icon: '\u{1F4CB}', label: 'Task', detail: typeof args?.description === 'string' ? args.description.slice(0, 40) : undefined };
    case 'Agent':
      return { icon: '\u{1F916}', label: 'Agent', detail: typeof args?.description === 'string' ? args.description.slice(0, 40) : undefined };
    case 'web_search':
      return { icon: '\u{1F310}', label: 'Web search', detail: typeof args?.query === 'string' ? args.query : undefined };
    case 'web_fetch':
      return { icon: '\u{1F310}', label: 'Fetch URL', detail: typeof args?.url === 'string' ? args.url : undefined };
    case 'calculator':
      return { icon: '\u{1F522}', label: 'Calculate', detail: typeof args?.expression === 'string' ? args.expression : undefined };
    case 'timer_set':
      return { icon: '\u{23F1}\u{FE0F}', label: 'Timer', detail: typeof args?.seconds === 'number' ? `${args.seconds}s` : undefined };
    case 'skill':
      return { icon: '\u{1F3AF}', label: 'Skill', detail: typeof args?.skill === 'string' ? args.skill : undefined };
    default:
      return { icon: '\u{1F527}', label: name, detail: undefined };
  }
}

export default function AgentStepCard({ step }: Props) {
  const [expanded, setExpanded] = useState(false);
  const isRunning = step.status === 'running';
  const isCompleted = step.status === 'completed';
  const isFailed = step.status === 'failed';
  const isSkipped = step.status === 'skipped';
  const isThought = step.type === 'thought';

  if (isThought) {
    const content = step.content || step.message || '';
    const displayText = isRunning && !content ? '\u{1F4AD} Thinking...' : content;

    return (
      <TouchableOpacity
        style={[styles.card, styles.thoughtCard]}
        onPress={() => setExpanded(!expanded)}
        activeOpacity={0.7}
      >
        <View style={styles.thoughtHeader}>
          <Text style={styles.thoughtMarker}>{isRunning ? '\u{25C9}' : '\u{25EF}'}</Text>
          <Text style={styles.thoughtText} numberOfLines={expanded ? undefined : 4}>
            {displayText}
          </Text>
          {step.duration_ms != null && (
            <Text style={styles.thoughtDuration}>{formatDuration(step.duration_ms)}</Text>
          )}
        </View>
      </TouchableOpacity>
    );
  }

  const tl = toolLabel(step);
  const statusIcon = isRunning ? '\u{25B6}' : isCompleted ? '\u{2714}\u{FE0F}' : isFailed ? '\u{2716}\u{FE0F}' : isSkipped ? '\u{23ED}\u{FE0F}' : '\u{25A1}';
  const statusColor = isRunning ? '#4A90D9' : isCompleted ? '#4CAF50' : isFailed ? '#F44336' : isSkipped ? '#FF9800' : '#888';

  return (
    <TouchableOpacity style={styles.card} onPress={() => setExpanded(!expanded)} activeOpacity={0.7}>
      <View style={styles.header}>
        <Text style={[styles.icon, { color: statusColor }]}>{statusIcon}</Text>
        <Text style={styles.toolIcon}>{tl.icon}</Text>
        <View style={styles.headerText}>
          <View style={styles.toolRow}>
            <Text style={styles.typeLabel}>{tl.label}</Text>
            {tl.detail && (
              <Text style={styles.toolDetail} numberOfLines={1}>  {tl.detail}</Text>
            )}
          </View>
        </View>
        {step.duration_ms != null && (
          <Text style={styles.duration}>{formatDuration(step.duration_ms)}</Text>
        )}
      </View>
      {expanded && (
        <View style={styles.detail}>
          {step.args && Object.keys(step.args).length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Args</Text>
              <Text style={styles.code}>{JSON.stringify(step.args, null, 2)}</Text>
            </View>
          )}
          {step.result && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Result</Text>
              <Text style={styles.code}>{JSON.stringify(step.result, null, 2)}</Text>
            </View>
          )}
          {step.error && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Error</Text>
              <Text style={styles.errorText}>{step.error}</Text>
            </View>
          )}
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#fff', marginHorizontal: 16, marginVertical: 4, borderRadius: 10, padding: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 2 },
  thoughtCard: { backgroundColor: '#fafaf5', borderLeftWidth: 3, borderLeftColor: '#e0d5c0' },
  thoughtHeader: { flexDirection: 'row', alignItems: 'flex-start' },
  thoughtMarker: { fontSize: 10, color: '#c0b090', marginRight: 8, marginTop: 3, width: 16, textAlign: 'center' },
  thoughtText: { flex: 1, fontSize: 13, color: '#555', lineHeight: 20 },
  thoughtDuration: { fontSize: 10, color: '#bbb', marginLeft: 8, marginTop: 2 },
  header: { flexDirection: 'row', alignItems: 'center' },
  icon: { fontSize: 12, marginRight: 6, width: 16, textAlign: 'center' },
  toolIcon: { fontSize: 16, marginRight: 8 },
  headerText: { flex: 1 },
  toolRow: { flexDirection: 'row', alignItems: 'baseline' },
  typeLabel: { fontSize: 14, fontWeight: '600', color: '#333' },
  toolDetail: { fontSize: 13, color: '#888', flex: 1 },
  duration: { fontSize: 11, color: '#aaa' },
  detail: { marginTop: 10, borderTopWidth: 1, borderTopColor: '#f0f0f0', paddingTop: 10 },
  section: { marginBottom: 8 },
  sectionTitle: { fontSize: 12, fontWeight: '600', color: '#666', marginBottom: 4 },
  code: { fontSize: 11, color: '#555', backgroundColor: '#f8f8f8', padding: 8, borderRadius: 6, fontFamily: 'monospace' },
  errorText: { fontSize: 12, color: '#F44336' },
});
