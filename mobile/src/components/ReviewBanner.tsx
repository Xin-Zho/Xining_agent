import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export interface ReviewEntry {
  missing_steps?: Array<{ description: string; why_important: string; suggested_insert_after: string }>;
  flawed_logic?: Array<{ step_reference: string; issue: string; suggested_fix: string }>;
  boundary_gaps?: Array<{ scenario: string; impact: string; suggested_handling: string }>;
  suggestions?: Array<{ aspect: string; current_approach: string; alternative: string }>;
}

interface Props {
  entry: ReviewEntry;
}

export default function ReviewBanner({ entry }: Props) {
  const [expanded, setExpanded] = useState(false);

  const totalFindings =
    (entry.missing_steps?.length || 0) +
    (entry.flawed_logic?.length || 0) +
    (entry.boundary_gaps?.length || 0) +
    (entry.suggestions?.length || 0);

  if (totalFindings === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.icon}>✓</Text>
          <Text style={styles.label}>方案已优化 — 未发现结构性问题</Text>
        </View>
      </View>
    );
  }

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => setExpanded(!expanded)}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <Text style={styles.icon}>↗</Text>
        <Text style={styles.label}>方案已优化 — {totalFindings} 条改进建议</Text>
        <Text style={styles.expandHint}>{expanded ? '收起' : '展开'}</Text>
      </View>

      {expanded && (
        <View style={styles.body}>
          {entry.missing_steps?.map((s, i) => (
            <View key={`ms-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[遗漏] {s.description}</Text>
              <Text style={styles.findingDetail}>重要性：{s.why_important}</Text>
              <Text style={styles.findingDetail}>建议位置：{s.suggested_insert_after}</Text>
            </View>
          ))}
          {entry.flawed_logic?.map((f, i) => (
            <View key={`fl-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[逻辑] {f.step_reference}</Text>
              <Text style={styles.findingDetail}>问题：{f.issue}</Text>
              <Text style={styles.findingDetail}>建议：{f.suggested_fix}</Text>
            </View>
          ))}
          {entry.boundary_gaps?.map((b, i) => (
            <View key={`bg-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[边界] {b.scenario}</Text>
              <Text style={styles.findingDetail}>影响：{b.impact}</Text>
              <Text style={styles.findingDetail}>处理：{b.suggested_handling}</Text>
            </View>
          ))}
          {entry.suggestions?.map((s, i) => (
            <View key={`sg-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[{s.aspect}] {s.current_approach}</Text>
              <Text style={styles.findingDetail}>替代：{s.alternative}</Text>
            </View>
          ))}
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginVertical: 4,
    borderRadius: 10,
    backgroundColor: '#F0FFF0',
    borderLeftWidth: 3,
    borderLeftColor: '#4CAF50',
    padding: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  icon: { fontSize: 18, marginRight: 8, color: '#4CAF50' },
  label: { flex: 1, fontSize: 13, fontWeight: '600', color: '#4CAF50' },
  expandHint: { fontSize: 12, color: '#888' },
  body: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#DCEFD5',
  },
  findingItem: {
    marginBottom: 10,
    paddingLeft: 8,
    borderLeftWidth: 2,
    borderLeftColor: '#E0E0E0',
  },
  findingType: { fontSize: 13, fontWeight: '600', color: '#333', marginBottom: 2 },
  findingDetail: { fontSize: 12, color: '#666', marginBottom: 1 },
});
