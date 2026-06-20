import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface Props {
  completed: number;
  total: number;
}

export default function AgentProgressBar({ completed, total }: Props) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <View style={styles.container}>
      <View style={styles.barBg}>
        <View style={[styles.barFill, { width: `${Math.max(pct, 2)}%` }]} />
      </View>
      <Text style={styles.text}>
        {completed}/{total} 步骤 ({pct}%)
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: 16, paddingVertical: 8 },
  barBg: { height: 6, backgroundColor: '#e0e0e0', borderRadius: 3, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: '#4A90D9', borderRadius: 3 },
  text: { textAlign: 'center', fontSize: 12, color: '#888', marginTop: 4 },
});
