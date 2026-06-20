import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export interface DigestEntry {
  id: string;
  decision: 'adopted' | 'rejected' | 'partial' | 'unknown';
  summary: string;
  userContent?: string;
}

interface Props {
  entry: DigestEntry;
}

const decisionConfig: Record<string, { label: string; color: string; icon: string }> = {
  adopted: { label: '已采纳', color: '#4CAF50', icon: '✓' },
  rejected: { label: '未采纳', color: '#F44336', icon: '✗' },
  partial: { label: '部分采纳', color: '#FF9800', icon: '△' },
  unknown: { label: '已收到', color: '#9E9E9E', icon: '?' },
};

export default function FeedbackDigestBanner({ entry }: Props) {
  const [expanded, setExpanded] = useState(false);
  const config = decisionConfig[entry.decision] || decisionConfig.unknown;

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => setExpanded(!expanded)}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <Text style={styles.icon}>↗</Text>
        <View style={styles.headerText}>
          <Text style={styles.label}>方向调整</Text>
          {entry.userContent && (
            <Text style={styles.userText} numberOfLines={expanded ? 0 : 1}>
              "{entry.userContent}"
            </Text>
          )}
        </View>
        <View style={[styles.badge, { backgroundColor: config.color }]}>
          <Text style={styles.badgeText}>{config.icon} {config.label}</Text>
        </View>
      </View>

      {expanded && (
        <View style={styles.body}>
          <Text style={styles.summaryLabel}>Agent 回应：</Text>
          <Text style={styles.summary}>{entry.summary}</Text>
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
    backgroundColor: '#F0F4FF',
    borderLeftWidth: 3,
    borderLeftColor: '#4A90D9',
    padding: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  icon: { fontSize: 18, marginRight: 8, color: '#4A90D9' },
  headerText: { flex: 1 },
  label: { fontSize: 13, fontWeight: '600', color: '#4A90D9' },
  userText: { fontSize: 12, color: '#666', marginTop: 2, fontStyle: 'italic' },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  badgeText: { fontSize: 11, color: '#fff', fontWeight: '600' },
  body: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#DCE4F5',
  },
  summaryLabel: { fontSize: 12, fontWeight: '600', color: '#333', marginBottom: 4 },
  summary: { fontSize: 13, color: '#555', lineHeight: 18 },
});
