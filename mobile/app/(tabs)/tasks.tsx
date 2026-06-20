import React, { useEffect, useState, useCallback } from 'react';
import { View, FlatList, TouchableOpacity, Text, StyleSheet, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { listAgentTasks, AgentTaskSummary } from '../../src/api/agent';
import { formatDate, statusLabel, statusColor } from '../../src/utils/format';
import { showAlert } from '../../src/utils/notify';
import { ui } from '../../src/theme/ui';

export default function TasksScreen() {
  const router = useRouter();
  const [tasks, setTasks] = useState<AgentTaskSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await listAgentTasks();
      setTasks(data);
    } catch (e: any) {
      if (e?.response?.status === 401) return;
      showAlert('错误', e?.response?.data?.detail || e?.message || '加载任务列表失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  const completedCount = tasks.filter((item) => item.status === 'completed').length;
  const runningCount = tasks.filter((item) => item.status === 'planning' || item.status === 'executing' || item.status === 'observing').length;

  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        <View style={styles.center}>
          <View style={styles.loadingCard}>
            <Text style={styles.loadingLabel}>正在整理任务面板...</Text>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.bgPanel} />
        <FlatList
          data={tasks}
          keyExtractor={(item) => String(item.id)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={ui.colors.success} />}
          contentContainerStyle={styles.content}
          ListHeaderComponent={(
            <View>
              <View style={styles.heroCard}>
                <Text style={styles.heroKicker}>AGENT BOARD</Text>
                <Text style={styles.heroTitle}>把复杂工作拆成可追踪的步骤</Text>
                <Text style={styles.heroSubtitle}>这里展示多步骤任务的状态、结果和基本消耗，适合回看执行过程。</Text>
                <View style={styles.summaryRow}>
                  <View style={styles.summaryCard}>
                    <Text style={styles.summaryValue}>{tasks.length}</Text>
                    <Text style={styles.summaryLabel}>全部任务</Text>
                  </View>
                  <View style={styles.summaryCard}>
                    <Text style={styles.summaryValue}>{runningCount}</Text>
                    <Text style={styles.summaryLabel}>进行中</Text>
                  </View>
                  <View style={styles.summaryCard}>
                    <Text style={styles.summaryValue}>{completedCount}</Text>
                    <Text style={styles.summaryLabel}>已完成</Text>
                  </View>
                </View>
              </View>

              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>任务记录</Text>
                <Text style={styles.sectionMeta}>{tasks.length} 条</Text>
              </View>
            </View>
          )}
          ListEmptyComponent={(
            <View style={styles.emptyCard}>
              <Text style={styles.emptyTitle}>还没有 Agent 任务</Text>
              <Text style={styles.emptyText}>去对话页点击“启动任务”，这里就会开始累积执行历史。</Text>
            </View>
          )}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.item}
              onPress={() => router.push(`/agent/${item.id}`)}
            >
              <View style={styles.itemHeader}>
                <Text style={styles.itemTitle} numberOfLines={1}>{item.title}</Text>
                <View style={[styles.badge, { backgroundColor: statusColor(item.status) }]}>
                  <Text style={styles.badgeText}>{statusLabel(item.status)}</Text>
                </View>
              </View>
              <View style={styles.itemStats}>
                <Text style={styles.itemMeta}>Token {item.total_tokens}</Text>
                <Text style={styles.dot}>•</Text>
                <Text style={styles.itemMeta}>{formatDate(item.created_at)}</Text>
              </View>
            </TouchableOpacity>
          )}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: ui.colors.background },
  container: { flex: 1, backgroundColor: ui.colors.background },
  content: { padding: 20, paddingBottom: 120 },
  bgPanel: {
    position: 'absolute',
    top: -40,
    left: 0,
    right: 0,
    height: 220,
    backgroundColor: ui.colors.backgroundAlt,
    borderBottomLeftRadius: 36,
    borderBottomRightRadius: 36,
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: ui.colors.background },
  loadingCard: {
    backgroundColor: ui.colors.card,
    borderRadius: 22,
    paddingHorizontal: 24,
    paddingVertical: 20,
    borderWidth: 1,
    borderColor: ui.colors.border,
    ...ui.shadow,
  },
  loadingLabel: { color: ui.colors.text, fontSize: 15, fontWeight: '700' },
  heroCard: {
    backgroundColor: ui.colors.success,
    borderRadius: 28,
    padding: 22,
    marginBottom: 22,
    ...ui.shadow,
  },
  heroKicker: { color: '#d8eee6', fontSize: 11, fontWeight: '800', letterSpacing: 1.4, marginBottom: 10 },
  heroTitle: { color: ui.colors.white, fontSize: 28, lineHeight: 34, fontWeight: '800', maxWidth: 280 },
  heroSubtitle: { color: '#ecf7f2', fontSize: 14, lineHeight: 22, marginTop: 10, maxWidth: 300 },
  summaryRow: { flexDirection: 'row', gap: 10, marginTop: 22 },
  summaryCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.14)',
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.18)',
  },
  summaryValue: { color: ui.colors.white, fontSize: 22, fontWeight: '800' },
  summaryLabel: { color: '#d8eee6', fontSize: 12, marginTop: 5 },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  sectionTitle: { fontSize: 20, fontWeight: '800', color: ui.colors.ink },
  sectionMeta: { fontSize: 12, fontWeight: '700', color: ui.colors.muted },
  emptyCard: {
    backgroundColor: ui.colors.surface,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: ui.colors.border,
    padding: 24,
    alignItems: 'center',
  },
  emptyTitle: { fontSize: 18, fontWeight: '800', color: ui.colors.ink, marginBottom: 8 },
  emptyText: { color: ui.colors.muted, fontSize: 14, lineHeight: 21, textAlign: 'center', maxWidth: 260 },
  item: {
    backgroundColor: ui.colors.card,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: ui.colors.border,
    padding: 18,
    marginBottom: 12,
  },
  itemHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  itemTitle: { fontSize: 16, fontWeight: '700', color: ui.colors.text, flex: 1 },
  badge: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: ui.radii.pill },
  badgeText: { color: ui.colors.white, fontSize: 11, fontWeight: '800' },
  itemStats: { flexDirection: 'row', alignItems: 'center', marginTop: 10 },
  itemMeta: { fontSize: 12, color: ui.colors.muted },
  dot: { marginHorizontal: 8, color: ui.colors.muted },
});
