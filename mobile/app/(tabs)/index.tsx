import React, { useEffect, useState, useCallback } from 'react';
import { View, FlatList, TouchableOpacity, Text, StyleSheet, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { listConversations, createConversation, Conversation } from '../../src/api/conversations';
import { useAuth } from '../../src/contexts/AuthContext';
import { formatDate } from '../../src/utils/format';
import { showAlert } from '../../src/utils/notify';
import { ui } from '../../src/theme/ui';

export default function ConversationsScreen() {
  const router = useRouter();
  const { logout } = useAuth();
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listConversations();
      setConvs(data);
    } catch (e: any) {
      if (e?.response?.status === 401) return;
      showAlert('错误', e?.response?.data?.detail || e?.message || '加载对话列表失败');
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

  const handleNewConversation = async () => {
    try {
      const conv = await createConversation();
      router.push(`/chat/${conv.id}`);
    } catch (e: any) {
      showAlert('错误', e?.response?.data?.detail || e?.message || '创建对话失败');
    }
  };

  const handleNewAgentTask = () => {
    router.push('/agent/new');
  };

  const handleLogout = async () => {
    await logout();
    router.replace('/login');
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.bgOrbLarge} />
        <View style={styles.bgOrbSmall} />
        <FlatList
          data={convs}
          keyExtractor={(item) => String(item.id)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={ui.colors.accent} />}
          contentContainerStyle={styles.content}
          ListHeaderComponent={(
            <View>
              <View style={styles.heroCard}>
                <View style={styles.heroTopRow}>
                  <View>
                    <Text style={styles.eyebrow}>WORKSPACE</Text>
                    <Text style={styles.heroTitle}>把对话和任务放到一个工作台里</Text>
                    <Text style={styles.heroSubtitle}>这里是你的日常入口。新建对话适合快速问答，Agent 任务适合多步骤执行。</Text>
                  </View>
                  <TouchableOpacity style={styles.logoutChip} onPress={handleLogout}>
                    <Text style={styles.logoutChipText}>退出</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{convs.length}</Text>
                    <Text style={styles.statLabel}>总对话数</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{loading ? '...' : '就绪'}</Text>
                    <Text style={styles.statLabel}>当前状态</Text>
                  </View>
                </View>
              </View>

              <View style={styles.actionRow}>
                <TouchableOpacity style={styles.primaryAction} onPress={handleNewConversation}>
                  <Text style={styles.actionKicker}>CHAT</Text>
                  <Text style={styles.primaryActionTitle}>新建对话</Text>
                  <Text style={styles.primaryActionText}>开始一段新的即时交流。</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.secondaryAction} onPress={handleNewAgentTask}>
                  <Text style={styles.actionKickerAlt}>AGENT</Text>
                  <Text style={styles.secondaryActionTitle}>启动任务</Text>
                  <Text style={styles.secondaryActionText}>让 Agent 处理多步骤工作流。</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>最近对话</Text>
                <Text style={styles.sectionMeta}>{loading ? '加载中' : `${convs.length} 条记录`}</Text>
              </View>
            </View>
          )}
          ListEmptyComponent={(
            <View style={styles.emptyCard}>
              <Text style={styles.emptyTitle}>还没有任何对话</Text>
              <Text style={styles.emptyText}>先从一个问题开始，或者直接启动一个 Agent 任务。</Text>
            </View>
          )}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.item}
              onPress={() => router.push(`/chat/${item.id}`)}
            >
              <View style={styles.itemBadge}>
                <Text style={styles.itemBadgeText}>对话</Text>
              </View>
              <View style={styles.itemBody}>
                <Text style={styles.itemTitle} numberOfLines={1}>{item.title}</Text>
                <Text style={styles.itemMeta}>创建于 {formatDate(item.created_at)}</Text>
              </View>
              <Text style={styles.itemArrow}>↗</Text>
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
  bgOrbLarge: {
    position: 'absolute',
    top: -70,
    right: -40,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: ui.colors.accentSoft,
    opacity: 0.55,
  },
  bgOrbSmall: {
    position: 'absolute',
    top: 110,
    left: -30,
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: ui.colors.successSoft,
    opacity: 0.7,
  },
  heroCard: {
    backgroundColor: ui.colors.card,
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: ui.colors.border,
    marginBottom: 18,
    ...ui.shadow,
  },
  heroTopRow: { flexDirection: 'row', justifyContent: 'space-between', gap: 16 },
  eyebrow: { fontSize: 11, fontWeight: '800', letterSpacing: 1.4, color: ui.colors.accent, marginBottom: 10 },
  heroTitle: { fontSize: 28, lineHeight: 34, fontWeight: '800', color: ui.colors.ink, maxWidth: 260 },
  heroSubtitle: { marginTop: 10, fontSize: 14, lineHeight: 22, color: ui.colors.muted, maxWidth: 290 },
  logoutChip: {
    alignSelf: 'flex-start',
    backgroundColor: ui.colors.surface,
    borderWidth: 1,
    borderColor: ui.colors.border,
    borderRadius: ui.radii.pill,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  logoutChipText: { color: ui.colors.danger, fontSize: 13, fontWeight: '700' },
  statRow: { flexDirection: 'row', gap: 12, marginTop: 22 },
  statCard: {
    flex: 1,
    backgroundColor: ui.colors.surface,
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: ui.colors.border,
  },
  statValue: { color: ui.colors.ink, fontSize: 24, fontWeight: '800' },
  statLabel: { color: ui.colors.muted, fontSize: 12, marginTop: 6 },
  actionRow: { gap: 12, marginBottom: 24 },
  primaryAction: {
    backgroundColor: ui.colors.accent,
    borderRadius: 24,
    padding: 20,
  },
  secondaryAction: {
    backgroundColor: ui.colors.success,
    borderRadius: 24,
    padding: 20,
  },
  actionKicker: { color: '#f9dccb', fontSize: 11, fontWeight: '800', letterSpacing: 1.4, marginBottom: 8 },
  actionKickerAlt: { color: '#d7ece5', fontSize: 11, fontWeight: '800', letterSpacing: 1.4, marginBottom: 8 },
  primaryActionTitle: { color: ui.colors.white, fontSize: 24, fontWeight: '800' },
  secondaryActionTitle: { color: ui.colors.white, fontSize: 24, fontWeight: '800' },
  primaryActionText: { color: '#fff0e6', fontSize: 14, lineHeight: 20, marginTop: 8 },
  secondaryActionText: { color: '#eef8f4', fontSize: 14, lineHeight: 20, marginTop: 8 },
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
  emptyText: { fontSize: 14, lineHeight: 21, color: ui.colors.muted, textAlign: 'center', maxWidth: 260 },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: ui.colors.card,
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: ui.colors.border,
    marginBottom: 12,
  },
  itemBadge: {
    backgroundColor: ui.colors.accentSoft,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: ui.radii.pill,
  },
  itemBadgeText: { color: ui.colors.accentDark, fontSize: 11, fontWeight: '800', letterSpacing: 0.8 },
  itemBody: { flex: 1 },
  itemTitle: { fontSize: 17, fontWeight: '700', color: ui.colors.text },
  itemMeta: { fontSize: 13, color: ui.colors.muted, marginTop: 6 },
  itemArrow: { fontSize: 18, color: ui.colors.accentDark, fontWeight: '700' },
});
