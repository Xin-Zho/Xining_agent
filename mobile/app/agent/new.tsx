import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { createAgentTask } from '../../src/api/agent';

export default function NewAgentTaskScreen() {
  const router = useRouter();
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    const trimmed = description.trim();
    if (!trimmed) {
      Alert.alert('提示', '请输入任务描述');
      return;
    }
    setLoading(true);
    try {
      const { task_id } = await createAgentTask(trimmed);
      router.replace(`/agent/${task_id}`);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e.message || '创建任务失败';
      Alert.alert('错误', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.label}>描述你想要完成的任务</Text>
        <TextInput
          style={styles.textarea}
          placeholder={'例如：\n搜索北京到上海的航班，比较各大航空公司的价格，整理成表格。\n\n或者：\n计算 (123 + 456) * 789，然后搜索这个结果是否有什么特殊的数学含义。'}
          placeholderTextColor="#bbb"
          value={description}
          onChangeText={setDescription}
          multiline
          numberOfLines={8}
          textAlignVertical="top"
        />
        <View style={styles.toolsBox}>
          <Text style={styles.toolsTitle}>可用工具</Text>
          <Text style={styles.toolItem}>🔍 网络搜索 — 搜索互联网获取最新信息</Text>
          <Text style={styles.toolItem}>📄 网页抓取 — 获取指定网页的内容</Text>
          <Text style={styles.toolItem}>🔢 数学计算 — 执行数学表达式计算</Text>
          <Text style={styles.toolItem}>⏰ 计时器 — 设置延时等待</Text>
        </View>
        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleStart}
          disabled={loading}
        >
          <Text style={styles.buttonText}>{loading ? '启动中...' : '启动 Agent 任务'}</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 16 },
  label: { fontSize: 16, fontWeight: '600', color: '#333', marginBottom: 8 },
  textarea: {
    borderWidth: 1, borderColor: '#ddd', borderRadius: 10, padding: 14,
    fontSize: 15, color: '#333', minHeight: 160, marginBottom: 16,
  },
  toolsBox: { backgroundColor: '#f8f8f8', borderRadius: 10, padding: 14, marginBottom: 20 },
  toolsTitle: { fontSize: 14, fontWeight: '600', color: '#666', marginBottom: 8 },
  toolItem: { fontSize: 13, color: '#888', marginBottom: 4 },
  button: { backgroundColor: '#FF9800', borderRadius: 10, padding: 16, alignItems: 'center' },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
});
