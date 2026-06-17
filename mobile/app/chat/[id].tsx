import React, { useEffect, useState, useRef, useCallback } from 'react';
import { View, FlatList, StyleSheet, ActivityIndicator, Text, TouchableOpacity } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { getConversation, Message } from '../../src/api/conversations';
import { sendMessage } from '../../src/api/chat';
import MessageBubble from '../../src/components/MessageBubble';
import ChatInput from '../../src/components/ChatInput';

export default function ChatScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const convId = Number(id);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const flatRef = useRef<FlatList>(null);

  const loadMessages = useCallback(async () => {
    if (!Number.isFinite(convId) || convId <= 0) {
      setError('无效的对话 ID');
      setLoading(false);
      return;
    }

    setError(null);
    try {
      const conv = await getConversation(convId);
      setMessages(conv.messages || []);
    } catch (e: any) {
      const message = e?.response?.data?.detail || e?.message || '加载对话失败';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [convId]);

  useEffect(() => { loadMessages(); }, [loadMessages]);

  const handleSend = async (text: string) => {
    if (!Number.isFinite(convId) || convId <= 0) {
      setError('无效的对话 ID');
      return;
    }

    const userMsg: Message = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setSending(true);
    try {
      const data = await sendMessage(convId, text);
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }]);
    } catch {
      setMessages((prev) => [...prev, { role: 'assistant', content: '发送失败，请重试' }]);
    } finally {
      setSending(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#4A90D9" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorTitle}>加载失败</Text>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={() => {
          setLoading(true);
          loadMessages();
        }}>
          <Text style={styles.retryText}>重试</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatRef}
        data={messages}
        keyExtractor={(_, i) => String(i)}
        renderItem={({ item }) => <MessageBubble role={item.role} content={item.content} />}
        onContentSizeChange={() => flatRef.current?.scrollToEnd({ animated: true })}
        ListEmptyComponent={<Text style={styles.empty}>开始对话吧</Text>}
      />
      <ChatInput onSend={handleSend} disabled={sending} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  empty: { textAlign: 'center', marginTop: 200, color: '#999', fontSize: 16 },
  errorTitle: { fontSize: 18, fontWeight: '700', color: '#333', marginBottom: 8 },
  errorText: { fontSize: 14, color: '#666', marginBottom: 16, paddingHorizontal: 24, textAlign: 'center' },
  retryBtn: { backgroundColor: '#4A90D9', borderRadius: 8, paddingHorizontal: 20, paddingVertical: 12 },
  retryText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});
