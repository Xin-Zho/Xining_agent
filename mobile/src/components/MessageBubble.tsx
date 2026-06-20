import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface Props {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export default function MessageBubble({ role, content }: Props) {
  if (role === 'system') return null;
  const isUser = role === 'user';
  return (
    <View style={[styles.wrapper, isUser ? styles.userWrapper : styles.assistantWrapper]}>
      <View style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        <Text style={[styles.text, isUser ? styles.userText : styles.assistantText]}>
          {content}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { paddingHorizontal: 16, paddingVertical: 6 },
  userWrapper: { alignItems: 'flex-end' },
  assistantWrapper: { alignItems: 'flex-start' },
  bubble: { maxWidth: '80%', borderRadius: 12, padding: 12 },
  userBubble: { backgroundColor: '#4A90D9' },
  assistantBubble: { backgroundColor: '#f0f0f0' },
  text: { fontSize: 15, lineHeight: 22 },
  userText: { color: '#fff' },
  assistantText: { color: '#333' },
});
