import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [text, setText] = useState('');

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText('');
  };

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <View style={styles.row}>
        <TextInput
          style={styles.input}
          placeholder="输入消息..."
          placeholderTextColor="#999"
          value={text}
          onChangeText={setText}
          multiline
          maxLength={4000}
          editable={!disabled}
        />
        <TouchableOpacity
          style={[styles.btn, disabled && styles.btnDisabled]}
          onPress={handleSend}
          disabled={disabled || !text.trim()}
        >
          <Text style={styles.btnText}>发送</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'flex-end', padding: 12, borderTopWidth: 1, borderTopColor: '#eee', backgroundColor: '#fff' },
  input: { flex: 1, borderWidth: 1, borderColor: '#ddd', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10, fontSize: 15, maxHeight: 100, color: '#333' },
  btn: { backgroundColor: '#4A90D9', borderRadius: 20, paddingHorizontal: 20, paddingVertical: 10, marginLeft: 8 },
  btnDisabled: { opacity: 0.4 },
  btnText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});
