import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import MD from 'react-native-markdown-display';

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  return (
    <ScrollView style={styles.container} horizontal={false}>
      <MD style={mdStyles}>{content}</MD>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16 },
});

const mdStyles = {
  body: { fontSize: 15, color: '#333', lineHeight: 24 },
  heading1: { fontSize: 22, fontWeight: '700' as const, color: '#222', marginBottom: 8 },
  heading2: { fontSize: 18, fontWeight: '600' as const, color: '#222', marginBottom: 6 },
  heading3: { fontSize: 16, fontWeight: '600' as const, color: '#333', marginBottom: 4 },
  code_inline: { backgroundColor: '#f0f0f0', color: '#e83e8c', fontSize: 13 },
  fence: { backgroundColor: '#f8f8f8', padding: 12, borderRadius: 6 },
  link: { color: '#4A90D9' },
  list_item: { marginBottom: 4 },
};
