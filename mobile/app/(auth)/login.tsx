import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Link } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../../src/contexts/AuthContext';
import { showAlert } from '../../src/utils/notify';
import { ui } from '../../src/theme/ui';

export default function LoginScreen() {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!username.trim() || !password) {
      showAlert('提示', '请输入用户名和密码');
      return;
    }
    setLoading(true);
    try {
      await login(username.trim(), password);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e.message || '登录失败';
      showAlert('登录失败', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
          <View style={styles.decorTop} />
          <View style={styles.decorSide} />
          <View style={styles.hero}>
            <Text style={styles.eyebrow}>AI WORKBENCH</Text>
            <Text style={styles.title}>让聊天和 Agent 任务拥有更像产品的入口</Text>
            <Text style={styles.subtitle}>登录后进入统一工作台，快速对话、启动任务、回看执行记录都在一个界面完成。</Text>
          </View>

          <View style={styles.card}>
            <View style={styles.cardTag}>
              <Text style={styles.cardTagText}>SIGN IN</Text>
            </View>
            <Text style={styles.formTitle}>登录你的账号</Text>
            <Text style={styles.formSubtitle}>继续进入你的 AI 工作区</Text>
            <TextInput
              style={styles.input}
              placeholder="用户名"
              placeholderTextColor={ui.colors.muted}
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
            />
            <TextInput
              style={styles.input}
              placeholder="密码"
              placeholderTextColor={ui.colors.muted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={handleLogin}
              disabled={loading}
            >
              <Text style={styles.buttonText}>{loading ? '登录中...' : '进入工作台'}</Text>
            </TouchableOpacity>
            <Link href="/register" style={styles.link}>
              没有账号？去创建一个新的工作区身份
            </Link>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: ui.colors.background },
  container: { flex: 1, backgroundColor: ui.colors.background },
  scrollContent: { flexGrow: 1, justifyContent: 'center', padding: 24 },
  decorTop: {
    position: 'absolute',
    top: 18,
    right: -36,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: ui.colors.accentSoft,
  },
  decorSide: {
    position: 'absolute',
    bottom: 110,
    left: -42,
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: ui.colors.successSoft,
  },
  hero: { marginBottom: 22, maxWidth: 540 },
  eyebrow: { color: ui.colors.accent, fontSize: 11, fontWeight: '800', letterSpacing: 1.6, marginBottom: 12 },
  title: { fontSize: 34, lineHeight: 40, fontWeight: '800', color: ui.colors.ink, marginBottom: 12 },
  subtitle: { fontSize: 15, lineHeight: 24, color: ui.colors.muted },
  card: {
    width: '100%',
    maxWidth: 440,
    backgroundColor: ui.colors.card,
    borderRadius: 28,
    padding: 24,
    borderWidth: 1,
    borderColor: ui.colors.border,
    ...ui.shadow,
  },
  cardTag: {
    alignSelf: 'flex-start',
    backgroundColor: ui.colors.accentSoft,
    borderRadius: ui.radii.pill,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 16,
  },
  cardTagText: { color: ui.colors.accentDark, fontSize: 11, fontWeight: '800', letterSpacing: 1.1 },
  formTitle: { fontSize: 24, fontWeight: '800', color: ui.colors.ink },
  formSubtitle: { fontSize: 14, color: ui.colors.muted, marginTop: 6, marginBottom: 18 },
  input: {
    borderWidth: 1,
    borderColor: ui.colors.border,
    backgroundColor: ui.colors.surface,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    marginBottom: 12,
    color: ui.colors.text,
  },
  button: { backgroundColor: ui.colors.accent, borderRadius: 18, padding: 16, alignItems: 'center', marginTop: 8 },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: ui.colors.white, fontSize: 16, fontWeight: '800' },
  link: { textAlign: 'center', color: ui.colors.accentDark, marginTop: 16, fontSize: 14, lineHeight: 21 },
});
