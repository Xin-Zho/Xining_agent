import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Link } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../../src/contexts/AuthContext';
import { showAlert } from '../../src/utils/notify';
import { ui } from '../../src/theme/ui';

export default function RegisterScreen() {
  const { register } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRegister = async () => {
    if (!username.trim() || !password) {
      showAlert('提示', '请输入用户名和密码');
      return;
    }
    if (password.length < 4) {
      showAlert('提示', '密码至少4位');
      return;
    }
    if (password !== confirm) {
      showAlert('提示', '两次密码不一致');
      return;
    }
    setLoading(true);
    try {
      await register(username.trim(), password);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e.message || '注册失败';
      showAlert('注册失败', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
          <View style={styles.decorTop} />
          <View style={styles.decorBottom} />
          <View style={styles.hero}>
            <Text style={styles.eyebrow}>CREATE ACCESS</Text>
            <Text style={styles.title}>给你的 AI 工作流创建一个正式入口</Text>
            <Text style={styles.subtitle}>注册后你会获得对话、任务和执行记录的一体化空间，适合长期使用和积累。</Text>
          </View>

          <View style={styles.card}>
            <View style={styles.cardTag}>
              <Text style={styles.cardTagText}>REGISTER</Text>
            </View>
            <Text style={styles.formTitle}>创建账号</Text>
            <Text style={styles.formSubtitle}>注册后即可进入统一工作台</Text>
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
              placeholder="密码（至少4位）"
              placeholderTextColor={ui.colors.muted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
            <TextInput
              style={styles.input}
              placeholder="确认密码"
              placeholderTextColor={ui.colors.muted}
              value={confirm}
              onChangeText={setConfirm}
              secureTextEntry
            />
            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={handleRegister}
              disabled={loading}
            >
              <Text style={styles.buttonText}>{loading ? '注册中...' : '创建并进入'}</Text>
            </TouchableOpacity>
            <Link href="/login" style={styles.link}>
              已有账号？直接回到登录页
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
    top: -50,
    right: -20,
    width: 170,
    height: 170,
    borderRadius: 85,
    backgroundColor: ui.colors.warningSoft,
  },
  decorBottom: {
    position: 'absolute',
    bottom: 80,
    left: -34,
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: ui.colors.accentSoft,
  },
  hero: { marginBottom: 22, maxWidth: 540 },
  eyebrow: { color: ui.colors.warning, fontSize: 11, fontWeight: '800', letterSpacing: 1.6, marginBottom: 12 },
  title: { fontSize: 34, lineHeight: 40, fontWeight: '800', color: ui.colors.ink, marginBottom: 12 },
  subtitle: { fontSize: 15, lineHeight: 24, color: ui.colors.muted },
  card: {
    width: '100%',
    maxWidth: 460,
    backgroundColor: ui.colors.card,
    borderRadius: 28,
    padding: 24,
    borderWidth: 1,
    borderColor: ui.colors.border,
    ...ui.shadow,
  },
  cardTag: {
    alignSelf: 'flex-start',
    backgroundColor: ui.colors.warningSoft,
    borderRadius: ui.radii.pill,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 16,
  },
  cardTagText: { color: '#8b5f12', fontSize: 11, fontWeight: '800', letterSpacing: 1.1 },
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
  button: { backgroundColor: ui.colors.success, borderRadius: 18, padding: 16, alignItems: 'center', marginTop: 8 },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: ui.colors.white, fontSize: 16, fontWeight: '800' },
  link: { textAlign: 'center', color: ui.colors.success, marginTop: 16, fontSize: 14, lineHeight: 21 },
});
