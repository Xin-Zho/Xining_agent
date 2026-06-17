import { Stack } from 'expo-router';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { AuthProvider } from '../src/contexts/AuthContext';
import { useAuthStore } from '../src/store';
import { ui } from '../src/theme/ui';

function RootNavigator() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isLoading = useAuthStore((s) => s.isLoading);

  if (isLoading) {
    return (
      <View style={styles.splash}>
        <ActivityIndicator size="large" color="#4A90D9" />
      </View>
    );
  }

  return (
    <Stack
      screenOptions={{
        headerShown: false,
        headerStyle: { backgroundColor: ui.colors.tab },
        headerTintColor: ui.colors.ink,
        headerShadowVisible: false,
        contentStyle: { backgroundColor: ui.colors.background },
      }}
    >
      <Stack.Protected guard={isAuthenticated}>
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="chat/[id]" options={{ headerShown: true, title: '对话' }} />
        <Stack.Screen name="agent/new" options={{ headerShown: true, title: '新建任务' }} />
        <Stack.Screen name="agent/[id]" options={{ headerShown: true, title: '任务详情' }} />
      </Stack.Protected>

      <Stack.Protected guard={!isAuthenticated}>
        <Stack.Screen name="(auth)" />
      </Stack.Protected>
    </Stack>
  );
}

const styles = StyleSheet.create({
  splash: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: ui.colors.background },
});

export default function RootLayout() {
  return (
    <AuthProvider>
      <RootNavigator />
    </AuthProvider>
  );
}
