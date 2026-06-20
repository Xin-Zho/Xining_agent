import { Redirect, Tabs } from 'expo-router';
import { Text } from 'react-native';
import { useAuthStore } from '../../src/store';
import { ui } from '../../src/theme/ui';

export default function TabLayout() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  if (!isAuthenticated) {
    return <Redirect href="/login" />;
  }

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: ui.colors.accent,
        tabBarInactiveTintColor: ui.colors.muted,
        tabBarStyle: {
          backgroundColor: ui.colors.tab,
          borderTopColor: ui.colors.border,
          height: 72,
          paddingTop: 10,
          paddingBottom: 10,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '700',
          letterSpacing: 0.3,
        },
        sceneStyle: {
          backgroundColor: ui.colors.background,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: '对话',
          tabBarIcon: ({ color }) => <Text style={{ fontSize: 20, color }}>💬</Text>,
        }}
      />
      <Tabs.Screen
        name="tasks"
        options={{
          title: '任务',
          tabBarIcon: ({ color }) => <Text style={{ fontSize: 20, color }}>⚡</Text>,
        }}
      />
    </Tabs>
  );
}
