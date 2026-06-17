import { Platform } from 'react-native';

function getWebStorage() {
  if (typeof localStorage === 'undefined') {
    return null;
  }

  return localStorage;
}

// SecureStore only works on native. Fall back to localStorage on web.
const storage = {
  getItem: async (key: string): Promise<string | null> => {
    if (Platform.OS === 'web') {
      try {
        return getWebStorage()?.getItem(key) ?? null;
      } catch {
        return null;
      }
    }
    const { getItemAsync } = await import('expo-secure-store');
    return getItemAsync(key);
  },
  setItem: async (key: string, value: string): Promise<void> => {
    if (Platform.OS === 'web') {
      try {
        getWebStorage()?.setItem(key, value);
      } catch {
        // Ignore storage write failures in restricted browser contexts.
      }
      return;
    }
    const { setItemAsync } = await import('expo-secure-store');
    return setItemAsync(key, value);
  },
  deleteItem: async (key: string): Promise<void> => {
    if (Platform.OS === 'web') {
      try {
        getWebStorage()?.removeItem(key);
      } catch {
        // Ignore storage delete failures in restricted browser contexts.
      }
      return;
    }
    const { deleteItemAsync } = await import('expo-secure-store');
    return deleteItemAsync(key);
  },
};

export default storage;
