import { Alert, Platform } from 'react-native';

export function showAlert(title: string, message: string) {
  if (Platform.OS === 'web' && typeof window !== 'undefined' && typeof window.alert === 'function') {
    window.alert(`${title}\n\n${message}`);
    return;
  }

  Alert.alert(title, message);
}
