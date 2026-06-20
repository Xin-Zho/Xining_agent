import axios from 'axios';
import { Platform } from 'react-native';
import storage from '../utils/storage';

let unauthorizedHandler: (() => void) | null = null;

function getConfiguredBaseUrl() {
  const baseUrl = process.env.EXPO_PUBLIC_API_BASE_URL?.trim();
  return baseUrl ? baseUrl.replace(/\/+$/, '') : null;
}

function getDevHost() {
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return window.location.hostname || 'localhost';
  }

  return Platform.select({
    // Android emulators reach the host machine through 10.0.2.2.
    // Physical devices can override this with EXPO_PUBLIC_API_BASE_URL.
    android: '10.0.2.2',
    ios: 'localhost',
    default: 'localhost',
  });
}

const API_BASE = getConfiguredBaseUrl()
  ?? (__DEV__ ? `http://${getDevHost()}:8000` : 'https://your-production-server.com');

const api = axios.create({ baseURL: API_BASE, timeout: 30000 });

api.interceptors.request.use(async (config) => {
  const token = await storage.getItem('jwt');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    if (err.response?.status === 401) {
      await storage.deleteItem('jwt');
      await storage.deleteItem('username');
      unauthorizedHandler?.();
    }
    return Promise.reject(err);
  }
);

export function setUnauthorizedHandler(handler: (() => void) | null) {
  unauthorizedHandler = handler;
}

export { API_BASE };
export default api;
