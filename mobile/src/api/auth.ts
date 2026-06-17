import api from './client';

export async function login(username: string, password: string) {
  const { data } = await api.post('/api/login', { username, password });
  return data as { token: string; username: string };
}

export async function register(username: string, password: string) {
  const { data } = await api.post('/api/register', { username, password });
  return data as { token: string; username: string };
}
