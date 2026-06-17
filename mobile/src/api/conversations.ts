import api from './client';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at?: string;
}

export interface Conversation {
  id: number;
  title: string;
  created_at: string;
  messages?: Message[];
}

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await api.get('/api/conversations');
  return data;
}

export async function createConversation(title: string = '新对话'): Promise<{ id: number; title: string }> {
  const { data } = await api.post('/api/conversations', { title });
  return data;
}

export async function getConversation(id: number): Promise<Conversation> {
  const { data } = await api.get(`/api/conversations/${id}`);
  return data;
}
