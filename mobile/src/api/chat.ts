import api from './client';

export async function sendMessage(conversationId: number, message: string): Promise<{ reply: string; conversation_id: number }> {
  const { data } = await api.post('/api/chat', {
    conversation_id: conversationId,
    message,
  });
  return data;
}
