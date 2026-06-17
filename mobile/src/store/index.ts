import { create } from 'zustand';

interface AuthState {
  token: string | null;
  username: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setAuth: (token: string, username: string) => void;
  logout: () => void;
  setLoaded: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  username: null,
  isAuthenticated: false,
  isLoading: true,
  setAuth: (token, username) => set({ token, username, isAuthenticated: true }),
  logout: () => set({ token: null, username: null, isAuthenticated: false }),
  setLoaded: () => set({ isLoading: false }),
}));

interface Conversation {
  id: number;
  title: string;
  created_at: string;
}

interface ConversationState {
  conversations: Conversation[];
  setConversations: (convs: Conversation[]) => void;
  addConversation: (conv: Conversation) => void;
}

export const useConversationStore = create<ConversationState>((set) => ({
  conversations: [],
  setConversations: (convs) => set({ conversations: convs }),
  addConversation: (conv) => set((s) => ({ conversations: [conv, ...s.conversations] })),
}));

interface AgentTask {
  id: number;
  title: string;
  status: string;
  total_tokens: number;
  duration_ms: number;
  created_at: string;
}

interface AgentTaskState {
  tasks: AgentTask[];
  setTasks: (tasks: AgentTask[]) => void;
}

export const useAgentTaskStore = create<AgentTaskState>((set) => ({
  tasks: [],
  setTasks: (tasks) => set({ tasks }),
}));
