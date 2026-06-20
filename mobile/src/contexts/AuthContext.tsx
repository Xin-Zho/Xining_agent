import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useAuthStore } from '../store';
import storage from '../utils/storage';
import { login as loginApi, register as registerApi } from '../api/auth';
import { setUnauthorizedHandler } from '../api/client';

interface AuthContextType {
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true);
  const { setAuth, logout: logoutStore, setLoaded } = useAuthStore();

  useEffect(() => {
    (async () => {
      try {
        const token = await storage.getItem('jwt');
        const username = await storage.getItem('username');
        if (token && username) {
          setAuth(token, username);
        }
      } finally {
        setIsLoading(false);
        setLoaded();
      }
    })();
  }, [setAuth, setLoaded]);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      logoutStore();
    });

    return () => {
      setUnauthorizedHandler(null);
    };
  }, [logoutStore]);

  const login = useCallback(async (username: string, password: string) => {
    const data = await loginApi(username, password);
    await storage.setItem('jwt', data.token);
    await storage.setItem('username', data.username);
    setAuth(data.token, data.username);
  }, [setAuth]);

  const register = useCallback(async (username: string, password: string) => {
    const data = await registerApi(username, password);
    await storage.setItem('jwt', data.token);
    await storage.setItem('username', data.username);
    setAuth(data.token, data.username);
  }, [setAuth]);

  const logout = useCallback(async () => {
    await storage.deleteItem('jwt');
    await storage.deleteItem('username');
    logoutStore();
  }, [logoutStore]);

  return (
    <AuthContext.Provider value={{ isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
