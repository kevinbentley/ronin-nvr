/**
 * Authentication context for managing user login state.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import { api, setAuthChangeCallback, type User } from '../services/api';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(api.isAuthenticated());
  const [isLoading, setIsLoading] = useState(true);

  // Fetch user info if we have a token
  useEffect(() => {
    const fetchUser = async () => {
      if (api.isAuthenticated()) {
        try {
          const userData = await api.getMe();
          setUser(userData);
          setIsAuthenticated(true);
        } catch {
          // Token invalid - clear it
          api.logout();
          setUser(null);
          setIsAuthenticated(false);
        }
      }
      setIsLoading(false);
    };

    fetchUser();
  }, []);

  // Listen for auth changes from API client (e.g., 401 errors)
  useEffect(() => {
    setAuthChangeCallback((authenticated: boolean) => {
      setIsAuthenticated(authenticated);
      if (!authenticated) {
        setUser(null);
      }
    });

    return () => {
      setAuthChangeCallback(() => {});
    };
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    await api.login({ username, password });
    const userData = await api.getMe();
    setUser(userData);
    setIsAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
