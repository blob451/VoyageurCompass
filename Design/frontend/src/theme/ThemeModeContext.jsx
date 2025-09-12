import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { getAppTheme } from './index.js';

const ThemeModeContext = createContext({
  mode: 'light',
  toggleMode: () => {},
  setMode: () => {},
});

export const ThemeModeProvider = ({ children }) => {
  const prefersDark = typeof window !== 'undefined'
    ? window.matchMedia('(prefers-color-scheme: dark)').matches
    : false;

  const [mode, setMode] = useState(() => {
    try {
      const saved = localStorage.getItem('themeMode');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch (_) {}
    return prefersDark ? 'dark' : 'light';
  });

  useEffect(() => {
    try { localStorage.setItem('themeMode', mode); } catch (_) {}
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-color-scheme', mode);
    }
  }, [mode]);

  const toggleMode = () => setMode((prev) => (prev === 'light' ? 'dark' : 'light'));

  const theme = useMemo(() => getAppTheme({ mode }), [mode]);

  return (
    <ThemeModeContext.Provider value={{ mode, setMode, toggleMode }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
};

export const useThemeMode = () => useContext(ThemeModeContext);

