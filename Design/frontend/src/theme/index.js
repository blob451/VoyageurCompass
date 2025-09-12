import { createTheme } from '@mui/material/styles';

// App theme factory with teal as secondary and support for light/dark modes
export const getAppTheme = ({ mode = 'light', direction = 'ltr' } = {}) =>
  createTheme({
    direction,
    palette: {
      mode,
      // Keep a deep navy as primary for structure and contrast
      primary: {
        main: '#0D1B2A',
        contrastText: '#FFFFFF',
      },
      // Teal-forward secondary palette as requested
      secondary: {
        main: '#006D77',
        dark: '#035C65',
        light: '#0A9396',
        contrastText: '#FFFFFF',
      },
      background: {
        default: mode === 'light' ? '#F7FAFC' : '#0B1220',
        paper: mode === 'light' ? '#FFFFFF' : '#0F172A',
      },
      text: {
        primary: mode === 'light' ? '#0B1F2A' : '#E5E7EB',
        secondary: mode === 'light' ? '#4B5563' : '#B6C2CF',
      },
      success: { main: '#2E7D32' },
      warning: { main: '#ED6C02' },
      error: { main: '#D32F2F' },
      info: { main: '#0288D1' },
      contrastThreshold: 3,
      tonalOffset: 0.1,
    },
    shape: { borderRadius: 12 },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h2: { fontWeight: 700 },
      h3: { fontWeight: 700 },
      h4: { fontWeight: 600 },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
      MuiButton: { styleOverrides: { root: { borderRadius: 10 } } },
      MuiCard: { styleOverrides: { root: { borderRadius: 12 } } },
    },
  });

export default getAppTheme;

