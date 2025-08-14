import React, { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { CircularProgress, Box, Typography } from '@mui/material';
import Layout from './components/Layout/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import SessionProvider from './components/SessionProvider';
import AuthInitializer from './components/AuthInitializer';

// Lazy load page components for code splitting
const HomePage = lazy(() => import('./pages/HomePage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const LogoutPage = lazy(() => import('./pages/LogoutPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const StockSearchPage = lazy(() => import('./pages/StockSearchPage'));
const ComparisonPage = lazy(() => import('./pages/ComparisonPage'));
const SectorPage = lazy(() => import('./pages/SectorPage'));
const StorePage = lazy(() => import('./pages/StorePage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const HelpPage = lazy(() => import('./pages/HelpPage'));

// Loading component with accessibility support
const PageLoader = () => (
  <Box
    role="status"
    aria-live="polite"
    sx={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      backgroundColor: '#f5f5f5',
    }}
  >
    <CircularProgress 
      size={60} 
      thickness={4} 
      aria-label="Loading page"
    />
    <Typography 
      variant="srOnly" 
      component="span"
      sx={{ position: 'absolute', left: '-9999px' }}
    >
      Loading page content...
    </Typography>
  </Box>
);

// 404 Not Found component
const NotFound = () => (
  <Box
    sx={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      backgroundColor: '#f5f5f5',
    }}
  >
    <Typography variant="h1" component="h1" gutterBottom>
      404
    </Typography>
    <Typography variant="h5" component="h2" gutterBottom>
      Page Not Found
    </Typography>
    <Typography variant="body1" color="text.secondary" paragraph>
      The page you are looking for doesn't exist.
    </Typography>
    <Navigate to="/dashboard" replace />
  </Box>
);

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1a1a2e',
    },
    secondary: {
      main: '#16213e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AuthInitializer>
          <SessionProvider>
            <Suspense fallback={<PageLoader />}>
              <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<HomePage />} />
              
              {/* Public routes */}
              <Route path="login" element={<LoginPage />} />
              <Route path="register" element={<RegisterPage />} />
              <Route path="logout" element={<LogoutPage />} />
              <Route path="help" element={<HelpPage />} />
              
              {/* Protected routes */}
              <Route element={<ProtectedRoute />}>
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="stocks" element={<StockSearchPage />} />
                <Route path="compare" element={<ComparisonPage />} />
                <Route path="sectors" element={<SectorPage />} />
                <Route path="store" element={<StorePage />} />
                <Route path="settings" element={<SettingsPage />} />
              </Route>
              
              {/* Catch-all route for 404 */}
              <Route path="*" element={<NotFound />} />
            </Route>
              </Routes>
            </Suspense>
          </SessionProvider>
        </AuthInitializer>
      </Router>
    </ThemeProvider>
  );
}

export default App;