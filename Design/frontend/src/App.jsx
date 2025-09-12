import React, { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { CircularProgress, Box, Typography } from '@mui/material';
import Layout from './components/Layout/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import SessionProvider from './components/SessionProvider';
import AuthInitializer from './components/AuthInitializer';
import { ThemeModeProvider } from './theme/ThemeModeContext.jsx';

// Lazy-loaded page components for code splitting optimisation
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
const AnalysisResultsPage = lazy(() => import('./pages/AnalysisResultsPage'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));

// Accessible loading component for page transitions
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

// HTTP 404 error page component
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

function App() {
  return (
    <ThemeModeProvider>
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
              
              {/* Authentication-protected routes */}
              <Route element={<ProtectedRoute />}>
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="stocks" element={<StockSearchPage />} />
                <Route path="reports" element={<ReportsPage />} />
                <Route path="compare" element={<ComparisonPage />} />
                <Route path="sectors" element={<SectorPage />} />
                <Route path="store" element={<StorePage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="analysis/:analysisId" element={<AnalysisResultsPage />} />
              </Route>
              
              {/* Catch-all route for 404 */}
              <Route path="*" element={<NotFound />} />
            </Route>
              </Routes>
            </Suspense>
          </SessionProvider>
        </AuthInitializer>
      </Router>
    </ThemeModeProvider>
  );
}

export default App;
