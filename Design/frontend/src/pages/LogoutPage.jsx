import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  LinearProgress,
  Alert,
  Button,
} from '@mui/material';
import { CheckCircle, LogoutOutlined } from '@mui/icons-material';

const LogoutPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [countdown, setCountdown] = useState(10);
  const [redirectCancelled, setRedirectCancelled] = useState(false);

  // Get logout reason from state or default to 'manual'
  const logoutReason = location.state?.reason || 'manual';
  const sessionExpired = logoutReason === 'timeout' || logoutReason === 'expired';

  useEffect(() => {
    // Countdown timer for automatic redirect
    const timer = setInterval(() => {
      if (!redirectCancelled) {
        setCountdown((prev) => {
          if (prev <= 1) {
            navigate('/');
            return 0;
          }
          return prev - 1;
        });
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [navigate, redirectCancelled]);

  const handleStayOnPage = () => {
    setRedirectCancelled(true);
  };

  const handleGoHome = () => {
    navigate('/');
  };

  const handleGoLogin = () => {
    navigate('/login');
  };

  const getLogoutMessage = () => {
    switch (logoutReason) {
      case 'timeout':
        return {
          title: 'Session Expired',
          message: 'Your session has expired due to inactivity for security reasons.',
          icon: <LogoutOutlined sx={{ fontSize: 60, color: 'warning.main' }} />
        };
      case 'expired':
        return {
          title: 'Session Ended',
          message: 'Your session has ended. Please log in again to continue.',
          icon: <LogoutOutlined sx={{ fontSize: 60, color: 'warning.main' }} />
        };
      case 'manual':
      default:
        return {
          title: 'Logged Out Successfully',
          message: 'You have been successfully logged out. Thank you for using VoyageurCompass.',
          icon: <CheckCircle sx={{ fontSize: 60, color: 'success.main' }} />
        };
    }
  };

  const { title, message, icon } = getLogoutMessage();
  const progressValue = ((10 - countdown) / 10) * 100;

  return (
    <Container component="main" maxWidth="sm">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          minHeight: '60vh',
          justifyContent: 'center',
        }}
      >
        <Paper 
          elevation={3} 
          sx={{ 
            padding: 4, 
            width: '100%', 
            textAlign: 'center',
            background: sessionExpired ? 'linear-gradient(145deg, #fff3e0 0%, #ffffff 100%)' : 'linear-gradient(145deg, #e8f5e8 0%, #ffffff 100%)'
          }}
        >
          {/* Icon */}
          <Box sx={{ mb: 3 }}>
            {icon}
          </Box>

          {/* Title */}
          <Typography component="h1" variant="h4" gutterBottom>
            {title}
          </Typography>

          {/* Message */}
          <Typography variant="body1" color="textSecondary" paragraph sx={{ mb: 3 }}>
            {message}
          </Typography>

          {/* Session expired alert */}
          {sessionExpired && (
            <Alert severity="warning" sx={{ mb: 3, textAlign: 'left' }}>
              <Typography variant="body2">
                For your security, we automatically log you out after 15 minutes of inactivity. 
                Your data is safe and you can log back in at any time.
              </Typography>
            </Alert>
          )}

          {/* Countdown section */}
          {!redirectCancelled && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Redirecting to homepage in {countdown} seconds...
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={progressValue} 
                sx={{ mt: 1, height: 6, borderRadius: 3 }}
              />
            </Box>
          )}

          {/* Action buttons */}
          <Box 
            sx={{ 
              display: 'flex', 
              gap: 2, 
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}
          >
            {!redirectCancelled ? (
              <>
                <Button
                  variant="outlined"
                  onClick={handleStayOnPage}
                  sx={{ minWidth: 120 }}
                >
                  Stay Here
                </Button>
                <Button
                  variant="contained"
                  onClick={handleGoHome}
                  sx={{ minWidth: 120 }}
                >
                  Go to Homepage
                </Button>
              </>
            ) : (
              <>
                <Button
                  variant="outlined"
                  onClick={handleGoHome}
                  sx={{ minWidth: 120 }}
                >
                  Homepage
                </Button>
                <Button
                  variant="contained"
                  onClick={handleGoLogin}
                  sx={{ minWidth: 120 }}
                >
                  Login Again
                </Button>
              </>
            )}
          </Box>

          {/* Additional info */}
          <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid #e0e0e0' }}>
            <Typography variant="caption" color="textSecondary" display="block">
              VoyageurCompass - Your Financial Navigation Partner
            </Typography>
            <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 0.5 }}>
              Questions? Visit our Help section or contact support.
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default LogoutPage;