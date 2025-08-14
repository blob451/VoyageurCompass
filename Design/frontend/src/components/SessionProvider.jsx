import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { logout, selectCurrentToken, selectCurrentUser } from '../features/auth/authSlice';
import { useLogoutMutation } from '../features/api/apiSlice';
import SessionWarningDialog from './SessionWarningDialog';
import sessionManager, { handlePageUnload, handlePageVisibility } from '../utils/sessionManager';

const SessionProvider = ({ children }) => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const token = useSelector(selectCurrentToken);
  const user = useSelector(selectCurrentUser);
  const [logoutApi] = useLogoutMutation();
  
  const [showWarning, setShowWarning] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize session management when user is authenticated
  useEffect(() => {
    if (token && user && !isInitialized) {
      
      sessionManager.init({
        onWarning: () => {
          setShowWarning(true);
        },
        onTimeout: () => {
          handleLogout('timeout');
        },
        onActivity: () => {
          // Optional: could show a subtle indicator of activity
        }
      });
      
      setIsInitialized(true);
    } else if (!token && isInitialized) {
      // User logged out, cleanup session manager
      sessionManager.cleanup();
      setIsInitialized(false);
      setShowWarning(false);
    }

    return () => {
      if (!token) {
        sessionManager.cleanup();
      }
    };
  }, [token, user, isInitialized]);

  // Handle page visibility changes
  useEffect(() => {
    if (!token) return;

    const cleanup = handlePageVisibility(
      () => {
        // Page became visible - check if session is still valid
        if (sessionManager.isSessionActive()) {
          sessionManager.resetSession();
        } else {
          handleLogout('expired');
        }
      },
      () => {
        // Page became hidden - no action needed for now
      }
    );

    return cleanup;
  }, [token]);

  // Handle page unload (browser close/refresh)
  useEffect(() => {
    if (!token) return;

    const cleanup = handlePageUnload((event) => {
      // Call logout API in background (best effort)
      try {
        const refreshToken = localStorage.getItem('refreshToken');
        if (refreshToken) {
          // Use navigator.sendBeacon for reliable logout on page unload
          const logoutData = JSON.stringify({ refresh_token: refreshToken });
          const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
          
          navigator.sendBeacon(
            `${apiUrl}/auth/logout/`,
            new Blob([logoutData], { type: 'application/json' })
          );
        }
      } catch (error) {
        console.warn('Failed to send logout beacon:', error);
      }
    });

    return cleanup;
  }, [token]);

  const handleLogout = async (reason = 'manual') => {
    try {
      setShowWarning(false);
      
      // Call logout API if we have a refresh token
      const refreshToken = localStorage.getItem('refreshToken');
      if (refreshToken) {
        try {
          await logoutApi(refreshToken).unwrap();
        } catch (error) {
          console.warn('Logout API call failed:', error);
          // Continue with client-side logout even if API fails
        }
      }
      
      // Dispatch logout action with reason
      dispatch(logout({ reason }));
      
      // Navigate to logout page with reason
      navigate('/logout', { 
        replace: true, 
        state: { reason } 
      });
      
    } catch (error) {
      console.error('Logout error:', error);
      // Fallback: force logout anyway
      dispatch(logout({ reason }));
      navigate('/logout', { 
        replace: true, 
        state: { reason: 'error' } 
      });
    }
  };

  const handleExtendSession = () => {
    console.log('Extending session');
    setShowWarning(false);
    sessionManager.extendSession();
  };

  const handleWarningLogout = () => {
    handleLogout('manual');
  };

  // Don't render session warning if user is not authenticated
  if (!token || !user) {
    return <>{children}</>;
  }

  return (
    <>
      {children}
      
      <SessionWarningDialog
        open={showWarning}
        onExtendSession={handleExtendSession}
        onLogout={handleWarningLogout}
        warningDuration={3 * 60 * 1000} // 3 minutes
      />
    </>
  );
};

export default SessionProvider;