import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Box, CircularProgress, Typography } from '@mui/material';
import { 
  selectCurrentToken,
  selectCurrentUser,
  startValidation,
  validationSuccess,
  validationFailure,
  logout
} from '../features/auth/authSlice';
import { useValidateTokenQuery } from '../features/api/apiSlice';
import { getValidTokensFromStorage, cleanupInvalidTokens } from '../utils/tokenValidation';
import { authLogger } from '../utils/logger';

const AuthInitializer = ({ children }) => {
  const dispatch = useDispatch();
  const token = useSelector(selectCurrentToken);
  const user = useSelector(selectCurrentUser);
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Validate token with backend if we have one
  const { 
    data: validationData, 
    error: validationError, 
    isLoading: isValidating 
  } = useValidateTokenQuery(undefined, {
    skip: !token || !!user, // Skip if no token or already have user
  });

  useEffect(() => {
    const initializeAuth = async () => {
      authLogger.info('Initializing authentication');
      
      // Clean up any invalid tokens from localStorage
      const tokensRemoved = cleanupInvalidTokens();
      if (tokensRemoved > 0) {
        authLogger.info('Token cleanup', { tokensRemoved });
      }
      
      // Get valid tokens after cleanup
      const { accessToken, hasExpiredTokens } = getValidTokensFromStorage();
      
      if (hasExpiredTokens) {
        authLogger.info('Found expired tokens, clearing auth state');
        dispatch(logout({ reason: 'expired' }));
        setIsInitialized(true);
        return;
      }
      
      if (!accessToken) {
        authLogger.info('No valid tokens found, user not authenticated');
        setIsInitialized(true);
        return;
      }
      
      if (accessToken && !user) {
        authLogger.info('Valid token found, validating with backend');
        dispatch(startValidation());
        // API call will be made by useValidateTokenQuery hook
      } else {
        authLogger.info('User already authenticated');
        setIsInitialized(true);
      }
    };

    if (!isInitialized) {
      initializeAuth();
    }
  }, [dispatch, isInitialized, user]); // Include user dependency for proper hook behavior

  useEffect(() => {
    // Handle successful validation
    if (validationData && !user) {
      authLogger.info('Token validation successful', { user: validationData.user });
      dispatch(validationSuccess({ user: validationData.user }));
      setIsInitialized(true);
    }
  }, [validationData, user, dispatch]);

  useEffect(() => {
    // Handle validation failure
    if (validationError) {
      authLogger.error('Token validation failed', { error: validationError });
      dispatch(validationFailure({ 
        error: validationError.data?.detail || 'Token validation failed' 
      }));
      setIsInitialized(true);
    }
  }, [validationError, dispatch]);

  // If we have a token but validation is in progress, show loading
  if (token && !user && (isValidating || !isInitialized)) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          gap: 2,
          backgroundColor: '#f5f5f5'
        }}
      >
        <CircularProgress size={80} thickness={4} />
        <Typography variant="h5" color="textSecondary" gutterBottom>
          VoyageurCompass
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Initializing your session...
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Please wait while we verify your authentication.
        </Typography>
      </Box>
    );
  }

  // Initialization complete, render the app
  return <>{children}</>;
};

export default AuthInitializer;