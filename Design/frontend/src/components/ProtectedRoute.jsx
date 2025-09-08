import React, { useEffect } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { Box, CircularProgress, Typography, Alert, Button } from '@mui/material';
import { 
  selectCurrentToken, 
  selectCurrentUser, 
  selectIsValidating, 
  selectValidationError,
  startValidation,
  validationSuccess,
  validationFailure,
  logout
} from '../features/auth/authSlice';
import { useValidateTokenQuery } from '../features/api/apiSlice';
import { isTokenExpired } from '../utils/tokenValidation';

const ProtectedRoute = () => {
  const dispatch = useDispatch();
  const location = useLocation();
  const token = useSelector(selectCurrentToken);
  const user = useSelector(selectCurrentUser);
  const isValidating = useSelector(selectIsValidating);
  const validationError = useSelector(selectValidationError);
  
  // Only call validation API if we have a token
  const { 
    data: validationData, 
    error: validationApiError, 
    isLoading: isLoadingValidation,
    refetch: refetchValidation 
  } = useValidateTokenQuery(undefined, {
    skip: !token,
    refetchOnMountOrArgChange: true,
  });

  useEffect(() => {
    // If no token, redirect immediately
    if (!token) {
      return;
    }

    // Check if token is expired on client side first
    if (isTokenExpired(token)) {
      console.log('Token expired on client side, logging out');
      dispatch(logout({ reason: 'expired' }));
      return;
    }

    // If we have a token but no user, start validation
    if (token && !user && !isValidating) {
      dispatch(startValidation());
    }
  }, [token, user, isValidating, dispatch]);

  useEffect(() => {
    // Handle validation API response
    if (validationData) {
      dispatch(validationSuccess({ user: validationData.user }));
    }
  }, [validationData, dispatch]);

  useEffect(() => {
    // Handle validation API error
    if (validationApiError) {
      console.log('Token validation failed:', validationApiError);
      dispatch(validationFailure({ error: validationApiError.data?.detail || 'Invalid token' }));
    }
  }, [validationApiError, dispatch]);

  // No token - redirect to login
  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Token expired on client side - redirect to logout page
  if (isTokenExpired(token)) {
    return <Navigate to="/logout" state={{ reason: 'expired' }} replace />;
  }

  // Validation in progress
  if (isValidating || isLoadingValidation) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          gap: 2
        }}
      >
        <CircularProgress size={60} />
        <Typography variant="h6" color="textSecondary">
          Verifying authentication...
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Please wait while we validate your session.
        </Typography>
      </Box>
    );
  }

  // Validation failed - show error with option to retry or logout
  if (validationError) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          gap: 2,
          maxWidth: 500,
          margin: '0 auto',
          padding: 3
        }}
      >
        <Alert severity="error" sx={{ width: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Authentication Error
          </Typography>
          <Typography variant="body2" paragraph>
            {validationError}
          </Typography>
          <Typography variant="body2">
            Your session may have expired or become invalid. Please log in again.
          </Typography>
        </Alert>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            variant="outlined" 
            onClick={() => refetchValidation()}
            disabled={isLoadingValidation}
          >
            Retry
          </Button>
          <Button 
            variant="contained" 
            onClick={() => {
              dispatch(logout({ reason: 'validation_failed' }));
            }}
          >
            Go to Login
          </Button>
        </Box>
      </Box>
    );
  }

  // Valid token and user - render protected content
  if (token && user) {
    return <Outlet />;
  }

  // Fallback - should not reach here, but redirect to login just in case
  return <Navigate to="/login" state={{ from: location }} replace />;
};

export default ProtectedRoute;