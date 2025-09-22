import { createSlice } from '@reduxjs/toolkit';
import { cleanupInvalidTokens, getValidTokensFromStorage } from '../../utils/tokenValidation';

// Clean up any invalid tokens on initialization
cleanupInvalidTokens();

// Get valid tokens for initial state
const { accessToken, refreshToken, hasValidTokens, hasExpiredTokens } = getValidTokensFromStorage();

// Log initial token state
console.log('[AUTH] Initial token state:', {
  hasValidTokens,
  hasExpiredTokens,
  accessTokenLength: accessToken?.length || 0,
  refreshTokenLength: refreshToken?.length || 0
});

const authSlice = createSlice({
  name: 'auth',
  initialState: {
    user: null,
    token: accessToken,
    refreshToken: refreshToken,
    isValidating: false,
    validationError: null,
    lastValidated: null,
  },
  reducers: {
    setCredentials: (state, action) => {
      const { user, access, refresh } = action.payload;
      state.user = user;
      state.token = access;
      state.refreshToken = refresh;

      // Persist tokens to localStorage with validation
      if (access) {
        localStorage.setItem('token', access);
        console.log('[AUTH] Access token stored in localStorage');
      }
      if (refresh) {
        localStorage.setItem('refreshToken', refresh);
        console.log('[AUTH] Refresh token stored in localStorage');
      }

      // Log token storage status
      const storedAccess = localStorage.getItem('token');
      const storedRefresh = localStorage.getItem('refreshToken');
      console.log('[AUTH] Token persistence status:', {
        accessStored: !!storedAccess,
        refreshStored: !!storedRefresh,
        accessLength: storedAccess?.length || 0,
        refreshLength: storedRefresh?.length || 0
      });
    },
    logout: (state, action) => {
      // Store logout reason for the logout page
      const reason = action.payload?.reason || 'manual';
      
      state.user = null;
      state.token = null;
      state.refreshToken = null;
      state.isValidating = false;
      state.validationError = null;
      state.lastValidated = null;
      
      // Clear localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      
      // Store logout reason temporarily for logout page
      sessionStorage.setItem('logoutReason', reason);
    },
    startValidation: (state) => {
      state.isValidating = true;
      state.validationError = null;
    },
    validationSuccess: (state, action) => {
      const { user } = action.payload;
      state.isValidating = false;
      state.validationError = null;
      state.user = user;
      state.lastValidated = Date.now();
    },
    validationFailure: (state, action) => {
      state.isValidating = false;
      state.validationError = action.payload?.error || 'Validation failed';
      state.user = null;
      state.token = null;
      state.refreshToken = null;
      
      // Clear invalid tokens from localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
    },
    clearValidationError: (state) => {
      state.validationError = null;
    },
  },
});

export const { 
  setCredentials, 
  logout, 
  startValidation, 
  validationSuccess, 
  validationFailure, 
  clearValidationError 
} = authSlice.actions;

export default authSlice.reducer;

export const selectCurrentUser = (state) => state.auth.user;
export const selectCurrentToken = (state) => state.auth.token;
export const selectIsValidating = (state) => state.auth.isValidating;
export const selectValidationError = (state) => state.auth.validationError;
export const selectLastValidated = (state) => state.auth.lastValidated;
export const selectIsAuthenticated = (state) => !!(state.auth.user && state.auth.token);