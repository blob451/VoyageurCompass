import { createSlice } from '@reduxjs/toolkit';

const authSlice = createSlice({
  name: 'auth',
  initialState: {
    isAuthenticated: false,
    user: null,
    tokens: null,
    loading: false,
    error: null
  },
  reducers: {
    loginStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    loginSuccess: (state, action) => {
      const { user, tokens } = action.payload;
      state.isAuthenticated = true;
      state.user = user;
      state.tokens = tokens;
      state.loading = false;
      state.error = null;
      
      // Persist tokens to localStorage
      if (tokens?.access) {
        localStorage.setItem('token', tokens.access);
      }
      if (tokens?.refresh) {
        localStorage.setItem('refreshToken', tokens.refresh);
      }
    },
    loginFailure: (state, action) => {
      state.isAuthenticated = false;
      state.user = null;
      state.tokens = null;
      state.loading = false;
      state.error = action.payload;
    },
    registerStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    registerSuccess: (state, action) => {
      const { user, tokens } = action.payload;
      state.isAuthenticated = true;
      state.user = user;
      state.tokens = tokens;
      state.loading = false;
      state.error = null;
      
      // Persist tokens to localStorage
      if (tokens?.access) {
        localStorage.setItem('token', tokens.access);
      }
      if (tokens?.refresh) {
        localStorage.setItem('refreshToken', tokens.refresh);
      }
    },
    registerFailure: (state, action) => {
      state.isAuthenticated = false;
      state.user = null;
      state.tokens = null;
      state.loading = false;
      state.error = action.payload;
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.user = null;
      state.tokens = null;
      state.loading = false;
      state.error = null;
      
      // Clear localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
    },
    updateProfile: (state, action) => {
      if (state.isAuthenticated && state.user) {
        state.user = { ...state.user, ...action.payload };
      }
    },
    clearError: (state) => {
      state.error = null;
    },
    setTokens: (state, action) => {
      state.tokens = action.payload;
      
      // Persist tokens to localStorage
      if (action.payload?.access) {
        localStorage.setItem('token', action.payload.access);
      }
      if (action.payload?.refresh) {
        localStorage.setItem('refreshToken', action.payload.refresh);
      }
    },
    // Legacy action for backward compatibility
    setCredentials: (state, action) => {
      const { user, access, refresh } = action.payload;
      state.isAuthenticated = true;
      state.user = user;
      state.tokens = { access, refresh };
      
      // Persist tokens to localStorage
      if (access) {
        localStorage.setItem('token', access);
      }
      if (refresh) {
        localStorage.setItem('refreshToken', refresh);
      }
    },
  },
});

export const { 
  loginStart, 
  loginSuccess, 
  loginFailure, 
  registerStart, 
  registerSuccess, 
  registerFailure,
  logout, 
  updateProfile, 
  clearError, 
  setTokens,
  setCredentials 
} = authSlice.actions;

export default authSlice.reducer;

export const selectCurrentUser = (state) => state.auth.user;
export const selectCurrentToken = (state) => state.auth.tokens?.access;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;
export const selectAuthLoading = (state) => state.auth.loading;
export const selectAuthError = (state) => state.auth.error;
export const selectTokens = (state) => state.auth.tokens;