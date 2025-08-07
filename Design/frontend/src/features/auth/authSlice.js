import { createSlice } from '@reduxjs/toolkit';

const authSlice = createSlice({
  name: 'auth',
  initialState: {
    user: null,
    token: localStorage.getItem('token'),
    refreshToken: localStorage.getItem('refreshToken'),
  },
  reducers: {
    setCredentials: (state, action) => {
      const { user, access, refresh } = action.payload;
      state.user = user;
      state.token = access;
      state.refreshToken = refresh;
      
      // Persist tokens to localStorage
      if (access) {
        localStorage.setItem('token', access);
      }
      if (refresh) {
        localStorage.setItem('refreshToken', refresh);
      }
    },
    logout: (state) => {
      state.user = null;
      state.token = null;
      state.refreshToken = null;
      
      // Clear localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
    },
  },
});

export const { setCredentials, logout } = authSlice.actions;

export default authSlice.reducer;

export const selectCurrentUser = (state) => state.auth.user;
export const selectCurrentToken = (state) => state.auth.token;