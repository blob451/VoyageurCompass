/**
 * Tests for authSlice Redux slice
 */

import { describe, it, expect } from 'vitest'
import authReducer, { 
  loginStart, 
  loginSuccess, 
  loginFailure, 
  logout, 
  registerStart, 
  registerSuccess, 
  registerFailure,
  updateProfile,
  clearError,
  setTokens
} from './authSlice'

describe('authSlice', () => {
  const initialState = {
    isAuthenticated: false,
    user: null,
    tokens: null,
    loading: false,
    error: null
  }


  describe('initial state', () => {
    it('should return the initial state', () => {
      expect(authReducer(undefined, { type: 'unknown' })).toEqual(initialState)
    })
  })

  describe('login actions', () => {
    it('should handle loginStart', () => {
      const action = loginStart()
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        ...initialState,
        loading: true,
        error: null
      })
    })

    it('should handle loginSuccess', () => {
      const mockUser = { username: 'testuser', email: 'test@example.com' }
      const mockTokens = { access: 'access-token', refresh: 'refresh-token' }
      
      const action = loginSuccess({ user: mockUser, tokens: mockTokens })
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        isAuthenticated: true,
        user: mockUser,
        tokens: mockTokens,
        loading: false,
        error: null
      })
    })

    it('should handle loginFailure', () => {
      const errorMessage = 'Invalid credentials'
      const loadingState = {
        ...initialState,
        loading: true
      }
      
      const action = loginFailure(errorMessage)
      const state = authReducer(loadingState, action)
      
      expect(state).toEqual({
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: false,
        error: errorMessage
      })
    })
  })

  describe('register actions', () => {
    it('should handle registerStart', () => {
      const action = registerStart()
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        ...initialState,
        loading: true,
        error: null
      })
    })

    it('should handle registerSuccess', () => {
      const mockUser = { username: 'newuser', email: 'new@example.com' }
      const mockTokens = { access: 'access-token', refresh: 'refresh-token' }
      
      const action = registerSuccess({ user: mockUser, tokens: mockTokens })
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        isAuthenticated: true,
        user: mockUser,
        tokens: mockTokens,
        loading: false,
        error: null
      })
    })

    it('should handle registerFailure', () => {
      const errorMessage = 'Username already exists'
      const loadingState = {
        ...initialState,
        loading: true
      }
      
      const action = registerFailure(errorMessage)
      const state = authReducer(loadingState, action)
      
      expect(state).toEqual({
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: false,
        error: errorMessage
      })
    })
  })

  describe('logout action', () => {
    it('should handle logout', () => {
      const authenticatedState = {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'token', refresh: 'refresh' },
        loading: false,
        error: null
      }
      
      const action = logout()
      const state = authReducer(authenticatedState, action)
      
      expect(state).toEqual(initialState)
    })
  })

  describe('updateProfile action', () => {
    it('should handle updateProfile', () => {
      const authenticatedState = {
        isAuthenticated: true,
        user: { username: 'testuser', email: 'test@example.com', first_name: 'Test' },
        tokens: { access: 'token', refresh: 'refresh' },
        loading: false,
        error: null
      }
      
      const updatedFields = { first_name: 'Updated', last_name: 'User' }
      const action = updateProfile(updatedFields)
      const state = authReducer(authenticatedState, action)
      
      expect(state).toEqual({
        ...authenticatedState,
        user: {
          ...authenticatedState.user,
          ...updatedFields
        }
      })
    })

    it('should not update profile when user is not authenticated', () => {
      const updatedFields = { first_name: 'Updated' }
      const action = updateProfile(updatedFields)
      const state = authReducer(initialState, action)
      
      expect(state).toEqual(initialState)
    })
  })

  describe('clearError action', () => {
    it('should handle clearError', () => {
      const errorState = {
        ...initialState,
        error: 'Some error message'
      }
      
      const action = clearError()
      const state = authReducer(errorState, action)
      
      expect(state).toEqual({
        ...errorState,
        error: null
      })
    })
  })

  describe('setTokens action', () => {
    it('should handle setTokens', () => {
      const newTokens = { access: 'new-access', refresh: 'new-refresh' }
      const action = setTokens(newTokens)
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        ...initialState,
        tokens: newTokens
      })
    })

    it('should update existing tokens', () => {
      const authenticatedState = {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'old-access', refresh: 'old-refresh' },
        loading: false,
        error: null
      }
      
      const newTokens = { access: 'new-access', refresh: 'new-refresh' }
      const action = setTokens(newTokens)
      const state = authReducer(authenticatedState, action)
      
      expect(state).toEqual({
        ...authenticatedState,
        tokens: newTokens
      })
    })
  })

  describe('complex scenarios', () => {
    it('should handle multiple actions in sequence', () => {
      let state = authReducer(undefined, { type: 'unknown' })
      
      // Start login
      state = authReducer(state, loginStart())
      expect(state.loading).toBe(true)
      expect(state.error).toBe(null)
      
      // Login fails
      state = authReducer(state, loginFailure('Network error'))
      expect(state.loading).toBe(false)
      expect(state.error).toBe('Network error')
      expect(state.isAuthenticated).toBe(false)
      
      // Clear error
      state = authReducer(state, clearError())
      expect(state.error).toBe(null)
      
      // Successful login
      const user = { username: 'testuser' }
      const tokens = { access: 'token', refresh: 'refresh' }
      state = authReducer(state, loginSuccess({ user, tokens }))
      
      expect(state.isAuthenticated).toBe(true)
      expect(state.user).toEqual(user)
      expect(state.tokens).toEqual(tokens)
      expect(state.loading).toBe(false)
      expect(state.error).toBe(null)
    })

    it('should preserve user data when updating tokens', () => {
      const initialUser = { username: 'testuser', email: 'test@example.com' }
      const initialTokens = { access: 'access1', refresh: 'refresh1' }
      
      let state = authReducer(initialState, loginSuccess({ 
        user: initialUser, 
        tokens: initialTokens 
      }))
      
      const newTokens = { access: 'access2', refresh: 'refresh2' }
      state = authReducer(state, setTokens(newTokens))
      
      expect(state.user).toEqual(initialUser)
      expect(state.tokens).toEqual(newTokens)
      expect(state.isAuthenticated).toBe(true)
    })

    it('should handle profile updates correctly', () => {
      const initialUser = { 
        username: 'testuser', 
        email: 'test@example.com',
        first_name: 'Test',
        last_name: 'User'
      }
      
      let state = authReducer(initialState, loginSuccess({ 
        user: initialUser, 
        tokens: { access: 'token', refresh: 'refresh' }
      }))
      
      // Update only some fields
      state = authReducer(state, updateProfile({ 
        first_name: 'Updated',
        phone: '+1234567890' 
      }))
      
      expect(state.user).toEqual({
        username: 'testuser',
        email: 'test@example.com',
        first_name: 'Updated', // Updated
        last_name: 'User', // Preserved
        phone: '+1234567890' // Added
      })
    })
  })

  describe('error handling', () => {
    it('should handle various error types', () => {
      const testCases = [
        { action: loginFailure('Network error'), expectedError: 'Network error' },
        { action: registerFailure('Validation failed'), expectedError: 'Validation failed' },
        { action: loginFailure(null), expectedError: null },
        { action: registerFailure(''), expectedError: '' }
      ]
      
      testCases.forEach(({ action, expectedError }) => {
        const state = authReducer(initialState, action)
        expect(state.error).toBe(expectedError)
        expect(state.loading).toBe(false)
        expect(state.isAuthenticated).toBe(false)
      })
    })
  })

  describe('state immutability', () => {
    it('should not mutate the original state', () => {
      const originalState = {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'token' },
        loading: false,
        error: null
      }
      
      const stateCopy = { ...originalState }
      const newState = authReducer(originalState, logout())
      
      // Original state should be unchanged
      expect(originalState).toEqual(stateCopy)
      
      // New state should be different
      expect(newState).not.toBe(originalState)
      expect(newState.isAuthenticated).toBe(false)
    })

    it('should not mutate nested objects', () => {
      const originalUser = { username: 'testuser', profile: { age: 25 } }
      const originalState = {
        isAuthenticated: true,
        user: originalUser,
        tokens: { access: 'token' },
        loading: false,
        error: null
      }
      
      const newState = authReducer(originalState, updateProfile({ 
        profile: { age: 26, city: 'New York' }
      }))
      
      // Original user object should be unchanged
      expect(originalUser).toEqual({ username: 'testuser', profile: { age: 25 } })
      
      // New state should have updated user
      expect(newState.user.profile).toEqual({ age: 26, city: 'New York' })
    })
  })
})