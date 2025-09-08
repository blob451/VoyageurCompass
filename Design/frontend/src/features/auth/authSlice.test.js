/**
 * Tests for authSlice Redux slice
 */

import { describe, it, expect } from 'vitest'
import authReducer, { 
  setCredentials, 
  logout, 
  startValidation, 
  validationSuccess, 
  validationFailure,
  clearValidationError
} from './authSlice'

describe('authSlice', () => {
  const initialState = {
    user: null,
    token: null,
    refreshToken: null,
    isValidating: false,
    validationError: null,
    lastValidated: null
  }

  describe('initial state', () => {
    it('should return the initial state', () => {
      expect(authReducer(undefined, { type: 'unknown' })).toEqual(initialState)
    })
  })

  describe('authentication actions', () => {
    it('should handle setCredentials', () => {
      const mockUser = { username: 'testuser', email: 'test@example.com' }
      const mockTokens = { access: 'access-token', refresh: 'refresh-token' }
      
      const action = setCredentials({ user: mockUser, access: mockTokens.access, refresh: mockTokens.refresh })
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        user: mockUser,
        token: mockTokens.access,
        refreshToken: mockTokens.refresh,
        isValidating: false,
        validationError: null,
        lastValidated: null
      })
    })

    it('should handle startValidation', () => {
      const action = startValidation()
      const state = authReducer(initialState, action)
      
      expect(state).toEqual({
        ...initialState,
        isValidating: true,
        validationError: null
      })
    })

    it('should handle validationSuccess', () => {
      const mockUser = { username: 'testuser', email: 'test@example.com' }
      const validatingState = {
        ...initialState,
        isValidating: true,
        token: 'existing-token'
      }
      
      const action = validationSuccess({ user: mockUser })
      const state = authReducer(validatingState, action)
      
      expect(state).toEqual({
        user: mockUser,
        token: 'existing-token',
        refreshToken: null,
        isValidating: false,
        validationError: null,
        lastValidated: expect.any(Number)
      })
    })

    it('should handle validationFailure', () => {
      const errorMessage = 'Token validation failed'
      const validatingState = {
        ...initialState,
        isValidating: true,
        token: 'invalid-token'
      }
      
      const action = validationFailure({ error: errorMessage })
      const state = authReducer(validatingState, action)
      
      expect(state).toEqual({
        user: null,
        token: null,
        refreshToken: null,
        isValidating: false,
        validationError: errorMessage,
        lastValidated: null
      })
    })
  })

  describe('error handling', () => {
    it('should handle clearValidationError', () => {
      const errorState = {
        ...initialState,
        validationError: 'Some validation error'
      }
      
      const action = clearValidationError()
      const state = authReducer(errorState, action)
      
      expect(state).toEqual({
        ...errorState,
        validationError: null
      })
    })
  })

  describe('logout action', () => {
    it('should handle logout', () => {
      const authenticatedState = {
        user: { username: 'testuser' },
        token: 'access-token',
        refreshToken: 'refresh-token',
        isValidating: false,
        validationError: null,
        lastValidated: Date.now()
      }
      
      const action = logout({ reason: 'manual' })
      const state = authReducer(authenticatedState, action)
      
      expect(state).toEqual(initialState)
    })
  })

  describe('token management', () => {
    it('should handle credentials update', () => {
      const mockUser = { username: 'testuser' }
      const mockTokens = { access: 'new-access', refresh: 'new-refresh' }
      
      const action = setCredentials({ user: mockUser, access: mockTokens.access, refresh: mockTokens.refresh })
      const state = authReducer(initialState, action)
      
      expect(state.user).toEqual(mockUser)
      expect(state.token).toBe(mockTokens.access)
      expect(state.refreshToken).toBe(mockTokens.refresh)
    })
  })

  describe('state immutability', () => {
    it('should not mutate the original state', () => {
      const originalState = {
        user: { username: 'testuser' },
        token: 'access-token',
        refreshToken: 'refresh-token',
        isValidating: false,
        validationError: null,
        lastValidated: Date.now()
      }
      
      const stateCopy = JSON.parse(JSON.stringify(originalState))
      const newState = authReducer(originalState, logout({ reason: 'manual' }))
      
      // Original state should be unchanged
      expect(originalState).toEqual(stateCopy)
      
      // New state should be different
      expect(newState).not.toBe(originalState)
      expect(newState.user).toBe(null)
      expect(newState.token).toBe(null)
    })
  })
})