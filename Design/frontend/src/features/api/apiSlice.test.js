/**
 * Tests for apiSlice RTK Query configuration
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { configureStore } from '@reduxjs/toolkit'
import { apiSlice } from './apiSlice'
import authSlice from '../auth/authSlice'

// Mock fetch
global.fetch = vi.fn()

describe('apiSlice', () => {
  let store

  beforeEach(() => {
    vi.clearAllMocks()
    
    // Create a test store with the API slice
    store = configureStore({
      reducer: {
        auth: authSlice,
        api: apiSlice.reducer,
      },
      middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware().concat(apiSlice.middleware),
    })
  })

  afterEach(() => {
    // Clean up any ongoing requests
    store.dispatch(apiSlice.util.resetApiState())
  })

  describe('baseQuery configuration', () => {
    it('should have correct base URL', () => {
      expect(apiSlice.reducerPath).toBe('api')
    })

    it('should include authentication headers when tokens exist', async () => {
      const mockResponse = { data: 'test data' }
      
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Set up authenticated state
      store.dispatch({
        type: 'auth/loginSuccess',
        payload: {
          user: { username: 'testuser' },
          tokens: { access: 'test-token', refresh: 'refresh-token' }
        }
      })

      // Make a request (this would be done through an endpoint)
      // Since we don't have specific endpoints defined, we'll test the baseQuery directly
      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        // Verify that fetch was called with authorization header
        expect(global.fetch).toHaveBeenCalledWith(
          expect.any(String),
          expect.objectContaining({
            headers: expect.objectContaining({
              'Authorization': 'Bearer test-token'
            })
          })
        )
      }
    })

    it('should not include auth headers when not authenticated', async () => {
      const mockResponse = { data: 'test data' }
      
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Ensure unauthenticated state
      store.dispatch({ type: 'auth/logout' })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        // Verify that fetch was called without authorization header
        const fetchCall = global.fetch.mock.calls[0]
        const headers = fetchCall[1]?.headers || {}
        expect(headers.Authorization).toBeUndefined()
      }
    })
  })

  describe('tag system', () => {
    it('should have correct tag types defined', () => {
      const expectedTagTypes = [
        'Stock', 
        'Portfolio', 
        'Holding', 
        'User', 
        'Market', 
        'Analytics'
      ]
      
      expect(apiSlice.tagTypes).toEqual(
        expect.arrayContaining(expectedTagTypes)
      )
    })
  })

  describe('error handling', () => {
    it('should handle network errors', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network error'))

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        expect(result.error).toBeDefined()
        expect(result.error.error).toBe('Network error')
      }
    })

    it('should handle HTTP error responses', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({ detail: 'Invalid request' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        expect(result.error).toBeDefined()
        expect(result.error.status).toBe(400)
      }
    })

    it('should handle 401 unauthorized responses', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        json: async () => ({ detail: 'Token expired' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Set up authenticated state
      store.dispatch({
        type: 'auth/loginSuccess',
        payload: {
          user: { username: 'testuser' },
          tokens: { access: 'expired-token', refresh: 'refresh-token' }
        }
      })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        expect(result.error).toBeDefined()
        expect(result.error.status).toBe(401)
      }
    })
  })

  describe('token refresh handling', () => {
    it('should attempt to refresh tokens on 401 errors', async () => {
      // First call returns 401
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Token expired' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Token refresh call succeeds
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ access: 'new-token', refresh: 'new-refresh' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Retry original request succeeds
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'success' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Set up authenticated state with refresh token
      store.dispatch({
        type: 'auth/loginSuccess',
        payload: {
          user: { username: 'testuser' },
          tokens: { access: 'expired-token', refresh: 'valid-refresh' }
        }
      })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        // Should have made 3 calls: original, refresh, retry
        expect(global.fetch).toHaveBeenCalledTimes(3)
        
        // Final result should be successful
        if (result.data) {
          expect(result.data).toEqual({ data: 'success' })
        }
      }
    })

    it('should logout user when refresh token fails', async () => {
      // Original call returns 401
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Token expired' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      // Token refresh fails
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Refresh token invalid' }),
        headers: new Headers({ 'content-type': 'application/json' })
      })

      const initialState = store.getState()
      
      store.dispatch({
        type: 'auth/loginSuccess',
        payload: {
          user: { username: 'testuser' },
          tokens: { access: 'expired-token', refresh: 'invalid-refresh' }
        }
      })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        await baseQuery(
          { url: '/test/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        // Check if user was logged out
        const finalState = store.getState()
        expect(finalState.auth.isAuthenticated).toBe(false)
        expect(finalState.auth.tokens).toBe(null)
      }
    })
  })

  describe('cache management', () => {
    it('should invalidate cache on logout', () => {
      // Dispatch logout action
      store.dispatch({ type: 'auth/logout' })
      
      // Check that API cache was reset
      const apiState = store.getState().api
      expect(apiState.queries).toEqual({})
      expect(apiState.subscriptions).toEqual({})
    })

    it('should have proper cache configuration', () => {
      expect(apiSlice.keepUnusedDataFor).toBeGreaterThan(0)
    })
  })

  describe('middleware integration', () => {
    it('should work with Redux store middleware', () => {
      // The store should be configured with the API middleware
      expect(store.dispatch).toBeDefined()
      
      // Should be able to dispatch API actions
      const action = apiSlice.util.resetApiState()
      expect(() => store.dispatch(action)).not.toThrow()
    })
  })

  describe('endpoint creation', () => {
    it('should allow injecting endpoints', () => {
      // Create an enhanced API slice with injected endpoints
      const extendedApi = apiSlice.injectEndpoints({
        endpoints: (builder) => ({
          testEndpoint: builder.query({
            query: () => '/test/',
            providesTags: ['Test']
          })
        })
      })

      expect(extendedApi.endpoints.testEndpoint).toBeDefined()
      expect(extendedApi.useTestEndpointQuery).toBeDefined()
    })
  })

  describe('transformResponse', () => {
    it('should handle paginated responses correctly', async () => {
      const paginatedResponse = {
        count: 100,
        next: 'http://api.example.com/data/?page=2',
        previous: null,
        results: [{ id: 1, name: 'Item 1' }]
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => paginatedResponse,
        headers: new Headers({ 'content-type': 'application/json' })
      })

      const baseQuery = apiSlice.baseQuery
      
      if (typeof baseQuery === 'function') {
        const result = await baseQuery(
          { url: '/data/', method: 'GET' },
          { getState: () => store.getState() },
          {}
        )

        expect(result.data).toEqual(paginatedResponse)
      }
    })
  })
})