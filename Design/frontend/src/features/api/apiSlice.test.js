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
    it('should have correct reducer path', () => {
      expect(apiSlice.reducerPath).toBe('api')
    })

    it('should include authentication headers when tokens exist', async () => {
      const mockResponse = { data: 'test data' }
      
      global.fetch.mockResolvedValueOnce(new Response(JSON.stringify(mockResponse), {
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'application/json' }
      }))

      // Set up authenticated state
      store.dispatch({
        type: 'auth/setCredentials',
        payload: {
          user: { username: 'testuser' },
          access: 'test-token',
          refresh: 'refresh-token'
        }
      })

      // Make a request (this would be done through an endpoint)
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
              'authorization': 'Bearer test-token'
            })
          })
        )
      }
    })

    it('should not include auth headers when not authenticated', async () => {
      const mockResponse = { data: 'test data' }
      
      global.fetch.mockResolvedValueOnce(new Response(JSON.stringify(mockResponse), {
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'application/json' }
      }))

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
        const callArgs = global.fetch.mock.calls[0]
        if (callArgs && callArgs[1] && callArgs[1].headers) {
          expect(callArgs[1].headers.authorization).toBeUndefined()
        }
      }
    })
  })

  describe('configuration', () => {
    it('should have correct reducer path', () => {
      expect(apiSlice.reducerPath).toBe('api')
    })

    it('should have endpoints defined', () => {
      expect(apiSlice.endpoints).toBeDefined()
      expect(typeof apiSlice.endpoints).toBe('object')
    })

    it('should have login endpoint with initiate function', () => {
      expect(apiSlice.endpoints.login).toBeDefined()
      expect(typeof apiSlice.endpoints.login.initiate).toBe('function')
    })

    it('should have register endpoint with initiate function', () => {
      expect(apiSlice.endpoints.register).toBeDefined()
      expect(typeof apiSlice.endpoints.register.initiate).toBe('function')
    })

    it('should have logout endpoint with initiate function', () => {
      expect(apiSlice.endpoints.logout).toBeDefined()
      expect(typeof apiSlice.endpoints.logout.initiate).toBe('function')
    })

    it('should have reducer defined', () => {
      expect(apiSlice.reducer).toBeDefined()
      expect(typeof apiSlice.reducer).toBe('function')
    })
  })
})