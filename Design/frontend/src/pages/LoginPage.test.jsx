/**
 * Tests for LoginPage component
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import LoginPage from './LoginPage'
import authReducer from '../store/authSlice'

// Mock the useNavigate hook for React Router v7
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

// Create a test store
const createTestStore = () => {
  return configureStore({
    reducer: {
      auth: authReducer,
    },
  })
}

// Helper function to render with providers
const renderWithProviders = (component) => {
  const store = createTestStore()
  return render(
    <Provider store={store}>
      <BrowserRouter>
        {component}
      </BrowserRouter>
    </Provider>
  )
}

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.fetch = vi.fn()
  })

  it('renders login form with all required elements', () => {
    renderWithProviders(<LoginPage />)
    
    expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByText(/don't have an account/i)).toBeInTheDocument()
  })

  it('allows user to type in username and password fields', () => {
    renderWithProviders(<LoginPage />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    
    fireEvent.change(usernameInput, { target: { value: 'testuser' } })
    expect(usernameInput.value).toBe('testuser')
    
    fireEvent.change(passwordInput, { target: { value: 'testpass123' } })
    expect(passwordInput.value).toBe('testpass123')
  })

  it('shows validation error when submitting empty form', async () => {
    renderWithProviders(<LoginPage />)
    
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    fireEvent.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText(/username is required/i)).toBeInTheDocument()
      expect(screen.getByText(/password is required/i)).toBeInTheDocument()
    })
  })

  it('successfully logs in and redirects on valid credentials', async () => {
    window.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        access: 'fake-jwt-token',
        refresh: 'fake-refresh-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@example.com'
        }
      })
    })
    
    renderWithProviders(<LoginPage />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    fireEvent.change(usernameInput, { target: { value: 'testuser' } })
    fireEvent.change(passwordInput, { target: { value: 'testpass123' } })
    fireEvent.click(submitButton)
    
    await waitFor(() => {
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/auth/login/'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({
            username: 'testuser',
            password: 'testpass123'
          })
        })
      )
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard')
    })
  })
})