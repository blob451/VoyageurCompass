/**
 * Tests for Navbar component
 */

import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import Navbar from './Navbar'
import authSlice from '../../features/auth/authSlice'
import { apiSlice } from '../../features/api/apiSlice'

// Create a mock store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      auth: authSlice,
      api: apiSlice.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(apiSlice.middleware),
    preloadedState: initialState
  })
}

// Create a test theme
const theme = createTheme()

// Test wrapper component
const TestWrapper = ({ children, store = createMockStore() }) => (
  <Provider store={store}>
    <MemoryRouter>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </MemoryRouter>
  </Provider>
)

describe('Navbar', () => {
  it('renders navbar with brand name', () => {
    const mockStore = createMockStore({
      auth: {
        user: null,
        token: null,
        refreshToken: null,
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Check for app name/brand
    expect(screen.getByText('VoyageurCompass')).toBeInTheDocument()
  })

  it('shows login/register links when not authenticated', () => {
    const mockStore = createMockStore({
      auth: {
        user: null,
        token: null,
        refreshToken: null,
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Look for login and register links
    expect(screen.getByText('Login')).toBeInTheDocument()
    expect(screen.getByText('Register')).toBeInTheDocument()
  })

  it('shows user menu when authenticated', async () => {
    const mockStore = createMockStore({
      auth: {
        user: { username: 'testuser', email: 'test@example.com' },
        token: 'mock-token',
        refreshToken: 'mock-refresh',
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Should show user's username
    expect(screen.getByText('testuser')).toBeInTheDocument()
    
    // Should show Dashboard button
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    
    // Should show Tools menu
    expect(screen.getByText('Tools')).toBeInTheDocument()
    
    // Should show Help button
    expect(screen.getByText('Help')).toBeInTheDocument()
    
    // Should show credits
    expect(screen.getByText(/25 Credits/)).toBeInTheDocument()
  })

  it('opens user menu when clicking username', async () => {
    const user = userEvent.setup()
    const mockStore = createMockStore({
      auth: {
        user: { username: 'testuser', email: 'test@example.com' },
        token: 'mock-token',
        refreshToken: 'mock-refresh',
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Click on username to open menu
    await user.click(screen.getByText('testuser'))
    
    // Should show logout option in menu
    await waitFor(() => {
      expect(screen.getByText('Logout')).toBeInTheDocument()
    })
    
    // Should show settings option in menu
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  it('opens tools menu when clicking Tools button', async () => {
    const user = userEvent.setup()
    const mockStore = createMockStore({
      auth: {
        user: { username: 'testuser', email: 'test@example.com' },
        token: 'mock-token',
        refreshToken: 'mock-refresh',
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Click on Tools button to open menu
    await user.click(screen.getByText('Tools'))
    
    // Should show tools menu items
    await waitFor(() => {
      expect(screen.getByText('Stock Analysis')).toBeInTheDocument()
    })
    
    expect(screen.getByText('Analysis Reports')).toBeInTheDocument()
    expect(screen.getByText('Compare Stocks')).toBeInTheDocument()
    expect(screen.getByText('Sector Analysis')).toBeInTheDocument()
    expect(screen.getByText('Credit Store')).toBeInTheDocument()
  })

  it('has proper navigation role', () => {
    render(
      <TestWrapper>
        <Navbar />
      </TestWrapper>
    )

    const navElement = screen.getByRole('navigation')
    expect(navElement).toBeInTheDocument()
  })

  it('displays correct brand logo', () => {
    render(
      <TestWrapper>
        <Navbar />
      </TestWrapper>
    )

    // Check for trending up icon (logo)
    const logoButton = screen.getByLabelText('logo')
    expect(logoButton).toBeInTheDocument()
  })

  it('shows correct user state when switching authentication', () => {
    const { rerender } = render(
      <TestWrapper store={createMockStore({
        auth: {
          user: null,
          token: null,
          refreshToken: null,
          isValidating: false,
          validationError: null,
          lastValidated: null
        }
      })}>
        <Navbar />
      </TestWrapper>
    )

    // Initially should show login/register
    expect(screen.getByText('Login')).toBeInTheDocument()
    expect(screen.getByText('Register')).toBeInTheDocument()

    // Re-render with authenticated state
    rerender(
      <TestWrapper store={createMockStore({
        auth: {
          user: { username: 'testuser' },
          token: 'mock-token',
          refreshToken: 'mock-refresh',
          isValidating: false,
          validationError: null,
          lastValidated: null
        }
      })}>
        <Navbar />
      </TestWrapper>
    )

    // Should now show authenticated UI
    expect(screen.getByText('testuser')).toBeInTheDocument()
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
  })

  it('handles navigation correctly', async () => {
    const mockStore = createMockStore({
      auth: {
        user: null,
        token: null,
        refreshToken: null,
        isValidating: false,
        validationError: null,
        lastValidated: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Click on brand name (should navigate to home)
    const brandElement = screen.getByText('VoyageurCompass')
    expect(brandElement).toBeInTheDocument()
    
    // Brand should be clickable
    expect(brandElement).toHaveStyle('cursor: pointer')
  })
})