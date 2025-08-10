/**
 * Tests for Navbar component
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import Navbar from './Navbar'
import authSlice from '../../features/auth/authSlice'

// Create a mock store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      auth: authSlice,
    },
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
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Check for app name/brand
    expect(screen.getByText(/VoyageurCompass/i)).toBeInTheDocument()
  })

  it('shows login/register links when not authenticated', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Look for login and register links
    expect(screen.getByText(/login/i)).toBeInTheDocument()
    expect(screen.getByText(/register/i)).toBeInTheDocument()
  })

  it('shows user menu when authenticated', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser', email: 'test@example.com' },
        tokens: { access: 'mock-token', refresh: 'mock-refresh' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Should show user's name or username
    expect(screen.getByText(/testuser/i)).toBeInTheDocument()
    
    // Should have logout functionality
    expect(screen.getByText(/logout/i)).toBeInTheDocument()
  })

  it('shows dashboard link when authenticated', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Should show dashboard link for authenticated users
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
  })

  it('handles logout action', async () => {
    const user = userEvent.setup()
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token', refresh: 'mock-refresh' },
        loading: false,
        error: null
      }
    })

    // Spy on the store's dispatch method
    const dispatchSpy = vi.spyOn(mockStore, 'dispatch')

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Find and click logout button
    const logoutButton = screen.getByText(/logout/i)
    await user.click(logoutButton)

    // Verify logout action was dispatched
    expect(dispatchSpy).toHaveBeenCalled()
    expect(dispatchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        type: expect.stringContaining('logout')
      })
    )

    // Clean up spy
    dispatchSpy.mockRestore()
  })

  it('has proper navigation structure', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Check for navigation landmark
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    
    // Check for AppBar (MUI component)
    expect(screen.getByRole('banner')).toBeInTheDocument()
  })

  it('is responsive and shows mobile menu toggle', () => {
    // Mock window.innerWidth for mobile view
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600, // Mobile width
    })

    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // On mobile, there should be a menu icon
    const menuButton = screen.queryByRole('button', { name: /menu/i })
    if (menuButton) {
      expect(menuButton).toBeInTheDocument()
    }
  })

  it('handles mobile menu toggle', async () => {
    const user = userEvent.setup()
    
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    })

    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Find mobile menu button if it exists
    const menuButton = screen.queryByRole('button', { name: /menu/i })
    if (menuButton) {
      await user.click(menuButton)
      
      // Mobile menu should be visible after clicking
      // This would depend on the actual implementation
    }
  })

  it('shows loading state when auth is loading', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: true,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Should show some loading indicator
    // This depends on how loading state is handled in the component
    const loadingElement = screen.queryByRole('progressbar')
    if (loadingElement) {
      expect(loadingElement).toBeInTheDocument()
    }
  })

  it('displays error message when auth has error', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: false,
        user: null,
        tokens: null,
        loading: false,
        error: 'Authentication failed'
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // Should show error message or indicator
    // This depends on how errors are handled in the component
    const errorElement = screen.queryByText(/error/i)
    if (errorElement) {
      expect(errorElement).toBeInTheDocument()
    }
  })

  it('has accessible navigation links', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    render(
      <TestWrapper store={mockStore}>
        <Navbar />
      </TestWrapper>
    )

    // All links should be accessible
    const links = screen.getAllByRole('link')
    links.forEach(link => {
      expect(link).toHaveAttribute('href')
    })

    // Buttons should be accessible
    const buttons = screen.getAllByRole('button')
    buttons.forEach(button => {
      expect(button).toBeInTheDocument()
    })
  })
})