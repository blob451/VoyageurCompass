/**
 * Layout component test suite.
 * Comprehensive testing for layout structure, accessibility, and responsive behaviour.
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import Layout from './Layout'
import authSlice from '../../features/auth/authSlice'

// Mock the Outlet component since Layout uses React Router's Outlet
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    Outlet: () => <div data-testid="outlet-content">Outlet Content</div>
  }
})

// Mock Redux store configuration
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      auth: authSlice,
    },
    preloadedState: initialState
  })
}

// Material-UI test theme
const theme = createTheme()

// Test wrapper with providers
const TestWrapper = ({ children, store = createMockStore() }) => (
  <Provider store={store}>
    <MemoryRouter>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </MemoryRouter>
  </Provider>
)

describe('Layout', () => {
  it('renders layout with navbar and outlet content', () => {
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
        <Layout />
      </TestWrapper>
    )

    // Verify navigation bar renders correctly
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    
    // Verify outlet content renders properly
    expect(screen.getByTestId('outlet-content')).toBeInTheDocument()
    expect(screen.getByText('Outlet Content')).toBeInTheDocument()
  })

  it('renders layout when user is not authenticated', () => {
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
        <Layout />
      </TestWrapper>
    )

    // Should still render the navigation and outlet
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    expect(screen.getByTestId('outlet-content')).toBeInTheDocument()
  })

  it('renders footer with copyright information', () => {
    render(
      <TestWrapper>
        <Layout />
      </TestWrapper>
    )

    // Verify footer content
    expect(screen.getByText(/© 2025 Voyageur Compass. All rights reserved./)).toBeInTheDocument()
  })

  it('has proper accessibility structure', () => {
    render(
      <TestWrapper>
        <Layout />
      </TestWrapper>
    )

    // Check for main content area
    const mainElement = screen.getByRole('main')
    expect(mainElement).toBeInTheDocument()

    // Check for navigation
    const navElement = screen.getByRole('navigation')
    expect(navElement).toBeInTheDocument()
  })
})