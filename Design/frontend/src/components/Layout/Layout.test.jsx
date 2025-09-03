/**
 * Layout component test suite.
 * Comprehensive testing for layout structure, accessibility, and responsive behaviour.
 */

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import Layout from './Layout'
import authSlice from '../../features/auth/authSlice'

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
  it('renders layout with navbar and children', () => {
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
        <Layout>
          <div data-testid="test-content">Test Content</div>
        </Layout>
      </TestWrapper>
    )

    // Verify navigation bar renders correctly
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    
    // Verify child components render properly
    expect(screen.getByTestId('test-content')).toBeInTheDocument()
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })

  it('renders layout when user is not authenticated', () => {
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
        <Layout>
          <div data-testid="test-content">Test Content</div>
        </Layout>
      </TestWrapper>
    )

    // Navigation bar should render with unauthenticated user interface
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    
    // Content should still render
    expect(screen.getByTestId('test-content')).toBeInTheDocument()
  })

  it('has correct semantic structure', () => {
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
        <Layout>
          <div data-testid="main-content">Main Content</div>
        </Layout>
      </TestWrapper>
    )

    // Check for semantic HTML elements
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
  })

  it('renders complete layout structure', () => {
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
        <Layout>
          <div data-testid="layout-content">Content</div>
        </Layout>
      </TestWrapper>
    )

    // Check that the layout contains all expected semantic regions
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByTestId('layout-content')).toBeInTheDocument()
    
    // Verify content is properly nested within main region
    const mainElement = screen.getByRole('main')
    const contentElement = screen.getByTestId('layout-content')
    expect(mainElement).toContainElement(contentElement)
  })
})