/**
 * Tests for DashboardPage component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import DashboardPage from './DashboardPage'
import authSlice from '../features/auth/authSlice'
import { apiSlice } from '../features/api/apiSlice'

// Mock the Chart components to avoid Canvas issues in tests
vi.mock('recharts', () => ({
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  AreaChart: ({ children }) => <div data-testid="area-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
}))

// Create a mock store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      auth: authSlice,
      api: apiSlice.reducer,
    },
    preloadedState: initialState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(apiSlice.middleware),
  })
}

// Create a test theme
const theme = createTheme()

// Test wrapper component
const TestWrapper = ({ children, store = createMockStore() }) => (
  <Provider store={store}>
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </BrowserRouter>
  </Provider>
)

// Mock API responses
const mockPortfolioData = {
  id: 1,
  name: 'My Portfolio',
  total_value: 15000,
  total_gain_loss: 2500,
  total_gain_loss_percent: 20.0,
  holdings: [
    {
      id: 1,
      stock: {
        symbol: 'AAPL',
        short_name: 'Apple Inc.',
        sector: 'Technology'
      },
      quantity: 10,
      current_value: 1500,
      unrealized_gain_loss: 200,
      unrealized_gain_loss_percent: 15.0
    },
    {
      id: 2,
      stock: {
        symbol: 'MSFT',
        short_name: 'Microsoft Corp.',
        sector: 'Technology'
      },
      quantity: 5,
      current_value: 1200,
      unrealized_gain_loss: 100,
      unrealized_gain_loss_percent: 9.0
    }
  ]
}

const mockMarketData = {
  indices: [
    { symbol: '^GSPC', name: 'S&P 500', price: 4200.0 },
    { symbol: '^DJI', name: 'Dow Jones', price: 34000.0 }
  ],
  top_gainers: [
    { symbol: 'AAPL', name: 'Apple Inc.', change_percent: 5.2 },
    { symbol: 'MSFT', name: 'Microsoft', change_percent: 3.8 }
  ],
  top_losers: [
    { symbol: 'TSLA', name: 'Tesla Inc.', change_percent: -2.1 }
  ]
}

describe('DashboardPage', () => {
  let originalInnerWidth

  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks()
    
    // Save original window.innerWidth
    originalInnerWidth = window.innerWidth
    
    // Mock fetch for API calls
    window.fetch = vi.fn()
  })

  afterEach(() => {
    // Restore original window.innerWidth after each test
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth
    })
  })

  it('renders dashboard for authenticated user', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser', email: 'test@example.com' },
        tokens: { access: 'mock-token', refresh: 'mock-refresh' },
        loading: false,
        error: null
      }
    })

    // Mock successful API responses
    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should show dashboard title
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
    
    // Should show welcome message with username
    expect(screen.getByText(/welcome/i)).toBeInTheDocument()
    expect(screen.getByText(/testuser/i)).toBeInTheDocument()
  })

  it('shows loading state while fetching data', () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    // Mock delayed API response
    window.fetch.mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 1000))
    )

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should show loading indicators
    const loadingElements = screen.getAllByRole('progressbar')
    expect(loadingElements.length).toBeGreaterThan(0)
  })

  it('displays portfolio overview when data is loaded', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText(/portfolio overview/i)).toBeInTheDocument()
    })

    // Should display portfolio value
    await waitFor(() => {
      expect(screen.getByText(/\$15,000/)).toBeInTheDocument()
    })

    // Should display gain/loss
    await waitFor(() => {
      expect(screen.getByText(/\$2,500/)).toBeInTheDocument()
      expect(screen.getByText(/20\.0%/)).toBeInTheDocument()
    })
  })

  it('displays holdings list', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Wait for holdings to load
    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('MSFT')).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
      expect(screen.getByText('Microsoft Corp.')).toBeInTheDocument()
    })
  })

  it('displays market overview', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Wait for market data to load
    await waitFor(() => {
      expect(screen.getByText(/market overview/i)).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByText('S&P 500')).toBeInTheDocument()
      expect(screen.getByText('Dow Jones')).toBeInTheDocument()
    })
  })

  it('shows charts when data is available', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Wait for charts to render
    await waitFor(() => {
      expect(screen.getByTestId('line-chart')).toBeInTheDocument()
    })
  })

  it('handles API errors gracefully', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    // Mock failed API responses
    window.fetch.mockRejectedValue(new Error('API Error'))

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should show error message or fallback to default content
    await waitFor(() => {
      // Check if the page still renders with default content when API fails
      expect(screen.getByText('Welcome back, testuser!')).toBeInTheDocument()
      // The component should gracefully handle API failures and still render
      expect(screen.getByText("Here's your portfolio overview and market insights")).toBeInTheDocument()
    })
  })

  it('shows empty state when no portfolio data', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [] }) // No portfolios
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should show empty state message for stocks
    await waitFor(() => {
      expect(screen.getByText('No stocks tracked yet. Start by searching and adding stocks to your watchlist.')).toBeInTheDocument()
    })
  })

  it('is responsive and adapts to different screen sizes', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should render without errors on mobile
    await waitFor(() => {
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
    })
  })

  it('has proper accessibility attributes', async () => {
    const mockStore = createMockStore({
      auth: {
        isAuthenticated: true,
        user: { username: 'testuser' },
        tokens: { access: 'mock-token' },
        loading: false,
        error: null
      }
    })

    window.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [mockPortfolioData] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockMarketData
      })

    render(
      <TestWrapper store={mockStore}>
        <DashboardPage />
      </TestWrapper>
    )

    // Should have proper heading structure
    await waitFor(() => {
      const headings = screen.getAllByRole('heading')
      expect(headings.length).toBeGreaterThan(0)
    })

    // Should have accessible content and landmarks
    await waitFor(() => {
      // Main landmark should always be present
      expect(screen.getByRole('main')).toBeInTheDocument()
      
      // Check for proper landmark structure
      const main = screen.getByRole('main')
      expect(main).toHaveAccessibleName()
      
      // Should have proper heading hierarchy (h1 should be present)
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()
      
      // Check that interactive elements have accessible names
      const buttons = screen.queryAllByRole('button')
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName()
      })
      
      // Check that any links have accessible names
      const links = screen.queryAllByRole('link')
      links.forEach(link => {
        expect(link).toHaveAccessibleName()
      })
    })
  })
})