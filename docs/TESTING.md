# VoyageurCompass Testing Guide

## Overview
This document provides comprehensive information about testing in the VoyageurCompass project, including both backend Django API tests and frontend React component tests.

## Test Structure

### Backend Tests (Django)
```
VoyageurCompass/
├── Data/tests/
│   ├── test_models.py        # Model unit tests
│   ├── test_views.py         # API endpoint tests
│   └── test_services.py      # Service layer tests
├── Analytics/tests/
│   ├── test_views.py         # Analytics API tests
│   └── test_engine.py        # Analytics engine tests
├── Core/tests/
│   ├── test_views.py         # Core API tests
│   └── test_auth.py          # Authentication tests
└── tests/
    └── test_integration.py   # Integration tests
```

### Frontend Tests (React)
```
Design/frontend/src/
├── components/
│   ├── Layout/
│   │   ├── Layout.test.jsx
│   │   └── Navbar.test.jsx
│   └── ...
├── pages/
│   ├── DashboardPage.test.jsx
│   └── LoginPage.test.jsx
├── features/
│   ├── auth/
│   │   └── authSlice.test.js
│   └── api/
│       └── apiSlice.test.js
└── test/
    └── setup.js              # Test setup configuration
```

## Running Tests

### Quick Commands

#### Backend Tests
```bash
# Run all backend tests
pytest

# Run with coverage
pytest --cov=Data --cov=Analytics --cov=Core --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m api          # API tests only
pytest -m models       # Model tests only
pytest -m "not slow"   # Exclude slow tests

# Run specific app tests
pytest Data/tests
pytest Analytics/tests
pytest Core/tests
```

#### Frontend Tests
```bash
cd Design/frontend

# Run all frontend tests
npm run test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch

# Run with UI
npm run test:ui
```

#### All Tests
```bash
# Run comprehensive test runner
python scripts/run_tests.py --all --coverage --lint

# Backend only
python scripts/run_tests.py --backend --coverage

# Frontend only
python scripts/run_tests.py --frontend --coverage

# Fast tests only
python scripts/run_tests.py --test-type fast
```

## Test Categories

### 1. Unit Tests
Test individual components/functions in isolation.

**Backend Examples:**
- Model methods and properties
- Utility functions
- Service class methods

**Frontend Examples:**
- Redux reducers
- Utility functions
- Individual component rendering

### 2. Integration Tests
Test interactions between multiple components.

**Backend Examples:**
- API endpoint workflows
- Database operations with business logic
- Service integrations

**Frontend Examples:**
- Component interactions
- Redux store with components
- API integration with UI

### 3. API Tests
Test REST API endpoints end-to-end.

**Coverage:**
- Authentication endpoints
- CRUD operations
- Error handling
- Permissions and authorization
- Data validation

### 4. Component Tests
Test React components with user interactions.

**Coverage:**
- Component rendering
- User interactions (clicks, form submissions)
- Props handling
- State management
- Event handlers

## Backend Testing Details

### Test Configuration
The backend tests use pytest with the following configuration:
- **Database**: SQLite in-memory for tests
- **Coverage**: Minimum 70% overall, 80% for critical components
- **Fixtures**: Django fixtures and factory classes
- **Mocking**: unittest.mock for external services

### Test Markers
```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test  
@pytest.mark.api          # API endpoint test
@pytest.mark.models       # Model test
@pytest.mark.slow         # Slow-running test
```

### Example Test Structure
```python
@pytest.mark.django_db
class TestStockModel:
    def test_create_stock(self):
        stock = Stock.objects.create(symbol='AAPL', short_name='Apple')
        assert stock.symbol == 'AAPL'
    
    def test_get_latest_price(self):
        # Test implementation
        pass
```

### API Testing Patterns
```python
class StockAPITestCase(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='test', password='test')
        
    def test_authenticated_request(self):
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/stocks/')
        self.assertEqual(response.status_code, 200)
```

## Frontend Testing Details

### Test Configuration
Frontend tests use Vitest with React Testing Library:
- **Environment**: jsdom
- **Coverage**: Minimum 70% overall, 80% for components
- **Mocking**: Vitest mocks for API calls and external dependencies
- **Setup**: Global test setup in `src/test/setup.js`

### Testing Utilities
```javascript
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
```

### Component Test Patterns
```javascript
describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent prop="value" />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })
  
  it('handles user interaction', async () => {
    const user = userEvent.setup()
    render(<MyComponent />)
    
    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Updated Text')).toBeInTheDocument()
  })
})
```

### Redux Testing
```javascript
describe('authSlice', () => {
  it('should handle login success', () => {
    const initialState = { user: null, isAuthenticated: false }
    const action = loginSuccess({ user: { username: 'test' } })
    const state = authReducer(initialState, action)
    
    expect(state.isAuthenticated).toBe(true)
    expect(state.user.username).toBe('test')
  })
})
```

## Coverage Requirements

### Backend Coverage Thresholds
- **Overall**: 70% minimum
- **Critical Components** (models, views): 80% minimum
- **Business Logic** (services, utilities): 85% minimum

### Frontend Coverage Thresholds
- **Overall**: 70% minimum
- **Components**: 80% minimum  
- **Redux Logic**: 75% minimum
- **Utilities**: 85% minimum

### Coverage Reports
Coverage reports are generated in:
- **Backend**: `htmlcov/index.html`
- **Frontend**: `coverage/index.html`

## Mocking Strategies

### Backend Mocking
```python
from unittest.mock import patch, MagicMock

@patch('Data.services.yahoo_finance.yahoo_finance_service.get_stock_data')
def test_stock_sync(self, mock_get_data):
    mock_get_data.return_value = {'symbol': 'AAPL', 'price': 150.00}
    # Test implementation
```

### Frontend Mocking
```javascript
import { vi } from 'vitest'

// Mock API calls
global.fetch = vi.fn()
fetch.mockResolvedValue({
  ok: true,
  json: async () => ({ data: 'test' })
})

// Mock components
vi.mock('recharts', () => ({
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>
}))
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov --cov-report=xml
      
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
      - name: Install dependencies
        run: cd Design/frontend && npm ci
      - name: Run tests
        run: cd Design/frontend && npm run test:coverage
```

## Test Data Management

### Fixtures and Factories
```python
# conftest.py
import pytest
from django.contrib.auth.models import User

@pytest.fixture
def test_user():
    return User.objects.create_user(
        username='testuser',
        password='testpass123',
        email='test@example.com'
    )

@pytest.fixture
def test_stock():
    return Stock.objects.create(
        symbol='AAPL',
        short_name='Apple Inc.',
        sector='Technology'
    )
```

### Test Database
- Backend tests use a separate test database
- Database is recreated for each test run
- Transactions are rolled back after each test

## Performance Testing

### Load Testing
```python
@pytest.mark.slow
def test_large_portfolio_performance():
    # Create portfolio with 1000 holdings
    # Measure response times
    # Assert performance thresholds
```

### Memory Testing
```python
import psutil
import gc

def test_memory_usage():
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform operations
    
    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Debugging Tests

### Backend Debugging
```bash
# Run with pdb debugger
pytest --pdb

# Run specific test with verbose output
pytest -v Data/tests/test_models.py::TestStock::test_create_stock

# Show print statements
pytest -s
```

### Frontend Debugging
```bash
# Run in debug mode
npm run test -- --reporter=verbose

# Run single test file
npm run test Layout.test.jsx
```

## Best Practices

### General Guidelines
1. **Write tests first** (TDD approach when possible)
2. **Test behavior, not implementation**
3. **Use descriptive test names**
4. **Keep tests independent and isolated**
5. **Mock external dependencies**
6. **Test edge cases and error conditions**

### Backend Best Practices
1. Use Django's TestCase for database tests
2. Use APITestCase for API endpoint tests
3. Mock external API calls (Yahoo Finance)
4. Test permissions and authentication
5. Use factories for test data creation

### Frontend Best Practices
1. Test component behavior, not internal state
2. Use screen queries over container queries
3. Test user interactions with userEvent
4. Mock API calls and external dependencies
5. Test error boundaries and loading states

## Troubleshooting

### Common Backend Issues
- **Database errors**: Ensure proper test database setup
- **Import errors**: Check Django settings and app configuration
- **Async issues**: Use proper async test decorators

### Common Frontend Issues
- **Component not found**: Check test setup and imports
- **Async operations**: Use waitFor and findBy queries
- **Mock issues**: Ensure mocks are properly configured

### Performance Issues
- **Slow tests**: Use pytest markers to skip slow tests during development
- **Memory leaks**: Check for proper cleanup in tearDown methods
- **Database locks**: Ensure tests don't interfere with each other

## Conclusion

This testing framework provides comprehensive coverage for both backend and frontend components. Regular test execution and maintenance ensure code quality and reliability throughout the development process.

For questions or issues with tests, refer to the project documentation or contact the development team.