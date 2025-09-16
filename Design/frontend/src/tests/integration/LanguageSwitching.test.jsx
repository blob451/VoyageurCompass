import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { I18nextProvider, useTranslation } from 'react-i18next';
import { configureStore } from '@reduxjs/toolkit';
import i18n from '../../i18n';
import { apiSlice } from '../../features/api/apiSlice';
import authReducer from '../../features/auth/authSlice';
import DashboardPage from '../../pages/DashboardPage';
import StockSearchPage from '../../pages/StockSearchPage';
import LanguageSwitcher from '../../components/LanguageSwitcher';

// Mock store setup
const createTestStore = () => {
  return configureStore({
    reducer: {
      auth: authReducer,
      api: apiSlice.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(apiSlice.middleware),
  });
};

// Test wrapper component
const TestWrapper = ({ children, store = createTestStore() }) => {
  return (
    <Provider store={store}>
      <BrowserRouter>
        <I18nextProvider i18n={i18n}>
          {children}
        </I18nextProvider>
      </BrowserRouter>
    </Provider>
  );
};

describe('Language Switching Integration Tests', () => {
  beforeEach(() => {
    // Reset i18n to English
    i18n.changeLanguage('en');
    localStorage.removeItem('voyageur-language');
  });

  describe('LanguageSwitcher Component', () => {
    test('should render language options correctly', () => {
      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      expect(screen.getByText('English')).toBeInTheDocument();
      expect(screen.getByText('Français')).toBeInTheDocument();
      expect(screen.getByText('Español')).toBeInTheDocument();
    });

    test('should change language when option is selected', async () => {
      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      const select = screen.getByDisplayValue('English');
      fireEvent.change(select, { target: { value: 'fr' } });

      await waitFor(() => {
        expect(i18n.language).toBe('fr');
        expect(localStorage.getItem('voyageur-language')).toBe('fr');
      });
    });

    test('should persist language selection in localStorage', async () => {
      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      const select = screen.getByDisplayValue('English');
      fireEvent.change(select, { target: { value: 'es' } });

      await waitFor(() => {
        expect(localStorage.getItem('voyageur-language')).toBe('es');
      });
    });
  });

  describe('Dashboard Language Switching', () => {
    test('should update dashboard text when language changes', async () => {
      const store = createTestStore();

      render(
        <TestWrapper store={store}>
          <DashboardPage />
        </TestWrapper>
      );

      // Check English text
      expect(screen.getByText(/Welcome back/i)).toBeInTheDocument();

      // Switch to French
      i18n.changeLanguage('fr');

      await waitFor(() => {
        expect(screen.getByText(/Bon retour/i)).toBeInTheDocument();
      });
    });

    test('should format numbers according to language locale', async () => {
      const store = createTestStore();

      render(
        <TestWrapper store={store}>
          <DashboardPage />
        </TestWrapper>
      );

      // Switch to French and check number formatting
      i18n.changeLanguage('fr');

      await waitFor(() => {
        // Look for French number formatting (comma as decimal separator)
        const elements = screen.getAllByText(/\d+,\d+/);
        expect(elements.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Stock Search Language Integration', () => {
    test('should update search page text when language changes', async () => {
      render(
        <TestWrapper>
          <StockSearchPage />
        </TestWrapper>
      );

      // Check English text
      expect(screen.getByText(/Stock Search/i)).toBeInTheDocument();

      // Switch to Spanish
      i18n.changeLanguage('es');

      await waitFor(() => {
        expect(screen.getByText(/Búsqueda de Acciones/i)).toBeInTheDocument();
      });
    });

    test('should include language parameter in API calls', async () => {
      // Mock fetch for API calls
      global.fetch = jest.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ data: 'test' }),
        })
      );

      render(
        <TestWrapper>
          <StockSearchPage />
        </TestWrapper>
      );

      // Switch to French
      i18n.changeLanguage('fr');

      const searchInput = screen.getByPlaceholderText(/Enter stock symbol/i);
      const searchButton = screen.getByText(/Search/i);

      fireEvent.change(searchInput, { target: { value: 'AAPL' } });
      fireEvent.click(searchButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('language=fr'),
          expect.any(Object)
        );
      });

      global.fetch.mockRestore();
    });
  });

  describe('Browser Language Detection', () => {
    test('should detect browser language on first visit', () => {
      // Mock navigator.language
      Object.defineProperty(navigator, 'language', {
        writable: true,
        value: 'fr-FR',
      });

      // Create new i18n instance to trigger detection
      const testI18n = i18n.cloneInstance();

      expect(testI18n.language).toBe('fr');
    });

    test('should fallback to English for unsupported languages', () => {
      Object.defineProperty(navigator, 'language', {
        writable: true,
        value: 'de-DE',
      });

      const testI18n = i18n.cloneInstance();

      expect(testI18n.language).toBe('en');
    });
  });

  describe('Translation Fallbacks', () => {
    test('should fallback to English when translation key is missing', async () => {
      // Switch to French
      i18n.changeLanguage('fr');

      render(
        <TestWrapper>
          <div>{i18n.t('nonexistent.key', { fallbackLng: 'en' })}</div>
        </TestWrapper>
      );

      // Should not crash and should show key or fallback
      expect(screen.getByText(/nonexistent\.key/)).toBeInTheDocument();
    });

    test('should handle interpolation correctly across languages', async () => {
      // Test interpolation with dynamic values
      i18n.changeLanguage('en');
      expect(i18n.t('dashboard.greeting', { name: 'John' })).toContain('John');

      i18n.changeLanguage('fr');
      expect(i18n.t('dashboard.greeting', { name: 'Jean' })).toContain('Jean');
    });
  });

  describe('Performance Tests', () => {
    test('should not cause excessive re-renders on language change', async () => {
      let renderCount = 0;

      const TestComponent = () => {
        renderCount++;
        const { t } = useTranslation('common');
        return <div>{t('dashboard.welcome')}</div>;
      };

      render(
        <TestWrapper>
          <TestComponent />
        </TestWrapper>
      );

      const initialRenderCount = renderCount;

      // Change language
      i18n.changeLanguage('fr');

      await waitFor(() => {
        // Should only re-render once for language change
        expect(renderCount).toBeLessThanOrEqual(initialRenderCount + 2);
      });
    });

    test('should load translations quickly', async () => {
      const startTime = performance.now();

      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      i18n.changeLanguage('es');

      await waitFor(() => {
        expect(screen.getByText(/Bienvenido/i)).toBeInTheDocument();
      });

      const endTime = performance.now();
      const loadTime = endTime - startTime;

      // Translation switch should be fast (under 100ms)
      expect(loadTime).toBeLessThan(100);
    });
  });
});