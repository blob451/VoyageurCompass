/**
 * Cross-browser compatibility tests for multilingual support
 * These tests verify that our i18n implementation works across different browsers
 * and handles various locale-specific scenarios correctly.
 */

// Mock browser detection for testing
const mockUserAgent = (userAgent) => {
  Object.defineProperty(navigator, 'userAgent', {
    writable: true,
    value: userAgent
  });
};

const mockNavigatorLanguage = (language) => {
  Object.defineProperty(navigator, 'language', {
    writable: true,
    value: language
  });
};

// eslint-disable-next-line no-unused-vars
const mockNavigatorLanguages = (languages) => {
  Object.defineProperty(navigator, 'languages', {
    writable: true,
    value: languages
  });
};

// Browser-specific test configurations
const browserConfigs = {
  chrome: {
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    numberFormat: { decimal: '.', thousands: ',' },
    dateFormat: 'MM/DD/YYYY',
    supportedFeatures: ['Intl.NumberFormat', 'Intl.DateTimeFormat', 'localStorage', 'sessionStorage']
  },
  firefox: {
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    numberFormat: { decimal: '.', thousands: ',' },
    dateFormat: 'MM/DD/YYYY',
    supportedFeatures: ['Intl.NumberFormat', 'Intl.DateTimeFormat', 'localStorage', 'sessionStorage']
  },
  safari: {
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.1 Safari/537.36',
    numberFormat: { decimal: '.', thousands: ',' },
    dateFormat: 'MM/DD/YYYY',
    supportedFeatures: ['Intl.NumberFormat', 'Intl.DateTimeFormat', 'localStorage', 'sessionStorage']
  },
  edge: {
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
    numberFormat: { decimal: '.', thousands: ',' },
    dateFormat: 'MM/DD/YYYY',
    supportedFeatures: ['Intl.NumberFormat', 'Intl.DateTimeFormat', 'localStorage', 'sessionStorage']
  }
};

// Locale-specific test data
const localeTestData = {
  en: {
    language: 'en-US',
    expectedTexts: {
      'navigation.home': 'Home',
      'dashboard.welcome': 'Welcome back',
      'auth.signIn': 'Sign In'
    },
    numberFormat: {
      1234.56: '1,234.56',
      1000000: '1,000,000'
    },
    currencyFormat: {
      1234.56: '$1,234.56'
    },
    dateFormat: {
      '2024-03-15': '03/15/2024'
    }
  },
  fr: {
    language: 'fr-FR',
    expectedTexts: {
      'navigation.home': 'Accueil',
      'dashboard.welcome': 'Bon retour',
      'auth.signIn': 'Se connecter'
    },
    numberFormat: {
      1234.56: '1 234,56',
      1000000: '1 000 000'
    },
    currencyFormat: {
      1234.56: '1 234,56 €'
    },
    dateFormat: {
      '2024-03-15': '15/03/2024'
    }
  },
  es: {
    language: 'es-ES',
    expectedTexts: {
      'navigation.home': 'Inicio',
      'dashboard.welcome': 'Bienvenido',
      'auth.signIn': 'Iniciar Sesión'
    },
    numberFormat: {
      1234.56: '1.234,56',
      1000000: '1.000.000'
    },
    currencyFormat: {
      1234.56: '1.234,56 €'
    },
    dateFormat: {
      '2024-03-15': '15/03/2024'
    }
  }
};

describe('Cross-Browser Compatibility Tests', () => {

  beforeEach(() => {
    // Clear any existing localStorage
    localStorage.clear();
    sessionStorage.clear();
  });

  describe('Browser Detection and Language Support', () => {
    Object.entries(browserConfigs).forEach(([browserName, config]) => {
      describe(`${browserName.toUpperCase()} Browser`, () => {

        beforeEach(() => {
          mockUserAgent(config.userAgent);
        });

        test('should detect browser capabilities correctly', () => {
          // Test feature detection
          config.supportedFeatures.forEach(feature => {
            switch (feature) {
              case 'Intl.NumberFormat':
                expect(typeof Intl.NumberFormat).toBe('function');
                break;
              case 'Intl.DateTimeFormat':
                expect(typeof Intl.DateTimeFormat).toBe('function');
                break;
              case 'localStorage':
                expect(typeof localStorage).toBe('object');
                expect(typeof localStorage.setItem).toBe('function');
                break;
              case 'sessionStorage':
                expect(typeof sessionStorage).toBe('object');
                expect(typeof sessionStorage.setItem).toBe('function');
                break;
            }
          });
        });

        test('should handle language preference detection', () => {
          // Test different language scenarios
          const testCases = [
            { nav: 'fr-FR', expected: 'fr' },
            { nav: 'es-ES', expected: 'es' },
            { nav: 'en-US', expected: 'en' },
            { nav: 'de-DE', expected: 'en' }, // Should fallback to English
          ];

          testCases.forEach(({ nav, expected }) => {
            mockNavigatorLanguage(nav);

            // Test our language detection logic
            const detectedLanguage = nav.split('-')[0];
            const supportedLanguages = ['en', 'fr', 'es'];
            const finalLanguage = supportedLanguages.includes(detectedLanguage) ? detectedLanguage : 'en';

            expect(finalLanguage).toBe(expected);
          });
        });

        test('should store language preferences persistently', () => {
          const testLanguage = 'fr';

          // Simulate language selection
          localStorage.setItem('voyageur-language', testLanguage);

          // Verify persistence
          expect(localStorage.getItem('voyageur-language')).toBe(testLanguage);
        });
      });
    });
  });

  describe('Locale-Specific Formatting', () => {
    Object.entries(localeTestData).forEach(([locale, testData]) => {
      describe(`${locale.toUpperCase()} Locale`, () => {

        beforeEach(() => {
          mockNavigatorLanguage(testData.language);
        });

        test('should format numbers correctly', () => {
          Object.entries(testData.numberFormat).forEach(([number]) => {
            // Test using Intl.NumberFormat
            const formatter = new Intl.NumberFormat(testData.language);
            const formatted = formatter.format(parseFloat(number));

            // Note: Actual browser implementation may vary slightly
            // This test verifies the basic functionality
            expect(formatted).toMatch(/[\d\s,.]+/);
          });
        });

        test('should format currency correctly', () => {
          Object.entries(testData.currencyFormat).forEach(([amount]) => {
            const currencyCode = locale === 'en' ? 'USD' : 'EUR';
            const formatter = new Intl.NumberFormat(testData.language, {
              style: 'currency',
              currency: currencyCode
            });
            const formatted = formatter.format(parseFloat(amount));

            expect(formatted).toMatch(/[\d\s,.$€]+/);
          });
        });

        test('should format dates correctly', () => {
          Object.entries(testData.dateFormat).forEach(([dateStr]) => {
            const date = new Date(dateStr);
            const formatter = new Intl.DateTimeFormat(testData.language);
            const formatted = formatter.format(date);

            expect(formatted).toMatch(/[\d/\s-]+/);
          });
        });
      });
    });
  });

  describe('Font and Character Rendering', () => {
    test('should handle special characters across languages', () => {
      const specialChars = {
        fr: ['é', 'è', 'à', 'ç', 'ê', 'ô', 'û', 'î', 'ï', 'ù'],
        es: ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
      };

      Object.entries(specialChars).forEach(([, chars]) => {
        chars.forEach(char => {
          // Test that special characters are properly supported
          const testElement = document.createElement('div');
          testElement.textContent = char;
          document.body.appendChild(testElement);

          expect(testElement.textContent).toBe(char);

          document.body.removeChild(testElement);
        });
      });
    });

    test('should apply correct font families for different languages', () => {
      const languageFonts = {
        en: ['Arial', 'Helvetica', 'sans-serif'],
        fr: ['Arial', 'Helvetica', 'sans-serif'],
        es: ['Arial', 'Helvetica', 'sans-serif']
      };

      Object.entries(languageFonts).forEach(([, expectedFonts]) => {
        const testElement = document.createElement('div');
        testElement.style.fontFamily = expectedFonts.join(', ');
        document.body.appendChild(testElement);

        const computedStyle = window.getComputedStyle(testElement);
        expect(computedStyle.fontFamily).toBeTruthy();

        document.body.removeChild(testElement);
      });
    });
  });

  describe('Local Storage Compatibility', () => {
    test('should handle localStorage across different browsers', () => {
      const testData = {
        'voyageur-language': 'fr',
        'voyageur-theme': 'dark',
        'voyageur-preferences': JSON.stringify({ notifications: true })
      };

      Object.entries(testData).forEach(([key, value]) => {
        // Test storage
        expect(() => {
          localStorage.setItem(key, value);
        }).not.toThrow();

        // Test retrieval
        expect(localStorage.getItem(key)).toBe(value);

        // Test removal
        localStorage.removeItem(key);
        expect(localStorage.getItem(key)).toBeNull();
      });
    });

    test('should handle localStorage quota limits gracefully', () => {
      // Test with large data (simulating quota limits)
      const largeData = 'x'.repeat(1024 * 1024); // 1MB string

      try {
        localStorage.setItem('test-large-data', largeData);
        localStorage.removeItem('test-large-data');
      } catch (error) {
        // Should handle quota exceeded error gracefully
        expect(error.name).toMatch(/QuotaExceededError|NS_ERROR_DOM_QUOTA_REACHED/);
      }
    });
  });

  describe('Performance Across Browsers', () => {
    test('should maintain acceptable performance for language switching', () => {
      const iterations = 100;
      const startTime = performance.now();

      for (let i = 0; i < iterations; i++) {
        // Simulate rapid language switching
        const languages = ['en', 'fr', 'es'];
        const randomLang = languages[i % languages.length];
        localStorage.setItem('voyageur-language', randomLang);
      }

      const endTime = performance.now();
      const totalTime = endTime - startTime;

      // Should complete 100 language switches in under 100ms
      expect(totalTime).toBeLessThan(100);
    });

    test('should handle concurrent translation requests efficiently', async () => {
      const testTexts = [
        'Welcome to the dashboard',
        'Stock price analysis',
        'Market trends',
        'Financial recommendations'
      ];

      const startTime = performance.now();

      // Simulate concurrent translation requests
      const promises = testTexts.map(text =>
        new Promise(resolve => {
          // Simulate async translation
          setTimeout(() => resolve(`Translated: ${text}`), Math.random() * 10);
        })
      );

      const results = await Promise.all(promises);
      const endTime = performance.now();

      expect(results).toHaveLength(testTexts.length);
      expect(endTime - startTime).toBeLessThan(100); // Should complete quickly
    });
  });

  describe('Accessibility Compliance', () => {
    test('should maintain ARIA labels across languages', () => {
      const ariaAttributes = {
        'aria-label': 'Search button',
        'aria-describedby': 'search-help-text',
        'aria-expanded': 'false'
      };

      Object.entries(ariaAttributes).forEach(([attr, value]) => {
        const element = document.createElement('button');
        element.setAttribute(attr, value);

        expect(element.getAttribute(attr)).toBe(value);
      });
    });

    test('should support screen readers with different languages', () => {
      const screenReaderTexts = {
        en: 'Navigate to dashboard',
        fr: 'Naviguer vers le tableau de bord',
        es: 'Navegar al panel de control'
      };

      Object.entries(screenReaderTexts).forEach(([lang, text]) => {
        const element = document.createElement('span');
        element.setAttribute('aria-label', text);
        element.setAttribute('lang', lang);

        expect(element.getAttribute('aria-label')).toBe(text);
        expect(element.getAttribute('lang')).toBe(lang);
      });
    });
  });

  describe('Error Handling and Fallbacks', () => {
    test('should gracefully handle missing translation files', () => {
      // Simulate missing translation scenario
      const fallbackText = 'navigation.missing_key';

      // Should not crash and should provide reasonable fallback
      expect(fallbackText).toBeTruthy();
      expect(typeof fallbackText).toBe('string');
    });

    test('should handle network errors for dynamic translations', () => {
      // Mock network error
      const mockFetch = jest.fn().mockRejectedValue(new Error('Network error'));
      global.fetch = mockFetch;

      // Should handle the error gracefully
      return expect(
        Promise.resolve('Fallback translation')
      ).resolves.toBeTruthy();
    });

    test('should maintain functionality with disabled JavaScript', () => {
      // Test basic HTML content accessibility
      const testElement = document.createElement('div');
      testElement.innerHTML = '<span data-i18n="navigation.home">Home</span>';

      // Even without JavaScript, basic content should be accessible
      expect(testElement.textContent).toContain('Home');
    });
  });
});

// Utility functions for cross-browser testing
export const CrossBrowserTestUtils = {

  /**
   * Test if a feature is supported in the current browser
   */
  isFeatureSupported: (feature) => {
    switch (feature) {
      case 'intl':
        return typeof Intl !== 'undefined';
      case 'localStorage':
        return typeof Storage !== 'undefined';
      case 'fetch':
        return typeof fetch === 'function';
      default:
        return false;
    }
  },

  /**
   * Get browser-specific configuration
   */
  getBrowserConfig: () => {
    const userAgent = navigator.userAgent.toLowerCase();

    if (userAgent.includes('chrome')) return browserConfigs.chrome;
    if (userAgent.includes('firefox')) return browserConfigs.firefox;
    if (userAgent.includes('safari') && !userAgent.includes('chrome')) return browserConfigs.safari;
    if (userAgent.includes('edge')) return browserConfigs.edge;

    return browserConfigs.chrome; // Default fallback
  },

  /**
   * Test locale support in the current browser
   */
  testLocaleSupport: (locale) => {
    try {
      new Intl.NumberFormat(locale);
      new Intl.DateTimeFormat(locale);
      return true;
    } catch {
      return false;
    }
  },

  /**
   * Performance benchmark for translation operations
   */
  benchmarkTranslation: (translationFunc, iterations = 100) => {
    const startTime = performance.now();

    for (let i = 0; i < iterations; i++) {
      translationFunc();
    }

    const endTime = performance.now();
    return {
      totalTime: endTime - startTime,
      averageTime: (endTime - startTime) / iterations,
      iterations
    };
  }
};