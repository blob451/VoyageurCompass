import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.js',
        '**/*.config.ts',
        'dist/',
        'build/',
        'coverage/',
        '**/*.test.{js,jsx,ts,tsx}',
        '**/*.spec.{js,jsx,ts,tsx}',
      ],
      include: [
        'src/**/*.{js,jsx,ts,tsx}',
      ],
      thresholds: {
        global: {
          branches: 70,
          functions: 70,
          lines: 70,
          statements: 70
        },
        'src/components/**': {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        },
        'src/features/**': {
          branches: 75,
          functions: 75,
          lines: 75,
          statements: 75
        }
      }
    },
    // Test file patterns
    include: [
      '**/*.{test,spec}.{js,jsx,ts,tsx}'
    ],
    // Mock patterns
    mockReset: true,
    clearMocks: true,
    restoreMocks: true,
    // Timeout for long-running tests
    testTimeout: 10000,
    // Parallel test execution
    threads: true,
    // Test environment options
    environmentOptions: {
      jsdom: {
        // Disabled 'resources: usable' for better test performance
        // External resources are mocked in tests instead of loaded
        // resources: 'usable', // Enable per test if external resources needed
        url: 'http://localhost:3000'
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@features': path.resolve(__dirname, './src/features'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@test': path.resolve(__dirname, './src/test')
    }
  }
})