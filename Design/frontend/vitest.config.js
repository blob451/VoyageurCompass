import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.jsx'],
    css: true,
    // CI optimizations
    watch: false,
    run: process.env.CI === 'true',
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
          branches: 60,
          functions: 60,
          lines: 60,
          statements: 60
        },
        'src/components/**': {
          branches: 70,
          functions: 35,
          lines: 70,
          statements: 70
        },
        'src/features/**': {
          branches: 70,
          functions: 55,
          lines: 70,
          statements: 70
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
    // Parallel test execution - reduced for CI stability
    threads: process.env.CI === 'true' ? 2 : true,
    // CI-specific performance optimizations
    minWorkers: process.env.CI === 'true' ? 1 : undefined,
    maxWorkers: process.env.CI === 'true' ? 2 : undefined,
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
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
      '@pages': fileURLToPath(new URL('./src/pages', import.meta.url)),
      '@features': fileURLToPath(new URL('./src/features', import.meta.url)),
      '@utils': fileURLToPath(new URL('./src/utils', import.meta.url)),
      '@hooks': fileURLToPath(new URL('./src/hooks', import.meta.url)),
      '@test': fileURLToPath(new URL('./src/test', import.meta.url))
    }
  }
})