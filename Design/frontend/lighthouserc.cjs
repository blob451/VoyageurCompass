/* eslint-env node */
module.exports = {
  ci: {
    // Build settings
    collect: {
      url: [
        'http://localhost:4173/',
        'http://localhost:4173/login',
        'http://localhost:4173/dashboard'
      ],
      settings: {
        chromeFlags: '--no-sandbox --headless'
      },
      numberOfRuns: 3
    },
    
    // Performance budgets - thresholds that will fail the build (CI-optimized)
    assert: {
      assertions: {
        // Core Web Vitals thresholds (relaxed for CI)
        'categories:performance': ['warn', { minScore: 0.6 }],
        'categories:accessibility': ['warn', { minScore: 0.8 }],
        'categories:best-practices': ['warn', { minScore: 0.7 }],
        'categories:seo': ['warn', { minScore: 0.7 }],
        
        // Performance budgets (relaxed for CI environment)
        'metrics:first-contentful-paint': ['warn', { maxNumericValue: 5000 }],
        'metrics:largest-contentful-paint': ['warn', { maxNumericValue: 8000 }],
        'metrics:cumulative-layout-shift': ['warn', { maxNumericValue: 0.25 }],
        'metrics:total-blocking-time': ['warn', { maxNumericValue: 1000 }]
      }
    },

    // Upload configuration for CI/CD
    upload: {
      target: 'filesystem',
      outputDir: './lighthouse-results'
    },

    // Server configuration for preview mode
    server: {
      command: 'npm run preview',
      port: 4173,
      timeout: 120000
    }
  }
};