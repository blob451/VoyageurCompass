module.exports = {
  ci: {
    // Build settings
    collect: {
      staticDistDir: './dist',
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/login',
        'http://localhost:3000/dashboard'
      ],
      settings: {
        chromeFlags: '--no-sandbox --headless'
      },
      numberOfRuns: 3
    },
    
    // Performance budgets - thresholds that will fail the build
    assert: {
      assertions: {
        // Core Web Vitals thresholds
        'categories:performance': ['error', { minScore: 0.85 }],
        'categories:accessibility': ['error', { minScore: 0.90 }],
        'categories:best-practices': ['error', { minScore: 0.85 }],
        'categories:seo': ['error', { minScore: 0.80 }],
        
        // Performance budgets (prevent >10% regression)
        'metrics:first-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'metrics:largest-contentful-paint': ['error', { maxNumericValue: 4000 }],
        'metrics:cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'metrics:total-blocking-time': ['error', { maxNumericValue: 500 }],
        
        // Resource budgets
        'resource-summary:document:size': ['error', { maxNumericValue: 50000 }],
        'resource-summary:script:size': ['error', { maxNumericValue: 500000 }],
        'resource-summary:stylesheet:size': ['error', { maxNumericValue: 100000 }],
        'resource-summary:image:size': ['error', { maxNumericValue: 1000000 }],
        'resource-summary:font:size': ['error', { maxNumericValue: 200000 }]
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