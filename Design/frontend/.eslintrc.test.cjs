module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  globals: {
    vi: 'readonly',
    describe: 'readonly',
    it: 'readonly',
    expect: 'readonly',
    beforeEach: 'readonly',
    afterEach: 'readonly',
    beforeAll: 'readonly',
    afterAll: 'readonly',
  },
  overrides: [
    {
      files: ['**/*.test.js', '**/*.test.jsx', '**/test/**/*.js', '**/test/**/*.jsx'],
      env: {
        jest: true,
      },
    },
  ],
}