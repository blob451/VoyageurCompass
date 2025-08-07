import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        // When running in Docker, the backend service is accessible as 'backend'
        // When running locally without Docker, change this to 'http://localhost:8000'
        target: 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
})