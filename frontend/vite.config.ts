import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development, proxy API calls to the FastAPI server so we don't
// need CORS headers or a separate origin in the browser.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/caption': 'http://localhost:8000',
      '/health':  'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
    },
  },
});
