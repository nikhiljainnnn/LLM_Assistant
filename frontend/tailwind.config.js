/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--bg-0)',
        'primary-container': 'var(--accent-4)',
        'on-primary-container': '#ffffff',
        'secondary-container': 'var(--accent)',
        'surface-container-highest': 'var(--bg-3)',
        'surface-container-low': 'var(--bg-1)',
        'on-surface': 'var(--text-0)',
        'on-surface-variant': 'var(--text-1)',
        primary: 'var(--accent)',
        error: 'var(--error)',
        'error-container': 'var(--error)'
      }
    },
  },
  plugins: [],
}
