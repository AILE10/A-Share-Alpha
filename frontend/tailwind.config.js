/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        obsidian: '#0B0F19',
        panel: '#121826',
        panelHover: '#192235',
        borderDark: '#1E293B',
        up: '#F43F5E',
        down: '#10B981',
        accent: '#38BDF8',
        gold: '#F59E0B',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
