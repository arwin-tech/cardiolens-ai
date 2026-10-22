/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: '#121110', 2: '#171615', 3: '#1E1C1A' },
        cream: '#F1E6D3',
        parchment: '#DCD3C6',
        stone: { DEFAULT: '#A8A199', 2: '#8A847C' },
        gold: { DEFAULT: '#B79A6C', bright: '#E4CC9F', dim: '#5E5546' },
        crimson: { DEFAULT: '#922F2C', deep: '#5B0B0D', ember: '#B3261E' },
        rose: '#B97A73',
        verdigris: '#5E9E6E',
        amber: '#C89A52',
      },
      fontFamily: {
        display: ['"EB Garamond"', 'Garamond', '"Times New Roman"', 'serif'],
        sans: ['"Alegreya Sans"', '"Segoe UI"', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
