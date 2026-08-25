/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: '#F9F7F4',
        card: '#FFFBF8',
        primary: '#8B2D2D',
        accent: '#D17F7F',
        text: '#2C2C2C',
        lightBorder: '#E8DFD9',
        success: '#6B9E7E',
        warning: '#C89968',
        critical: '#A84040',
      },
      borderRadius: {
        'medical': '12px',
      }
    },
  },
  plugins: [],
}