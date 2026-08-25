import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Analyze Risk', path: '/analyze' },
    { name: 'Results', path: '/results' },
    { name: 'About Model', path: '/about' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#FFFBF8]/90 backdrop-blur-md border-b border-[#E8DFD9]">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-3">
          <div className="p-2 bg-[#8B2D2D] text-white rounded-medical">
            <Activity className="h-5 w-5" />
          </div>
          <span className="font-bold text-xl tracking-tight text-[#8B2D2D]">CardioLens<span className="text-[#2C2C2C] font-normal">.AI</span></span>
        </Link>
        <nav className="flex items-center space-x-8">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`text-sm font-medium transition-colors ${
                location.pathname === link.path ? 'text-[#8B2D2D] font-semibold' : 'text-gray-600 hover:text-[#8B2D2D]'
              }`}
            >
              {link.name}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}