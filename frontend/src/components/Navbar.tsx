import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import medallion from '../assets/logo-medallion.png';
import filigree from '../assets/filigree-corner.webp';

const LINKS = [
  { name: 'Home', path: '/' },
  { name: 'Analyze Risk', path: '/analyze' },
  { name: 'Results', path: '/results' },
  { name: 'About Model', path: '/about' },
];

export default function Navbar() {
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <header className="site-header">
      <img src={filigree} alt="" aria-hidden="true" className="header-filigree header-filigree--left" />
      <img src={filigree} alt="" aria-hidden="true" className="header-filigree header-filigree--right" />

      <div className="site-header__inner">
        <Link to="/" className="brand" aria-label="CardioLens AI, home">
          <img src={medallion} alt="" className="brand__medallion" width={56} height={56} />
          <span className="brand__name">CardioLens.AI</span>
        </Link>

        <nav className="nav" aria-label="Main">
          {LINKS.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className="nav__link"
              aria-current={pathname === link.path ? 'page' : undefined}
            >
              {link.name}
            </Link>
          ))}
        </nav>

        <button
          type="button"
          className="nav-toggle"
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <nav id="mobile-nav" className="mobile-nav" aria-label="Main">
          {LINKS.map((link) => (
            <Link key={link.path} to={link.path} aria-current={pathname === link.path ? 'page' : undefined}>
              {link.name}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
