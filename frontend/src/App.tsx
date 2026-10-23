import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ServiceStatus from './components/ServiceStatus';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import About from './pages/About';
import Monitoring from './pages/Monitoring';
import { BackendStatusProvider } from './context/BackendStatus';

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <Router>
      <BackendStatusProvider>
        <ScrollToTop />
        <div className="flex min-h-screen flex-col bg-ink text-stone">
          <Navbar />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/results" element={<Results />} />
              <Route path="/about" element={<About />} />
              <Route path="/monitoring" element={<Monitoring />} />
            </Routes>
          </main>
          <Footer />
          <ServiceStatus />
        </div>
      </BackendStatusProvider>
    </Router>
  );
}
