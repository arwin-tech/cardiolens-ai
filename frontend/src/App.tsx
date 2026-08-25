import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ServiceStatus from './components/ServiceStatus';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import About from './pages/About';

export default function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-[#F9F7F4] text-[#2C2C2C]">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/results" element={<Results />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
        <ServiceStatus />
      </div>
    </Router>
  );
}