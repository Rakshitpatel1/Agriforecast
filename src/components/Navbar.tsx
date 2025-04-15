
import React, { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import Logo from './Logo';
import { Button } from '@/components/ui/button';

const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300 ease-in-out py-4 px-6 md:px-10",
        scrolled ? "bg-white/80 backdrop-blur-md shadow-sm" : "bg-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Logo />
        
        <nav className="hidden md:flex items-center space-x-8">
          <a 
            href="#" 
            className="text-foreground/80 hover:text-foreground transition-colors font-medium"
          >
            Predictions
          </a>
          <a 
            href="#" 
            className="text-foreground/80 hover:text-foreground transition-colors font-medium"
          >
            About
          </a>
          <a 
            href="#" 
            className="text-foreground/80 hover:text-foreground transition-colors font-medium"
          >
            Markets
          </a>
          <a 
            href="#" 
            className="text-foreground/80 hover:text-foreground transition-colors font-medium"
          >
            Contact
          </a>
        </nav>
        
        <div className="flex items-center gap-4">
          <Button 
            variant="ghost" 
            className="hidden md:flex"
          >
            Sign In
          </Button>
          <Button 
            className="bg-agri-green hover:bg-agri-green-dark text-white"
          >
            Get Started
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
