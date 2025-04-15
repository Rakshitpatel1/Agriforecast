
import React from 'react';
import { ArrowUpRight } from 'lucide-react';
import Logo from './Logo';

const Footer: React.FC = () => {
  return (
    <footer className="bg-agri-green-light/30 py-16 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10">
          <div className="space-y-6">
            <Logo className="mb-4" />
            <p className="text-foreground/70 max-w-xs">
              Advanced AI-powered crop price prediction platform for farmers and agricultural businesses.
            </p>
          </div>
          
          <div className="space-y-4">
            <h3 className="font-display font-semibold text-lg">Services</h3>
            <ul className="space-y-3">
              {["Price Predictions", "Market Analysis", "Crop Insights", "Seasonal Trends"].map((item) => (
                <li key={item}>
                  <a 
                    href="#" 
                    className="text-foreground/70 hover:text-agri-green flex items-center transition-colors"
                  >
                    {item}
                    <ArrowUpRight className="ml-1 h-3 w-3 opacity-70" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          <div className="space-y-4">
            <h3 className="font-display font-semibold text-lg">Company</h3>
            <ul className="space-y-3">
              {["About Us", "How It Works", "Our Team", "Testimonials", "Contact"].map((item) => (
                <li key={item}>
                  <a 
                    href="#" 
                    className="text-foreground/70 hover:text-agri-green flex items-center transition-colors"
                  >
                    {item}
                    <ArrowUpRight className="ml-1 h-3 w-3 opacity-70" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          <div className="space-y-4">
            <h3 className="font-display font-semibold text-lg">Stay Updated</h3>
            <p className="text-foreground/70">
              Subscribe to our newsletter for the latest agricultural market insights.
            </p>
            <div className="flex overflow-hidden rounded-lg">
              <input 
                type="email" 
                placeholder="Your email" 
                className="px-4 py-3 flex-1 bg-white focus:outline-none"
              />
              <button className="bg-agri-green hover:bg-agri-green-dark text-white px-4 transition-colors">
                Subscribe
              </button>
            </div>
          </div>
        </div>
        
        <div className="border-t border-agri-green/10 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-foreground/60">
            © {new Date().getFullYear()} AgriForecast. All rights reserved.
          </p>
          <div className="flex items-center gap-6 mt-4 md:mt-0">
            <a href="#" className="text-sm text-foreground/60 hover:text-agri-green">Privacy Policy</a>
            <a href="#" className="text-sm text-foreground/60 hover:text-agri-green">Terms of Service</a>
            <a href="#" className="text-sm text-foreground/60 hover:text-agri-green">FAQ</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
