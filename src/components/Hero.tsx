
import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[85vh] flex items-center justify-center px-6 py-16 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-[25%] right-[20%] w-[25vw] h-[25vw] rounded-full bg-agri-green/5 blur-3xl" />
        <div className="absolute bottom-[10%] left-[15%] w-[20vw] h-[20vw] rounded-full bg-agri-earth/5 blur-3xl" />
      </div>
      
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
        <div className="space-y-8 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="space-y-2">
            <div className="inline-block px-3 py-1 rounded-full bg-agri-green/10 text-agri-green text-sm font-medium">
              Precision Agriculture
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-bold tracking-tight text-balance">
              <span className="block">Predict Crop Prices</span>
              <span className="block text-agri-green">With Confidence</span>
            </h1>
          </div>
          
          <p className="text-lg md:text-xl text-foreground/80 max-w-xl leading-relaxed text-balance">
            Make informed decisions with our advanced AI-powered crop price predictions. Get accurate market forecasts tailored to your region and crops.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4">
            <Button 
              size="lg"
              className="bg-agri-green hover:bg-agri-green-dark text-white rounded-full px-8"
            >
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              className="rounded-full px-8 border-agri-green/30 text-agri-green hover:border-agri-green/50 hover:bg-agri-green/5"
            >
              Learn More
            </Button>
          </div>
        </div>
        
        <div className="relative flex items-center justify-center">
          <div className={cn(
            "w-[90%] aspect-square rounded-full",
            "bg-gradient-to-br from-agri-earth-light via-agri-green-light/50 to-agri-green/10",
            "animate-pulse-subtle"
          )} />
          
          <div className="absolute inset-0 flex items-center justify-center animate-fade-in" style={{ animationDelay: '0.8s' }}>
            <div className="relative w-[85%] aspect-square rounded-3xl overflow-hidden p-1 bg-gradient-to-br from-white/80 to-white/20 backdrop-blur-sm shadow-glass-lg">
              <img 
                src="https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
                alt="Agricultural field" 
                className="w-full h-full object-cover rounded-3xl opacity-90"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
