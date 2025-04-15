
import React from 'react';
import { cn } from '@/lib/utils';
import { Sprout } from 'lucide-react';

interface LogoProps {
  className?: string;
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({ className, iconOnly = false }) => {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative">
        <Sprout className="h-7 w-7 text-agri-green animate-pulse-subtle" />
      </div>
      {!iconOnly && (
        <span className="font-display font-semibold text-xl tracking-tight">
          <span className="text-agri-green">Agri</span>Forecast
        </span>
      )}
    </div>
  );
};

export default Logo;
