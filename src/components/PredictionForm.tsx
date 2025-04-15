
import React, { useState, useEffect } from 'react';
import { DateRange } from "react-day-picker";
import { Search, TrendingUp, MapPin, Wheat, Building2, Loader2 } from 'lucide-react';
import { DatePickerWithRange } from './DatePickerWithRange';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { getPrediction } from '@/services/predictionApi';
import { PredictionData } from './PredictionResults';

const states = [
  "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
  "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
  "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", 
  "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
];

const crops = [
  "Rice", "Wheat", "Maize", "Jowar", "Bajra", "Ragi", "Barley", "Potato", "Onion", 
  "Tomato", "Brinjal", "Cabbage", "Cauliflower", "Lady Finger", "Peas", "Garlic", 
  "Ginger", "Turmeric", "Chilli", "Cotton", "Jute", "Sugarcane", "Groundnut", 
  "Mustard", "Soybean", "Sunflower", "Coconut", "Banana", "Mango", "Grapes"
];

// Market yards by state
const marketYardsByState: Record<string, string[]> = {
  "Andhra Pradesh": ["Guntur", "Vijayawada", "Kurnool", "Kadapa", "Anantapur"],
  "Karnataka": ["Bengaluru", "Mysore", "Hubli", "Mangalore", "Belgaum"],
  "Maharashtra": ["Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur"],
  "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
  "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"],
  "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
  "West Bengal": ["Kolkata", "Siliguri", "Asansol", "Durgapur", "Howrah"],
  // Default market yards for other states
  "default": ["Central Market", "Regional Market", "District Market", "Local Market", "Wholesale Market"]
};

interface PredictionFormProps {
  onPredictionResult?: (data: PredictionData) => void;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onPredictionResult }) => {
  const [state, setState] = useState<string>("");
  const [marketYard, setMarketYard] = useState<string>("");
  const [availableMarketYards, setAvailableMarketYards] = useState<string[]>([]);
  const [crop, setCrop] = useState<string>("");
  const [date, setDate] = useState<DateRange | undefined>({
    from: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
    to: new Date()
  });
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // Update available market yards when state changes
  useEffect(() => {
    if (state) {
      const yards = marketYardsByState[state] || marketYardsByState.default;
      setAvailableMarketYards(yards);
      setMarketYard(""); // Reset market yard when state changes
    }
  }, [state]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!state || !marketYard || !crop || !date?.from || !date?.to) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }
    
    setIsLoading(true);
    
    try {
      const result = await getPrediction({
        state,
        market_yard: marketYard,
        crop,
        start_date: date.from.toISOString().split('T')[0],
        end_date: date.to.toISOString().split('T')[0]
      });
      
      // Pass the result to the parent component
      if (onPredictionResult) {
        onPredictionResult(result);
      }
      
      toast({
        title: "Prediction Completed",
        description: "Your crop price prediction is ready",
        variant: "default"
      });
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction Error",
        description: "Failed to generate prediction. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section id="prediction-form" className="py-20 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <h2 className="text-3xl md:text-4xl font-display font-bold mb-6">
            Get Your <span className="text-agri-green">Price Predictions</span>
          </h2>
          <p className="text-lg text-foreground/70 max-w-2xl mx-auto">
            Enter your details below to get accurate price forecasts for your crops. Our AI-powered model
            considers historical trends and market conditions.
          </p>
        </div>

        <div className="relative animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
          <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
          
          <div className={cn(
            "bg-white/80 backdrop-blur-md rounded-[24px] p-8 md:p-12",
            "shadow-glass-lg border border-white/40"
          )}>
            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <MapPin className="h-5 w-5 text-agri-green" />
                  <Label htmlFor="state" className="text-base font-medium">State</Label>
                </div>
                <Select value={state} onValueChange={setState}>
                  <SelectTrigger id="state" className="h-14 rounded-xl bg-agri-green-light/70 border-0 focus:ring-2 focus:ring-agri-green/20">
                    <SelectValue placeholder="Select state" />
                  </SelectTrigger>
                  <SelectContent>
                    {states.map((s) => (
                      <SelectItem key={s} value={s} className="rounded-lg my-1 focus:bg-agri-green-light focus:text-agri-green-dark">
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Building2 className="h-5 w-5 text-agri-green" />
                  <Label htmlFor="market-yard" className="text-base font-medium">Market Yard</Label>
                </div>
                <Select 
                  value={marketYard} 
                  onValueChange={setMarketYard}
                  disabled={!state} // Disable if no state is selected
                >
                  <SelectTrigger id="market-yard" className="h-14 rounded-xl bg-agri-green-light/70 border-0 focus:ring-2 focus:ring-agri-green/20">
                    <SelectValue placeholder={state ? "Select market yard" : "Please select a state first"} />
                  </SelectTrigger>
                  <SelectContent>
                    {availableMarketYards.map((yard) => (
                      <SelectItem key={yard} value={yard} className="rounded-lg my-1 focus:bg-agri-green-light focus:text-agri-green-dark">
                        {yard}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Wheat className="h-5 w-5 text-agri-green" />
                  <Label htmlFor="crop" className="text-base font-medium">Crop</Label>
                </div>
                <Select value={crop} onValueChange={setCrop}>
                  <SelectTrigger id="crop" className="h-14 rounded-xl bg-agri-green-light/70 border-0 focus:ring-2 focus:ring-agri-green/20">
                    <SelectValue placeholder="Select crop" />
                  </SelectTrigger>
                  <SelectContent>
                    {crops.map((c) => (
                      <SelectItem key={c} value={c} className="rounded-lg my-1 focus:bg-agri-green-light focus:text-agri-green-dark">
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-agri-green" />
                  <Label className="text-base font-medium">Date Range</Label>
                </div>
                <DatePickerWithRange
                  dateRange={date}
                  onDateRangeChange={setDate}
                />
              </div>

              <div className="md:col-span-2 mt-6">
                <Button 
                  type="submit" 
                  className="w-full md:w-auto px-10 py-6 rounded-xl bg-agri-green hover:bg-agri-green-dark text-white font-medium text-lg shadow-md hover:shadow-lg transition-all duration-300"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-5 w-5" />
                      Get Price Prediction
                    </>
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PredictionForm;
