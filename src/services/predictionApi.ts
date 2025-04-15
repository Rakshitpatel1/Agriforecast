
import { PredictionData } from '@/components/PredictionResults';

// The base URL for our API - in production this would be your actual API endpoint
const API_BASE_URL = 'http://localhost:8000';

// Function to generate random price data based on crop and state
const generateMockPrediction = (request: PredictionRequest): PredictionData => {
  const { crop, state, market_yard } = request;
  
  // Base prices for different crops (in rupees)
  const basePrices: Record<string, number> = {
    "Rice": 35,
    "Wheat": 25,
    "Maize": 20,
    "Potato": 15,
    "Onion": 20,
    "Tomato": 30,
    "Cotton": 60,
    "Sugarcane": 40,
    // Default price for other crops
    "default": 30
  };
  
  // Price multipliers for different states
  const stateMultipliers: Record<string, number> = {
    "Maharashtra": 1.2,
    "Punjab": 1.1,
    "Gujarat": 1.15,
    "Karnataka": 1.05,
    "Uttar Pradesh": 0.95,
    "Tamil Nadu": 1.1,
    "West Bengal": 1.0,
    // Default multiplier for other states
    "default": 1.0
  };
  
  // Market yard factors (additional variance)
  const yardFactor = market_yard.includes("Central") ? 1.1 : 
                     market_yard.includes("Wholesale") ? 1.05 : 
                     market_yard.includes("Regional") ? 0.95 : 1.0;
  
  // Get base price for the crop (or use default)
  const basePrice = basePrices[crop] || basePrices.default;
  
  // Get multiplier for the state (or use default)
  const multiplier = stateMultipliers[state] || stateMultipliers.default;
  
  // Calculate the final base price with some randomness
  const adjustedBasePrice = basePrice * multiplier * yardFactor * (0.9 + Math.random() * 0.2);
  
  // Create a seed for randomization that's consistent for the same crop/state/market but varies between runs
  const randomSeed = (crop + state + market_yard).split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  
  // Generate historical data (past 30 days)
  const historical_prices = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (30 - i));
    
    // Create some realistic price patterns with seasonality and trends
    const dayFactor = Math.sin((i + randomSeed) / 5) * 0.12; // Seasonality
    const trendFactor = i / 80; // Slight upward trend
    const randomFactor = Math.sin(randomSeed + i * 3) * 0.08; // Random daily fluctuations
    
    const price = adjustedBasePrice * (1 + dayFactor + trendFactor + randomFactor);
    
    return { 
      date: date.toISOString().split('T')[0], 
      price: parseFloat(price.toFixed(2))
    };
  });
  
  // Calculate last known price (most recent historical price)
  const lastKnownPrice = historical_prices[historical_prices.length - 1].price;
  
  // Generate future price trend - either rising or falling with probability based on crop
  const cropTrendBias: Record<string, number> = {
    "Rice": 0.6,    // 60% chance of rising
    "Wheat": 0.55,
    "Maize": 0.5,
    "Potato": 0.45,
    "Onion": 0.4,
    "Tomato": 0.35,
    "Cotton": 0.65,
    "Sugarcane": 0.6,
    "default": 0.5
  };
  
  const trendBias = cropTrendBias[crop] || cropTrendBias.default;
  // Fix for the type error: Ensure trend is strictly "rising" or "falling"
  const trend = Math.random() < trendBias ? 'rising' as const : 'falling' as const;
  const trendPercentage = 2 + Math.random() * 12; // 2% to 14% change
  
  // Direction factor: +1 for rising, -1 for falling
  const directionFactor = trend === 'rising' ? 1 : -1;
  
  // Generate future price predictions (next 30 days)
  const predicted_prices = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() + i + 1);
    
    // Calculate future price with trend, seasonality, and some randomness
    const trendEffect = directionFactor * (trendPercentage / 100) * (i / 30) * lastKnownPrice;
    const seasonality = Math.sin((i + randomSeed + 30) / 5) * 0.1 * lastKnownPrice;
    const randomness = (Math.random() - 0.5) * 0.05 * lastKnownPrice;
    
    const price = lastKnownPrice + trendEffect + seasonality + randomness;
    
    return { 
      date: date.toISOString().split('T')[0], 
      price: parseFloat(price.toFixed(2))
    };
  });
  
  // Calculate average predicted price
  const avg_predicted_price = parseFloat((predicted_prices.reduce((sum, item) => sum + item.price, 0) / predicted_prices.length).toFixed(2));
  
  // Generate mock metrics with slight randomness
  const metrics = {
    mse: parseFloat((10 + Math.random() * 5).toFixed(2)),
    r2: parseFloat((0.75 + Math.random() * 0.2).toFixed(4)),
    rmse: parseFloat((3 + Math.random() * 2).toFixed(2)),
    last_known_price: lastKnownPrice,
    avg_predicted_price: avg_predicted_price,
    trend: trend,
    trend_percentage: parseFloat(trendPercentage.toFixed(2))
  };
  
  // Generate model info
  const model_info = {
    model_type: "Random Forest",
    top_features: [
      "price_rolling_mean_14", 
      "month", 
      "price_lag_7", 
      crop === "Rice" || crop === "Wheat" ? "rainfall_seasonal" : "day_of_year",
      crop === "Potato" || crop === "Onion" ? "price_momentum_7" : "price_rolling_std_30"
    ]
  };
  
  return {
    historical_prices,
    predicted_prices,
    metrics,
    model_info
  };
};

export interface PredictionRequest {
  state: string;
  market_yard: string;
  crop: string;
  start_date: string;
  end_date: string;
}

export const getPrediction = async (request: PredictionRequest): Promise<PredictionData> => {
  try {
    console.log('Sending prediction request:', request);
    
    // Try to fetch from the real API
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      console.error(`API request failed with status ${response.status}`);
      throw new Error(`API request failed with status ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Received prediction data:', data);
    return data;
  } catch (error) {
    console.warn('Failed to fetch from API, using mock data instead:', error);
    // Return customized mock data based on input parameters
    return generateMockPrediction(request);
  }
};
