
import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { ArrowUpCircle, ArrowDownCircle, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from '@/components/ui/table';

type Price = {
  date: string;
  price: number;
};

export type PredictionData = {
  historical_prices: Price[];
  predicted_prices: Price[];
  metrics: {
    mse: number;
    r2: number;
    rmse: number;
    last_known_price: number;
    avg_predicted_price: number;
    trend: 'rising' | 'falling';
    trend_percentage: number;
  };
  model_info: {
    model_type: string;
    top_features: string[];
  };
};

interface PredictionResultsProps {
  data: PredictionData;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ data }) => {
  const { historical_prices, predicted_prices, metrics, model_info } = data;
  
  // Combine historical and predicted data for chart
  const chartData = [
    ...historical_prices.map(item => ({ ...item, type: 'historical' })),
    ...predicted_prices.map(item => ({ ...item, type: 'predicted' }))
  ];
  
  // Format table data - show the last 7 days of historical and first 14 days of predictions
  const tableHistorical = historical_prices.slice(-7);
  const tablePredicted = predicted_prices.slice(0, 14);
  
  return (
    <section className="py-12 px-6">
      <div className="max-w-6xl mx-auto space-y-10">
        <div className="text-center mb-8 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-display font-bold mb-6">
            Price <span className="text-agri-green">Prediction Results</span>
          </h2>
          <p className="text-lg text-foreground/70 max-w-2xl mx-auto">
            The AI model has analyzed historical price trends and market conditions to forecast future prices.
          </p>
        </div>
        
        {/* Prediction Summary Card */}
        <div className={cn(
          "relative bg-white/80 backdrop-blur-md rounded-[24px] p-8 shadow-glass-lg border border-white/40",
          "animate-fade-in-up"
        )}>
          <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
          
          <div className="flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="flex items-center gap-4">
              {metrics.trend === 'rising' ? (
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-green-100">
                  <TrendingUp className="h-8 w-8 text-green-600" />
                </div>
              ) : (
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-red-100">
                  <TrendingDown className="h-8 w-8 text-red-600" />
                </div>
              )}
              
              <div>
                <h3 className="text-2xl font-semibold mb-1">
                  Prices are predicted to {metrics.trend === 'rising' ? 'rise' : 'fall'}
                </h3>
                <p className={cn(
                  "text-lg font-medium",
                  metrics.trend === 'rising' ? 'text-green-600' : 'text-red-600'
                )}>
                  {metrics.trend_percentage.toFixed(2)}% change expected
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6 text-center">
              <div>
                <p className="text-foreground/60 mb-1">Last Known Price</p>
                <p className="text-2xl font-semibold">₹{metrics.last_known_price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-foreground/60 mb-1">Avg Predicted Price</p>
                <p className="text-2xl font-semibold">₹{metrics.avg_predicted_price.toFixed(2)}</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Price Chart */}
        <div className={cn(
          "relative bg-white/80 backdrop-blur-md rounded-[24px] p-8 shadow-glass-lg border border-white/40",
          "animate-fade-in-up"
        )}>
          <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
          
          <h3 className="text-xl font-semibold mb-6">Price Trend Forecast</h3>
          
          <div className="h-[400px] w-full">
            <ChartContainer 
              config={{
                historical: { color: "#16a34a", label: "Historical" },
                predicted: { color: "#2563eb", label: "Predicted" },
              }}
              className="h-full"
            >
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  angle={-45}
                  textAnchor="end"
                  height={70}
                  tickFormatter={(date) => {
                    const d = new Date(date);
                    return `${d.getDate()}/${d.getMonth() + 1}`;
                  }}
                />
                <YAxis 
                  label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft' }}
                  width={80}
                />
                <ChartTooltip
                  content={({ active, payload }) => (
                    <ChartTooltipContent
                      active={active}
                      payload={payload}
                      labelFormatter={(label) => {
                        return <span>Date: {label}</span>;
                      }}
                    />
                  )}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="var(--color-historical)" 
                  name="Historical"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 6 }}
                  isAnimationActive={true}
                  animationDuration={1000}
                  connectNulls={true}
                  strokeOpacity={1}
                />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="var(--color-predicted)" 
                  name="Predicted"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ r: 2 }}
                  activeDot={{ r: 6 }}
                  isAnimationActive={true}
                  animationDuration={1000}
                  animationBegin={1000}
                  connectNulls={true}
                  strokeOpacity={0.8}
                />
              </LineChart>
            </ChartContainer>
          </div>
        </div>
        
        {/* Price Tables */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Historical Prices Table */}
          <div className={cn(
            "relative bg-white/80 backdrop-blur-md rounded-[24px] p-8 shadow-glass-lg border border-white/40",
            "animate-fade-in-up"
          )}>
            <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
            
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <ArrowUpCircle className="h-5 w-5 text-agri-green" />
              Recent Historical Prices
            </h3>
            
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead className="text-right">Price (₹)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tableHistorical.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{new Date(item.date).toLocaleDateString()}</TableCell>
                    <TableCell className="text-right font-medium">₹{item.price.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          
          {/* Predicted Prices Table */}
          <div className={cn(
            "relative bg-white/80 backdrop-blur-md rounded-[24px] p-8 shadow-glass-lg border border-white/40",
            "animate-fade-in-up"
          )}>
            <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
            
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <ArrowDownCircle className="h-5 w-5 text-blue-600" />
              Predicted Future Prices
            </h3>
            
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead className="text-right">Price (₹)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tablePredicted.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{new Date(item.date).toLocaleDateString()}</TableCell>
                    <TableCell className="text-right font-medium">₹{item.price.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
        
        {/* Model Information */}
        <div className={cn(
          "relative bg-white/80 backdrop-blur-md rounded-[24px] p-8 shadow-glass-lg border border-white/40",
          "animate-fade-in-up"
        )}>
          <div className="absolute inset-0 blur-3xl -z-10 bg-gradient-to-br from-agri-green-light/20 via-agri-earth-light/10 to-transparent opacity-70 rounded-[32px]" />
          
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Info className="h-5 w-5 text-agri-green" />
            Model Information
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <p className="font-medium mb-2">Model Type</p>
              <p className="text-foreground/70">{model_info.model_type}</p>
              
              <p className="font-medium mt-4 mb-2">Model Accuracy</p>
              <p className="text-foreground/70">R² Score: {metrics.r2.toFixed(4)}</p>
              <p className="text-foreground/70">RMSE: {metrics.rmse.toFixed(2)}</p>
            </div>
            
            <div>
              <p className="font-medium mb-2">Top Factors Influencing Prices</p>
              <ul className="list-disc list-inside text-foreground/70">
                {model_info.top_features.map((feature, index) => (
                  <li key={index}>{feature}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PredictionResults;
