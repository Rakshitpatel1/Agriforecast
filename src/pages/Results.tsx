
import React from 'react';
import Navbar from '@/components/Navbar';
import PredictionResults from '@/components/PredictionResults';
import Footer from '@/components/Footer';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';

const Results: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const predictionData = location.state?.predictionData;
  
  if (!predictionData) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-white to-agri-earth-light/30 flex flex-col">
        <Navbar />
        <div className="flex-1 flex flex-col items-center justify-center p-6">
          <h1 className="text-2xl font-semibold mb-4">No prediction data found</h1>
          <p className="text-muted-foreground mb-8">Please go back and submit the prediction form first.</p>
          <Button 
            variant="outline" 
            onClick={() => navigate('/')}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
        </div>
        <Footer />
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-agri-earth-light/30 flex flex-col">
      <Navbar />
      
      <main className="flex-1">
        <div className="container px-4 py-8">
          <Button 
            variant="outline" 
            onClick={() => navigate('/')}
            className="mb-8 flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
          
          <PredictionResults data={predictionData} />
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Results;
