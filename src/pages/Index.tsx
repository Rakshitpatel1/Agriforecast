
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import PredictionForm from '@/components/PredictionForm';
import Footer from '@/components/Footer';
import { PredictionData } from '@/components/PredictionResults';

const Index: React.FC = () => {
  const navigate = useNavigate();
  
  // Add smooth scroll behavior
  useEffect(() => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        if (targetId) {
          document.querySelector(targetId)?.scrollIntoView({
            behavior: 'smooth'
          });
        }
      });
    });
  }, []);

  const handlePredictionResult = (data: PredictionData) => {
    // Navigate to results page with the prediction data
    navigate('/results', { state: { predictionData: data } });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-agri-earth-light/30">
      <Navbar />
      <Hero />
      <PredictionForm onPredictionResult={handlePredictionResult} />
      <Footer />
    </div>
  );
};

export default Index;
