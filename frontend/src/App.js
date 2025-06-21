import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import MainLayout from './components/layout/MainLayout';
import PlaceholderModule from './components/modules/PlaceholderModule';
import { 
  TrendingUp as TrendingUpIcon, 
  Timeline as TimelineIcon, 
  VideoCall as VideoCallIcon,
  Dashboard as DashboardIcon 
} from '@mui/icons-material';

/**
 * Main App Component
 * Root component that manages global state and coordinates all modules
 */
function App() {
  // Global state
  const [videoHistory, setVideoHistory] = useState([]);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Simulate loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
      // Add loading complete class to hide loading screen
      document.body.classList.add('app-loaded');
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  // Mock video history data for testing
  useEffect(() => {
    setVideoHistory([
      {
        id: '1',
        title: 'Team Meeting Discussion',
        date: new Date('2024-06-20'),
        duration: 1200,
        emotions: ['happiness', 'neutral', 'surprise'],
      },
      {
        id: '2',
        title: 'Product Demo Presentation',
        date: new Date('2024-06-19'),
        duration: 800,
        emotions: ['happiness', 'excitement', 'confidence'],
      },
      {
        id: '3',
        title: 'Customer Feedback Session',
        date: new Date('2024-06-18'),
        duration: 950,
        emotions: ['concern', 'neutral', 'sadness'],
      },
    ]);
  }, []);

  // Handlers
  const handleAddVideo = () => {
    console.log('Add video clicked');
    // TODO: Open add video modal
  };

  const handleSettings = () => {
    console.log('Settings clicked');
    // TODO: Open settings modal
  };

  const handleVideoSelect = (video) => {
    console.log('Video selected:', video);
    setCurrentVideo(video);
    // TODO: Load video and analysis data
  };

  // Show loading screen
  if (isLoading) {
    return (
      <motion.div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          color: '#f8fafc',
          fontFamily: "'Inter', sans-serif",
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          style={{
            width: '60px',
            height: '60px',
            border: '3px solid rgba(139, 92, 246, 0.2)',
            borderTop: '3px solid #8B5CF6',
            borderRadius: '50%',
            marginBottom: '24px',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <motion.h1
          style={{
            fontSize: '2rem',
            fontWeight: 600,
            marginBottom: '8px',
            background: 'linear-gradient(135deg, #8B5CF6, #06B6D4)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          Emotion Analysis Dashboard
        </motion.h1>
        <motion.p
          style={{
            fontSize: '1rem',
            color: '#cbd5e1',
            margin: 0,
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
        >
          Loading your premium experience...
        </motion.p>
      </motion.div>
    );
  }

  return (
    <div className="App">
      <MainLayout
        // Top left - placeholder for future features
        topLeft={
          <PlaceholderModule
            title="Analytics"
            subtitle="Advanced analytics coming soon"
            icon={<DashboardIcon />}
          />
        }
        
        // Bottom left - Real-Time Emotion Tracker
        bottomLeft={
          <PlaceholderModule
            title="Real-Time Tracker"
            subtitle="Live emotion detection will appear here"
            icon={<TimelineIcon />}
          />
        }
        
        // Center - Video player and controls
        center={
          <PlaceholderModule
            title="Video Center"
            subtitle="Video player, controls, and transcript will be displayed here"
            icon={<VideoCallIcon />}
            onClick={handleAddVideo}
          />
        }
        
        // Top right - Emotion Pulse
        topRight={
          <PlaceholderModule
            title="Emotion Pulse"
            subtitle="Current emotion visualization"
            icon={<TrendingUpIcon />}
          />
        }
        
        // Bottom right - placeholder for future features
        bottomRight={
          <PlaceholderModule
            title="Insights"
            subtitle="AI-powered insights coming soon"
            icon={<TrendingUpIcon />}
          />
        }
        
        // Sidebar props
        videoHistory={videoHistory}
        onAddVideo={handleAddVideo}
        onSettings={handleSettings}
        onVideoSelect={handleVideoSelect}
      />
    </div>
  );
}

export default App;
