import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import customTheme from '../theme';

// Import distribution chart components
import EmotionDistributionChart from './EmotionDistributionChart';
import SubEmotionDistributionChart from './SubEmotionDistributionChart';
import IntensityDistributionChart from './IntensityDistributionChart';

/**
 * EmotionDistributionAnalytics Component - Advanced Data Distribution Analytics
 * 
 * Provides comprehensive data distribution visualization across three key dimensions:
 * - Emotion Distribution: Primary emotion categories analysis
 * - Sub-emotion Distribution: Detailed emotional nuances
 * - Intensity Distribution: Emotional intensity patterns
 * 
 * Features:
 * - Tabbed interface for organized data exploration
 * - Animated transitions between views
 * - Responsive chart rendering
 * - Empty state handling
 */
const EmotionDistributionAnalytics = ({ analysisData, currentTime = 0 }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const tabPanelStyle = {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative'
  };
  const tabData = [
    {
      label: 'Emotions',
      icon: '○',
      description: 'Primary emotion categories distribution',
      component: EmotionDistributionChart
    },
    {
      label: 'Sub-emotions',
      icon: '◇',
      description: 'Detailed emotional nuances analysis',
      component: SubEmotionDistributionChart
    },
    {
      label: 'Intensity',
      icon: '△',
      description: 'Emotional intensity patterns',
      component: IntensityDistributionChart
    }
  ];

  // Show empty state when no data is available
  if (!analysisData || !analysisData.transcript || analysisData.transcript.length === 0) {
    return (
      <Box sx={{
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 2,
        color: customTheme.colors.text.secondary,
        textAlign: 'center',
        position: 'relative'
      }}>
        {/* Animated Background Elements */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          gap: 1,
          alignItems: 'center'
        }}>
          {['🧠 Deep Learning', '⚡ Real-time Processing', '🎯 Precision Analytics'].map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2, duration: 0.6 }}
            >
              <Box sx={{
                px: 1.5,
                py: 0.6,
                borderRadius: '16px',
                background: `linear-gradient(135deg, ${customTheme.colors.secondary.main}20, ${customTheme.colors.primary.main}15)`,
                border: `1px solid ${customTheme.colors.secondary.main}30`,
                fontSize: '0.7rem',
                fontWeight: 600,
                color: customTheme.colors.text.primary,
                animation: `featurePill ${2 + index * 0.5}s ease-in-out infinite`,
                animationDelay: `${index * 0.3}s`,
                '@keyframes featurePill': {
                  '0%, 100%': { transform: 'scale(1)', opacity: 0.8 },
                  '50%': { transform: 'scale(1.05)', opacity: 1 }
                }
              }}>
                {feature}
              </Box>
            </motion.div>
          ))}
        </Box>
          <Typography variant="body2" sx={{ 
          opacity: 0.9,
          maxWidth: '200px',
          lineHeight: 1.5,
          color: customTheme.colors.text.primary,
          fontWeight: 500,
          fontSize: '0.8rem',
          mt: 1
        }}>
          Emotion distribution analytics
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      position: 'relative'
    }}>      {/* Tabs Navigation */}
      <Box sx={{ 
        mb: 2
      }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="fullWidth"
          sx={{
            '& .MuiTab-root': { 
              color: 'text.secondary',
              textTransform: 'none',
              fontSize: '0.75rem',
              backgroundColor: 'rgba(30, 41, 59, 0.4)',
              borderRadius: customTheme.borderRadius.md,
              mx: 0.5,
              border: `1px solid ${customTheme.colors.border}`,
              minHeight: '40px',
              '&:hover': {
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                color: customTheme.colors.primary.main,
              }
            },
            '& .Mui-selected': { 
              color: customTheme.colors.primary.main,
              backgroundColor: 'rgba(99, 102, 241, 0.25)',
              fontWeight: 600,
              border: `1px solid ${customTheme.colors.primary.main}40`,
            },
            '& .MuiTabs-indicator': {
              backgroundColor: customTheme.colors.primary.main,
              height: 3,
              borderRadius: '2px',
            }
          }}
        >
          {tabData.map((tab, index) => (
            <Tab
              key={index}
              label={
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1
                }}>
                  <Box sx={{ fontSize: '1rem' }}>{tab.icon}</Box>
                  <Typography variant="caption" sx={{ 
                    fontWeight: 600,
                    fontSize: '0.7rem'
                  }}>
                    {tab.label}
                  </Typography>
                </Box>
              }
            />
          ))}
        </Tabs>
      </Box>

      {/* Tab Panels with Animation */}
      <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <AnimatePresence mode="wait">
          {tabData.map((tab, index) => {
            if (index !== activeTab) return null;
              const ChartComponent = tab.component;
            
            return (
              <motion.div
                key={`tab-panel-${index}`}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}                transition={{ 
                  duration: 0.3, 
                  ease: "easeOut"
                }}
                style={tabPanelStyle}
              >                <Box sx={{ 
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column'
                }}>
                  {/* Chart Container - Maximized */}
                  <Box sx={{ 
                    flex: 1,
                    height: '100%'
                  }}>
                    <ChartComponent 
                      key={`chart-${index}-${activeTab}`}
                      analysisData={analysisData}
                      currentTime={currentTime}
                    />
                  </Box>
                </Box>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </Box>
    </Box>
  );
};

export default EmotionDistributionAnalytics;
