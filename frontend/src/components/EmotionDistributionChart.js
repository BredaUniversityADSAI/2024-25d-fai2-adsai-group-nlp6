import React, { memo } from 'react';
import { Box, Typography } from '@mui/material';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { motion } from 'framer-motion';
import { getEmotionColor, processEmotionData } from '../utils';
import customTheme from '../theme';
import InsightsIcon from '@mui/icons-material/Insights';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

/**
 * EmotionDistributionChart Component
 * 
 * Displays the distribution of primary emotions using both bar and doughnut charts
 * for comprehensive data visualization.
 * 
 * Features:
 * - Dual chart view (bar and doughnut)
 * - Animated rendering
 * - Responsive design
 * - Empty state handling
 */
const EmotionDistributionChart = memo(({ analysisData }) => {
  // Process emotion data
  const { emotionDistribution } = analysisData ? processEmotionData(analysisData) : { emotionDistribution: {} };

  // Check if we have data
  const hasData = Object.keys(emotionDistribution).length > 0;

  if (!hasData) {
    return (
      <Box sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: customTheme.colors.text.secondary,
        textAlign: 'center'
      }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <Box sx={{
            width: 80,
            height: 80,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${customTheme.colors.secondary.main}20, ${customTheme.colors.primary.main}15)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
            position: 'relative',
            border: `2px dashed ${customTheme.colors.secondary.main}30`
          }}>
            <InsightsIcon sx={{
              fontSize: '2rem',
              color: customTheme.colors.text.tertiary,
            }} />
          </Box>
          <Typography variant="body2" sx={{ 
            color: customTheme.colors.text.tertiary,
            fontSize: '0.8rem',
            fontWeight: 500
          }}>
            No emotion data available
          </Typography>
        </motion.div>
      </Box>
    );
  }

  // Prepare data for charts
  const sortedEmotions = Object.entries(emotionDistribution)
    .sort((a, b) => b[1] - a[1])
    .map(([emotion, value]) => ({ emotion, value }));

  const labels = sortedEmotions.map(item => 
    item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)
  );
  const values = sortedEmotions.map(item => item.value);
  const colors = sortedEmotions.map(item => getEmotionColor(item.emotion));

  // Bar chart configuration
  const barChartData = {
    labels,
    datasets: [{
      label: 'Distribution',
      data: values,
      backgroundColor: colors.map(color => `${color}80`),
      borderColor: colors,
      borderWidth: 2,
      borderRadius: 6,
      borderSkipped: false,
    }]
  };
  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 800,
      easing: 'easeOutQuart',
      loop: false,
      animateRotate: false,
      animateScale: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: customTheme.colors.surface.glass,
        titleColor: customTheme.colors.text.primary,
        bodyColor: customTheme.colors.text.secondary,
        borderColor: customTheme.colors.border,
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        callbacks: {
          label: (context) => `${Math.round(context.parsed.y * 100)}%`
        }
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          color: customTheme.colors.text.tertiary,
          font: { size: 10, weight: '500' }
        },
        border: { display: false }
      },
      y: {
        beginAtZero: true,
        max: 1,
        grid: {
          color: `${customTheme.colors.secondary.main}20`,
        },
        ticks: {
          color: customTheme.colors.text.tertiary,
          font: { size: 10 },
          callback: (value) => `${Math.round(value * 100)}%`
        },
        border: { display: false }
      },    },    layout: {
      padding: { top: 5, right: 5, bottom: 5, left: 5 }
    }
  };
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Single Chart Container */}
      <Box sx={{ 
        flex: 1,
        minHeight: 0
      }}>
        {/* Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          style={{ height: '100%' }}
        >
          <Box sx={{
            height: '100%',            background: `linear-gradient(135deg, ${customTheme.colors.secondary.main}05, transparent)`,
            borderRadius: customTheme.borderRadius.lg,
            border: `1px solid ${customTheme.colors.secondary.main}20`,
            p: 1
          }}>            <Box sx={{ 
              height: '100%',
              overflow: 'hidden'
            }}>
              <Bar 
                key="emotion-bar-chart"
                data={barChartData} 
                options={barChartOptions} 
              />
            </Box>
          </Box>
        </motion.div>
      </Box>
    </Box>
  );
});

EmotionDistributionChart.displayName = 'EmotionDistributionChart';

export default EmotionDistributionChart;
