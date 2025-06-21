import React from 'react';
import { Box, Typography } from '@mui/material';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { motion } from 'framer-motion';
import { getEmotionColor } from '../utils';
import ShowChartIcon from '@mui/icons-material/ShowChart';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// Create gradient background for bars
const createGradient = (ctx, chartArea, color) => {
  if (!chartArea) return color;
  const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
  gradient.addColorStop(0, `${color}90`);
  gradient.addColorStop(0.5, `${color}D0`);
  gradient.addColorStop(1, color);
  return gradient;
};

// Enhanced chart options
const options = {
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    duration: 1500,
    easing: 'easeOutQuart',
  },
  plugins: {
    legend: {
      display: false,
    },
    tooltip: {
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      titleColor: '#111827',
      bodyColor: '#374151',
      titleFont: {
        size: 14,
        weight: 'bold',
        family: 'Inter, sans-serif',
      },
      bodyFont: {
        size: 12,
        family: 'Inter, sans-serif',
      },
      padding: 12,
      borderColor: 'rgba(229, 231, 235, 0.8)',
      borderWidth: 1,
      cornerRadius: 8,
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
      boxPadding: 6,
      usePointStyle: true,
      callbacks: {
        title: (items) => {
          return items[0].label;
        },
        label: (context) => {
          return `Frequency: ${Math.round(context.parsed.y * 100)}%`;
        }
      }
    },
  },
  scales: {
    x: {
      grid: {
        display: false,
        drawBorder: false,
      },
      ticks: {
        font: {
          family: 'Inter, sans-serif',
          size: 11,
        },
        color: '#6B7280',
      },
      border: {
        display: false,
      },
    },
    y: {
      beginAtZero: true,
      max: 1,
      grid: {
        color: 'rgba(243, 244, 246, 0.8)',
        drawBorder: false,
      },
      border: {
        display: false,
      },
      ticks: {
        font: {
          family: 'Inter, sans-serif',
          size: 11,
        },
        color: '#6B7280',
        callback: (value) => {
          return `${Math.round(value * 100)}%`;
        },
      },
    },
  },
  layout: {
    padding: {
      top: 10,
      right: 10,
      bottom: 0,
      left: 10,
    },
  },
  barPercentage: 0.7,
  categoryPercentage: 0.7,
};

const EmotionBarChart = ({ data = {} }) => {
  // Format data for the chart
  const emotionData = Object.entries(data).map(([emotion, value]) => ({
    emotion,
    value: parseFloat(value.toFixed(2)),
  }));

  // Sort by value in descending order
  emotionData.sort((a, b) => b.value - a.value);

  // If no data or empty object, show empty state
  if (!data || Object.keys(data).length === 0) {
    return (
      <Box sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(255, 255, 255, 0.5)',
        borderRadius: '12px',
      }}>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
          }}
        >
          <Box sx={{
            width: 80,
            height: 80,
            borderRadius: '50%',
            background: 'rgba(240, 242, 245, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
            position: 'relative'
          }}>
            <ShowChartIcon sx={{
              fontSize: '2.5rem',
              color: 'rgba(99, 102, 241, 0.4)',
              opacity: 0.7,
            }} />

            <motion.div
              animate={{
                rotate: 360,
                opacity: [0.3, 0.8, 0.3],
                scale: [0.95, 1.05, 0.95],
              }}
              transition={{
                duration: 8,
                repeat: Infinity,
                ease: "linear"
              }}
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                border: '1px dashed rgba(99, 102, 241, 0.3)',
              }}
            />
          </Box>

          <Typography
            variant="body2"
            align="center"
            color="text.secondary"
            sx={{ fontWeight: 500 }}
          >
            No emotion data to display
          </Typography>
        </motion.div>
      </Box>
    );
  }

  // Sort emotions by frequency (highest to lowest)
  const sortedEmotions = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .map(entry => entry[0]);

  const chartData = {
    labels: sortedEmotions.map(emotion => emotion.charAt(0).toUpperCase() + emotion.slice(1)),
    datasets: [
      {
        label: 'Frequency',
        data: sortedEmotions.map(emotion => data[emotion]),
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          if (!chartArea) {
            // Fallback for initial render if chartArea is not defined
            return sortedEmotions.map(emotion => getEmotionColor(emotion));
          }
          return sortedEmotions.map(emotion => createGradient(ctx, chartArea, getEmotionColor(emotion)));
        },
        borderColor: sortedEmotions.map(emotion => getEmotionColor(emotion)),
        borderWidth: 1,
        borderRadius: 8, // More rounded bars
        hoverBackgroundColor: sortedEmotions.map(emotion => getEmotionColor(emotion) + 'E0'),
        borderSkipped: false, // Don't skip any sides for fully rounded corners
      },
    ],
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.7 }}
      style={{ height: '100%', position: 'relative' }}
    >
      <Bar data={chartData} options={options} />
    </motion.div>
  );
};

export default EmotionBarChart;
