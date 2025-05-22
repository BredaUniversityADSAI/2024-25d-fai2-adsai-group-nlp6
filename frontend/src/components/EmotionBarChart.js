import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { useTheme } from '@mui/material/styles';
import { getEmotionColor } from '../utils';
import { motion } from 'framer-motion';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Helper to create a gradient
const createGradient = (ctx, chartArea, color) => {
  if (!chartArea) { // Return a fallback if chartArea is not available
    return color;
  }
  const gradient = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
  gradient.addColorStop(0, color + 'E0'); // Slightly more opaque start
  gradient.addColorStop(1, color + 'B0'); // Slightly more transparent end
  return gradient;
};

const EmotionBarChart = ({ data = {} }) => {
  const theme = useTheme();

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
        justifyContent: 'center'
      }}>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
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
            <AccessTimeIcon sx={{ fontSize: '2.5rem', color: 'rgba(99, 102, 241, 0.4)' }} />

            <Box
              sx={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                border: '1px dashed rgba(99, 102, 241, 0.2)',
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
        borderRadius: 4, // Rounded bars
        hoverBackgroundColor: sortedEmotions.map(emotion => getEmotionColor(emotion) + 'E0'),
      },
    ],
  };

  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Emotion Distribution',
        font: {
          size: 18, // Increased title font size
          weight: '600', // Bolder title
          family: 'Roboto, sans-serif', // Consistent font
        },
        color: '#333', // Darker title color
        padding: {
          top: 15,
          bottom: 25,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0,0,0,0.75)', // Darker tooltip
        titleFont: {
          size: 14,
          family: 'Roboto, sans-serif',
        },
        bodyFont: {
          size: 12,
          family: 'Roboto, sans-serif',
        },
        callbacks: {
          label: (context) => {
            const total = Object.values(data).reduce((sum, value) => sum + value, 0);
            const percentage = total > 0 ? Math.round((context.raw / total) * 100) : 0;
            return `${context.dataset.label}: ${context.raw} (${percentage}%)`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        grid: {
          display: true,
          drawBorder: false, // Cleaner look
          color: 'rgba(0, 0, 0, 0.08)', // Lighter grid lines
        },
        ticks: {
          precision: 0,
          font: {
            size: 12,
            family: 'Roboto, sans-serif',
          },
          color: '#555', // Axis tick color
        },
      },
      y: {
        grid: {
          display: false, // Hide y-axis grid lines for horizontal bar
        },
        ticks: {
          font: {
            size: 13, // Slightly larger emotion labels
            family: 'Roboto, sans-serif',
          },
          color: '#555', // Axis tick color
        },
      },
    },
    animation: {
      duration: 800, // Slightly faster animation
      easing: 'easeOutCubic',
    },
  };

  return (
    // Increased height for better readability with horizontal bars
    <Box sx={{ height: { xs: 300, sm: 350, md: 400 }, p: 2, backgroundColor: '#fff', borderRadius: 2, boxShadow: '0 3px 10px rgb(0 0 0 / 0.1)' }}>
      <Bar data={chartData} options={options} />
    </Box>
  );
};

export default EmotionBarChart;
