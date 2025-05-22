import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { getEmotionColor } from '../utils';

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

const EmotionBarChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <Paper elevation={0} sx={{ p: 3, bgcolor: 'grey.100', borderRadius: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography align="center" color="textSecondary">
          No emotion distribution data available
        </Typography>
      </Paper>
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
