import React, { useRef, useEffect } from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { formatTimestamp, getEmotionColor } from '../utils';
import { useTheme } from '@mui/material/styles';
import { motion } from 'framer-motion';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  annotationPlugin,
  Filler
);

// Helper to create a gradient for area fill
const createAreaGradient = (ctx, chartArea, baseColor) => {
  if (!chartArea) return baseColor + '33'; // Fallback if chartArea is not defined
  const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
  gradient.addColorStop(0, baseColor + '00'); // Transparent at the bottom
  gradient.addColorStop(1, baseColor + '4D'); // Semi-transparent at the top (adjust opacity as needed)
  return gradient;
};

const EmotionTimeline = ({ data = [], currentTime = 0 }) => {
  const theme = useTheme();
  const chartRef = useRef(null);
  const containerRef = useRef();

  useEffect(() => {
    if (chartRef.current && currentTime !== undefined) { // Ensure currentTime is defined
      const chart = chartRef.current;
      if (chart.options.plugins.annotation?.annotations?.currentTime) {
        chart.options.plugins.annotation.annotations.currentTime.value = currentTime;
        chart.update('none'); // Use 'none' for no animation during update
      }
    }
  }, [currentTime]);

  if (!data || data.length === 0) {
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
            No emotion timeline data
          </Typography>
        </motion.div>
      </Box>
    );
  }

  const groupedByEmotion = data.reduce((acc, item) => {
    if (!acc[item.emotion]) {
      acc[item.emotion] = [];
    }
    acc[item.emotion].push(item);
    return acc;
  }, {});

  const datasets = Object.entries(groupedByEmotion).map(([emotion, points]) => {
    const color = getEmotionColor(emotion);
    return {
      label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      data: points.map(point => ({
        x: point.time, // Assuming point.time is in seconds
        y: point.intensity,
      })),
      borderColor: color,
      backgroundColor: (context) => {
        const chart = context.chart;
        const { ctx, chartArea } = chart;
        return createAreaGradient(ctx, chartArea, color);
      },
      borderWidth: 2,
      pointRadius: 2, // Smaller default points
      pointHoverRadius: 6, // Larger on hover
      tension: 0.4,
      fill: true, // Fill area under line
    };
  });

  const createAnnotation = (currentVal) => ({
    type: 'line',
    scaleID: 'x',
    value: currentVal,
    borderColor: 'rgba(255, 0, 0, 0.7)', // Brighter red for visibility
    borderWidth: 2,
    borderDash: [6, 6], // Dashed line
    label: {
      display: true,
      content: 'Current',
      position: 'end',
      backgroundColor: 'rgba(255, 0, 0, 0.7)',
      font: { size: 10, weight: 'bold', family: 'Roboto, sans-serif' },
      color: '#fff',
      padding: 4,
      borderRadius: 4,
    },
  });

  const chartData = {
    datasets
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          boxWidth: 8,
          padding: 20,
          font: { size: 13, family: 'Roboto, sans-serif' },
          color: '#333',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0,0,0,0.8)',
        titleFont: { size: 14, family: 'Roboto, sans-serif', weight: '600' },
        bodyFont: { size: 12, family: 'Roboto, sans-serif' },
        padding: 10,
        callbacks: {
          title: (tooltipItems) => {
            return `Time: ${formatTimestamp(tooltipItems[0].parsed.x)}`;
          },
          label: (context) => {
            const intensity = context.parsed.y;
            let intensityLabel = 'Low'; // Changed from Mild for broader appeal
            if (intensity >= 0.4 && intensity < 0.7) intensityLabel = 'Medium'; // Adjusted thresholds
            else if (intensity >= 0.7) intensityLabel = 'High';

            return `${context.dataset.label}: ${intensityLabel} (${Math.round(intensity * 100)}%)`;
          }
        }
      },
      annotation: {
        annotations: {
          currentTime: createAnnotation(currentTime !== undefined ? currentTime : data.length > 0 ? data[0].time : 0),
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: 'Time',
          font: { size: 14, family: 'Roboto, sans-serif', weight: '500' },
          color: '#444',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)', // Lighter grid
        },
        ticks: {
          callback: (value) => formatTimestamp(value),
          // consider adding adapter for dynamic ticks if videos vary greatly in length
          font: { size: 12, family: 'Roboto, sans-serif' },
          color: '#555',
        },
        min: data.length > 0 ? Math.max(0, data[0].time - 5) : 0, // Adjusted min padding
        max: data.length > 0 ? (data[data.length - 1].time + 5) : 100, // Adjusted max padding
      },
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Emotion Intensity',
          font: { size: 14, family: 'Roboto, sans-serif', weight: '500' },
          color: '#444',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)', // Lighter grid
        },
        ticks: {
          callback: (value) => `${Math.round(value * 100)}%`,
          font: { size: 12, family: 'Roboto, sans-serif' },
          color: '#555',
        },
      },
    },
    animation: {
      duration: 800,
      easing: 'easeOutCubic',
    },
  };

  return (
    <Box sx={{ height: { xs: 300, sm: 350, md: 400 }, p: 2, backgroundColor: '#fff', borderRadius: 2, boxShadow: '0 3px 10px rgb(0 0 0 / 0.1)' }}>
      <Line ref={chartRef} data={chartData} options={options} />
    </Box>
  );
};

export default EmotionTimeline;
