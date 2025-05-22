import React from 'react';
import { Box } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

const EmotionTimeline = ({ data, currentTime }) => {
  // Format timestamp display for x-axis
  const formatTimestamp = (seconds) => {
    if (seconds === undefined || seconds === null || isNaN(seconds)) {
      return "0:00";
    }

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // general animation time
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          usePointStyle: true,
          pointStyle: 'circle',
          font: {
            size: 10,
            family: "'Inter', sans-serif",
            weight: 500
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        titleColor: '#333',
        bodyColor: '#555',
        bodyFont: {
          family: "'Inter', sans-serif",
          size: 12
        },
        titleFont: {
          family: "'Inter', sans-serif",
          size: 13,
          weight: 600
        },
        padding: 12,
        borderColor: 'rgba(0, 0, 0, 0.05)',
        borderWidth: 1,
        boxPadding: 4,
        usePointStyle: true,
        boxWidth: 10,
        boxHeight: 10,
        callbacks: {
          title: (items) => {
            if (!items.length) return '';
            const seconds = parseFloat(items[0].label);
            return `Time: ${formatTimestamp(seconds)}`;
          }
        },
        mode: 'index',
        intersect: false,
      },
      annotation: {
        annotations: {}
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Timeline (seconds)',
          color: 'rgba(0, 0, 0, 0.6)',
          font: {
            weight: 500,
            size: 11
          }
        },
        ticks: {
          callback: (value, index, ticks) => {
            return formatTimestamp(value);
          },
          maxRotation: 0,
          autoSkip: true,
          font: {
            size: 10,
            family: "'Inter', sans-serif",
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.04)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Emotion Intensity',
          color: 'rgba(0, 0, 0, 0.6)',
          font: {
            weight: 500,
            size: 11
          }
        },
        suggestedMin: 0,
        suggestedMax: 1,
        ticks: {
          stepSize: 0.2,
          font: {
            size: 10,
            family: "'Inter', sans-serif",
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.04)',
        },
      },
    },
    elements: {
      line: {
        tension: 0.4, // Smoother curves
      },
      point: {
        radius: 2,
        hitRadius: 6,
        hoverRadius: 5,
      },
    },
    hover: {
      mode: 'nearest',
      intersect: false
    },
  };

  // Add vertical line for current time position when available
  if (currentTime !== undefined && currentTime !== null) {
    options.plugins.annotation.annotations.currentTime = {
      type: 'line',
      scaleID: 'x',
      value: currentTime,
      borderColor: 'rgba(0, 0, 0, 0.7)',
      borderWidth: 2,
      label: {
        content: 'Current',
        enabled: true,
        position: 'top',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        color: '#fff',
        font: {
          size: 9
        },
        padding: 2
      }
    };
  }

  return (
    <Box sx={{
      width: '100%',
      height: 280,
      position: 'relative',
      pb: 3,
      pt: 1,
    }}>
      {data && data.datasets && data.datasets.length > 0 ? (
        <Line
          options={options}
          data={data}
        />
      ) : (
        <Box sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
          fontSize: '0.875rem',
          opacity: 0.7,
          fontStyle: 'italic'
        }}>
          No timeline data available
        </Box>
      )}
    </Box>
  );
};

export default EmotionTimeline;
