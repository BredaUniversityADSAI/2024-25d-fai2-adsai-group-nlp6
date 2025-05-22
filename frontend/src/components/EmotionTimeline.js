import React from 'react';
import { Box, Typography } from '@mui/material';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import 'chartjs-adapter-date-fns';
import { getEmotionColor } from '../utils';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
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

  const isValidData = data && data.datasets && data.datasets.length > 0;
  const emotionLabels = isValidData ? data.emotionLabels : [];

  const modifiedChartData = React.useMemo(() => {
    if (!isValidData) {
      // Return a structure that matches what Scatter expects, even if empty
      return { datasets: [], emotionLabels: [] };
    }

    return {
      ...data, // Spreads other properties from data, like emotionLabels
      datasets: data.datasets.map(dataset => {
        const originalBackgroundColor = dataset.backgroundColor;
        const originalBorderColor = dataset.borderColor;
        const originalPointRadius = dataset.pointRadius !== undefined ? dataset.pointRadius : 6;
        const originalHoverRadius = dataset.pointHoverRadius !== undefined ? dataset.pointHoverRadius : 8;

        return {
          ...dataset,
          pointRadius: function(context) {
            const xValue = context.raw?.x ?? context.parsed?.x;
            // If xValue is undefined (e.g., point is somehow invalid), use original, else apply logic
            if (xValue === undefined) return originalPointRadius;
            return xValue > currentTime ? 0 : originalPointRadius;
          },
          pointBackgroundColor: function(context) {
            const xValue = context.raw?.x ?? context.parsed?.x;
            if (xValue === undefined) return originalBackgroundColor;
            return xValue > currentTime ? 'transparent' : originalBackgroundColor;
          },
          pointBorderColor: function(context) {
            const xValue = context.raw?.x ?? context.parsed?.x;
            if (xValue === undefined) return originalBorderColor;
            return xValue > currentTime ? 'transparent' : originalBorderColor;
          },
          pointHoverRadius: function(context) {
            const xValue = context.raw?.x ?? context.parsed?.x;
            if (xValue === undefined) return originalHoverRadius;
            return xValue > currentTime ? 0 : originalHoverRadius;
          }
        };
      })
    };
  }, [data, currentTime, isValidData]);

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 200, // Faster animation for smoother currentTime updates
    },
    interaction: {
      mode: 'nearest',
      intersect: false,
      axis: 'x'
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          pointStyle: 'circle',
          font: {
            size: 10,
            family: "'Inter', sans-serif",
            weight: 500
          },
          boxWidth: 8,
          boxHeight: 8
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
            const seconds = parseFloat(items[0].parsed.x);
            return `Time: ${formatTimestamp(seconds)}`;
          },
          label: (context) => {
            const dataset = context.dataset;
            const emotion = dataset.label;
            // Only show tooltip for visible points
            const xValue = context.raw?.x ?? context.parsed?.x;
            if (xValue !== undefined && xValue > currentTime) {
              return null; // Suppress tooltip for hidden points
            }
            return emotion;
          }
        },
        mode: 'nearest',
        intersect: false,
        // Filter out tooltips for points that are effectively hidden
        filter: function(tooltipItem) {
          const xValue = tooltipItem.raw?.x ?? tooltipItem.parsed?.x;
          if (xValue === undefined) return true; // if no x-value, don't filter (shouldn't happen often)
          return xValue <= currentTime;
        }
      },
      annotation: {
        annotations: {}
      }
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: 'Timeline (MM:SS)',
          color: 'rgba(0, 0, 0, 0.6)',
          font: {
            weight: 500,
            size: 11
          }
        },
        ticks: {
          callback: (value) => formatTimestamp(value),
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
        type: 'category',
        labels: emotionLabels,
        offset: true,
        position: 'left',
        title: {
          display: true,
          text: 'Emotions',
          color: 'rgba(0, 0, 0, 0.6)',
          font: {
            weight: 500,
            size: 11
          }
        },
        ticks: {
          font: {
            size: 11,
            family: "'Inter', sans-serif",
          },
          color: (context) => {
            if (context.tick && typeof context.tick.label === 'string') {
              return getEmotionColor(context.tick.label.toLowerCase());
            }
            return '#666';
          },
          padding: 8
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.04)',
          z: 1
        },
      }
    },
    elements: {
      point: {
        // Default radius and hoverRadius are now controlled per dataset by scriptable options
        // radius: 6,
        // hoverRadius: 8,
        pointStyle: 'rectRot',
      },
    },
    parsing: {
      xAxisKey: 'x',
      yAxisKey: 'y'
    }
  };

  // Add vertical line for current time position when available
  if (currentTime !== undefined && currentTime !== null) {
    options.plugins.annotation.annotations.currentTimeLine = { // Renamed for clarity
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
      },
      // Ensure the line is drawn above the new background box but below tooltips/points if needed
      // z: 1 // Optional: z-index if layering becomes an issue, default should be fine
    };

    // Add a box annotation for the grayed-out future area
    options.plugins.annotation.annotations.futureBackground = {
      type: 'box',
      xMin: currentTime,
      // xMax is omitted to extend to the right edge of the chart
      yScaleID: 'y', // Span the full height of the y-axis
      xScaleID: 'x', // Associate with the x-axis for positioning
      backgroundColor: 'rgba(220, 220, 220, 0.3)', // Light gray, semi-transparent
      borderColor: 'transparent', // No border for the box
      drawTime: 'beforeDatasetsDraw' // Draw behind datasets and grid lines
    };
  }

  // Add horizontal lines to separate emotions more clearly
  // This section is removed as drawing lines precisely between string-based categories
  // with the annotation plugin is complex and might not render as expected.
  // The default y-axis grid lines will provide some separation.
  /*
  if (emotionLabels.length > 0) {
    emotionLabels.forEach((label, index) => {
      if (index < emotionLabels.length - 1) {
        options.plugins.annotation.annotations[`line${index}`] = {
          type: 'line',
          scaleID: 'y',
          value: label,
          yMax: label,
          yMin: label,
          borderColor: 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1,
          borderDash: [3, 3],
        };
      }
    });
  }
  */

  return (
    <Box sx={{
      width: '100%',
      height: 280,
      position: 'relative',
      pb: 3,
      pt: 1,
    }}>
      {isValidData ? (
        <Scatter
          options={options}
          data={modifiedChartData}
        />
      ) : (
        <Box sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
          fontSize: '0.875rem',
          opacity: 0.7,
          fontStyle: 'italic'
        }}>
          <Typography variant="body2" sx={{ mb: 1 }}>
            No timeline data available
          </Typography>
          <Typography variant="caption">
            Process a video to see emotion analysis over time
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default EmotionTimeline;
