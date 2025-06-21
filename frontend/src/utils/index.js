/**
 * Utility Functions
 * Common utilities for the emotion analysis dashboard
 */

import { colors } from '../constants/theme';

/**
 * Format timestamp from seconds to MM:SS format
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export const formatTimestamp = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

/**
 * Convert HH:MM:SS time string to seconds
 * @param {string|number} timeString - Time string or number
 * @returns {number} Time in seconds
 */
export const timeStringToSeconds = (timeString) => {
  if (typeof timeString === 'number') {
    return timeString; // Already in seconds
  }
  
  if (typeof timeString !== 'string') {
    return 0; // Invalid input
  }
  
  // Handle HH:MM:SS, MM:SS, or SS formats
  const parts = timeString.split(':').map(part => parseInt(part, 10) || 0);
  
  if (parts.length === 3) {
    // HH:MM:SS format
    const [hours, minutes, seconds] = parts;
    return hours * 3600 + minutes * 60 + seconds;
  } else if (parts.length === 2) {
    // MM:SS format
    const [minutes, seconds] = parts;
    return minutes * 60 + seconds;
  } else if (parts.length === 1) {
    // SS format
    return parts[0];
  }
  
  return 0; // Invalid format
};

/**
 * Get color based on emotion using our design system
 * @param {string} emotion - Emotion name
 * @returns {string} Hex color value
 */
export const getEmotionColor = (emotion) => {
  return colors.emotions[emotion] || colors.emotions.neutral;
};

/**
 * Get intensity value (0-1) from string representation
 * @param {string} intensity - Intensity level (mild, moderate, intense)
 * @returns {number} Normalized intensity value
 */
export const getIntensityValue = (intensity) => {
  const intensityValues = {
    mild: 0.3,
    moderate: 0.6,
    intense: 0.9
  };

  return intensityValues[intensity] || 0.5;
};

/**
 * Process emotion data for visualization
 * @param {Object} analysisData - Raw analysis data from API
 * @returns {Object} Processed data for charts and visualizations
 */
export const processEmotionData = (analysisData) => {
  if (!analysisData || !analysisData.transcript) {
    return { 
      emotionDistribution: {}, 
      intensityTimeline: { datasets: [], emotionLabels: [] } 
    };
  }

  // Group by emotion categories for the bar chart
  const emotionCounts = analysisData.transcript.reduce((acc, item) => {
    acc[item.emotion] = (acc[item.emotion] || 0) + 1;
    return acc;
  }, {});

  // Calculate total count to normalize to percentages
  const totalCount = Object.values(emotionCounts).reduce((sum, count) => sum + count, 0);

  // Convert counts to normalized percentages (0-1)
  const emotionDistribution = {};
  Object.entries(emotionCounts).forEach(([emotion, count]) => {
    emotionDistribution[emotion] = totalCount > 0 ? count / totalCount : 0;
  });

  // Get all unique emotions present in the transcript
  const uniqueEmotions = [...new Set(analysisData.transcript.map(item => item.emotion))];

  // Sort emotions in a meaningful order
  const emotionOrder = ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'];
  uniqueEmotions.sort((a, b) => {
    const indexA = emotionOrder.indexOf(a) !== -1 ? emotionOrder.indexOf(a) : 999;
    const indexB = emotionOrder.indexOf(b) !== -1 ? emotionOrder.indexOf(b) : 999;
    return indexA - indexB;
  });

  // Create data for categorical timeline
  const timelineData = {
    datasets: uniqueEmotions.map((emotion) => {
      // Filter transcript items for this emotion
      const emotionSegments = analysisData.transcript
        .filter(item => item.emotion === emotion)
        .map(item => ({
          x: item.start_time,
          y: emotion.charAt(0).toUpperCase() + emotion.slice(1),
          duration: item.end_time - item.start_time,
          intensity: getIntensityValue(item.intensity)
        }));

      return {
        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
        data: emotionSegments,
        backgroundColor: getEmotionColor(emotion),
        borderColor: getEmotionColor(emotion),
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false, // Scatter plot style
      };
    }),
    emotionLabels: uniqueEmotions.map(e => e.charAt(0).toUpperCase() + e.slice(1))
  };

  return { emotionDistribution, intensityTimeline: timelineData };
};

/**
 * Debounce function to limit API calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Generate a unique ID
 * @returns {string} Unique identifier
 */
export const generateId = () => {
  return Math.random().toString(36).substr(2, 9);
};

/**
 * Validate URL format
 * @param {string} url - URL to validate
 * @returns {boolean} Whether URL is valid
 */
export const isValidUrl = (url) => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Format file size in human readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Calculate emotion statistics
 * @param {Array} emotionData - Array of emotion data points
 * @returns {Object} Emotion statistics
 */
export const calculateEmotionStats = (emotionData) => {
  if (!emotionData || emotionData.length === 0) {
    return {
      dominant: 'neutral',
      confidence: 0,
      duration: 0,
      segments: 0,
    };
  }

  // Find dominant emotion
  const emotionCounts = emotionData.reduce((acc, item) => {
    acc[item.emotion] = (acc[item.emotion] || 0) + 1;
    return acc;
  }, {});

  const dominant = Object.entries(emotionCounts)
    .sort(([,a], [,b]) => b - a)[0][0];

  // Calculate average confidence
  const totalConfidence = emotionData.reduce((sum, item) => 
    sum + (item.confidence || 0.5), 0);
  const confidence = totalConfidence / emotionData.length;

  // Calculate total duration
  const duration = emotionData.reduce((sum, item) => 
    sum + (item.end_time - item.start_time), 0);

  return {
    dominant,
    confidence,
    duration,
    segments: emotionData.length,
  };
};
