// Format timestamp from seconds to MM:SS format
export const formatTimestamp = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

// Get color based on emotion
export const getEmotionColor = (emotion) => {
  const emotionColors = {
    happiness: '#FFD700',  // Gold
    sadness: '#4169E1',    // Royal Blue
    anger: '#FF4500',      // Red Orange
    fear: '#800080',       // Purple
    disgust: '#008000',    // Green
    surprise: '#FFA500',   // Orange
    neutral: '#A9A9A9',    // Dark Gray
  };

  return emotionColors[emotion] || '#A9A9A9';
};

// Get intensity value (0-1) from string representation
export const getIntensityValue = (intensity) => {
  const intensityValues = {
    mild: 0.3,
    moderate: 0.6,
    intense: 0.9
  };

  return intensityValues[intensity] || 0.5;
};

// Process emotion data for visualization
export const processEmotionData = (analysisData) => {
  if (!analysisData || !analysisData.transcript) {
    return { emotionDistribution: {}, intensityTimeline: [] };
  }

  // Group by emotion categories for the bar chart
  const emotionDistribution = analysisData.transcript.reduce((acc, item) => {
    acc[item.emotion] = (acc[item.emotion] || 0) + 1;
    return acc;
  }, {});

  // Process timeline data for the intensity graph
  const intensityTimeline = analysisData.transcript.map(item => ({
    time: item.start_time,
    emotion: item.emotion,
    intensity: getIntensityValue(item.intensity)
  }));

  return { emotionDistribution, intensityTimeline };
};
