import axios from 'axios';

const API_BASE_URL = 'http://localhost:3120';

// Submit a YouTube video URL for analysis
export const analyzeVideo = async (youtubeUrl) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      url: youtubeUrl
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};

// Get analysis results for a specific video
export const getVideoAnalysis = async (videoId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/${videoId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching analysis:', error);
    throw error;
  }
};
