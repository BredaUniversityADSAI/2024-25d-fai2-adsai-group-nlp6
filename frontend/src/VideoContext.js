import React, { createContext, useContext, useState } from 'react';
import { analyzeVideo, getVideoAnalysis } from './api';
import { processEmotionData } from './utils';

// Create context
const VideoContext = createContext();

// Hook to use video context
export const useVideo = () => {
  return useContext(VideoContext);
};

// Provider component
export const VideoProvider = ({ children }) => {
  const [videoUrl, setVideoUrl] = useState('');
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [videoHistory, setVideoHistory] = useState([]);

  // Process video URL and get analysis
  const processVideo = async (url) => {
    if (!url) return;

    setVideoUrl(url);
    setIsLoading(true);
    setError(null);

    try {
      // For development, use mock data instead of API call
      // In production, uncomment the API call below

      // Mock data simulation
      // await new Promise(resolve => setTimeout(resolve, 3000));
      // const mockData = {
      //   videoId: Date.now().toString(),
      //   title: 'Sample Video Analysis',
      //   transcript: [
      //     {
      //       sentence: "Hang on to your seats because Asia's next Top Model is back with a vengeance.",
      //       start_time: 1,
      //       end_time: 5,
      //       emotion: "happiness",
      //       sub_emotion: "excitement",
      //       intensity: "mild"
      //     },
      //     {
      //       sentence: "Do you want to be on top?",
      //       start_time: 5,
      //       end_time: 6,
      //       emotion: "happiness",
      //       sub_emotion: "curiosity",
      //       intensity: "intense"
      //     },
      //     {
      //       sentence: "I am Filipino espresso.",
      //       start_time: 6,
      //       end_time: 7,
      //       emotion: "neutral",
      //       sub_emotion: "neutral",
      //       intensity: "intense"
      //     },
      //     {
      //       sentence: "This show is really exciting!",
      //       start_time: 7,
      //       end_time: 10,
      //       emotion: "happiness",
      //       sub_emotion: "excitement",
      //       intensity: "intense"
      //     },
      //     {
      //       sentence: "But sometimes I feel a bit nervous.",
      //       start_time: 10,
      //       end_time: 13,
      //       emotion: "fear",
      //       sub_emotion: "anxiety",
      //       intensity: "moderate"
      //     },
      //   ]
      // };

      // Real API call
      const result = await analyzeVideo(url);

      setAnalysisData(result);

      // Add to history
      setVideoHistory(prev => {
        const newHistory = [
          {
            id: result.videoId,
            title: result.title || 'Untitled Video',
            url: url,
            date: new Date().toISOString().split('T')[0],
            emotions: processEmotionData(result).emotionDistribution, // Ensure processEmotionData expects 'result'
          },
          ...prev
        ];

        // Keep only the most recent 10 videos
        return newHistory.slice(0, 10);
      });

    } catch (err) {
      console.error('Error processing video:', err);
      setError(err.message || 'Failed to process video');
    } finally {
      setIsLoading(false);
    }
  };

  // Load a video from history
  const loadFromHistory = (historyItem) => {
    setVideoUrl(historyItem.url);
    // In a real app, you'd fetch the analysis from backend here
    // For now, we'll just use mock data
    setIsLoading(true);
    setTimeout(() => {
      setAnalysisData({
        videoId: historyItem.id,
        title: historyItem.title,
        transcript: [
          {
            sentence: "This is a previously analyzed video.",
            start_time: 1,
            end_time: 3,
            emotion: Object.keys(historyItem.emotions)[0] || "neutral",
            sub_emotion: "neutral",
            intensity: "moderate"
          }
        ]
      });
      setIsLoading(false);
    }, 1000);
  };

  // Get current emotion based on timestamp
  const getCurrentEmotion = () => {
    if (!analysisData || !analysisData.transcript || analysisData.transcript.length === 0) {
      return null;
    }

    const current = analysisData.transcript.find(
      item => currentTime >= item.start_time && currentTime <= item.end_time
    );

    return current || null;
  };

  // Value to be provided by the context
  const value = {
    videoUrl,
    currentTime,
    setCurrentTime,
    isPlaying,
    setIsPlaying,
    isLoading,
    error,
    analysisData,
    videoHistory,
    processVideo,
    loadFromHistory,
    getCurrentEmotion,
  };

  return (
    <VideoContext.Provider value={value}>
      {children}
    </VideoContext.Provider>
  );
};

export default VideoContext;
