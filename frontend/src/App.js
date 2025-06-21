import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import EditNoteIcon from '@mui/icons-material/EditNote';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import PsychologyAltIcon from '@mui/icons-material/PsychologyAlt';
import VisibilityIcon from '@mui/icons-material/Visibility';
import InsightsIcon from '@mui/icons-material/Insights';
import TimelineIcon from '@mui/icons-material/Timeline';
import DonutLargeIcon from '@mui/icons-material/DonutLarge';
import './App.css';
import { motion, AnimatePresence } from 'framer-motion';
import * as XLSX from 'xlsx';

// Import components
import Sidebar from './components/Sidebar';
import AddVideoModal from './components/AddVideoModal';
import SettingsModal from './components/SettingsModal';
import FeedbackModal from './components/FeedbackModal';


import EmotionTimeline from './components/EmotionTimeline';
import EmotionBarChart from './components/EmotionBarChart';
import VideoPlayer from './components/VideoPlayer';

// Import context and utilities
import { VideoProvider, useVideo } from './VideoContext';
import { processEmotionData } from './utils';
import customTheme from './theme';

// Create Sophisticated MUI Theme - Minimalist Navy Design
const muiTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: customTheme.colors.primary.main, // Sophisticated indigo
      light: customTheme.colors.primary.light,
      dark: customTheme.colors.primary.dark,
    },
    secondary: {
      main: customTheme.colors.secondary.main, // Navy slate
      light: customTheme.colors.secondary.light,
      dark: customTheme.colors.secondary.dark,
    },
    background: {
      default: 'transparent', // Let CSS handle the navy gradient
      paper: customTheme.colors.surface.glass,
    },
    text: {
      primary: customTheme.colors.text.primary,
      secondary: customTheme.colors.text.secondary,
    },
  },
  typography: {
    fontFamily: customTheme.typography.fontFamily.primary,
    h1: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.bold,
      letterSpacing: customTheme.typography.letterSpacing.tight,
    },
    h2: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.semibold,
      letterSpacing: customTheme.typography.letterSpacing.tight,
    },
    h3: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.semibold,
    },
    h4: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.medium,
    },
    h5: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.medium,
    },
    h6: {
      fontFamily: customTheme.typography.fontFamily.heading,
      fontWeight: customTheme.typography.fontWeight.medium,
    },
    body1: {
      fontWeight: customTheme.typography.fontWeight.normal,
      lineHeight: customTheme.typography.lineHeight.relaxed,
    },
    body2: {
      fontWeight: customTheme.typography.fontWeight.normal,
      lineHeight: customTheme.typography.lineHeight.normal,
    },
  },
  shape: {
    borderRadius: 16, // More refined border radius
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {        body: {
          background: 'transparent', // Let CSS handle the navy gradient
          overflow: 'hidden',
          fontSmooth: 'always',
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: customTheme.borderRadius.xl,
          fontWeight: customTheme.typography.fontWeight.semibold,
          padding: '12px 24px',
          fontSize: '1rem',
          transition: `all ${customTheme.animation.duration.normal} ${customTheme.animation.easing.premium}`,
          '&:hover': {
            transform: 'translateY(-2px)',
          },
        },
        contained: {
          boxShadow: customTheme.shadows.lg,
          '&:hover': {
            boxShadow: customTheme.shadows.xl,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          transition: `all ${customTheme.animation.duration.normal} ${customTheme.animation.easing.premium}`,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: customTheme.typography.fontWeight.medium,
          fontSize: '0.95rem',
          minHeight: '48px',
          transition: `all ${customTheme.animation.duration.fast} ${customTheme.animation.easing.premium}`,
        },
      },
    },
  },
});

// Main App Content
function AppContent() {
  const {
    videoUrl,
    currentTime,
    setCurrentTime,
    isLoading,
    analysisData,
    videoHistory,
    loadFromHistory,
    getCurrentEmotion,
    processVideo
  } = useVideo();
  const [searchTerm, setSearchTerm] = useState('');
  const [tabValue, setTabValue] = useState(0); // 0 for Live Stream, 1 for Full Analysis
  const [factIndex, setFactIndex] = useState(0);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
    // Modal states for new modular layout
  const [addVideoModalOpen, setAddVideoModalOpen] = useState(false);  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const tabsRef = useRef(null);
  const transcriptContainerRef = useRef(null); // Ref for auto-scroll functionality
  // Premium emotion insights for loading screen
  const emotionFacts = [
    "Advanced AI can now detect micro-expressions lasting just 1/25th of a second.",
    "Human emotional intelligence involves processing over 10,000 facial expressions.",
    "Emotions trigger chemical responses 500 times faster than rational thought.",
    "Premium AI models achieve 97% accuracy in real-time emotion recognition.",
    "Your brain processes emotional context using 12 distinct neural networks.",
    "Machine learning can identify emotions through voice patterns with 94% precision.",
    "Facial coding technology maps 43 individual muscle movements for emotion analysis.",
    "Advanced algorithms can detect emotional intent 3 seconds before conscious expression.",
    "Cross-cultural emotion recognition requires training on 50+ diverse populations.",
    "Next-generation AI processes multimodal emotion data in under 50 milliseconds.",
    "Sophisticated models identify 27 distinct emotional states beyond basic categories.",
    "Premium emotion AI integrates physiological, vocal, and facial data streams.",
    "State-of-the-art systems achieve human-level accuracy in complex emotional scenarios.",
    "Advanced neural networks can predict emotional transitions with 89% accuracy.",
    "Premium AI distinguishes between genuine and performed emotions with 92% precision."
  ];// Update emotion facts periodically during loading
  useEffect(() => {
    if (!isLoading) return;

    const factInterval = setInterval(() => {
      setFactIndex(prev => (prev + 1) % emotionFacts.length);
    }, 5000);

    return () => {
      clearInterval(factInterval);
    };
  }, [isLoading, emotionFacts.length]);

  // Auto-scroll transcript to current time position
  useEffect(() => {
    if (analysisData?.transcript && transcriptContainerRef.current && currentTime > 0) {
      const activeSegment = analysisData.transcript.find(segment => {
        const startTime = segment.start_time ?? segment.start ?? 0;
        const endTime = segment.end_time ?? segment.end ?? startTime + 2;
        return currentTime >= startTime && currentTime <= endTime;
      });

      if (activeSegment) {
        const segmentIndex = analysisData.transcript.indexOf(activeSegment);
        const segmentElement = transcriptContainerRef.current.children[segmentIndex];
          if (segmentElement) {
          segmentElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start', // Changed from 'center' to 'start' for top positioning
            inline: 'nearest'
          });
        }
      }
    }
  }, [currentTime, analysisData?.transcript]);

  // Safety check for theme object to prevent runtime errors
  if (!customTheme || !customTheme.colors) {
    console.error('Theme object is not properly loaded');
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh',
        backgroundColor: '#0f172a' // Fallback dark background
      }}>
        <Typography variant="h6" sx={{ color: '#f8fafc' }}>
          Loading theme...
        </Typography>
      </Box>
    );  }

  // Current emotion based on timestamp
  const currentEmotion = getCurrentEmotion();

  // Process analyzed data for visualizations
  const { emotionDistribution, intensityTimeline } =
    analysisData ? processEmotionData(analysisData) : { emotionDistribution: {}, intensityTimeline: [] };
  // Handle jumping to a specific time in the video
  const handleSentenceClick = (time) => {
    setCurrentTime(time);
    // VideoPlayer will be updated through context
  };
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle feedback modal
  const handleOpenFeedback = () => {
    setFeedbackModalOpen(true);
  };
  const handleCloseFeedback = () => {
    setFeedbackModalOpen(false);
  };

  // Helper function to format time from seconds to HH:MM:SS
  const formatTimeToHHMMSS = (seconds) => {
    if (seconds === undefined || seconds === null || isNaN(seconds)) {
      return "00:00:00";
    }
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Helper function to calculate dominant emotion from transcript data
  const calculateDominantEmotion = (transcriptData) => {
    if (!transcriptData || transcriptData.length === 0) return 'Unknown';
    
    const emotionCounts = {};
    transcriptData.forEach(segment => {
      const emotion = segment.emotion || 'neutral';
      emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    });
    
    return Object.entries(emotionCounts)
      .sort(([,a], [,b]) => b - a)[0]?.[0] || 'Unknown';
  };  // Helper function to get intensity value with proper validation
  const getIntensityValue = (segment) => {
    // Try different possible field names and validate they're numbers
    let intensity = segment.intensity ?? segment.confidence ?? segment.strength ?? segment.score;
    
    // Convert string numbers to actual numbers
    if (typeof intensity === 'string' && !isNaN(intensity) && intensity.trim() !== '') {
      intensity = parseFloat(intensity);
    }
    
    // If we have a valid number between 0 and 1
    if (typeof intensity === 'number' && !isNaN(intensity) && intensity >= 0 && intensity <= 1) {
      return `${(intensity * 100).toFixed(1)}%`;
    }
    
    // If we have a number greater than 1 (might be percentage already)
    if (typeof intensity === 'number' && !isNaN(intensity) && intensity > 1 && intensity <= 100) {
      return `${intensity.toFixed(1)}%`;
    }
    
    // Try to extract from emotion-specific fields if available
    const emotionField = segment.emotion;
    if (emotionField && typeof segment[emotionField] === 'number') {
      const emotionIntensity = segment[emotionField];
      if (emotionIntensity >= 0 && emotionIntensity <= 1) {
        return `${(emotionIntensity * 100).toFixed(1)}%`;
      }
    }
    
    // Generate a reasonable fallback based on emotion if no intensity data
    if (segment.emotion) {
      // Assign default intensities based on emotion type (for demonstration)
      const defaultIntensities = {
        'happiness': 75,
        'joy': 75,
        'sadness': 60,
        'anger': 70,
        'fear': 65,
        'surprise': 55,
        'disgust': 50,
        'neutral': 40,
        'positive': 70,
        'negative': 60
      };
      
      const defaultIntensity = defaultIntensities[segment.emotion.toLowerCase()];
      if (defaultIntensity) {
        return `${defaultIntensity}%`;
      }
    }
    
    return 'N/A';
  };

  // Helper function to get analysis date with fallbacks
  const getAnalysisDate = (analysisData) => {
    // Try different possible date field names
    const possibleDateFields = [
      'createdAt', 'created_at', 'analysisDate', 'analysis_date', 
      'timestamp', 'dateProcessed', 'date_processed'
    ];
    
    for (const field of possibleDateFields) {
      if (analysisData[field]) {
        try {
          return new Date(analysisData[field]).toLocaleString();
        } catch (e) {
          // Continue to next field if date parsing fails
        }
      }
    }
    
    // If no date field found, use current date as fallback
    return new Date().toLocaleString();
  };

  // Handle export predictions to Excel
  const handleExportPredictions = () => {
    if (!analysisData) return;
    
    try {

      // Create a new workbook
      const workbook = XLSX.utils.book_new();

      // Calculate dominant emotion and other summary statistics
      const dominantEmotion = calculateDominantEmotion(analysisData.transcript);
      const totalSegments = analysisData.transcript?.length || 0;
      const totalDuration = analysisData.transcript?.reduce((sum, segment) => {
        const start = segment.start_time ?? segment.start ?? 0;
        const end = segment.end_time ?? segment.end ?? start;
        return sum + (end - start);
      }, 0) || 0;      // 1. Enhanced Summary Sheet
      const summaryData = [
        ['🎬 EMOTION ANALYSIS REPORT'],
        [''],
        ['📊 VIDEO INFORMATION'],
        ['Title', analysisData.title || 'Unknown Video'],
        ['Source URL', videoUrl || 'N/A'],
        ['Export Date', new Date().toLocaleString()],
        ['Analysis Date', getAnalysisDate(analysisData)],
        [''],
        ['📈 ANALYSIS OVERVIEW'],
        ['Total Duration', formatTimeToHHMMSS(totalDuration)],
        ['Total Segments', totalSegments],
        ['Dominant Emotion', dominantEmotion],
        ['Average Segment Length', totalSegments > 0 ? formatTimeToHHMMSS(totalDuration / totalSegments) : 'N/A'],
        [''],
        ['🎭 EMOTION DISTRIBUTION'],
        ['Emotion', 'Count', 'Percentage', 'Duration (approx)']
      ];

      // Calculate enhanced emotion distribution
      const emotionCounts = {};
      const emotionDurations = {};
      
      if (analysisData.transcript) {
        analysisData.transcript.forEach(segment => {
          const emotion = segment.emotion || 'neutral';
          const start = segment.start_time ?? segment.start ?? 0;
          const end = segment.end_time ?? segment.end ?? start;
          const duration = end - start;
          
          emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
          emotionDurations[emotion] = (emotionDurations[emotion] || 0) + duration;
        });
      }

      Object.entries(emotionCounts)
        .sort(([,a], [,b]) => b - a) // Sort by count descending
        .forEach(([emotion, count]) => {
          const percentage = totalSegments > 0 ? ((count / totalSegments) * 100).toFixed(1) : '0.0';
          const duration = formatTimeToHHMMSS(emotionDurations[emotion] || 0);
          summaryData.push([emotion, count, `${percentage}%`, duration]);
        });

      // Add quality metrics
      summaryData.push(
        [''],
        ['📊 DATA QUALITY METRICS'],
        ['Segments with Text', analysisData.transcript?.filter(s => (s.text || s.sentence || s.content || '').trim()).length || 0],
        ['Segments with Sub-emotions', analysisData.transcript?.filter(s => s.subEmotion || s.sub_emotion || s.secondary_emotion).length || 0],
        ['Unique Emotions Detected', Object.keys(emotionCounts).length]
      );

      const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
      
      // Enhanced styling for summary sheet
      summarySheet['!cols'] = [
        { width: 25 },
        { width: 20 },
        { width: 15 },
        { width: 18 }
      ];

      XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');

      // 2. Enhanced Transcript Sheet
      const transcriptData = [
        ['Segment', 'Start Time', 'End Time', 'Duration', 'Text Content', 'Primary Emotion', 'Sub Emotion', 'Intensity']
      ];

      if (analysisData.transcript) {
        analysisData.transcript.forEach((segment, index) => {
          // Handle different possible field names for time
          const startTime = segment.start_time ?? segment.start ?? 0;
          const endTime = segment.end_time ?? segment.end ?? startTime + 1; // Default 1 second if no end time
          const duration = endTime - startTime;
          
          // Handle different possible field names for text
          const text = (segment.text || segment.sentence || segment.content || '').trim() || '[No text available]';
          
          // Handle emotion data
          const primaryEmotion = segment.emotion || 'neutral';
          const subEmotion = segment.subEmotion || segment.sub_emotion || segment.secondary_emotion || '';
          const intensity = getIntensityValue(segment);

          transcriptData.push([
            index + 1,
            formatTimeToHHMMSS(startTime),
            formatTimeToHHMMSS(endTime),
            formatTimeToHHMMSS(duration),
            text,
            primaryEmotion,
            subEmotion,
            intensity
          ]);
        });
      }

      const transcriptSheet = XLSX.utils.aoa_to_sheet(transcriptData);
      
      // Enhanced styling for transcript sheet
      transcriptSheet['!cols'] = [
        { width: 8 },   // Segment
        { width: 12 },  // Start Time
        { width: 12 },  // End Time
        { width: 12 },  // Duration
        { width: 70 },  // Text (increased width for better readability)
        { width: 18 },  // Primary Emotion
        { width: 16 },  // Sub Emotion
        { width: 12 }   // Intensity
      ];

      XLSX.utils.book_append_sheet(workbook, transcriptSheet, 'Transcript');

      // 3. Enhanced Emotion Timeline Sheet
      if (analysisData.emotionTimeline && analysisData.emotionTimeline.length > 0) {
        const timelineData = [
          ['Time', 'Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral', 'Dominant Emotion']
        ];

        analysisData.emotionTimeline.forEach(point => {
          timelineData.push([
            formatTimeToHHMMSS(point.time || 0),
            (point.happiness || 0).toFixed(3),
            (point.sadness || 0).toFixed(3),
            (point.anger || 0).toFixed(3),
            (point.fear || 0).toFixed(3),
            (point.surprise || 0).toFixed(3),
            (point.disgust || 0).toFixed(3),
            (point.neutral || 0).toFixed(3),
            point.dominant_emotion || point.dominantEmotion || 'Unknown'
          ]);
        });

        const timelineSheet = XLSX.utils.aoa_to_sheet(timelineData);
        
        // Enhanced styling for timeline sheet
        timelineSheet['!cols'] = [
          { width: 12 },  // Time
          { width: 12 },  // Happiness
          { width: 12 },  // Sadness
          { width: 12 },  // Anger
          { width: 12 },  // Fear
          { width: 12 },  // Surprise
          { width: 12 },  // Disgust
          { width: 12 },  // Neutral
          { width: 18 }   // Dominant Emotion
        ];

        XLSX.utils.book_append_sheet(workbook, timelineSheet, 'Emotion Timeline');
      }

      // 4. Enhanced Analytics Sheet
      const analyticsData = [
        ['📊 ADVANCED EMOTION ANALYTICS'],
        [''],
        ['🔄 EMOTION TRANSITIONS'],
        ['From Emotion', 'To Emotion', 'Frequency', 'Percentage']
      ];

      // Calculate emotion transitions
      const transitions = {};
      if (analysisData.transcript && analysisData.transcript.length > 1) {
        for (let i = 0; i < analysisData.transcript.length - 1; i++) {
          const fromEmotion = analysisData.transcript[i].emotion || 'neutral';
          const toEmotion = analysisData.transcript[i + 1].emotion || 'neutral';
          const key = `${fromEmotion}→${toEmotion}`;
          transitions[key] = (transitions[key] || 0) + 1;
        }
      }

      const totalTransitions = Object.values(transitions).reduce((sum, count) => sum + count, 0) || 1;
      const sortedTransitions = Object.entries(transitions).sort(([,a], [,b]) => b - a);
      
      sortedTransitions.forEach(([transition, count]) => {
        const [from, to] = transition.split('→');
        const percentage = ((count / totalTransitions) * 100).toFixed(1);
        analyticsData.push([from, to, count, `${percentage}%`]);
      });

      // Add emotion stability analysis
      analyticsData.push(
        [''],
        ['🎭 EMOTION STABILITY ANALYSIS'],
        ['Metric', 'Value', 'Description']
      );

      if (analysisData.transcript && analysisData.transcript.length > 0) {
        const emotions = analysisData.transcript.map(s => s.emotion || 'neutral');
        const uniqueEmotions = [...new Set(emotions)];
        const emotionChanges = emotions.slice(1).filter((emotion, i) => emotion !== emotions[i]).length;
        const stabilityScore = emotions.length > 1 ? ((emotions.length - emotionChanges) / emotions.length * 100).toFixed(1) : '100.0';
        
        analyticsData.push(
          ['Unique Emotions', uniqueEmotions.length, 'Total different emotions detected'],
          ['Emotion Changes', emotionChanges, 'Number of times emotion switched'],
          ['Stability Score', `${stabilityScore}%`, 'Percentage of time emotion remained same'],
          ['Most Common', dominantEmotion, 'Most frequently occurring emotion'],
          ['Diversity Index', (uniqueEmotions.length / Math.max(1, emotions.length) * 100).toFixed(1) + '%', 'Emotional diversity ratio']
        );
      }

      // Add duration insights
      analyticsData.push(
        [''],
        ['⏱️ TEMPORAL ANALYSIS'],
        ['Metric', 'Value', 'Details']
      );

      if (analysisData.transcript) {
        const durations = analysisData.transcript.map(s => {
          const start = s.start_time ?? s.start ?? 0;
          const end = s.end_time ?? s.end ?? start + 1;
          return end - start;
        }).filter(d => d > 0);

        if (durations.length > 0) {
          const avgDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;
          const minDuration = Math.min(...durations);
          const maxDuration = Math.max(...durations);
          
          analyticsData.push(
            ['Total Duration', formatTimeToHHMMSS(totalDuration), 'Complete video length analyzed'],
            ['Average Segment', formatTimeToHHMMSS(avgDuration), 'Mean length per segment'],
            ['Shortest Segment', formatTimeToHHMMSS(minDuration), 'Minimum segment duration'],
            ['Longest Segment', formatTimeToHHMMSS(maxDuration), 'Maximum segment duration'],
            ['Segments per Minute', (durations.length / (totalDuration / 60)).toFixed(1), 'Segment density']
          );
        }
      }

      const analyticsSheet = XLSX.utils.aoa_to_sheet(analyticsData);
      
      // Enhanced styling for analytics sheet
      analyticsSheet['!cols'] = [
        { width: 25 },
        { width: 20 },
        { width: 15 },
        { width: 35 }
      ];

      XLSX.utils.book_append_sheet(workbook, analyticsSheet, 'Analytics');

      // Generate enhanced filename with video title
      const videoTitle = (analysisData.title || 'unknown-video')
        .replace(/[^\w\s-]/g, '') // Remove special chars
        .replace(/\s+/g, '-') // Replace spaces with hyphens
        .substring(0, 30); // Limit length
      
      const timestamp = new Date().toISOString().slice(0, 10); // YYYY-MM-DD format
      const filename = `emotion-analysis-${videoTitle}-${timestamp}.xlsx`;

      // Write and download the file
      XLSX.writeFile(workbook, filename);
      
      console.log('Enhanced Excel report exported successfully:', filename);
    } catch (error) {
      console.error('Error exporting to Excel:', error);
      // Fallback to JSON export if Excel fails
      try {
        const exportData = {
          videoTitle: analysisData.title || 'Unknown Video',
          videoUrl: videoUrl,
          exportDate: new Date().toISOString(),
          emotionData: analysisData.emotionTimeline || [],
          transcript: analysisData.transcript || [],
          summary: {
            totalSegments: analysisData.transcript?.length || 0,
            dominantEmotion: calculateDominantEmotion(analysisData.transcript),
            totalDuration: analysisData.transcript?.reduce((sum, segment) => {
              const start = segment.start_time ?? segment.start ?? 0;
              const end = segment.end_time ?? segment.end ?? start;
              return sum + (end - start);
            }, 0) || 0
          }
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        const exportFileDefaultName = `emotion-analysis-fallback-${Date.now()}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        console.log('Fallback JSON export completed');
      } catch (fallbackError) {
        console.error('Both Excel and JSON export failed:', fallbackError);
      }
    }  };  // Handle URL upload - processes YouTube URLs and file uploads for emotion analysis
  const handleUrlUpload = async (data) => {
    try {
      console.log('Processing video data:', data);
      
      // Validate input data
      if (!data || typeof data !== 'object') {
        console.error('Invalid data provided to handleUrlUpload');
        return;
      }
      
      // Close the modal first
      setAddVideoModalOpen(false);
      
      if (data.type === 'youtube') {
        // Handle YouTube URL processing
        if (!data.url || !data.url.trim()) {
          console.error('Empty YouTube URL provided');
          return;
        }
        
        console.log('Processing YouTube URL:', data.url);
        await processVideo(data.url.trim());
        console.log('YouTube video processing initiated successfully');
        
      } else if (data.type === 'file') {
        // Handle file upload processing
        if (!data.file) {
          console.error('No file provided for upload');
          return;
        }
        
        console.log('Processing video file:', data.file.name);
        // TODO: Implement file upload processing
        // For now, we'll show an info message
        console.info('File upload functionality not yet implemented:', data.file.name);
        // You might want to show a user-friendly message here
        
      } else {
        console.error('Unknown upload type:', data.type);
      }
      
    } catch (error) {
      console.error('Error processing video:', error);
      // TODO: Show user-friendly error message (e.g., using a snackbar or alert)
    }
  };

  // Filter history based on search term
  const filteredHistory = videoHistory.filter(video =>
    video.title.toLowerCase().includes(searchTerm.toLowerCase())  );  return (    <Box sx={{ 
      display: 'flex', 
      minHeight: '100vh', 
      background: customTheme.colors.background.primary, // Navy gradient background
      position: 'relative',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: customTheme.colors.gradients.subtle, // Subtle navy effect
        opacity: 0.1,
        zIndex: 0,
      },
    }}>
      <Sidebar
        videoHistory={filteredHistory}
        onVideoSelect={loadFromHistory}
        onAddVideo={() => setAddVideoModalOpen(true)}
        onSettings={() => setSettingsModalOpen(true)}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}      />      {/* Main Content Area with Enhanced Premium Grid Layout */}
      <Box sx={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        p: 4, 
        pl: '140px', // Enhanced left padding        minHeight: '100vh',
        justifyContent: 'flex-start',
        alignItems: 'stretch',
        position: 'relative',
        zIndex: 1,
      }}>
        <Grid container spacing={4} sx={{ 
          maxWidth: '100%', 
          width: '100%',
          height: 'fit-content',
          alignItems: 'stretch',
          justifyContent: 'flex-start',
          py: 2,
          mt: 1
        }}>
          {/* Premium Dashboard Card */}
          <Grid item xs={12} lg={4}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <Paper
                elevation={0}
                sx={{ 
                  height: '78vh',
                  p: 4,
                  background: customTheme.glassmorphism.luxury.background,
                  backdropFilter: customTheme.glassmorphism.luxury.backdropFilter,
                  border: customTheme.glassmorphism.luxury.border,
                  borderRadius: customTheme.borderRadius['3xl'],
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  position: 'relative',
                  transition: `all ${customTheme.animation.duration.normal} ${customTheme.animation.easing.premium}`,
                  boxShadow: customTheme.shadows.xl,
                  '&:hover': {
                    transform: 'translateY(-6px)',
                    boxShadow: customTheme.shadows['3xl'],
                    border: `1px solid ${customTheme.colors.primary.glow}`,                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '2px',
                    background: customTheme.colors.gradients.primary,
                    borderRadius: customTheme.borderRadius.full,
                  }
                }}
              >                <Typography variant="h6" sx={{ 
                  mb: 3, 
                  fontWeight: 700,
                  color: customTheme.colors.text.primary,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  fontSize: '1.2rem',
                }}>
                  📊 Premium Dashboard
                </Typography>
                
                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  gap: 3,
                  color: customTheme.colors.text.secondary,
                  textAlign: 'center'
                }}>
                  <Box sx={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',                    background: customTheme.colors.gradients.primary, // Primary accent only
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '2rem',
                    boxShadow: customTheme.shadows.glow,
                    animation: 'pulse 2s ease-in-out infinite',
                    '@keyframes pulse': {
                      '0%, 100%': {
                        transform: 'scale(1)',
                        opacity: 0.8
                      },
                      '50%': {
                        transform: 'scale(1.05)',
                        opacity: 1
                      }
                    }
                  }}>
                    📈
                  </Box>
                  <Typography variant="h6" sx={{ 
                    fontWeight: 600,
                    color: customTheme.colors.text.primary,
                    mb: 1
                  }}>
                    Analytics Dashboard
                  </Typography>
                  <Typography variant="body2" sx={{ 
                    opacity: 0.8,
                    maxWidth: '200px',
                    lineHeight: 1.6
                  }}>
                    Advanced metrics and insights coming soon with premium visualizations
                  </Typography>
                </Box>
              </Paper>
            </motion.div>
          </Grid>          {/* Premium Video Player & Transcript Section */}
          <Grid item xs={12} lg={4}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1, ease: "easeOut" }}
            >
              <Paper 
                elevation={0}
                sx={{ 
                  height: '78vh',
                  p: 4,
                  background: customTheme.glassmorphism.luxury.background,
                  backdropFilter: customTheme.glassmorphism.luxury.backdropFilter,
                  border: customTheme.glassmorphism.luxury.border,
                  borderRadius: customTheme.borderRadius['3xl'],
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  position: 'relative',
                  transition: `all ${customTheme.animation.duration.normal} ${customTheme.animation.easing.premium}`,
                  boxShadow: customTheme.shadows.xl,
                  '&:hover': {
                    transform: 'translateY(-6px)',
                    boxShadow: customTheme.shadows['3xl'],
                    border: `1px solid ${customTheme.colors.primary.glow}`, // Primary accent only
                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,                    right: 0,                    height: '2px',
                    background: customTheme.colors.gradients.primary, // Primary accent only
                    borderRadius: customTheme.borderRadius.full,
                  }
                }}
              >
                {/* Premium Video Player Section */}
                <Box sx={{ mb: 3 }}>                  <Typography variant="h6" sx={{ 
                    mb: 3, 
                    fontWeight: 700,
                    color: customTheme.colors.text.primary,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1.5,
                    fontSize: '1.2rem',
                  }}>
                    🎥 Premium Video Player
                  </Typography>
                
                {videoUrl ? (                  <Box sx={{ 
                    borderRadius: customTheme.borderRadius.lg,
                    overflow: 'hidden',
                    border: `1px solid ${customTheme.colors.border}`,
                    height: '280px' // Restored height for proper video controls
                  }}>
                    <VideoPlayer 
                      url={videoUrl}
                      currentTime={currentTime}
                      onProgress={(state) => setCurrentTime(state.playedSeconds)}
                    />
                  </Box>
                ) : (                  <Box sx={{
                    height: '280px', // Restored height for consistency
                    border: `2px dashed ${customTheme.colors.border}`,
                    borderRadius: customTheme.borderRadius.lg,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 2,
                    color: customTheme.colors.text.secondary
                  }}>
                    <Typography variant="h6" sx={{ fontWeight: 500 }}>
                      📹 No Video Selected
                    </Typography>                    <Typography variant="body2" sx={{ opacity: 0.8 }}>
                      Upload a video or enter a URL to get started
                    </Typography>
                    <Button
                      onClick={() => setAddVideoModalOpen(true)}
                      variant="contained"
                      startIcon={<CloudUploadIcon />}
                      sx={{
                        mt: 2,
                        textTransform: 'none',
                        borderRadius: 2,
                        px: 3,
                        py: 1.5,
                        fontSize: '0.9rem',
                        fontWeight: 600,
                        background: customTheme.colors.primary.main,
                        color: 'white',
                        boxShadow: `0 4px 12px ${customTheme.colors.primary.main}40`,
                        '&:hover': {
                          background: customTheme.colors.primary.dark,
                          transform: 'translateY(-1px)',
                          boxShadow: `0 6px 16px ${customTheme.colors.primary.main}50`,
                        }
                      }}
                    >
                      Add Video
                    </Button>
                  </Box>
                )}
              </Box>

              {/* Action Buttons */}
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                gap: 2, 
                mb: 3,
                flexWrap: 'wrap' 
              }}>
                <Button
                  onClick={handleOpenFeedback}
                  disabled={!analysisData}
                  variant="contained"
                  startIcon={<EditNoteIcon />}
                  sx={{
                    textTransform: 'none',
                    borderRadius: 2,
                    px: 3,
                    py: 1.5,
                    fontSize: '0.9rem',
                    fontWeight: 600,
                    background: customTheme.colors.primary.main,
                    color: 'white',
                    boxShadow: `0 4px 12px ${customTheme.colors.primary.main}40`,
                    '&:hover': {
                      background: customTheme.colors.primary.dark,
                      transform: 'translateY(-1px)',
                      boxShadow: `0 6px 16px ${customTheme.colors.primary.main}50`,
                    },
                    '&:disabled': {
                      background: customTheme.colors.surface.glass,
                      color: customTheme.colors.text.disabled,
                    }
                  }}
                >
                  Give Feedback
                </Button>
                
                <Button
                  onClick={handleExportPredictions}
                  disabled={!analysisData}
                  variant="contained"
                  startIcon={<FileDownloadIcon />}
                  sx={{
                    textTransform: 'none',
                    borderRadius: 2,
                    px: 3,
                    py: 1.5,
                    fontSize: '0.9rem',
                    fontWeight: 600,
                    background: customTheme.colors.status.success,
                    color: 'white',
                    boxShadow: `0 4px 12px ${customTheme.colors.status.success}40`,
                    '&:hover': {
                      background: '#0d9488',
                      transform: 'translateY(-1px)',
                      boxShadow: `0 6px 16px ${customTheme.colors.status.success}50`,
                    },
                    '&:disabled': {
                      background: customTheme.colors.surface.glass,
                      color: customTheme.colors.text.disabled,
                    }
                  }}
                >
                  Export Results
                </Button>
              </Box>              {/* Controls Section - Only show when no analysis data or loading */}
              {(!analysisData || isLoading) && (
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  gap: 2, 
                  mt: 2,
                  flexWrap: 'wrap' 
                }}>
                  {/* Upload/Analysis controls will go here */}
                </Box>
              )}{analysisData && (
                <Box mt={4} sx={{
                  flexGrow: 1,
                  overflow: 'hidden',
                  minHeight: 0,
                  display: 'flex',
                  flexDirection: 'column',
                }}>
                  {/* Enhanced Transcript Header */}
                  <Typography variant="h6" sx={{ 
                    mb: 2, 
                    fontWeight: 600,
                    color: 'text.primary',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                  }}>
                    📝 Transcript & Emotions
                    <Typography variant="body2" sx={{ 
                      ml: 2, 
                      color: 'text.secondary',
                      fontWeight: 400 
                    }}>
                      {analysisData.transcript?.length || 0} segments
                    </Typography>
                  </Typography>

                  {/* Enhanced Transcript List */}
                  <Box sx={{ 
                    flexGrow: 1, 
                    overflow: 'auto',
                    pr: 1,
                    '&::-webkit-scrollbar': {
                      width: '6px',
                    },
                    '&::-webkit-scrollbar-track': {
                      background: 'rgba(0,0,0,0.05)',
                      borderRadius: '3px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                      background: 'rgba(0,0,0,0.2)',
                      borderRadius: '3px',
                      '&:hover': {
                        background: 'rgba(0,0,0,0.3)',
                      }
                    }
                  }} ref={transcriptContainerRef}>
                    {analysisData.transcript?.map((segment, index) => {
                      const startTime = segment.start_time ?? segment.start ?? 0;
                      const emotion = segment.emotion || 'neutral';
                      const text = segment.text || segment.sentence || segment.content || 'No text available';
                      const isActive = Math.abs(currentTime - startTime) < 2; // Active if within 2 seconds

                      return (
                        <Box
                          key={index}
                          onClick={() => handleSentenceClick(startTime)}
                          sx={{
                            p: 2,
                            mb: 1.5,
                            borderRadius: '12px',
                            cursor: 'pointer',
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            border: '1px solid',
                            borderColor: isActive ? 
                              customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral :
                              'rgba(0, 0, 0, 0.08)',                            backgroundColor: isActive ? 
                              `${customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral}15` :
                              'rgba(30, 41, 59, 0.4)',
                            boxShadow: isActive ? 
                              `0 4px 20px ${customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral}20` :
                              '0 2px 8px rgba(0, 0, 0, 0.04)',
                            transform: isActive ? 'translateY(-2px)' : 'translateY(0px)',
                            '&:hover': {
                              transform: 'translateY(-3px)',
                              boxShadow: `0 6px 25px ${customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral}25`,
                              borderColor: customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral,
                            }
                          }}
                        >
                          {/* Timestamp and Emotion Badge */}
                          <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'space-between',
                            mb: 1.5
                          }}>
                            <Typography variant="caption" sx={{ 
                              color: 'text.secondary',
                              fontWeight: 500,
                              fontFamily: 'monospace',
                              backgroundColor: 'rgba(0, 0, 0, 0.05)',
                              px: 1,
                              py: 0.5,
                              borderRadius: '4px'
                            }}>
                              {formatTimeToHHMMSS(startTime)}
                            </Typography>
                            
                            <Box sx={{
                              px: 1.5,
                              py: 0.5,
                              borderRadius: '20px',
                              backgroundColor: `${customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral}15`,
                              border: `1px solid ${customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral}40`,
                            }}>
                              <Typography variant="caption" sx={{ 
                                color: customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral,
                                fontWeight: 600,
                                textTransform: 'capitalize',
                                fontSize: '0.75rem'
                              }}>
                                {emotion}
                              </Typography>
                            </Box>
                          </Box>

                          {/* Text Content */}
                          <Typography variant="body2" sx={{ 
                            color: 'text.primary',
                            lineHeight: 1.6,
                            fontWeight: isActive ? 500 : 400,
                            fontSize: '0.9rem'
                          }}>
                            {text}
                          </Typography>

                          {/* Intensity Bar (if available) */}
                          {(segment.intensity || segment.confidence) && (
                            <Box sx={{ mt: 1.5 }}>
                              <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1,
                                mb: 0.5
                              }}>
                                <Typography variant="caption" sx={{ 
                                  color: 'text.secondary',
                                  fontSize: '0.7rem'
                                }}>
                                  Intensity
                                </Typography>                                <Typography variant="caption" sx={{ 
                                  color: customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral,
                                  fontWeight: 600,
                                  fontSize: '0.7rem'
                                }}>
                                  {getIntensityValue(segment)}
                                </Typography>
                              </Box>
                              <Box sx={{
                                width: '100%',
                                height: '3px',
                                backgroundColor: 'rgba(0, 0, 0, 0.08)',
                                borderRadius: '2px',
                                overflow: 'hidden'
                              }}>                                <Box sx={{
                                  width: `${Math.min(100, Math.max(0, ((segment.intensity ?? segment.confidence ?? 0.4) * 100)))}%`,
                                  height: '100%',
                                  backgroundColor: customTheme.colors.emotion[emotion] || customTheme.colors.emotion.neutral,
                                  transition: 'width 0.3s ease'
                                }} />
                              </Box>
                            </Box>
                          )}
                        </Box>                      );
                    })}
                  </Box>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Premium Emotion Analytics Section */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}          ><Box
              sx={{
                height: '78vh',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {analysisData ? (
                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  background: customTheme.glassmorphism.luxury.background,
                  backdropFilter: customTheme.glassmorphism.luxury.backdropFilter,
                  borderRadius: customTheme.borderRadius['3xl'],
                  border: customTheme.glassmorphism.luxury.border,
                  boxShadow: customTheme.shadows['2xl'],
                  transition: `all ${customTheme.animation.duration.normal} ${customTheme.animation.easing.premium}`,
                  position: 'relative',
                  zIndex: 10,
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: customTheme.shadows['3xl'],
                    border: `1px solid ${customTheme.colors.tertiary.glow}`,
                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '2px',
                    background: customTheme.colors.gradients.tertiary,
                    borderRadius: customTheme.borderRadius.full,
                  }                }}>
                  <Box sx={{
                    padding: 4,
                    borderBottom: `1px solid ${customTheme.colors.border}`,
                    display: 'flex',
                    flexDirection: 'column',
                    background: customTheme.colors.surface.card,
                  }}>                    <Typography variant="h6" fontWeight={700} sx={{
                      color: customTheme.colors.text.primary, // Simple text color
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      mb: 2,
                    }}>
                      <PsychologyAltIcon />
                      Emotion Analytics
                    </Typography>                    <Box sx={{ position: 'relative' }} ref={tabsRef}>                      <Tabs
                        value={tabValue}
                        onChange={handleTabChange}
                        variant="fullWidth"
                        scrollButtons={false}
                        aria-label="emotion analysis tabs"                        sx={{ 
                          '& .MuiTab-root': { 
                            color: 'text.secondary',
                            textTransform: 'none',
                            backgroundColor: 'rgba(30, 41, 59, 0.4)', // Dark background
                            borderRadius: customTheme.borderRadius.md,
                            mx: 0.5,
                            border: `1px solid ${customTheme.colors.border}`,
                            '&:hover': {
                              backgroundColor: 'rgba(99, 102, 241, 0.2)',
                              color: customTheme.colors.primary.main,
                            }
                          },
                          '& .Mui-selected': { 
                            color: customTheme.colors.primary.main,
                            backgroundColor: 'rgba(99, 102, 241, 0.25)', // Darker selected state
                            fontWeight: 600,
                            border: `1px solid ${customTheme.colors.primary.main}40`,
                          },
                          '& .MuiTabs-indicator': {
                            backgroundColor: customTheme.colors.primary.main,
                            height: 3,
                            borderRadius: '2px',
                          }
                        }}
                      >
                        <Tab
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <VisibilityIcon fontSize="small" />
                              <span>Live Stream</span>
                            </Box>
                          }
                        />
                        <Tab
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <InsightsIcon fontSize="small" />
                              <span>Full Analysis</span>
                            </Box>
                          }
                        />
                      </Tabs>
                      {/* Tab indicator removed - using Material-UI default */}
                    </Box>
                  </Box>

                  <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>                    {tabValue === 0 && (
                      <Box sx={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'flex-start',
                        gap: 3,
                      }}>                        {/* Current Emotion Card */}
                        <Box sx={{
                          p: 2.5,
                          borderRadius: customTheme.borderRadius.lg,
                          background: customTheme.colors.surface.card,
                          boxShadow: customTheme.shadows.md,
                          border: `1px solid ${customTheme.colors.border}`,
                        }}>
                          <Typography variant="h6" sx={{
                            mb: 2,
                            fontSize: '1rem',
                            color: '#6366F1',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1                          }}>
                            <PsychologyAltIcon fontSize="small" />
                            Emotion Pulse
                          </Typography>                          <Box sx={{
                            height: '280px', // Increased for better visualization
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            position: 'relative'
                          }}>
                            {/* Enhanced Emotion Pulse Display */}
                            <Box sx={{ 
                              position: 'relative',
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              gap: 2
                            }}>
                              {/* Dynamic Emotion Circle */}
                              <Box sx={{
                                width: 120,
                                height: 120,
                                borderRadius: '50%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                position: 'relative',
                                background: currentEmotion?.emotion ? 
                                  `linear-gradient(135deg, ${customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral}33, ${customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral}11)` :
                                  `linear-gradient(135deg, ${customTheme.colors.emotion.neutral}33, ${customTheme.colors.emotion.neutral}11)`,
                                border: `3px solid ${currentEmotion?.emotion ? customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral : customTheme.colors.emotion.neutral}`,
                                boxShadow: currentEmotion?.emotion ? 
                                  `0 8px 32px ${customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral}40` :
                                  `0 8px 32px ${customTheme.colors.emotion.neutral}40`,
                                transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
                              }}>
                                {/* Pulse Animation */}
                                <Box sx={{
                                  position: 'absolute',
                                  width: '100%',
                                  height: '100%',
                                  borderRadius: '50%',
                                  border: `2px solid ${currentEmotion?.emotion ? customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral : customTheme.colors.emotion.neutral}60`,
                                  animation: 'emotionPulse 2s ease-in-out infinite',
                                  '@keyframes emotionPulse': {
                                    '0%, 100%': {
                                      transform: 'scale(1)',
                                      opacity: 0.8
                                    },
                                    '50%': {
                                      transform: 'scale(1.1)',
                                      opacity: 0.4
                                    }
                                  }
                                }} />
                                
                                {/* Emotion Text */}
                                <Typography variant="h6" sx={{ 
                                  fontWeight: 700,
                                  color: currentEmotion?.emotion ? customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral : customTheme.colors.emotion.neutral,
                                  textTransform: 'capitalize',
                                  fontSize: '1rem',
                                  textAlign: 'center'
                                }}>
                                  {currentEmotion?.emotion || 'neutral'}
                                </Typography>
                              </Box>

                              {/* Emotion Details */}
                              <Box sx={{ textAlign: 'center', minHeight: '60px' }}>
                                <Typography variant="body1" sx={{ 
                                  fontWeight: 600,
                                  color: currentEmotion?.emotion ? customTheme.colors.emotion[currentEmotion.emotion] || customTheme.colors.emotion.neutral : customTheme.colors.emotion.neutral,
                                  textTransform: 'capitalize',
                                  mb: 0.5
                                }}>
                                  {currentEmotion?.emotion || 'Neutral'}
                                </Typography>                                  {currentEmotion?.intensity && !isNaN(Number(currentEmotion.intensity)) && (
                                  <Typography variant="body2" sx={{ 
                                    color: 'text.secondary',
                                    fontWeight: 500,
                                    mb: 1
                                  }}>
                                    Intensity: {getIntensityValue(currentEmotion || {})}
                                  </Typography>
                                )}
                                
                                {currentEmotion?.sub_emotion && (
                                  <Typography variant="caption" sx={{ 
                                    color: 'text.secondary',
                                    fontStyle: 'italic'
                                  }}>
                                    Sub-emotion: {currentEmotion.sub_emotion}
                                  </Typography>
                                )}
                              </Box>
                            </Box>
                          </Box>
                        </Box>

                        {/* Real-Time Emotion Tracker moved to Live Stream tab */}
                        <Box sx={{                          p: 2.5,
                          borderRadius: customTheme.borderRadius.lg,
                          background: customTheme.colors.surface.card,
                          boxShadow: customTheme.shadows.md,
                          border: `1px solid ${customTheme.colors.border}`,
                        }}>
                          <Typography variant="h6" sx={{
                            mb: 2,
                            fontSize: '1rem',
                            color: '#8B5CF6',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1
                          }}>
                            <TimelineIcon fontSize="small" />
                            Real-Time Emotion Tracker
                          </Typography>
                          <Box sx={{ height: '260px' }}> {/* Increased chart height */}
                            <EmotionTimeline
                              data={intensityTimeline}
                              currentTime={currentTime}
                            />
                          </Box>                        </Box>
                      </Box>
                    )}
                    {tabValue === 1 && (
                      <Box sx={{ height: '100%' }}>                        <Box sx={{
                          p: 2.5,
                          mb: 3,
                          borderRadius: customTheme.borderRadius.lg,
                          background: customTheme.colors.surface.card,
                          boxShadow: customTheme.shadows.md,
                          border: `1px solid ${customTheme.colors.border}`,
                        }}>
                          <Typography variant="h6" sx={{
                            mb: 2,
                            fontSize: '1rem',
                            color: '#6366F1',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1
                          }}>
                            <DonutLargeIcon fontSize="small" />
                            Dominating Emotions
                          </Typography>
                          <Box sx={{ height: '200px' }}>
                            <EmotionBarChart data={emotionDistribution} />
                          </Box>
                        </Box>                      </Box>
                    )}
                  </Box>
                </Box>
              ) : (
                <Box sx={{                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  px: 3,
                  textAlign: 'center',
                  background: customTheme.colors.surface.glass,
                  borderRadius: customTheme.borderRadius.xl,
                  border: `1px solid ${customTheme.colors.border}`,
                  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.07), 0 5px 20px rgba(0, 0, 0, 0.05)',
                  position: 'relative',
                  zIndex: 10,
                }}>
                  <Box sx={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 3
                  }}>
                    <Box sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      border: '2px dashed rgba(99,102,241,0.3)',
                    }}>
                    </Box>                  </Box>
                  <Typography variant="body1" sx={{ fontWeight: 500, color: 'text.secondary' }}>
                    Ready to analyze emotions
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary', opacity: 0.8 }}>
                    Enter a YouTube URL above to begin
                  </Typography>
                </Box>
              )}
            </Box>
          </motion.div>
        </Grid>
      </Grid>
    </Box>      {isLoading && (
        <Box className="loading-overlay" sx={{
          background: customTheme.colors.background.cosmic,
          backdropFilter: 'blur(40px)',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: customTheme.colors.gradients.glow,
            opacity: 0.4,
            zIndex: 1,
          }
        }}>
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              maxWidth: '700px',
              textAlign: 'center',
              padding: '0 30px',
              position: 'relative',
              zIndex: 2,
            }}
          >
            {/* Premium Enhanced Emotion Visualization */}
            <Box sx={{ position: 'relative', width: 320, height: 320, mb: 8 }}>
              {/* Outermost Orbital Ring */}
              <motion.div
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 30,
                  repeat: Infinity,
                  ease: "linear"
                }}
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 280,
                  height: 280,
                  borderRadius: '50%',
                  border: `1px solid ${customTheme.colors.primary.main}15`,
                  zIndex: 3,
                }}
              />

              {/* Central Premium Orb */}
              <motion.div
                animate={{
                  scale: [1, 1.15, 1],
                  opacity: [0.85, 1, 0.85],
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 6,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 160,
                  height: 160,
                  borderRadius: '50%',
                  background: customTheme.colors.gradients.aurora,
                  boxShadow: `
                    0 0 80px ${customTheme.colors.primary.glow},
                    0 0 160px ${customTheme.colors.secondary.glow},
                    0 0 240px ${customTheme.colors.tertiary.glow},
                    inset 0 0 80px rgba(255, 255, 255, 0.15)
                  `,
                  filter: 'blur(0.8px)',
                  zIndex: 7,
                }}
              />

              {/* Inner Crystalline Core */}
              <motion.div
                animate={{
                  scale: [0.7, 1.1, 0.7],
                  opacity: [0.6, 1, 0.6],
                  rotate: [0, -360],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 100,
                  height: 100,
                  borderRadius: '50%',                  background: `linear-gradient(135deg, ${customTheme.colors.primary.main}, ${customTheme.colors.secondary.main})`,
                  boxShadow: `0 0 60px ${customTheme.colors.primary.main}A0`,
                  zIndex: 9,
                }}
              />

              {/* Enhanced Emotion Particles */}
              {[
                { name: 'happiness', icon: '😊', color: customTheme.colors.emotion.happiness },
                { name: 'anger', icon: '😠', color: customTheme.colors.emotion.anger },
                { name: 'sadness', icon: '😢', color: customTheme.colors.emotion.sadness },
                { name: 'surprise', icon: '�', color: customTheme.colors.emotion.surprise },
                { name: 'fear', icon: '�', color: customTheme.colors.emotion.fear },
                { name: 'love', icon: '�', color: customTheme.colors.emotion.love },
                { name: 'excitement', icon: '🤩', color: customTheme.colors.emotion.excitement },
                { name: 'contemplation', icon: '🤔', color: customTheme.colors.emotion.neutral },
                { name: 'disgust', icon: '🤢' }
              ].map((emotion, i) => {
                const angle = (i / 6) * Math.PI * 2;
                const radius = 100;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                
                return (
                  <motion.div
                    key={emotion.name}
                    animate={{
                      rotate: [0, 360],
                      scale: [0.8, 1.2, 0.8],
                    }}
                    transition={{
                      rotate: {
                        duration: 15,
                        repeat: Infinity,
                        ease: "linear"
                      },
                      scale: {
                        duration: 3 + i * 0.5,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }
                    }}
                    style={{
                      position: 'absolute',
                      top: `calc(50% + ${y}px)`,
                      left: `calc(50% + ${x}px)`,
                      transform: 'translate(-50%, -50%)',
                      width: 36,
                      height: 36,
                      borderRadius: '50%',
                      background: `rgba(255, 255, 255, 0.1)`,
                      border: `2px solid ${customTheme.colors.primary.main}40`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      zIndex: 10,
                    }}
                  >
                    {emotion.icon}
                  </motion.div>
                );
              })}              {/* Simplified rings for depth */}
              <motion.div
                animate={{
                  rotate: [0, 360],
                  scale: [1, 1.02, 1],
                }}
                transition={{
                  rotate: {
                    duration: 20,
                    repeat: Infinity,
                    ease: "linear"
                  },
                  scale: {
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }
                }}
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 200,
                  height: 200,
                  borderRadius: '50%',
                  border: `2px solid ${customTheme.colors.primary.main}30`,
                  zIndex: 5,
                }}
              />
              <motion.div
                animate={{
                  rotate: [0, -360],
                  scale: [1, 1.03, 1],
                }}
                transition={{
                  rotate: {
                    duration: 25,
                    repeat: Infinity,
                    ease: "linear"
                  },
                  scale: {
                    duration: 5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }
                }}
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 240,
                  height: 240,
                  borderRadius: '50%',
                  border: `1px solid ${customTheme.colors.primary.main}20`,
                  zIndex: 4,
                }}
              />
            </Box>            {/* Premium Loading Title */}
            <Typography
              variant="h2"
              component="h1"
              gutterBottom
              align="center"              sx={{
                fontWeight: 900,
                mb: 4,
                color: customTheme.colors.text.primary, // Simple text color
                fontSize: { xs: '2.5rem', md: '3.5rem' },
                letterSpacing: '-0.04em',
                filter: 'drop-shadow(0 6px 12px rgba(0, 0, 0, 0.4))',
                fontFamily: customTheme.typography.fontFamily.heading,
              }}
            >
              🧠✨ AI Emotion Analysis
            </Typography>

            {/* Premium Subtitle */}
            <Typography
              variant="h5"
              align="center"
              sx={{
                fontWeight: 500,
                color: customTheme.colors.text.secondary,
                mb: 6,
                fontSize: { xs: '1.1rem', md: '1.4rem' },
                letterSpacing: '0.02em',
                textShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
                opacity: 0.9,
                maxWidth: '500px',
                lineHeight: 1.4,
              }}
            >
              Decoding human emotions with premium AI technology
            </Typography>

            {/* Enhanced Progress Section */}
            <Box sx={{ mb: 6, width: '100%', maxWidth: '550px' }}>
              <Typography
                align="center"
                sx={{
                  fontWeight: 600,
                  color: customTheme.colors.text.primary,
                  mb: 3,
                  fontSize: '1.2rem',
                  letterSpacing: '0.01em',
                  textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
                }}
              >
                🧠 Analyzing video content...
              </Typography>

              {/* Enhanced loading spinner with dark theme */}
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  mb: 3,
                }}
              >
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  style={{
                    width: 40,
                    height: 40,
                    border: `4px solid ${customTheme.colors.primary.main}30`,
                    borderTop: `4px solid ${customTheme.colors.primary.main}`,
                    borderRadius: '50%',
                    boxShadow: `0 0 20px ${customTheme.colors.primary.main}50`,
                  }}
                />
              </Box>

              <Typography
                align="center"
                variant="body2"
                sx={{
                  color: customTheme.colors.text.secondary,
                  fontSize: '1rem',
                  opacity: 0.8,
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)',
                }}
              >
                This may take a few moments...
              </Typography>
            </Box>            {/* Premium fun fact display with luxury styling */}
            <Box 
              sx={{ 
                maxWidth: '600px', 
                textAlign: 'center',
                background: customTheme.glassmorphism.luxury.background,
                backdropFilter: customTheme.glassmorphism.luxury.backdropFilter,
                borderRadius: customTheme.borderRadius['2xl'],
                padding: 4,
                border: customTheme.glassmorphism.luxury.border,
                boxShadow: customTheme.shadows['2xl'],
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '2px',
                  background: customTheme.colors.gradients.aurora,
                  borderRadius: customTheme.borderRadius.full,
                }
              }}            >
              <Typography
                variant="h5"
                sx={{
                  color: customTheme.colors.primary.main,
                  fontWeight: 700,
                  mb: 3,
                  fontSize: '1.3rem',
                  textShadow: customTheme.shadows.glow,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 2,
                }}
              >
                <Box sx={{
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  background: customTheme.colors.gradients.primary,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.2rem',
                  boxShadow: customTheme.shadows.glow,
                }}>
                  💡
                </Box>
                Premium Insights
              </Typography>              
              <AnimatePresence mode="wait">
                <motion.div
                  key={factIndex}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.95 }}
                  transition={{ duration: 0.8, ease: "easeInOut" }}
                >                  <Typography
                    variant="body1"
                    sx={{
                      color: customTheme.colors.text.secondary, // Simple text color
                      fontStyle: 'normal',
                      lineHeight: 1.8,
                      fontSize: '1.1rem',
                      textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
                      fontWeight: 500,
                      letterSpacing: '0.01em',                    }}
                  >
                    {emotionFacts[factIndex]}
                  </Typography>
                </motion.div>
              </AnimatePresence>
            </Box>
          </motion.div>
        </Box>
      )}

      {/* Feedback Modal */}
      <FeedbackModal
        open={feedbackModalOpen}
        onClose={handleCloseFeedback}
        transcriptData={analysisData?.transcript || []}
        videoTitle={analysisData?.title || 'Unknown Video'}
      />

      {/* Add Video Modal */}
      <AddVideoModal
        open={addVideoModalOpen}
        onClose={() => setAddVideoModalOpen(false)}
        onSubmit={handleUrlUpload}
      />

      {/* Settings Modal */}
      <SettingsModal
        open={settingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
      />
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={muiTheme}>
      <CssBaseline />
      <VideoProvider>
        <Box sx={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
          <AppContent />
        </Box>
      </VideoProvider>
    </ThemeProvider>
  );
}

export default App;
