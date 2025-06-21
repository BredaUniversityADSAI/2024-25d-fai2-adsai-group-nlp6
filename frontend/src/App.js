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

// Create MUI theme based on our custom theme
const muiTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: customTheme.colors.primary.main,
      light: customTheme.colors.primary.light,
      dark: customTheme.colors.primary.dark,
    },
    secondary: {
      main: customTheme.colors.secondary.main,
      light: customTheme.colors.secondary.light,
      dark: customTheme.colors.secondary.dark,
    },
    background: {
      default: customTheme.colors.background.primary,
      paper: customTheme.colors.surface.glass,
    },
    text: {
      primary: customTheme.colors.text.primary,
      secondary: customTheme.colors.text.secondary,
    },
  },
  typography: {
    fontFamily: customTheme.typography.fontFamily.primary,
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: customTheme.colors.background.primary,
          overflow: 'hidden',
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
    getCurrentEmotion
  } = useVideo();

  const [searchTerm, setSearchTerm] = useState('');
  const [tabValue, setTabValue] = useState(0); // 0 for Live Stream, 1 for Full Analysis
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [factIndex, setFactIndex] = useState(0);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
    // Modal states for new modular layout
  const [addVideoModalOpen, setAddVideoModalOpen] = useState(false);  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const tabsRef = useRef(null);
  const transcriptContainerRef = useRef(null); // Ref for auto-scroll functionality

  // Emotion facts for loading screen
  const emotionFacts = [
    "Humans can express over 7,000 different facial expressions.",
    "The amygdala processes emotions before our conscious brain is aware of them.",
    "Smiling can actually make you feel happier due to facial feedback.",
    "Emotions are universal across cultures, but their expressions may vary.",
    "Your brain processes emotions in just 74 milliseconds.",
    "Music can trigger the same emotion centers as food and other rewards.",
    "Emotions can be contagious - you can 'catch' feelings from others.",
    "The average human experiences 27 distinct emotions.",
    "Suppressing emotions can weaken your immune system.",
    "Emotional intelligence predicts 58% of success in all types of jobs."
  ];

  // Progress phases for loading
  const loadingPhases = [
    "Extracting audio from video...",
    "Transcribing speech...",
    "Analyzing speech patterns...",
    "Detecting emotional cues...",
    "Mapping emotional journey...",
    "Finalizing analysis..."
  ];

  // Update loading phase and fact periodically
  useEffect(() => {
    if (!isLoading) return;

    const phaseInterval = setInterval(() => {
      setLoadingPhase(prev => (prev + 1) % loadingPhases.length);
    }, 3000);

    const factInterval = setInterval(() => {
      setFactIndex(prev => (prev + 1) % emotionFacts.length);
    }, 5000);

    return () => {
      clearInterval(phaseInterval);
      clearInterval(factInterval);
    };  }, [isLoading, loadingPhases.length, emotionFacts.length]);

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
    }  };
  // Handle URL upload - placeholder implementation
  const handleUrlUpload = (url) => {
    console.log('URL upload requested:', url);
    // TODO: Implement URL upload functionality
    // Note: videoUrl is managed by the VideoContext
  };

  // Filter history based on search term
  const filteredHistory = videoHistory.filter(video =>
    video.title.toLowerCase().includes(searchTerm.toLowerCase())
  );  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: customTheme.colors.background.primary }}>
      <Sidebar
        videoHistory={filteredHistory}
        onVideoSelect={loadFromHistory}
        onAddVideo={() => setAddVideoModalOpen(true)}
        onSettings={() => setSettingsModalOpen(true)}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}      />      {/* Main Content Area with Grid Layout */}
      <Box sx={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        p: 2, 
        pl: '120px', // Reduced padding for better positioning
        minHeight: '100vh',
        justifyContent: 'flex-start', // Changed from center to start
        alignItems: 'stretch' // Changed from center to stretch
      }}>
        <Grid container spacing={3} sx={{ 
          maxWidth: '100%', 
          width: '100%',
          height: 'fit-content',
          alignItems: 'stretch',
          justifyContent: 'flex-start', // Changed from space-between
          py: 3, // Increased vertical padding
          mt: 2 // Added top margin for better spacing
        }}>
          {/* Left Side - Dashboard */}
          <Grid item xs={12} lg={4}>            <Paper
              elevation={3}
              sx={{ 
                height: '78vh', // Increased height for better content display
                p: 3,
                background: customTheme.glassmorphism.primary.background,
                backdropFilter: customTheme.glassmorphism.primary.backdropFilter,
                border: customTheme.glassmorphism.primary.border,
                borderRadius: customTheme.borderRadius.xl,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }
              }}
            >
              <Typography variant="h6" sx={{ 
                mb: 2, 
                fontWeight: 600,
                color: customTheme.colors.text.primary,
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}>
                📊 Dashboard
              </Typography>
              
              <Box sx={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: customTheme.colors.text.secondary,
                textAlign: 'center'
              }}>
                <Typography variant="body2" sx={{ opacity: 0.7 }}>
                  Dashboard content coming soon...
                </Typography>
              </Box>
            </Paper>
          </Grid>          {/* Center - Video Player and Transcript */}
          <Grid item xs={12} lg={4}><Paper 
              elevation={3}              sx={{ 
                height: '78vh', // Increased height
                p: 3,
                background: customTheme.glassmorphism.primary.background,
                backdropFilter: customTheme.glassmorphism.primary.backdropFilter,
                border: customTheme.glassmorphism.primary.border,
                borderRadius: customTheme.borderRadius.xl,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }              }}>
              {/* Video Player Section */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" sx={{ 
                  mb: 2, 
                  fontWeight: 600,
                  color: customTheme.colors.text.primary,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1
                }}>
                  🎥 Video Player
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
                        </Box>
                      );
                    })}
                  </Box>
                </Box>
              )}
            </Paper>
          </Grid>          {/* Right Side - Emotion Analytics */}
          <Grid item xs={12} lg={4}>            <Box
              sx={{
                height: '78vh', // Increased height
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {analysisData ? (                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  background: customTheme.colors.surface.glass,
                  borderRadius: customTheme.borderRadius.xl,
                  border: `1px solid ${customTheme.colors.border}`,
                  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.07), 0 5px 20px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                  position: 'relative',
                  zIndex: 10, /* Ensure it's above any background elements */
                  '&:hover': {
                    transform: 'translateY(-6px)',
                    boxShadow: '0 20px 50px rgba(0, 0, 0, 0.12), 0 8px 25px rgba(0, 0, 0, 0.08)'
                  }
                }}>                  <Box sx={{
                    padding: 3,
                    borderBottom: `1px solid ${customTheme.colors.border}`,
                    display: 'flex',
                    flexDirection: 'column',
                    background: customTheme.colors.surface.card,
                  }}>
                    <Typography variant="h6" fontWeight={600} sx={{
                      background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
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
                    </Box>
                  </Box>
                  <Typography variant="body1" sx={{ fontWeight: 500, color: 'text.secondary' }}>
                    Ready to analyze emotions
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary', opacity: 0.8 }}>
                    Enter a YouTube URL above to begin
                  </Typography>                </Box>
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>

      {isLoading && (
        <Box className="loading-overlay">
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              maxWidth: '600px',
              textAlign: 'center',
              padding: '0 20px'
            }}
          >
            {/* Enhanced animated emotion visualization */}
            <Box sx={{ position: 'relative', width: 240, height: 240, mb: 6 }}>
              {/* Central pulsing orb */}
              <motion.div
                animate={{
                  scale: [1, 1.15, 1],
                  opacity: [0.8, 1, 0.8],
                  rotate: [0, 360],
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
                  width: 120,
                  height: 120,
                  borderRadius: '50%',
                  background: 'radial-gradient(circle at 30% 30%, #6366F1FF 0%, #8B5CF6DD 50%, #EC489988 100%)',
                  boxShadow: `
                    0 0 40px rgba(99, 102, 241, 0.5),
                    0 0 80px rgba(139, 92, 246, 0.3),
                    inset 0 0 40px rgba(255, 255, 255, 0.2)
                  `,
                }}
              />

              {/* Orbiting emotion particles */}
              {[
                { color: '#10B981', name: 'happiness' },
                { color: '#EF4444', name: 'anger' },
                { color: '#3B82F6', name: 'sadness' },
                { color: '#F59E0B', name: 'surprise' },
                { color: '#8B5CF6', name: 'fear' },
                { color: '#84CC16', name: 'disgust' }
              ].map((emotion, i) => {
                const angle = (i / 6) * Math.PI * 2;
                const radius = 85;
                return (
                  <motion.div
                    key={emotion.name}
                    animate={{
                      rotate: [0, 360],
                      scale: [0.8, 1.2, 0.8],
                    }}
                    transition={{
                      rotate: {
                        duration: 8,
                        repeat: Infinity,
                        ease: "linear"
                      },
                      scale: {
                        duration: 2 + i * 0.3,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }
                    }}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: `translate(-50%, -50%) translate(${Math.cos(angle) * radius}px, ${Math.sin(angle) * radius}px)`,
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${emotion.color}, ${emotion.color}CC)`,
                      boxShadow: `0 0 20px ${emotion.color}77`,
                    }}
                  />
                );
              })}

              {/* Outer ring effect */}
              <motion.div
                animate={{
                  rotate: [0, -360],
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  rotate: {
                    duration: 12,
                    repeat: Infinity,
                    ease: "linear"
                  },
                  scale: {
                    duration: 3,
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
                  border: '2px solid rgba(99, 102, 241, 0.2)',
                  background: 'radial-gradient(circle, transparent 70%, rgba(99, 102, 241, 0.1) 100%)',
                }}
              />
            </Box>            {/* Enhanced loading title */}
            <Typography
              variant="h3"
              component="h2"
              gutterBottom
              align="center"
              sx={{
                fontWeight: 800,
                mb: 3,
                background: 'linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                fontSize: { xs: '2rem', md: '2.5rem' },
                letterSpacing: '-0.02em',
              }}
            >
              Analyzing Emotions
            </Typography>

            {/* Enhanced processing phase indicator */}
            <Box sx={{ mb: 4, width: '100%', maxWidth: '400px' }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={loadingPhase}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.5 }}
                >
                  <Typography
                    align="center"
                    sx={{
                      fontWeight: 600,
                      color: 'text.secondary',
                      mb: 2,
                      fontSize: '1.1rem',
                      letterSpacing: '0.01em',
                    }}
                  >
                    {loadingPhases[loadingPhase]}
                  </Typography>
                </motion.div>
              </AnimatePresence>

              {/* Enhanced progress bar */}
              <Box
                sx={{
                  height: 8,
                  bgcolor: 'rgba(0, 0, 0, 0.08)',
                  borderRadius: 4,
                  overflow: 'hidden',
                  width: '100%',
                  position: 'relative',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
              >
                <motion.div
                  animate={{
                    width: [
                      `${(loadingPhase / loadingPhases.length) * 100}%`,
                      `${((loadingPhase + 1) / loadingPhases.length) * 100}%`
                    ],
                  }}
                  transition={{
                    duration: 3,
                    ease: "easeInOut"
                  }}
                  style={{
                    height: '100%',
                    background: 'linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899)',
                    borderRadius: '4px',
                    boxShadow: '0 0 12px rgba(99, 102, 241, 0.4)',
                  }}
                />
              </Box>
            </Box>

            {/* Fun fact display */}
            <Box sx={{ maxWidth: '450px', textAlign: 'center' }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={factIndex}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.5 }}
                >                  <Typography
                    variant="body2"
                    sx={{
                      color: 'text.secondary',
                      fontStyle: 'italic',
                      lineHeight: 1.6,
                      fontSize: '0.95rem',
                    }}
                  >
                    💡 Did you know? {emotionFacts[factIndex]}
                  </Typography>
                </motion.div>
              </AnimatePresence>
            </Box>
          </motion.div>
        </Box>
      )}      {/* Feedback Modal */}
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
