import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Button from '@mui/material/Button';
import './App.css';
import { motion, AnimatePresence } from 'framer-motion';
import PsychologyAltIcon from '@mui/icons-material/PsychologyAlt';
import VisibilityIcon from '@mui/icons-material/Visibility';
import InsightsIcon from '@mui/icons-material/Insights';
import DonutLargeIcon from '@mui/icons-material/DonutLarge';
import TimelineIcon from '@mui/icons-material/Timeline';
import EditNoteIcon from '@mui/icons-material/EditNote';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import styled from '@emotion/styled';
import * as XLSX from 'xlsx';

// Import components
import UrlInput from './components/UrlInput';
import VideoPlayer from './components/VideoPlayer';
import SearchBar from './components/SearchBar';
import VideoHistory from './components/VideoHistory';
import Transcript from './components/Transcript';
import EmotionCurrent from './components/EmotionCurrent';
import EmotionBarChart from './components/EmotionBarChart';
import EmotionTimeline from './components/EmotionTimeline';
import VideoMemoryHeader from './components/VideoMemoryHeader';
import FeedbackButton from './components/FeedbackButton';
import FeedbackModal from './components/FeedbackModal';

// Import context
import { VideoProvider, useVideo } from './VideoContext';
import { processEmotionData } from './utils';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6366F1', // Indigo
      light: '#8B5CF6',
      dark: '#4338CA',
    },
    secondary: {
      main: '#EC4899', // Pink
      light: '#F472B6',
      dark: '#DB2777',
    },
    accent: {
      teal: '#06B6D4', // Cyan
      purple: '#8B5CF6', // Purple
      success: '#10B981', // Emerald
      warning: '#F59E0B', // Amber
    },
    background: {
      default: 'linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)',
      paper: 'rgba(255, 255, 255, 0.95)',
    },
    emotion: {
      happiness: '#10B981', // Emerald - Joy, positivity
      sadness: '#3B82F6', // Blue - Calm, melancholy
      anger: '#EF4444', // Red - Intensity, passion
      fear: '#8B5CF6', // Purple - Mystery, anxiety
      surprise: '#F59E0B', // Amber - Energy, excitement
      disgust: '#84CC16', // Lime - Natural, aversion
      neutral: '#64748B', // Slate - Balance, neutrality
    },
    surface: {
      glass: 'rgba(255, 255, 255, 0.75)',
      elevated: 'rgba(255, 255, 255, 0.95)',
    }
  },  typography: {
    fontFamily: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontSize: '3rem',
      fontWeight: 800,
      letterSpacing: '-0.04em',
      lineHeight: 1.1,
      background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
    },
    h2: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.03em',
      lineHeight: 1.2,
    },
    h3: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      lineHeight: 1.3,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.015em',
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 600,
      letterSpacing: '-0.005em',
      lineHeight: 1.5,
    },
    subtitle1: {
      fontSize: '1.125rem',
      fontWeight: 500,
      lineHeight: 1.6,
      color: '#64748B',
    },
    subtitle2: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: '#64748B',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.7,
      fontWeight: 400,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      fontWeight: 400,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontSize: '0.95rem',
      letterSpacing: '0.01em',
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.5,
      fontWeight: 500,
      color: '#64748B',
    },
  },  shape: {
    borderRadius: 20,
  },
  spacing: 8,
  shadows: [
    'none',
    '0px 1px 3px rgba(0, 0, 0, 0.05), 0px 1px 2px rgba(0, 0, 0, 0.06)',
    '0px 2px 6px rgba(0, 0, 0, 0.06), 0px 2px 4px rgba(0, 0, 0, 0.07)',
    '0px 4px 12px rgba(0, 0, 0, 0.06), 0px 2px 8px rgba(0, 0, 0, 0.08)',
    '0px 8px 20px rgba(0, 0, 0, 0.08), 0px 4px 12px rgba(0, 0, 0, 0.10)',
    '0px 12px 28px rgba(0, 0, 0, 0.10), 0px 6px 16px rgba(0, 0, 0, 0.12)',
    '0px 16px 32px rgba(0, 0, 0, 0.12), 0px 8px 20px rgba(0, 0, 0, 0.14)',
    ...Array(18).fill('none'),
  ],  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          padding: '12px 28px',
          fontSize: '0.95rem',
          fontWeight: 600,
          textTransform: 'none',
          boxShadow: '0px 3px 12px rgba(99, 102, 241, 0.15)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
            transition: 'left 0.6s',
          },
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0px 8px 25px rgba(99, 102, 241, 0.25)',
            '&::before': {
              left: '100%',
            },
          },
          '&:active': {
            transform: 'translateY(0px)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
          color: 'white',
          '&:hover': {
            background: 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)',
          },
        },
        containedSecondary: {
          background: 'linear-gradient(135deg, #EC4899 0%, #F472B6 100%)',
          color: 'white',
          '&:hover': {
            background: 'linear-gradient(135deg, #DB2777 0%, #EC4899 100%)',
          },
        },
        outlined: {
          border: '2px solid rgba(99, 102, 241, 0.2)',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(12px)',
          '&:hover': {
            border: '2px solid rgba(99, 102, 241, 0.4)',
            backgroundColor: 'rgba(99, 102, 241, 0.05)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 24,
          backgroundImage: 'none',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0px 8px 32px rgba(0, 0, 0, 0.08), 0px 4px 16px rgba(0, 0, 0, 0.04)',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0px 16px 48px rgba(0, 0, 0, 0.12), 0px 8px 24px rgba(0, 0, 0, 0.08)',
          },
        },
        elevation1: {
          boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.04), 0px 2px 8px rgba(0, 0, 0, 0.02)',
        },
        elevation2: {
          boxShadow: '0px 8px 24px rgba(0, 0, 0, 0.06), 0px 4px 12px rgba(0, 0, 0, 0.04)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 24,
          overflow: 'hidden',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-6px) scale(1.02)',
            boxShadow: '0px 20px 40px rgba(0, 0, 0, 0.12)',
          },
        }
      }
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '0.95rem',
          minHeight: '52px',
          borderRadius: 16,
          margin: '4px 6px',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          color: '#64748B',
          '&.Mui-selected': {
            backgroundColor: 'rgba(99, 102, 241, 0.12)',
            color: '#6366F1',
            boxShadow: '0px 4px 12px rgba(99, 102, 241, 0.15)',
          },
          '&:hover': {
            backgroundColor: 'rgba(99, 102, 241, 0.05)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          borderRadius: 20,
          padding: '8px',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
        },
        indicator: {
          display: 'none',
        },
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          fontWeight: 600,
          fontSize: '0.875rem',
          height: '36px',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0px 6px 16px rgba(0, 0, 0, 0.15)',
          },
        },
        colorPrimary: {
          background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
          color: 'white',
          boxShadow: '0px 4px 12px rgba(99, 102, 241, 0.3)',
        },
        colorSecondary: {
          background: 'linear-gradient(135deg, #EC4899, #F472B6)',
          color: 'white',
          boxShadow: '0px 4px 12px rgba(236, 72, 153, 0.3)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 16,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(12px)',
            transition: 'all 0.3s ease',
            '& fieldset': {
              border: '2px solid rgba(99, 102, 241, 0.1)',
            },
            '&:hover fieldset': {
              border: '2px solid rgba(99, 102, 241, 0.2)',
            },
            '&.Mui-focused fieldset': {
              border: '2px solid #6366F1',
              boxShadow: '0px 0px 0px 4px rgba(99, 102, 241, 0.1)',
            },          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 16,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(12px)',
            transition: 'all 0.3s ease',
            '& fieldset': {
              border: '2px solid rgba(99, 102, 241, 0.1)',
            },
            '&:hover fieldset': {
              border: '2px solid rgba(99, 102, 241, 0.2)',
            },
            '&.Mui-focused fieldset': {
              border: '2px solid #6366F1',
              boxShadow: '0px 0px 0px 4px rgba(99, 102, 241, 0.1)',
            },
          },
        },
      },
    },
  },
});

// Tab panel component
const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`emotion-tabpanel-${index}`}
      aria-labelledby={`emotion-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      <AnimatePresence mode="wait">
        {value === index && (
          <motion.div
            key={`tab-content-${index}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{ height: '100%' }}
          >
            <Box sx={{ height: '100%' }}>
              {children}
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Add a styled component for the tab indicator animation
const StyledTabs = styled(Tabs)(({ theme }) => ({
  position: 'relative',
  minHeight: '40px',
  backgroundColor: '#FFFFFF',
  borderRadius: '16px',
  padding: '4px',
  width: '100%',
  border: '1px solid rgba(229, 231, 235, 0.6)',
  boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.02)',
  '& .MuiTab-root': {
    minHeight: '40px',
    fontSize: '0.85rem',
    fontWeight: 600,
    py: 0.5,
    px: 2,
    borderRadius: '12px',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    color: 'rgba(0, 0, 0, 0.6)',
    zIndex: 1,
    '&:hover': {
      backgroundColor: 'rgba(99, 102, 241, 0.05)',
      color: '#4F46E5',
    },
  },
  '& .Mui-selected': {
    color: '#6366F1 !important',
    fontWeight: 700,
  },
  '& .MuiTabs-indicator': {
    display: 'none',
  },
}));

// Create a tab indicator animation component
const TabIndicator = ({ activeIndex, tabsRef }) => {
  const [dimensions, setDimensions] = useState({ width: 0, left: 0 });

  useEffect(() => {
    if (tabsRef.current) {
      const activeTab = tabsRef.current.querySelector(`[aria-selected="true"]`);
      if (activeTab) {
        const { width, left } = activeTab.getBoundingClientRect();
        const parentLeft = tabsRef.current.getBoundingClientRect().left;
        setDimensions({
          width,
          left: left - parentLeft
        });
      }
    }
  }, [activeIndex, tabsRef]);

  return (
    <motion.div
      style={{
        position: 'absolute',
        height: 'calc(100% - 8px)',
        top: 4,
        borderRadius: 12,
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        zIndex: 0,
        left: dimensions.left,
        width: dimensions.width,
      }}
      initial={false}
      animate={{
        left: dimensions.left,
        width: dimensions.width,
      }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    />
  );
};

// Main App Content
function AppContent() {
  const {
    videoUrl,
    currentTime,
    setCurrentTime,
    isLoading,
    analysisData,
    videoHistory,
    processVideo,
    loadFromHistory,
    getCurrentEmotion
  } = useVideo();
  const [searchTerm, setSearchTerm] = useState('');
  const [tabValue, setTabValue] = useState(0); // 0 for Live Stream, 1 for Full Analysis
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [factIndex, setFactIndex] = useState(0);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
  const tabsRef = useRef(null);

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
    };
  }, [isLoading, loadingPhases.length, emotionFacts.length]);

  // Current emotion based on timestamp
  const currentEmotion = getCurrentEmotion();

  // Process analyzed data for visualizations
  const { emotionDistribution, intensityTimeline } =
    analysisData ? processEmotionData(analysisData) : { emotionDistribution: {}, intensityTimeline: [] };

  // Handle video progress
  const handleVideoProgress = (state) => {
    setCurrentTime(state.playedSeconds);
  };

  // Handle jumping to a specific time in the video
  const handleSentenceClick = (time) => {
    setCurrentTime(time);
    // VideoPlayer will be updated through context
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  // Handle search
  const handleSearch = (term) => {
    setSearchTerm(term);
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
  };
  // Helper function to get intensity value with proper validation
  const getIntensityValue = (segment) => {
    // Try different possible field names and validate they're numbers
    const intensity = segment.intensity ?? segment.confidence ?? segment.strength ?? segment.score;
    
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
    }
  };

  // Filter history based on search term
  const filteredHistory = videoHistory.filter(video =>
    video.title.toLowerCase().includes(searchTerm.toLowerCase())
  );
  return (
    <Box className="app-container" sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Container
        maxWidth="xl"
        sx={{
          py: { xs: 3, md: 5 },
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          minHeight: '100vh',
          position: 'relative',
          zIndex: 1,        }}
      >
        <Grid container spacing={4} sx={{ flexGrow: 1 }}>
          {/* Video History Panel - Left Side */}
          <Grid item xs={12} md={3}>
            <Paper
              elevation={0}
              className="panel"
              sx={{
                p: 3,
                height: '80vh',
                overflow: 'auto',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                borderRadius: '24px',
                border: '1px solid rgba(255, 255, 255, 0.9)',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.07), 0 5px 20px rgba(0, 0, 0, 0.05)',
                transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: '0 15px 45px rgba(0, 0, 0, 0.09), 0 7px 25px rgba(0, 0, 0, 0.07)'
                }
              }}
            >
              <VideoMemoryHeader
                searchValue={searchTerm}
                onSearchChange={(e) => handleSearch(e.target.value)}
                onSearchClear={() => handleSearch('')}
              />
              <Box mt={1}>
                <VideoHistory videos={filteredHistory} onVideoSelect={loadFromHistory} />
              </Box>
            </Paper>
          </Grid>

          {/* Video Player & Transcript - Center */}
          <Grid item xs={12} md={5}>
            <Paper
              elevation={0}
              className="panel"
              sx={{
                p: 3,
                height: '80vh',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                borderRadius: '24px',
                border: '1px solid rgba(255, 255, 255, 0.9)',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.07), 0 5px 20px rgba(0, 0, 0, 0.05)',
                transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: '0 15px 45px rgba(0, 0, 0, 0.09), 0 7px 25px rgba(0, 0, 0, 0.07)'
                }
              }}
            >              <Box sx={{
                borderRadius: 2,
                overflow: 'hidden',
                boxShadow: '0px 8px 20px rgba(0, 0, 0, 0.1)',
                flexShrink: 0,
              }}>
                <VideoPlayer
                  url={videoUrl}
                  onProgress={handleVideoProgress}
                  currentTime={currentTime}
                />
              </Box>              {/* Action Buttons - shows when analysis data is available */}
              {analysisData && (
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  gap: 2, 
                  mt: 2,
                  flexWrap: 'wrap' 
                }}>
                  {/* Toned down feedback button */}
                  <Button
                    onClick={handleOpenFeedback}
                    disabled={!analysisData}
                    variant="outlined"
                    startIcon={<EditNoteIcon />}
                    sx={{
                      textTransform: 'none',
                      borderRadius: 2,
                      px: 3,
                      py: 1,
                      fontSize: '0.9rem',
                      fontWeight: 500,
                      borderColor: 'divider',
                      color: 'text.secondary',
                      backgroundColor: 'transparent',
                      '&:hover': {
                        borderColor: 'primary.main',
                        backgroundColor: 'rgba(99, 102, 241, 0.04)',
                        color: 'primary.main',
                      },
                      '&:disabled': {
                        borderColor: 'divider',
                        color: 'text.disabled',
                      }
                    }}
                  >
                    Give Feedback
                  </Button>
                  
                  {/* Export predictions button */}
                  <Button
                    onClick={handleExportPredictions}
                    disabled={!analysisData}
                    variant="outlined"
                    startIcon={<FileDownloadIcon />}
                    sx={{
                      textTransform: 'none',
                      borderRadius: 2,
                      px: 3,
                      py: 1,
                      fontSize: '0.9rem',
                      fontWeight: 500,
                      borderColor: 'divider',
                      color: 'text.secondary',
                      backgroundColor: 'transparent',
                      '&:hover': {
                        borderColor: 'success.main',
                        backgroundColor: 'rgba(16, 185, 129, 0.04)',
                        color: 'success.main',
                      },
                      '&:disabled': {
                        borderColor: 'divider',
                        color: 'text.disabled',
                      }
                    }}
                  >
                    Export Results
                  </Button>
                </Box>
              )}

              {analysisData && (
                <Box mt={4} sx={{
                  flexGrow: 1,
                  overflow: 'hidden',
                  minHeight: 0,
                  display: 'flex',
                  flexDirection: 'column',
                }}>
                  <Transcript
                    data={analysisData.transcript}
                    currentTime={currentTime}
                    onSentenceClick={handleSentenceClick}
                  />
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Emotion Display - Right Side */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                height: '80vh',
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
                  background: '#FFFFFF',
                  borderRadius: '24px',
                  border: '1px solid rgba(229, 231, 235, 0.8)',
                  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.07), 0 5px 20px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                  position: 'relative',
                  zIndex: 10, /* Ensure it's above any background elements */
                  '&:hover': {
                    transform: 'translateY(-6px)',
                    boxShadow: '0 20px 50px rgba(0, 0, 0, 0.12), 0 8px 25px rgba(0, 0, 0, 0.08)'
                  }
                }}>
                  <Box sx={{
                    padding: 3,
                    borderBottom: '1px solid rgba(0, 0, 0, 0.06)',
                    display: 'flex',
                    flexDirection: 'column',
                    background: '#FFFFFF',
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
                    </Typography>

                    <Box sx={{ position: 'relative' }} ref={tabsRef}>
                      <StyledTabs
                        value={tabValue}
                        onChange={handleTabChange}
                        variant="fullWidth"
                        scrollButtons={false}
                        aria-label="emotion analysis tabs"
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
                      </StyledTabs>
                      <TabIndicator activeIndex={tabValue} tabsRef={tabsRef} />
                    </Box>
                  </Box>

                  <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                    <TabPanel value={tabValue} index={0}>
                      <Box sx={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'flex-start',
                        gap: 3,
                      }}>
                        {/* Current Emotion Card */}
                        <Box sx={{
                          p: 2.5,
                          borderRadius: '16px',
                          background: '#FFFFFF',
                          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.03), 0 1px 8px rgba(0, 0, 0, 0.02)',
                          border: '1px solid rgba(229, 231, 235, 0.8)',
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
                            <PsychologyAltIcon fontSize="small" />
                            Emotion Pulse
                          </Typography>
                          <Box sx={{
                            height: '250px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <EmotionCurrent
                              emotion={currentEmotion?.emotion}
                              subEmotion={currentEmotion?.sub_emotion}
                              intensity={currentEmotion?.intensity}
                              relatedEmotions={[]} // Would come from backend in a real app
                              compact={true} // Add a compact prop to make it smaller
                            />
                          </Box>
                        </Box>

                        {/* Real-Time Emotion Tracker moved to Live Stream tab */}
                        <Box sx={{
                          p: 2.5,
                          borderRadius: '16px',
                          background: '#FFFFFF',
                          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.03), 0 1px 8px rgba(0, 0, 0, 0.02)',
                          border: '1px solid rgba(229, 231, 235, 0.8)',
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
                          <Box sx={{ height: '250px' }}>
                            <EmotionTimeline
                              data={intensityTimeline}
                              currentTime={currentTime}
                            />
                          </Box>
                        </Box>
                      </Box>
                    </TabPanel>
                    <TabPanel value={tabValue} index={1}>
                      <Box sx={{ height: '100%' }}>
                        <Box sx={{
                          p: 2.5,
                          mb: 3,
                          borderRadius: '16px',
                          background: '#FFFFFF',
                          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.03), 0 1px 8px rgba(0, 0, 0, 0.02)',
                          border: '1px solid rgba(229, 231, 235, 0.8)',
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
                        </Box>
                      </Box>
                    </TabPanel>
                  </Box>
                </Box>
              ) : (
                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  px: 3,
                  textAlign: 'center',
                  background: '#FFFFFF',
                  borderRadius: '24px',
                  border: '1px solid rgba(229, 231, 235, 0.8)',
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
                  </Typography>
                </Box>
              )}
            </Box>
          </Grid>
        </Grid>
      </Container>

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
      )}

      {/* Feedback Modal */}
      <FeedbackModal
        open={feedbackModalOpen}
        onClose={handleCloseFeedback}
        transcriptData={analysisData?.transcript || []}
        videoTitle={analysisData?.title || 'Unknown Video'}
      />
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
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
