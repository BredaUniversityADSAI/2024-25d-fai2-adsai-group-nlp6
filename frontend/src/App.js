import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import './App.css';
import { motion, AnimatePresence } from 'framer-motion';

// Import components
import UrlInput from './components/UrlInput';
import VideoPlayer from './components/VideoPlayer';
import SearchBar from './components/SearchBar';
import VideoHistory from './components/VideoHistory';
import Transcript from './components/Transcript';
import EmotionCurrent from './components/EmotionCurrent';
import EmotionBarChart from './components/EmotionBarChart';
import EmotionTimeline from './components/EmotionTimeline';

// Import context
import { VideoProvider, useVideo } from './VideoContext';
import { processEmotionData } from './utils';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6466F1', // Slightly adjusted Indigo
      light: '#878DF9',
      dark: '#4F46E5',
    },
    secondary: {
      main: '#EC4899', // Pink
      light: '#F472B6',
      dark: '#DB2777',
    },
    accent: {
      teal: '#0EA5E9', // Bright blue
      purple: '#8B5CF6', // Medium purple
      success: '#10B981', // Emerald green
    },
    background: {
      default: '#F9FAFB',
      paper: 'rgba(255, 255, 255, 0.85)',
    },
    emotion: {
      happiness: '#10B981',
      sadness: '#60A5FA',
      anger: '#EF4444',
      fear: '#8B5CF6',
      surprise: '#F59E0B',
      disgust: '#65A30D',
      neutral: '#9CA3AF',
    }
  },
  typography: {
    fontFamily: '"Inter", "Manrope", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.75rem',
      fontWeight: 700,
      letterSpacing: '-0.025em',
      fontFamily: '"Manrope", sans-serif',
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2.25rem',
      fontWeight: 600,
      letterSpacing: '-0.025em',
      fontFamily: '"Manrope", sans-serif',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.015em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.015em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontFamily: '"Manrope", sans-serif',
    },
    subtitle1: {
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.95rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 16,
  },
  shadows: [
    'none',
    '0px 1px 2px rgba(0, 0, 0, 0.06), 0px 1px 3px rgba(0, 0, 0, 0.1)',
    '0px 2px 4px rgba(0, 0, 0, 0.06), 0px 1px 5px rgba(0, 0, 0, 0.1)',
    '0px 4px 6px rgba(0, 0, 0, 0.06), 0px 2px 10px rgba(0, 0, 0, 0.1)',
    '0px 6px 15px rgba(0, 0, 0, 0.06), 0px 3px 12px rgba(0, 0, 0, 0.1)',
    '0px 10px 20px rgba(0, 0, 0, 0.07), 0px 5px 15px rgba(0, 0, 0, 0.1)',
    ...Array(19).fill('none'),
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '10px 20px',
          boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.08)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          fontWeight: 600,
          '&:hover': {
            boxShadow: '0px 6px 16px rgba(0, 0, 0, 0.08), 0px 2px 4px rgba(0, 0, 0, 0.1)',
            transform: 'translateY(-1px)'
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #6366F1, #7C3AED)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
          }
        }
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.04), 0px 1px 3px rgba(0, 0, 0, 0.05)',
          backdropFilter: 'blur(12px)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0px 2px 6px rgba(0, 0, 0, 0.02), 0px 1px 2px rgba(0, 0, 0, 0.04)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          overflow: 'hidden',
        }
      }
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          minHeight: '48px',
          borderRadius: 10,
          transition: 'all 0.3s ease',
          marginRight: '4px',
          '&.Mui-selected': {
            backgroundColor: 'rgba(99, 102, 241, 0.08)',
          }
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.6)',
          borderRadius: 12,
          padding: 4,
        },
        indicator: {
          height: 0, // Hide the default indicator
        },
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
          '&.MuiChip-colorPrimary': {
            background: 'linear-gradient(135deg, #6366F1, #818CF8)',
          },
          '&.MuiChip-colorSecondary': {
            background: 'linear-gradient(135deg, #EC4899, #F472B6)',
          }
        },
      }
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
          }
        }
      }
    }
  },
});

// Tab panel component
function TabPanel(props) {
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
      {value === index && (
        <Box sx={{ height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

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

  // Filter history based on search term
  const filteredHistory = videoHistory.filter(video =>
    video.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Box className="app-container" sx={{ minHeight: '100vh' }}>
      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* <Typography
          variant="h1"
          component="h1"
          gutterBottom
          align="center"
          className="page-title"
          sx={{
            mb: 4,
            fontWeight: 800,
            position: 'relative',
          }}
        >
          Emotion Journey
        </Typography> */}

        <Box
          mb={5}
          sx={{
            maxWidth: '700px',
            mx: 'auto',
            position: 'relative',
            zIndex: 1,
          }}
        >
          <UrlInput onSubmit={processVideo} isLoading={isLoading} />
        </Box>

        <Grid container spacing={3}>
          {/* Video History Panel - Left Side */}
          <Grid item xs={12} md={3}>
            <Paper
              elevation={0}
              className="panel"
              sx={{
                p: 3,
                height: '75vh',
                overflow: 'auto',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Box
                  component="span"
                  sx={{
                    width: 12,
                    height: 12,
                    mr: 1.5,
                    borderRadius: '50%',
                    bgcolor: 'primary.main'
                  }}
                />
                Video Memory
              </Typography>
              <SearchBar                 value={searchTerm}                onChange={(e) => handleSearch(e.target.value)}                onClear={() => handleSearch('')}              />
              <Box mt={3}>
                <VideoHistory videos={filteredHistory} onVideoSelect={loadFromHistory} />
              </Box>
            </Paper>
          </Grid>

          {/* Video Player & Transcript - Center */}
          <Grid item xs={12} md={6}>
            <Paper
              elevation={0}
              className="panel"
              sx={{
                p: 3,
                height: '75vh',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
              }}
            >
              <Box sx={{
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
              </Box>

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
          <Grid item xs={12} md={3}>
            <Paper
              elevation={0}
              className="panel"
              sx={{
                p: 3,
                height: '75vh',
                overflow: 'hidden',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Box
                  component="span"
                  sx={{
                    width: 12,
                    height: 12,
                    mr: 1.5,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #6366F1, #EC4899)'
                  }}
                />
              </Typography>

              {analysisData ? (
                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden'
                }}>
                  <Box sx={{
                    borderBottom: 1,
                    borderColor: 'divider',
                    mb: 2,
                  }}>
                    <Tabs
                      value={tabValue}
                      onChange={handleTabChange}
                      variant="fullWidth"
                      sx={{
                        '& .MuiTabs-indicator': {
                          height: 3,
                          borderRadius: '3px 3px 0 0',
                        }
                      }}
                    >
                      <Tab label="Live Stream" />
                      <Tab label="Full Analysis" />
                    </Tabs>
                  </Box>

                  <Box sx={{ flex: 1, overflow: 'auto' }}>
                    <TabPanel value={tabValue} index={0}>
                      <Box sx={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        alignItems: 'center',
                      }}>
                        <EmotionCurrent
                          emotion={currentEmotion?.emotion}
                          subEmotion={currentEmotion?.sub_emotion}
                          intensity={currentEmotion?.intensity}
                          relatedEmotions={[]} // Would come from backend in a real app
                        />
                      </Box>
                    </TabPanel>
                    <TabPanel value={tabValue} index={1}>
                      <Box sx={{ height: '100%' }}>
                        <Typography variant="h6" sx={{
                          mb: 2,
                          fontSize: '0.9rem',
                          color: 'text.secondary',
                          fontWeight: 600
                        }}>
                          Emotion Distribution
                        </Typography>
                        <Box sx={{ mb: 4, height: '45%' }}>
                          <EmotionBarChart data={emotionDistribution} />
                        </Box>

                        <Typography variant="h6" sx={{
                          mb: 2,
                          mt: 2,
                          fontSize: '0.9rem',
                          color: 'text.secondary',
                          fontWeight: 600
                        }}>
                          Emotion Timeline
                        </Typography>
                        <Box sx={{ height: '45%' }}>
                          <EmotionTimeline
                            data={intensityTimeline}
                            currentTime={currentTime}
                          />
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
                  textAlign: 'center'
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
            </Paper>
          </Grid>
        </Grid>
      </Container>

      {isLoading && (
        <Box className="loading-overlay">
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: '500px' }}
          >
            {/* Animated emotion orbs */}
            <Box sx={{ position: 'relative', width: 200, height: 200, mb: 4 }}>
              {/* Main pulsing circle */}
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{
                  duration: 3,
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
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
                  boxShadow: '0 0 30px rgba(99, 102, 241, 0.4)'
                }}
              />

              {/* Orbiting emotion circles */}
              {['#10B981', '#EF4444', '#60A5FA', '#F59E0B', '#8B5CF6'].map((color, i) => {
                const angle = (i / 5) * Math.PI * 2;
                const radius = 70;
                return (
                  <motion.div
                    key={color}
                    initial={{
                      x: Math.cos(angle) * radius,
                      y: Math.sin(angle) * radius,
                      opacity: 0
                    }}
                    animate={{
                      x: [
                        Math.cos(angle) * radius,
                        Math.cos(angle + Math.PI * 2) * radius
                      ],
                      y: [
                        Math.sin(angle) * radius,
                        Math.sin(angle + Math.PI * 2) * radius
                      ],
                      opacity: [0, 1, 1, 0],
                      scale: [0.5, 1.2, 0.8]
                    }}
                    transition={{
                      duration: 8,
                      delay: i * 0.5,
                      repeat: Infinity,
                      times: [0, 0.2, 0.8, 1]
                    }}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      marginLeft: -15,
                      marginTop: -15,
                      width: 30,
                      height: 30,
                      borderRadius: '50%',
                      background: color,
                      boxShadow: `0 0 15px ${color}88`
                    }}
                  />
                );
              })}
            </Box>

            <Typography
              className="loading-text"
              variant="h5"
              sx={{
                mb: 3,
                fontWeight: 700,
                textAlign: 'center',
                background: 'linear-gradient(45deg, #6366F1, #EC4899)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              Analyzing emotions...
            </Typography>

            {/* Current processing phase */}
            <Box sx={{ mb: 4, width: '100%' }}>
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
                      fontFamily: '"Manrope", sans-serif',
                      fontWeight: 600,
                      color: 'rgba(0,0,0,0.7)',
                      mb: 1
                    }}
                  >
                    {loadingPhases[loadingPhase]}
                  </Typography>
                </motion.div>
              </AnimatePresence>

              {/* Progress bar */}
              <Box
                sx={{
                  height: 6,
                  bgcolor: 'rgba(255,255,255,0.4)',
                  borderRadius: 3,
                  overflow: 'hidden',
                  width: '100%',
                  position: 'relative'
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
                    background: 'linear-gradient(90deg, #6366F1, #EC4899)',
                    borderRadius: 8,
                    position: 'absolute',
                    left: 0,
                    top: 0
                  }}
                />
              </Box>
            </Box>

            {/* Rotating emotion facts */}
            <Box sx={{ maxWidth: 400, px: 3 }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={factIndex}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.8 }}
                  style={{ position: 'relative' }}
                >
                  <Typography
                    variant="body2"
                    align="center"
                    sx={{
                      fontStyle: 'italic',
                      color: 'text.secondary',
                      pb: 3,
                      minHeight: 60
                    }}
                  >
                    <Box component="span" sx={{
                      fontSize: '1.2rem',
                      color: 'primary.main',
                      display: 'block',
                      mb: 1,
                      fontWeight: 600
                    }}>
                      Did you know?
                    </Box>
                    {emotionFacts[factIndex]}
                  </Typography>
                </motion.div>
              </AnimatePresence>
            </Box>

            <Typography
              variant="caption"
              align="center"
              sx={{
                opacity: 0.6,
                mt: 2,
                fontFamily: '"Manrope", sans-serif',
              }}
            >
              This usually takes about 30 seconds
            </Typography>
          </motion.div>
        </Box>
      )}
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <VideoProvider>
        <AppContent />
      </VideoProvider>
    </ThemeProvider>
  );
}

export default App;
