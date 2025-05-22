import React from 'react';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Typography,
  Box,
  Divider,
  Tooltip,
  Chip,
  IconButton
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import YouTubeIcon from '@mui/icons-material/YouTube';
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded';
import CloseIcon from '@mui/icons-material/Close';
import { useVideo } from '../VideoContext';

const StyledListItem = styled(ListItem)(({ theme }) => ({
  padding: 0,
  marginBottom: theme.spacing(1.5),
}));

const StyledListItemButton = styled(ListItemButton)(({ theme }) => ({
  borderRadius: theme.shape.borderRadius,
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  padding: theme.spacing(1.5, 1.5),
  overflow: 'hidden',
  position: 'relative',
  border: '1px solid rgba(0,0,0,0.04)',
  '&:hover': {
    backgroundColor: 'rgba(0,0,0,0.03)',
    transform: 'translateY(-2px)',
    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
    '& .play-icon': {
      opacity: 1,
      transform: 'scale(1)',
    },
    '& .delete-icon': {
      opacity: 1,
    }
  },
  '&:after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, #6366F1, #EC4899)',
    opacity: 0,
    transition: 'opacity 0.3s ease',
  },
  '&:hover:after': {
    opacity: 1,
  }
}));

const VideoThumbnail = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: 60,
  height: 60,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  background: 'rgba(0,0,0,0.05)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const PlayIcon = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%) scale(0.8)',
  color: '#fff',
  backgroundColor: 'rgba(0,0,0,0.5)',
  borderRadius: '50%',
  padding: '4px',
  opacity: 0,
  transition: 'all 0.3s ease',
  zIndex: 2,
}));

const DeleteButton = styled(IconButton)(({ theme }) => ({
  position: 'absolute',
  top: 8,
  right: 8,
  opacity: 0,
  transition: 'all 0.2s ease',
  padding: '4px',
  background: 'rgba(0, 0, 0, 0.05)',
  '&:hover': {
    background: 'rgba(0, 0, 0, 0.1)',
  },
  '& .MuiSvgIcon-root': {
    fontSize: '0.95rem',
  },
}));

const EmotionStripe = styled(Box)(({ theme }) => ({
  display: 'flex',
  height: '4px',
  width: '100%',
  borderRadius: '4px',
  overflow: 'hidden',
  marginTop: '6px',
  boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
  '& > div': {
    height: '100%',
    transition: 'all 0.3s ease',
  },
}));

// SVG waveform component for the thumbnail
const WaveformSVG = ({ color }) => (
  <svg width="50" height="30" viewBox="0 0 50 30" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor={color} stopOpacity="0.7"/>
        <stop offset="100%" stopColor={color} stopOpacity="0.3"/>
      </linearGradient>
    </defs>
    <path fill="url(#waveGradient)" d="M0,15 Q2.5,12 5,15 T10,15 T15,15 T20,15 T25,15 T30,15 T35,15 T40,15 T45,15 T50,15 V30 H0 Z" />
    <path fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round"
          d="M0,15 Q2.5,5 5,15 T10,15 T15,5 T20,25 T25,10 T30,20 T35,15 T40,5 T45,25 T50,15" />
  </svg>
);

const VideoHistory = ({ videos = [], onVideoSelect }) => {
  const { removeFromHistory } = useVideo();

  // Maintain local storage for persistence between sessions
  React.useEffect(() => {
    if (videos.length > 0) {
      try {
        localStorage.setItem('videoAnalysisHistory', JSON.stringify(videos));
      } catch (error) {
        console.error("Failed to save video history to localStorage:", error);
      }
    }
  }, [videos]);

  // Use videos prop directly instead of filtering it here (filtering happens in parent)
  const displayVideos = videos.length > 0 ? videos : [];

  const renderEmotionStripe = (emotions) => {
    const emotionColors = {
      happiness: '#FFD700',
      sadness: '#4169E1',
      anger: '#FF4500',
      fear: '#800080',
      disgust: '#008000',
      surprise: '#FFA500',
      neutral: '#A9A9A9',
    };

    const stripes = Object.entries(emotions).map(([emotion, percentage]) => (
      <Box
        key={emotion}
        sx={{
          width: `${percentage}%`,
          backgroundColor: emotionColors[emotion] || '#A9A9A9',
        }}
      />
    ));

    return <EmotionStripe>{stripes}</EmotionStripe>;
  };

  // Calculate dominant emotion
  const getDominantEmotion = (emotions) => {
    return Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];
  };

  // Get emotion color
  const getEmotionColor = (emotion) => {
    const emotionColors = {
      happiness: '#10B981',
      sadness: '#60A5FA',
      anger: '#EF4444',
      fear: '#8B5CF6',
      disgust: '#65A30D',
      surprise: '#F59E0B',
      neutral: '#9CA3AF',
    };
    return emotionColors[emotion] || '#9CA3AF';
  };

  // Handle delete with stopPropagation to prevent selection
  const handleDelete = (e, videoId) => {
    e.stopPropagation();
    removeFromHistory(videoId);
  };

  return (
    <List disablePadding>
      {displayVideos.length > 0 ? (
        displayVideos.map((video, index) => {
          const dominantEmotion = getDominantEmotion(video.emotions);
          const dominantColor = getEmotionColor(dominantEmotion);
          return (
            <motion.div
              key={video.id || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <StyledListItem>
                <StyledListItemButton
                  onClick={() => onVideoSelect && onVideoSelect(video)}
                  sx={{
                    '&:after': {
                      background: `linear-gradient(90deg, ${dominantColor}, ${dominantColor}88)`,
                    }
                  }}
                >
                  <VideoThumbnail>
                    <WaveformSVG color={dominantColor} />
                    <PlayIcon className="play-icon">
                      <PlayArrowRoundedIcon />
                    </PlayIcon>
                  </VideoThumbnail>
                  <ListItemText
                    sx={{ ml: 2 }}
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="subtitle2" noWrap sx={{ fontWeight: 600, flex: 1 }}>
                          {video.title}
                        </Typography>
                        <Tooltip title={`Dominant emotion: ${dominantEmotion}`}>
                          <Chip
                            size="small"
                            label={dominantEmotion}
                            sx={{
                              height: 20,
                              backgroundColor: `${dominantColor}22`,
                              color: dominantColor,
                              fontWeight: 600,
                              fontSize: '0.65rem',
                              borderRadius: '10px',
                              border: `1px solid ${dominantColor}44`,
                              ml: 1,
                            }}
                          />
                        </Tooltip>
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                          {video.date}
                        </Typography>
                        <Tooltip title={
                          Object.entries(video.emotions)
                            .map(([emotion, percentage]) => `${emotion}: ${percentage}%`)
                            .join(', ')
                        }>
                          {renderEmotionStripe(video.emotions)}
                        </Tooltip>
                      </>
                    }
                  />
                  <DeleteButton
                    className="delete-icon"
                    onClick={(e) => handleDelete(e, video.id)}
                    aria-label="delete"
                    size="small"
                  >
                    <CloseIcon fontSize="inherit" />
                  </DeleteButton>
                </StyledListItemButton>
              </StyledListItem>
              {index < displayVideos.length - 1 &&
                <Box sx={{ ml: 2, mr: 2 }}>
                  <Divider light />
                </Box>
              }
            </motion.div>
          );
        })
      ) : (
        <Box
          sx={{
            p: 4,
            textAlign: 'center',
            borderRadius: 2,
            backgroundColor: 'rgba(0,0,0,0.02)',
            border: '1px dashed rgba(0,0,0,0.1)',
            my: 2
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
            No video history found
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', opacity: 0.7 }}>
            Analyzed videos will appear here
          </Typography>
        </Box>
      )}
    </List>
  );
};

export default VideoHistory;
