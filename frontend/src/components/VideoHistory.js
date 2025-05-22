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
import { getEmotionColor } from '../utils';
import HistoryIcon from '@mui/icons-material/History';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';

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
  zIndex: 10,
  '&:hover': {
    background: 'rgba(239, 68, 68, 0.1)',
    color: '#EF4444',
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

// Enhanced styled components
const HistoryItemContainer = styled(motion.div)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  borderRadius: '12px',
  overflow: 'hidden',
  cursor: 'pointer',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  border: '1px solid rgba(0, 0, 0, 0.06)',
  backgroundColor: 'rgba(255, 255, 255, 0.7)',
  position: 'relative',
  '&:hover': {
    transform: 'translateY(-3px)',
    boxShadow: '0 10px 20px rgba(0, 0, 0, 0.08), 0 5px 10px rgba(0, 0, 0, 0.04)',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    '& .delete-icon': {
      opacity: 1,
    },
  },
}));

const EmotionAvatar = styled(Avatar)(({ theme, emotion }) => {
  const emotionColor = getEmotionColor(emotion || 'neutral');

  return {
    backgroundColor: `${emotionColor}25`,
    border: `1px solid ${emotionColor}50`,
    color: emotionColor,
    '& svg': {
      fontSize: '1.2rem',
    },
  };
});

const VideoTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 600,
  fontSize: '0.95rem',
  lineHeight: 1.3,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: '-webkit-box',
  WebkitLineClamp: 2,
  WebkitBoxOrient: 'vertical',
}));

const DateText = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  color: theme.palette.text.secondary,
  fontWeight: 500,
}));

const EmotionBar = styled(Box)(({ theme }) => ({
  height: '5px',
  borderRadius: '3px',
  marginTop: theme.spacing(1),
  overflow: 'hidden',
  background: 'rgba(229, 231, 235, 0.5)',
}));

const EmotionBarItem = styled(Box)(({ width, color }) => ({
  height: '100%',
  backgroundColor: color,
  display: 'inline-block',
}));

const EmotionalIcon = ({ emotion }) => {
  switch(emotion) {
    case 'happiness':
      return <SentimentSatisfiedAltIcon />;
    default:
      return <HistoryIcon />;
  }
};

// Helper function to format date
const formatDate = (dateString) => {
  const date = new Date(dateString);
  const currentDate = new Date();
  const isToday = date.toDateString() === currentDate.toDateString();

  // Format: Today or MM-DD-YY
  if (isToday) {
    return 'Today';
  } else {
    return date.toLocaleDateString('en-US', {
      month: '2-digit',
      day: '2-digit',
      year: '2-digit'
    });
  }
};

// Function to create color bar from emotional data
const createEmotionBar = (videoData) => {
  // Example emotions and their proportion in the video
  // In a real implementation, this would come from the videoData
  const emotions = [
    { emotion: 'happiness', proportion: 0.3 },
    { emotion: 'sadness', proportion: 0.2 },
    { emotion: 'anger', proportion: 0.1 },
    { emotion: 'surprise', proportion: 0.15 },
    { emotion: 'neutral', proportion: 0.25 },
  ];

  return emotions.map((item, index) => (
    <EmotionBarItem
      key={index}
      width={`${item.proportion * 100}%`}
      color={getEmotionColor(item.emotion)}
      sx={{ width: `${item.proportion * 100}%` }}
    />
  ));
};

// Get the dominant emotion from a video
const getDominantEmotion = (video) => {
  return video.dominant_emotion || 'neutral';
};

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

  if (!videos.length) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          py: 4,
          textAlign: 'center',
        }}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{
            width: 70,
            height: 70,
            borderRadius: '50%',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
          }}>
            <HistoryIcon sx={{ fontSize: '2rem', color: '#6366F1', opacity: 0.6 }} />
          </Box>

          <Typography variant="body1" color="textSecondary" sx={{ fontWeight: 500 }}>
            No video history yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, maxWidth: 220 }}>
            Analyzed videos will appear here
          </Typography>
        </motion.div>
      </Box>
    );
  }

  const container = {
    hidden: { opacity: 1 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
    >
      {videos.map((video) => {
        const dominantEmotion = getDominantEmotion(video);

        return (
          <motion.div key={video.id} variants={item}>
            <HistoryItemContainer onClick={() => onVideoSelect(video)}>
              <Box sx={{ p: 2, position: 'relative' }}>
                <DeleteButton
                  className="delete-icon"
                  onClick={(e) => handleDelete(e, video.id)}
                  aria-label="delete"
                  size="small"
                >
                  <CloseIcon fontSize="inherit" />
                </DeleteButton>

                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <EmotionAvatar emotion={dominantEmotion} variant="rounded">
                    <EmotionalIcon emotion={dominantEmotion} />
                  </EmotionAvatar>
                  <Box sx={{ ml: 1.5 }}>
                    <VideoTitle>{video.title}</VideoTitle>
                    <DateText>{formatDate(video.date)}</DateText>
                  </Box>
                </Box>

                <EmotionBar>
                  {createEmotionBar(video)}
                </EmotionBar>
              </Box>
            </HistoryItemContainer>
          </motion.div>
        );
      })}
    </motion.div>
  );
};

export default VideoHistory;
