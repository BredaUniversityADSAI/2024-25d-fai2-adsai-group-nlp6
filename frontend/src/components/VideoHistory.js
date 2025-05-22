import React from 'react';import {   List,   ListItem,   ListItemButton,   ListItemText,   ListItemAvatar,   Avatar,   Typography,   Box,   Divider,  Tooltip,  Chip} from '@mui/material';import { styled } from '@mui/material/styles';import { motion } from 'framer-motion';import YouTubeIcon from '@mui/icons-material/YouTube';import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded';const StyledListItem = styled(ListItem)(({ theme }) => ({  padding: 0,  marginBottom: theme.spacing(1.5),}));const StyledListItemButton = styled(ListItemButton)(({ theme }) => ({  borderRadius: theme.shape.borderRadius,  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',  padding: theme.spacing(1.5, 1.5),  overflow: 'hidden',  position: 'relative',  border: '1px solid rgba(0,0,0,0.04)',  '&:hover': {    backgroundColor: 'rgba(0,0,0,0.03)',    transform: 'translateY(-2px)',    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',    '& .play-icon': {      opacity: 1,      transform: 'scale(1)',    }  },  '&:after': {    content: '""',    position: 'absolute',    top: 0,    left: 0,    right: 0,    height: '2px',    background: 'linear-gradient(90deg, #6366F1, #EC4899)',    opacity: 0,    transition: 'opacity 0.3s ease',  },  '&:hover:after': {    opacity: 1,  }}));const VideoThumbnail = styled(Box)(({ theme }) => ({  position: 'relative',  width: 60,  height: 60,  borderRadius: theme.shape.borderRadius,  overflow: 'hidden',  background: '#000',  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',  '& .MuiAvatar-root': {    width: '100%',    height: '100%',  }}));const PlayIcon = styled(Box)(({ theme }) => ({  position: 'absolute',  top: '50%',  left: '50%',  transform: 'translate(-50%, -50%) scale(0.8)',  color: '#fff',  backgroundColor: 'rgba(0,0,0,0.5)',  borderRadius: '50%',  padding: '4px',  opacity: 0,  transition: 'all 0.3s ease',}));const EmotionStripe = styled(Box)(({ theme }) => ({  display: 'flex',  height: '4px',  width: '100%',  borderRadius: '4px',  overflow: 'hidden',  marginTop: '6px',  boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',  '& > div': {    height: '100%',    transition: 'all 0.3s ease',  },}));

const VideoHistory = ({ videos = [], onVideoSelect }) => {
  // Mock data for demonstration
  const mockVideos = [
    {
      id: 'abc123',
      title: 'Introduction to Machine Learning',
      thumbnail: null,
      date: '2023-05-10',
      emotions: {
        happiness: 40,
        sadness: 10,
        anger: 5,
        neutral: 45,
      }
    },
    {
      id: 'def456',
      title: 'Advanced Data Science Techniques',
      thumbnail: null,
      date: '2023-05-08',
      emotions: {
        happiness: 20,
        sadness: 30,
        fear: 15,
        neutral: 35,
      }
    }
  ];

  const displayVideos = videos.length > 0 ? videos : mockVideos;

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

    // Calculate dominant emotion  const getDominantEmotion = (emotions) => {    return Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];  };  // Get emotion color  const getEmotionColor = (emotion) => {    const emotionColors = {      happiness: '#10B981',      sadness: '#60A5FA',      anger: '#EF4444',      fear: '#8B5CF6',      disgust: '#65A30D',      surprise: '#F59E0B',      neutral: '#9CA3AF',    };    return emotionColors[emotion] || '#9CA3AF';  };  return (    <List disablePadding>      {displayVideos.length > 0 ? (        displayVideos.map((video, index) => {          const dominantEmotion = getDominantEmotion(video.emotions);          const dominantColor = getEmotionColor(dominantEmotion);                    return (            <motion.div              key={video.id || index}              initial={{ opacity: 0, y: 20 }}              animate={{ opacity: 1, y: 0 }}              transition={{ duration: 0.3, delay: index * 0.05 }}            >              <StyledListItem>                <StyledListItemButton                   onClick={() => onVideoSelect && onVideoSelect(video)}                   sx={{                     '&:after': {                      background: `linear-gradient(90deg, ${dominantColor}, ${dominantColor}88)`,                    }                  }}                >                  <VideoThumbnail>                    <Avatar variant="rounded" src={video.thumbnail}>                      <YouTubeIcon fontSize="large" />                    </Avatar>                    <PlayIcon className="play-icon">                      <PlayArrowRoundedIcon />                    </PlayIcon>                  </VideoThumbnail>                  <ListItemText                     sx={{ ml: 2 }}                    primary={                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>                        <Typography variant="subtitle2" noWrap sx={{ fontWeight: 600, flex: 1 }}>                          {video.title}                        </Typography>                        <Tooltip title={`Dominant emotion: ${dominantEmotion}`}>                          <Chip                             size="small"                             label={dominantEmotion}                            sx={{                               height: 20,                               backgroundColor: `${dominantColor}22`,                              color: dominantColor,                              fontWeight: 600,                              fontSize: '0.65rem',                              borderRadius: '10px',                              border: `1px solid ${dominantColor}44`,                              ml: 1,                            }}                           />                        </Tooltip>                      </Box>                    }                    secondary={                      <>                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>                          {video.date}                        </Typography>                        <Tooltip title={                          Object.entries(video.emotions)                            .map(([emotion, percentage]) => `${emotion}: ${percentage}%`)                            .join(', ')                        }>                          {renderEmotionStripe(video.emotions)}                        </Tooltip>                      </>                    }                  />                </StyledListItemButton>              </StyledListItem>              {index < displayVideos.length - 1 &&                 <Box sx={{ ml: 2, mr: 2 }}>                  <Divider light />                </Box>              }            </motion.div>          );        })      ) : (        <Box           sx={{             p: 4,             textAlign: 'center',             borderRadius: 2,            backgroundColor: 'rgba(0,0,0,0.02)',            border: '1px dashed rgba(0,0,0,0.1)',            my: 2          }}        >          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>            No video history found          </Typography>          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', opacity: 0.7 }}>            Analyzed videos will appear here          </Typography>        </Box>      )}    </List>  );
};

export default VideoHistory;
