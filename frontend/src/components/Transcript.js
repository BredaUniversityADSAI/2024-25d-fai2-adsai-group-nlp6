import React, { useEffect, useRef, useState, useLayoutEffect, useMemo } from 'react';
import { Box, Typography, Paper, List, ListItem, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import { getEmotionColor } from '../utils';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

const TranscriptContainer = styled(Paper)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  background: 'rgba(255, 255, 255, 0.7)',
  backdropFilter: 'blur(10px)',
  borderRadius: '18px',
  border: '1px solid rgba(255, 255, 255, 0.8)',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 8px 25px rgba(0, 0, 0, 0.07)',
  },
}));

const TranscriptHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2, 3),
  borderBottom: '1px solid rgba(0, 0, 0, 0.05)',
  display: 'flex',
  alignItems: 'center',
}));

const TranscriptList = styled(List)(({ theme }) => ({
  flexGrow: 1,
  overflow: 'auto',
  padding: theme.spacing(2),
  backgroundColor: 'rgba(249, 250, 251, 0.5)',
  scrollBehavior: 'smooth',
  '&::-webkit-scrollbar': {
    width: '6px',
  },
  '&::-webkit-scrollbar-thumb': {
    backgroundColor: 'rgba(0,0,0,0.1)',
    borderRadius: '6px',
  },
  '&::-webkit-scrollbar-thumb:hover': {
    backgroundColor: 'rgba(0,0,0,0.2)',
  },
}));

const TranscriptItem = styled(motion.div)(({ theme, emotion, selected }) => {
  const baseColor = getEmotionColor(emotion);
  const bgColor = selected
    ? `rgba(${parseInt(baseColor.slice(1, 3), 16)}, ${parseInt(baseColor.slice(3, 5), 16)}, ${parseInt(baseColor.slice(5, 7), 16)}, 0.12)`
    : 'transparent';

  return {
    padding: theme.spacing(1.5, 2),
    borderRadius: '14px',
    marginBottom: theme.spacing(1.5),
    position: 'relative',
    backgroundColor: bgColor,
    border: selected ? `1px solid ${baseColor}30` : '1px solid rgba(229, 231, 235, 0.7)',
    transition: 'all 0.3s ease',
    overflow: 'hidden',
    '&:hover': {
      backgroundColor: selected
        ? `rgba(${parseInt(baseColor.slice(1, 3), 16)}, ${parseInt(baseColor.slice(3, 5), 16)}, ${parseInt(baseColor.slice(5, 7), 16)}, 0.17)`
        : 'rgba(0, 0, 0, 0.02)',
      transform: 'translateY(-1px)',
      boxShadow: '0 3px 10px rgba(0, 0, 0, 0.04)',
    },
    '&::before': selected ? {
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      width: '3px',
      backgroundColor: baseColor,
      borderRadius: '3px',
    } : {},
  };
});

const TimeChip = styled(Chip)(({ theme }) => ({
  fontSize: '0.75rem',
  height: '24px',
  backgroundColor: 'rgba(0, 0, 0, 0.05)',
  color: 'rgba(0, 0, 0, 0.6)',
  fontWeight: 500,
  marginRight: theme.spacing(1),
  '& .MuiChip-label': {
    padding: '0 8px',
  }
}));

const EmotionChip = styled(Chip)(({ theme, emotion }) => {
  const baseColor = getEmotionColor(emotion);
  return {
    fontSize: '0.75rem',
    height: '24px',
    backgroundColor: `${baseColor}20`,
    color: baseColor,
    fontWeight: 600,
    border: `1px solid ${baseColor}40`,
    '& .MuiChip-label': {
      padding: '0 8px',
      textTransform: 'capitalize',
    }
  };
});

const formatTime = (seconds) => {
  // Check if seconds is undefined or not a valid number
  if (seconds === undefined || isNaN(seconds)) {
    return "0:00";
  }

  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${min}:${sec.toString().padStart(2, '0')}`;
};

const Transcript = ({ data, currentTime, onSentenceClick }) => {
  const listRef = useRef();
  const activeItemRef = useRef(null);
  const itemsRef = useRef({});
  const prevIndexRef = useRef(-1);
  const prevTimeRef = useRef(-1);
  const [pulsingItem, setPulsingItem] = useState(null);

  // Log transcript data for debugging
  useEffect(() => {
    if (data) {
      console.log(`Transcript received ${data.length} sentences`);
      if (data.length === 1) {
        console.warn("Transcript only has one sentence:", data[0]);
      }
    } else {
      console.warn("Transcript data is null or undefined");
    }
  }, [data]);

  // Ensure data is properly structured before use
  const validatedData = useMemo(() => {
    if (!data || !Array.isArray(data)) {
      console.warn("Invalid transcript data format");
      return [];
    }

    // Filter out invalid entries
    return data.filter(item =>
      item && typeof item === 'object' &&
      typeof item.sentence === 'string' &&
      item.sentence.trim() !== '' &&
      typeof item.start_time !== 'undefined'
    );
  }, [data]);

  // Calculate which sentence is currently being shown based on timestamp
  const currentSentenceIndex = validatedData.length > 0
    ? validatedData.findIndex((item, index) => {
        const nextItemTime = index < validatedData.length - 1 ? validatedData[index + 1].start_time : Infinity;
        return currentTime >= item.start_time && currentTime < nextItemTime;
      })
    : -1;

  // Store a reference to each transcript item using a ref map
  const setItemRef = (index, element) => {
    itemsRef.current[index] = element;
    if (index === currentSentenceIndex) {
      activeItemRef.current = element;
    }
  };

  // Handle scrolling - using useLayoutEffect to ensure it happens before browser paint
  useLayoutEffect(() => {
    // Only scroll when index changes or time changes significantly
    const shouldScroll =
      currentSentenceIndex >= 0 &&
      (currentSentenceIndex !== prevIndexRef.current ||
       Math.abs(currentTime - prevTimeRef.current) > 1);

    if (shouldScroll) {
      prevIndexRef.current = currentSentenceIndex;
      prevTimeRef.current = currentTime;
      setPulsingItem(currentSentenceIndex);

      // Get the active element from our refs map
      const activeElement = itemsRef.current[currentSentenceIndex];

      if (activeElement && listRef.current) {
        // Ensure scroll happens in next frame
        requestAnimationFrame(() => {
          activeElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        });
      }

      const pulseTimer = setTimeout(() => {
        setPulsingItem(null);
      }, 1500);

      return () => {
        clearTimeout(pulseTimer);
      };
    }
  }, [currentSentenceIndex, currentTime, data]);

  if (!validatedData || validatedData.length === 0) {
    return (
      <TranscriptContainer elevation={0}>
        <TranscriptHeader>
          <Typography variant="h6" fontWeight={600} sx={{
            background: 'linear-gradient(90deg, #6366F1, #EC4899)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Emotional Transcript
          </Typography>
        </TranscriptHeader>
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          p: 3
        }}>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.2 }}
          >
            <Box sx={{
              mb: 3,
              width: 100,
              height: 100,
              borderRadius: '50%',
              background: 'rgba(245, 247, 250, 0.7)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <AccessTimeIcon
                sx={{
                  fontSize: '3rem',
                  color: 'rgba(156, 163, 175, 0.6)'
                }}
              />
              <Box sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '30%',
                background: 'linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.03) 100%)'
              }} />
              <Box
                sx={{
                  position: 'absolute',
                  width: 150,
                  height: 150,
                  borderRadius: '50%',
                  border: '2px dashed rgba(99, 102, 241, 0.15)',
                }}
              />
            </Box>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.4 }}
          >
            <Typography
              variant="h6"
              align="center"
              sx={{
                mb: 1,
                fontWeight: 500,
                background: 'linear-gradient(90deg, #6366F1, #818CF8)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              No emotion data available
            </Typography>

            <Typography
              variant="body2"
              align="center"
              color="text.secondary"
              sx={{ maxWidth: 300 }}
            >
              This video doesn't have any emotional transcript data to display. Try analyzing another video or check your connection.
            </Typography>
          </motion.div>
        </Box>
      </TranscriptContainer>
    );
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <TranscriptContainer elevation={0}>
      <TranscriptHeader>
        <Typography variant="h6" fontWeight={600} sx={{
          background: 'linear-gradient(90deg, #6366F1, #EC4899)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Emotional Transcript ({validatedData.length} sentences)
        </Typography>
      </TranscriptHeader>
      <TranscriptList ref={listRef}>
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
        >
          {validatedData.map((sentenceItem, index) => (
            <TranscriptItem
              key={index}
              variants={item}
              ref={(el) => setItemRef(index, el)}
              onClick={() => onSentenceClick(sentenceItem.start_time)}
              emotion={sentenceItem.emotion}
              selected={index === currentSentenceIndex}
              style={{
                cursor: 'pointer',
              }}
              animate={pulsingItem === index ? {
                scale: [1, 1.02, 1],
                boxShadow: [
                  '0 0 0 rgba(0,0,0,0)',
                  '0 0 10px rgba(0,0,0,0.1)',
                  '0 0 0 rgba(0,0,0,0)'
                ],
                transition: { duration: 1, repeat: 1 }
              } : {}}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                <TimeChip
                  label={formatTime(sentenceItem.start_time)}
                  size="small"
                />
                <EmotionChip
                  label={sentenceItem.emotion}
                  size="small"
                  emotion={sentenceItem.emotion}
                />
              </Box>
              <Typography variant="body2" sx={{
                opacity: index === currentSentenceIndex ? 1 : 0.85,
                fontWeight: index === currentSentenceIndex ? 500 : 400,
              }}>
                {sentenceItem.sentence}
              </Typography>
            </TranscriptItem>
          ))}
        </motion.div>
      </TranscriptList>
    </TranscriptContainer>
  );
};

export default Transcript;
