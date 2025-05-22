import React, { useEffect, useState } from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { styled } from '@mui/material/styles';
import { getEmotionColor, getIntensityValue } from '../utils';
import SentimentVeryDissatisfiedIcon from '@mui/icons-material/SentimentVeryDissatisfied';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import PsychologyAltIcon from '@mui/icons-material/PsychologyAlt';

// Updated for direct integration with parent container
const EmotionPulse = styled(Box)(({ theme, compact }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
  padding: compact ? theme.spacing(1, 0) : theme.spacing(4, 0),
}));

// Modern header with better spacing and alignment
const EmotionHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(0, 2, 2),
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}));

// Updated Emotion Orb with more vibrant and dynamic effects
const EmotionOrb = styled(motion.div)(({ color, size = 100, intensity = 0.5, compact }) => ({
  width: compact ? size * 0.7 : size,
  height: compact ? size * 0.7 : size,
  borderRadius: '50%',
  background: `radial-gradient(circle, ${color}DD 0%, ${color}99 50%, ${color}44 70%, ${color}11 100%)`,
  boxShadow: `
    0 0 ${40 * intensity}px ${30 * intensity}px ${color}33,
    0 0 ${25 * intensity}px ${15 * intensity}px ${color}55,
    inset 0 0 ${55 * intensity}px ${color}66
  `,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  position: 'relative',
  zIndex: 2,
  isolation: 'isolate',
  '&:before': {
    content: '""',
    position: 'absolute',
    top: -size * (compact ? 0.1 : 0.15),
    left: -size * (compact ? 0.1 : 0.15),
    right: -size * (compact ? 0.1 : 0.15),
    bottom: -size * (compact ? 0.1 : 0.15),
    borderRadius: '50%',
    border: `2px solid ${color}44`,
    opacity: 0.6,
    animation: 'pulse 4s infinite ease-in-out',
    zIndex: -1,
  },
  '&:after': {
    content: '""',
    position: 'absolute',
    top: -size * (compact ? 0.03 : 0.05),
    left: -size * (compact ? 0.03 : 0.05),
    right: -size * (compact ? 0.03 : 0.05),
    bottom: -size * (compact ? 0.03 : 0.05),
    borderRadius: '50%',
    border: `3px solid ${color}77`,
    opacity: 0.8,
    animation: 'pulse 5s 1s infinite ease-in-out',
    zIndex: -1,
  },
}));

const OrbParticle = styled(motion.div)(({ color }) => ({
  position: 'absolute',
  width: '4px',
  height: '4px',
  borderRadius: '50%',
  backgroundColor: color,
  filter: 'blur(1px)',
  opacity: 0.6,
}));

const InnerGlow = styled(motion.div)(({ color, size = 60 }) => ({
  position: 'absolute',
  width: size,
  height: size,
  borderRadius: '50%',
  background: `radial-gradient(circle, ${color}99 0%, ${color}00 70%)`,
  opacity: 0.7,
}));

// Enhanced sub-emotion orbs with better visual effects
const SubEmotionOrb = styled(motion.div)(({ color, size = 36, primary }) => {
  return {
    width: size,
    height: size,
    borderRadius: '50%',
    background: `radial-gradient(circle, ${color}DD 0%, ${color}99 60%, ${color}33 100%)`,
    boxShadow: primary
      ? `0 0 25px 10px ${color}33, inset 0 0 12px 2px rgba(255,255,255,0.35)`
      : `0 0 18px 6px ${color}22, inset 0 0 10px 1px rgba(255,255,255,0.25)`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: primary ? 3 : 1,
    backdropFilter: 'blur(4px)',
    border: `1px solid ${color}44`,
    position: 'absolute',
    '&:before': primary ? {
      content: '""',
      position: 'absolute',
      top: -3,
      left: -3,
      right: -3,
      bottom: -3,
      borderRadius: '50%',
      border: `1px solid ${color}44`,
      opacity: 0.7,
    } : {},
  };
});

// Enhanced emotion text with better typography
const EmotionText = styled(Typography)(({ theme, color }) => ({
  fontWeight: 700,
  fontSize: '1.1rem',
  textTransform: 'capitalize',
  color: color,
  marginTop: theme.spacing(2),
  textShadow: '0 1px 3px rgba(0,0,0,0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: 'Manrope, sans-serif',
  letterSpacing: '-0.01em',
}));

// Redesigned sub-emotion text with pill style
const SubEmotionText = styled(Typography)(({ theme, color }) => ({
  fontSize: '0.85rem',
  fontWeight: 600,
  textTransform: 'capitalize',
  color: color,
  position: 'absolute',
  padding: '6px 12px',
  borderRadius: '12px',
  backgroundColor: `${color}11`,
  border: `1px solid ${color}22`,
  boxShadow: `0 2px 8px ${color}15`,
  backdropFilter: 'blur(4px)',
}));

// Improved emotion label with more prominent styling
const EmotionLabel = styled(Typography)(({ color, compact }) => ({
  fontWeight: 800,
  fontSize: compact ? '1.9rem' : '2.4rem',
  color: color,
  textTransform: 'capitalize',
  background: `linear-gradient(45deg, ${color}, ${color}99)`,
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  textShadow: '0px 2px 2px rgba(0,0,0,0.05)',
  letterSpacing: '-0.02em',
  marginBottom: '0.3rem',
  fontFamily: '"Manrope", sans-serif',
}));

// Enhanced intensity indicator with animated glow effect
const IntensityIndicator = styled(motion.div)(({ color, intensity, compact }) => ({
  width: compact ? '120px' : '160px',
  height: compact ? '6px' : '8px',
  backgroundColor: 'rgba(0,0,0,0.05)',
  borderRadius: '4px',
  overflow: 'hidden',
  marginTop: compact ? '6px' : '10px',
  position: 'relative',
  boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.08)',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${intensity * 100}%`,
    background: `linear-gradient(90deg, ${color}DD, ${color})`,
    borderRadius: '4px',
    boxShadow: `0 0 20px ${color}99, 0 0 10px ${color}77`,
    animation: 'pulse-opacity 2s infinite ease-in-out',
  }
}));

// New component for intensity percentage display
const IntensityPercentage = styled(Typography)(({ color, compact }) => ({
  fontSize: compact ? '0.85rem' : '1rem',
  fontWeight: 700,
  color: color,
  marginLeft: '8px',
}));

// New styled component for a better metadata display
const EmotionMetaContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  padding: theme.spacing(2, 2.5),
  borderRadius: '16px',
  backgroundColor: '#FFFFFF',
  margin: theme.spacing(0, 2.5, 2.5),
  marginTop: 'auto',
  border: '1px solid rgba(229, 231, 235, 0.8)',
  boxShadow: '0 4px 15px rgba(0, 0, 0, 0.03), 0 1px 8px rgba(0, 0, 0, 0.02)',
  position: 'relative',
  zIndex: 5,
}));

// New loading animation
const LoadingPulse = styled(Box)(({ theme }) => ({
  position: 'absolute',
  width: 110,
  height: 110,
  borderRadius: '50%',
  background: 'linear-gradient(135deg, rgba(156, 163, 175, 0.1), rgba(156, 163, 175, 0.2))',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  border: '1px solid rgba(156, 163, 175, 0.15)',
  overflow: 'hidden',
}));

// New styled component for sub-emotion display under the main emotion
const SubEmotionDisplay = styled(Typography)(({ color, compact }) => ({
  fontWeight: 500,
  fontSize: compact ? '0.8rem' : '0.95rem',
  color: color,
  textTransform: 'capitalize',
  opacity: 0.85,
  marginTop: '-0.2rem',
  marginBottom: compact ? '0.5rem' : '0.7rem',
  fontFamily: '"Inter", sans-serif',
}));

const EmotionCurrent = ({ emotion, subEmotion, intensity = 0.5, relatedEmotions = [], compact = false }) => {
  const [infoTooltip, setInfoTooltip] = useState(false);

  useEffect(() => {
    // Reset tooltip state when emotion changes
    setInfoTooltip(false);
  }, [emotion]);

  if (!emotion) {
    return (
      <AnimatePresence mode="wait">
        <motion.div
          key="no-emotion"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.6 }}
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <Box sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%'
          }}>
            <LoadingPulse>
              <AccessTimeIcon sx={{
                fontSize: '3.2rem',
                opacity: 0.6,
                color: '#9CA3AF'
              }} />

              <Box sx={{
                position: 'absolute',
                width: '140%',
                height: '140%',
                pointerEvents: 'none'
              }}>
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={i}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '100%',
                      height: '100%',
                      borderRadius: '50%',
                      border: '1px dashed rgba(156, 163, 175, 0.3)',
                    }}
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.7, 0.3, 0.7],
                      rotate: [0, 180, 0],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 4 + i,
                      ease: "easeInOut",
                      delay: i * 0.7
                    }}
                  />
                ))}
              </Box>
            </LoadingPulse>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              style={{ marginTop: '1.5rem' }}
            >
              <Typography
                variant="h6"
                align="center"
                sx={{
                  mb: 1,
                  fontWeight: 600,
                  background: 'linear-gradient(90deg, #9CA3AF, #6B7280)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                No emotion detected
              </Typography>

              <Typography
                variant="body2"
                align="center"
                color="text.secondary"
                sx={{ maxWidth: 280, px: 2 }}
              >
                We couldn't detect any emotional data at this moment of the video
              </Typography>
            </motion.div>
          </Box>
        </motion.div>
      </AnimatePresence>
    );
  }

  const mainColor = getEmotionColor(emotion);
  const intensityValue = getIntensityValue(intensity);
  const size = compact ?
    (90 + (intensityValue * 30)) : // Smaller base size and less growth with intensity when compact
    (120 + (intensityValue * 40)); // Original size

  // Generate random particles for effect
  const particles = Array.from({ length: 12 }, (_, i) => ({
    id: i,
    x: Math.random() * 320 - 160,
    y: Math.random() * 320 - 160,
    scale: 0.8 + Math.random() * 0.5,
    duration: 3 + Math.random() * 4,
    delay: i * 0.2
  }));

  // Format intensity for display
  const intensityLabel = () => {
    if (intensityValue < 0.3) return "Low";
    if (intensityValue < 0.7) return "Moderate";
    return "High";
  };

  return (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative',
      justifyContent: 'center',
    }}>
      <EmotionPulse compact={compact}>
        <AnimatePresence mode="wait">
          <motion.div
            key={emotion + subEmotion}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              width: '100%',
              height: '100%',
              padding: '10px',
              position: 'relative',
            }}
          >
            <Box
              sx={{
                position: 'relative',
                width: compact ? (size + 80) : (size + 120),
                height: compact ? (size + 80) : (size + 120),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginY: compact ? 0 : 0.5,
              }}
            >
              {/* Background radial gradient */}
              <Box
                sx={{
                  position: 'absolute',
                  width: compact ? size * 2 : size * 2.5,
                  height: compact ? size * 2 : size * 2.5,
                  borderRadius: '50%',
                  background: `radial-gradient(circle, ${mainColor}15 0%, ${mainColor}08 50%, transparent 80%)`,
                  opacity: 0.8,
                }}
              />

              {/* Floating particles */}
              {particles.map(particle => (
                <OrbParticle
                  key={particle.id}
                  color={mainColor}
                  initial={{
                    x: 0,
                    y: 0,
                    scale: 0,
                    opacity: 0
                  }}
                  animate={{
                    x: particle.x,
                    y: particle.y,
                    scale: particle.scale,
                    opacity: [0, 0.8, 0]
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: particle.duration,
                    delay: particle.delay,
                    ease: "easeInOut"
                  }}
                />
              ))}

              {/* Main emotion orb */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{
                  scale: 1,
                  transition: {
                    type: 'spring',
                    damping: 15,
                    stiffness: 200,
                  }
                }}
              >
                <EmotionOrb
                  color={mainColor}
                  size={size}
                  intensity={intensityValue}
                  compact={compact}
                  animate={{
                    scale: [1, 1.05, 1],
                    boxShadow: [
                      `0 0 ${35 * intensityValue}px ${20 * intensityValue}px ${mainColor}33`,
                      `0 0 ${45 * intensityValue}px ${30 * intensityValue}px ${mainColor}44`,
                      `0 0 ${35 * intensityValue}px ${20 * intensityValue}px ${mainColor}33`
                    ]
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 4,
                    ease: "easeInOut"
                  }}
                >
                  <InnerGlow
                    color={mainColor}
                    animate={{
                      opacity: [0.6, 0.9, 0.6],
                      scale: [0.9, 1.2, 0.9],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 3,
                      ease: "easeInOut"
                    }}
                  />
                </EmotionOrb>
              </motion.div>
            </Box>

            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                mt: 1,
              }}>
                <EmotionLabel color={mainColor} compact={compact}>
                  {emotion}
                </EmotionLabel>

                {/* Add sub-emotion display here, below the main emotion */}
                {subEmotion && subEmotion !== 'neutral' && (
                  <SubEmotionDisplay color={getEmotionColor(subEmotion)} compact={compact}>
                    {subEmotion}
                  </SubEmotionDisplay>
                )}

                <Box sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mt: 0.5
                }}>
                  <IntensityIndicator
                    color={mainColor}
                    intensity={intensityValue}
                    compact={compact}
                    initial={{ width: 0 }}
                    animate={{ width: '160px' }}
                    transition={{ duration: 0.8, delay: 0.5, ease: 'easeOut' }}
                  />
                  <IntensityPercentage color={mainColor} compact={compact}>
                    {(intensityValue * 100).toFixed(0)}%
                  </IntensityPercentage>
                </Box>
              </Box>
            </motion.div>
          </motion.div>
        </AnimatePresence>
      </EmotionPulse>
    </Box>
  );
};

export default EmotionCurrent;
