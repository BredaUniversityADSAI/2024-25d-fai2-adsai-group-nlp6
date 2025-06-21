import React, { useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { styled } from '@mui/material/styles';
import { getEmotionColor, getIntensityValue } from '../utils';

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

// Enhanced Emotion Orb with sophisticated visual effects
const EmotionOrb = styled(motion.div)(({ color, size = 120, intensity = 0.5, compact }) => ({
  width: compact ? size * 0.8 : size,
  height: compact ? size * 0.8 : size,
  borderRadius: '50%',
  background: `
    radial-gradient(circle at 30% 30%, ${color}FF 0%, ${color}DD 25%, ${color}BB 50%, ${color}88 75%, ${color}33 100%),
    radial-gradient(circle at 70% 70%, ${color}77 0%, ${color}44 50%, ${color}11 100%)
  `,
  boxShadow: `
    0 0 ${30 * intensity}px ${20 * intensity}px ${color}44,
    0 0 ${60 * intensity}px ${40 * intensity}px ${color}22,
    0 0 ${90 * intensity}px ${60 * intensity}px ${color}11,
    inset 0 0 ${40 * intensity}px ${color}55,
    inset 0 ${5 * intensity}px ${20 * intensity}px rgba(255, 255, 255, 0.3)
  `,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  position: 'relative',
  zIndex: 2,
  isolation: 'isolate',
  cursor: 'pointer',
  transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: -size * (compact ? 0.12 : 0.18),
    left: -size * (compact ? 0.12 : 0.18),
    right: -size * (compact ? 0.12 : 0.18),
    bottom: -size * (compact ? 0.12 : 0.18),
    borderRadius: '50%',
    border: `3px solid ${color}33`,
    opacity: 0.8,
    animation: 'pulse 4s infinite ease-in-out',
    zIndex: -1,
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    top: -size * (compact ? 0.06 : 0.08),
    left: -size * (compact ? 0.06 : 0.08),
    right: -size * (compact ? 0.06 : 0.08),
    bottom: -size * (compact ? 0.06 : 0.08),
    borderRadius: '50%',
    border: `2px solid ${color}55`,
    opacity: 0.6,
    animation: 'pulse 3s 0.5s infinite ease-in-out',
    zIndex: -1,
  },
  '&:hover': {
    transform: 'scale(1.05)',
    boxShadow: `
      0 0 ${40 * intensity}px ${30 * intensity}px ${color}55,
      0 0 ${80 * intensity}px ${60 * intensity}px ${color}33,
      0 0 ${120 * intensity}px ${80 * intensity}px ${color}11,
      inset 0 0 ${50 * intensity}px ${color}66
    `,
  }
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

// Enhanced emotion label with modern gradient text
const EmotionLabel = styled(Typography)(({ color, compact }) => ({
  fontWeight: 800,
  fontSize: compact ? '2.2rem' : '2.8rem',
  background: `linear-gradient(135deg, ${color}, ${color}CC, ${color}99)`,
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  backgroundClip: 'text',
  textTransform: 'capitalize',
  letterSpacing: '-0.03em',
  marginBottom: '0.5rem',
  fontFamily: '"Inter", sans-serif',
  fontVariant: 'small-caps',
  position: 'relative',
  textAlign: 'center',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-8px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '60%',
    height: '3px',
    background: `linear-gradient(90deg, transparent, ${color}77, transparent)`,
    borderRadius: '2px',
  }
}));

// Modern intensity indicator with enhanced visual feedback
const IntensityIndicator = styled(motion.div)(({ color, intensity, compact }) => ({
  width: compact ? '140px' : '180px',
  height: compact ? '8px' : '12px',
  backgroundColor: 'rgba(0, 0, 0, 0.08)',
  borderRadius: '6px',
  overflow: 'hidden',
  marginTop: compact ? '8px' : '12px',
  position: 'relative',
  boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.1)',
  border: '1px solid rgba(255, 255, 255, 0.3)',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${Math.max(intensity * 100, 5)}%`,
    background: `linear-gradient(90deg, ${color}FF 0%, ${color}DD 50%, ${color}BB 100%)`,
    borderRadius: '6px',
    boxShadow: `0 0 12px ${color}88, inset 0 1px 0 rgba(255, 255, 255, 0.3)`,
    transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${Math.max(intensity * 100, 5)}%`,
    background: `linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent)`,
    borderRadius: '6px',
    animation: 'shimmer 2s infinite ease-in-out',
  }
}));

// Enhanced intensity percentage display
const IntensityPercentage = styled(Typography)(({ color, compact }) => ({
  fontSize: compact ? '0.9rem' : '1.1rem',
  fontWeight: 700,
  color: color,
  marginTop: '8px',
  textAlign: 'center',
  fontFamily: '"Inter", sans-serif',
  letterSpacing: '0.5px',
  marginLeft: '8px',
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
  useEffect(() => {
    // Reset tooltip state when emotion changes
  }, [emotion]);

  if (!emotion) {
    const noEmotionColor = '#9CA3AF'; // Gray color for no emotion

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
          <EmotionPulse compact={compact}>
            <Box
              sx={{
                position: 'relative',
                width: compact ? 160 : 200,
                height: compact ? 160 : 200,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {/* Empty gray circle */}
              <Box
                sx={{
                  width: compact ? 90 : 120,
                  height: compact ? 90 : 120,
                  borderRadius: '50%',
                  border: '2px solid rgba(156, 163, 175, 0.4)',
                  backgroundColor: 'rgba(156, 163, 175, 0.05)',
                  boxShadow: '0 0 15px rgba(156, 163, 175, 0.1)',
                }}
              />
            </Box>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
              <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                mt: 1,
              }}>
                <EmotionLabel color={noEmotionColor} compact={compact}>
                  No Emotion
                </EmotionLabel>

                <SubEmotionDisplay color={noEmotionColor} compact={compact}>
                  None
                </SubEmotionDisplay>

                <Box sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mt: 0.5
                }}>
                  <IntensityIndicator
                    color={noEmotionColor}
                    intensity={0}
                    compact={compact}
                    initial={{ width: 0 }}
                    animate={{ width: compact ? '120px' : '160px' }}
                    transition={{ duration: 0.8, delay: 0.5, ease: 'easeOut' }}
                  />
                  <IntensityPercentage color={noEmotionColor} compact={compact}>
                    0%
                  </IntensityPercentage>
                </Box>
              </Box>
            </motion.div>
          </EmotionPulse>
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
    delay: i * 0.2  }));

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
                {subEmotion && (
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
