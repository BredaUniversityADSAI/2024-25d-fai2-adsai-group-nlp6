import React, { useEffect } from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { styled } from '@mui/material/styles';
import { getEmotionColor, getIntensityValue } from '../utils';

const EmotionContainer = styled(Paper)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  background: 'rgba(255, 255, 255, 0.7)',
  backdropFilter: 'blur(12px)',
  borderRadius: '18px',
  border: '1px solid rgba(255, 255, 255, 0.8)',
  transition: 'transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1), 0 5px 15px rgba(0, 0, 0, 0.05)'
  }
}));

const EmotionHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2, 3),
  borderBottom: '1px solid rgba(0, 0, 0, 0.05)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between'
}));

const EmotionOrb = styled(motion.div)(({ color, size = 100, intensity = 0.5 }) => ({
  width: size,
  height: size,
  borderRadius: '50%',
  background: `radial-gradient(circle, ${color}CC 0%, ${color}99 35%, ${color}44 70%, ${color}11 100%)`,
  boxShadow: `
    0 0 ${30 * intensity}px ${20 * intensity}px ${color}33,
    0 0 ${15 * intensity}px ${10 * intensity}px ${color}55,
    inset 0 0 ${40 * intensity}px ${color}66
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
    top: -size * 0.15,
    left: -size * 0.15,
    right: -size * 0.15,
    bottom: -size * 0.15,
    borderRadius: '50%',
    border: `2px solid ${color}33`,
    opacity: 0.6,
    animation: 'pulse 4s infinite ease-in-out',
    zIndex: -1,
  },
  '&:after': {
    content: '""',
    position: 'absolute',
    top: -size * 0.05,
    left: -size * 0.05,
    right: -size * 0.05,
    bottom: -size * 0.05,
    borderRadius: '50%',
    border: `3px solid ${color}55`,
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

const SubEmotionOrb = styled(motion.div)(({ color, size = 36, primary }) => {
  return {
    width: size,
    height: size,
    borderRadius: '50%',
    background: `radial-gradient(circle, ${color}DD 0%, ${color}99 60%, ${color}33 100%)`,
    boxShadow: primary
      ? `0 0 20px 8px ${color}33, inset 0 0 10px 2px rgba(255,255,255,0.3)`
      : `0 0 15px 5px ${color}22, inset 0 0 8px 1px rgba(255,255,255,0.2)`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: primary ? 3 : 1,
    backdropFilter: 'blur(4px)',
    border: `1px solid ${color}33`,
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

const EmotionPulse = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
}));

const EmotionText = styled(Typography)(({ theme, color }) => ({
  fontWeight: 600,
  textTransform: 'capitalize',
  color: color,
  marginTop: theme.spacing(2),
  textShadow: '0 1px 3px rgba(0,0,0,0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: 'Manrope, sans-serif',
}));

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
}));

const EmotionLabel = styled(Typography)(({ color }) => ({
  fontWeight: 700,
  fontSize: '1.5rem',
  color: color,
  textTransform: 'capitalize',
  background: `linear-gradient(to right, ${color}, ${color}99)`,
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  textShadow: '0px 1px 1px rgba(0,0,0,0.05)',
}));

const IntensityIndicator = styled(motion.div)(({ color, intensity }) => ({
  width: '80px',
  height: '4px',
  backgroundColor: 'rgba(0,0,0,0.08)',
  borderRadius: '2px',
  overflow: 'hidden',
  marginTop: '8px',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${intensity * 100}%`,
    backgroundColor: color,
    borderRadius: '2px',
    boxShadow: `0 0 10px ${color}99`,
  }
}));

const EmotionCurrent = ({ emotion, subEmotion, intensity = 0.5, relatedEmotions = [] }) => {
  if (!emotion) {
    return (
      <EmotionContainer elevation={0}>
        <EmotionHeader>
          <Typography variant="h6" fontWeight={600}>
            Emotion Pulse
          </Typography>
        </EmotionHeader>
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          p: 3
        }}>
          <Typography align="center" color="text.secondary" sx={{ opacity: 0.7 }}>
            No emotion data available
          </Typography>
        </Box>
      </EmotionContainer>
    );
  }

  const mainColor = getEmotionColor(emotion);
  const intensityValue = getIntensityValue(intensity);
  const size = 120 + (intensityValue * 40); // Size varies based on intensity

  // Create a list that includes the main sub-emotion plus any related ones
  const allSubEmotions = subEmotion && subEmotion !== 'neutral'
    ? [{ name: subEmotion, primary: true }, ...relatedEmotions.slice(0, 3)]
    : [...relatedEmotions.slice(0, 4)];

  // Generate random particles for effect
  const particles = Array.from({ length: 8 }, (_, i) => ({
    id: i,
    x: Math.random() * 300 - 150,
    y: Math.random() * 300 - 150,
    scale: 0.8 + Math.random() * 0.5,
    duration: 3 + Math.random() * 4,
    delay: i * 0.3
  }));

  // Position sub-emotions in a circle around the main emotion
  const getOrbPosition = (index, total) => {
    const angle = (index / total) * 2 * Math.PI + Math.PI / total;
    const radius = size * 0.9; // Distance from main orb center
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    return { x, y };
  };

  return (
    <EmotionContainer elevation={0}>
      <EmotionHeader>
        <Typography variant="h6" fontWeight={600} sx={{
          background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Emotion Pulse
        </Typography>
      </EmotionHeader>
      <EmotionPulse>
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
              padding: '20px',
              position: 'relative',
            }}
          >
            <Box
              sx={{
                position: 'relative',
                width: size + 120,
                height: size + 120,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginY: 2,
              }}
            >
              {/* Background radial gradient */}
              <Box
                sx={{
                  position: 'absolute',
                  width: size * 2.5,
                  height: size * 2.5,
                  borderRadius: '50%',
                  background: `radial-gradient(circle, ${mainColor}11 0%, ${mainColor}05 50%, transparent 80%)`,
                  opacity: 0.7,
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

              {/* Sub emotions orbs */}
              {allSubEmotions.map((subEm, i) => {
                const pos = getOrbPosition(i, allSubEmotions.length);
                const subColor = getEmotionColor(subEm.name);
                const orbSize = subEm.primary ? 36 : 28;

                return (
                  <motion.div
                    key={`${subEm.name}-${i}`}
                    initial={{
                      x: 0,
                      y: 0,
                      opacity: 0,
                      scale: 0
                    }}
                    animate={{
                      x: pos.x,
                      y: pos.y,
                      opacity: 1,
                      scale: 1
                    }}
                    transition={{
                      type: 'spring',
                      damping: 15,
                      stiffness: 200,
                      delay: 0.2 + (i * 0.1)
                    }}
                  >
                    <SubEmotionOrb
                      color={subColor}
                      size={orbSize}
                      primary={subEm.primary}
                      animate={{
                        y: [0, -5, 0],
                        boxShadow: [
                          `0 0 15px 5px ${subColor}22`,
                          `0 0 20px 7px ${subColor}33`,
                          `0 0 15px 5px ${subColor}22`
                        ]
                      }}
                      transition={{
                        repeat: Infinity,
                        duration: 3 + i,
                        ease: "easeInOut"
                      }}
                    />
                    {subEm.primary && (
                      <SubEmotionText
                        color={subColor}
                        sx={{
                          top: pos.y > 0 ? '110%' : 'auto',
                          bottom: pos.y <= 0 ? '110%' : 'auto',
                          left: pos.x >= 0 ? '50%' : 'auto',
                          right: pos.x < 0 ? '50%' : 'auto',
                          transform: pos.x >= 0 ? 'translateX(-50%)' : 'translateX(50%)',
                        }}
                      >
                        {subEm.name}
                      </SubEmotionText>
                    )}
                  </motion.div>
                );
              })}

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
                  animate={{
                    scale: [1, 1.05, 1],
                    boxShadow: [
                      `0 0 ${30 * intensityValue}px ${20 * intensityValue}px ${mainColor}33`,
                      `0 0 ${40 * intensityValue}px ${25 * intensityValue}px ${mainColor}44`,
                      `0 0 ${30 * intensityValue}px ${20 * intensityValue}px ${mainColor}33`
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
                      opacity: [0.6, 0.8, 0.6],
                      scale: [0.9, 1.1, 0.9],
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

            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              mt: 2
            }}>
              <EmotionLabel color={mainColor}>
                {emotion}
              </EmotionLabel>
              <IntensityIndicator
                color={mainColor}
                intensity={intensityValue}
                initial={{ width: 0 }}
                animate={{ width: '80px' }}
                transition={{ duration: 0.8, delay: 0.5, ease: 'easeOut' }}
              />
            </Box>
          </motion.div>
        </AnimatePresence>
      </EmotionPulse>
    </EmotionContainer>
  );
};

export default EmotionCurrent;
