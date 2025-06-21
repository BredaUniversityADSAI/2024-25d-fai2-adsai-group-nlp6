import React from 'react';
import { motion } from 'framer-motion';
import { Add as AddIcon } from '@mui/icons-material';
import { colors, spacing, typography, borderRadius } from '../../constants/theme';
import GlassCard from '../shared/GlassCard';

/**
 * PlaceholderModule Component
 * Empty module placeholder with subtle animation
 * Used for top-left and bottom-right positions that will be filled later
 * 
 * @param {Object} props - Component props
 * @param {string} props.title - Placeholder title
 * @param {string} props.subtitle - Placeholder subtitle
 * @param {React.ReactNode} props.icon - Optional icon
 * @param {Function} props.onClick - Optional click handler
 */
const PlaceholderModule = ({
  title = 'Coming Soon',
  subtitle = 'This module will be available soon',
  icon = <AddIcon />,
  onClick,
  ...props
}) => {
  return (
    <GlassCard
      hover={!!onClick}
      onClick={onClick}
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        position: 'relative',
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px dashed rgba(255, 255, 255, 0.1)',
      }}
      {...props}
    >
      {/* Animated background pattern */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(circle at 20% 20%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%)
          `,
          borderRadius: borderRadius.xl,
        }}
      />

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        <motion.div
          style={{
            width: '60px',
            height: '60px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: borderRadius.full,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: spacing.md,
            color: colors.text.tertiary,
          }}
          animate={{
            scale: [1, 1.05, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        >
          {icon}
        </motion.div>

        <h3
          style={{
            color: colors.text.secondary,
            fontSize: typography.fontSize.lg,
            fontWeight: typography.fontWeight.medium,
            margin: `0 0 ${spacing.sm} 0`,
          }}
        >
          {title}
        </h3>

        <p
          style={{
            color: colors.text.tertiary,
            fontSize: typography.fontSize.sm,
            margin: 0,
            lineHeight: typography.lineHeight.relaxed,
          }}
        >
          {subtitle}
        </p>
      </div>
    </GlassCard>
  );
};

export default PlaceholderModule;
