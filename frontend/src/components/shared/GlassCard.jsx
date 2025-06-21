import React from 'react';
import { motion } from 'framer-motion';
import { colors, glassmorphism, borderRadius, spacing } from '../../constants/theme';

/**
 * GlassCard Component
 * Reusable glassmorphism card container with premium styling
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Child components
 * @param {string} props.className - Additional CSS classes
 * @param {Object} props.style - Additional inline styles
 * @param {boolean} props.hover - Enable hover effects
 * @param {string} props.variant - Card variant (primary, secondary)
 * @param {Function} props.onClick - Click handler
 */
const GlassCard = ({
  children,
  className = '',
  style = {},
  hover = false,
  variant = 'primary',
  onClick,
  ...props
}) => {
  const glassStyle = variant === 'secondary' ? glassmorphism.secondary : glassmorphism.primary;
  
  const cardStyle = {
    ...glassStyle,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    position: 'relative',
    overflow: 'hidden',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    cursor: onClick ? 'pointer' : 'default',
    ...style,
  };

  const hoverStyle = hover ? {
    transform: 'translateY(-2px)',
    boxShadow: `${glassStyle.boxShadow}, 0 0 30px rgba(139, 92, 246, 0.2)`,
    background: 'rgba(255, 255, 255, 0.12)',
  } : {};

  return (
    <motion.div
      className={`glass-card ${className}`}
      style={cardStyle}
      onClick={onClick}
      whileHover={hover ? hoverStyle : {}}
      whileTap={onClick ? { scale: 0.98 } : {}}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      {...props}
    >
      {/* Subtle gradient overlay for depth */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%)',
          pointerEvents: 'none',
          borderRadius: borderRadius.xl,
        }}
      />
      
      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        {children}
      </div>
    </motion.div>
  );
};

export default GlassCard;
