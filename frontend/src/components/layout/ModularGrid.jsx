import React from 'react';
import { motion } from 'framer-motion';
import { spacing } from '../../constants/theme';

/**
 * ModularGrid Component
 * 5-box grid layout system for the dashboard
 * Layout: 2 small boxes left, 1 large center, 2 small boxes right
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.topLeft - Top left module
 * @param {React.ReactNode} props.bottomLeft - Bottom left module  
 * @param {React.ReactNode} props.center - Center module (main content)
 * @param {React.ReactNode} props.topRight - Top right module
 * @param {React.ReactNode} props.bottomRight - Bottom right module
 * @param {string} props.className - Additional CSS classes
 */
const ModularGrid = ({
  topLeft,
  bottomLeft,
  center,
  topRight,
  bottomRight,
  className = '',
  ...props
}) => {
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr 2fr 1fr',
    gridTemplateRows: '1fr 1fr',
    gap: spacing.lg,
    height: '100vh',
    padding: spacing.lg,
    boxSizing: 'border-box',
    minHeight: '600px', // Ensure minimum height for proper layout
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const moduleVariants = {
    hidden: { 
      opacity: 0, 
      scale: 0.9,
      y: 20,
    },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: 'easeOut',
      },
    },
  };

  return (
    <motion.div
      className={`modular-grid ${className}`}
      style={gridStyle}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      {...props}
    >
      {/* Top Left Module */}
      <motion.div
        className="module module-top-left"
        style={{
          gridArea: '1 / 1',
          display: 'flex',
          flexDirection: 'column',
        }}
        variants={moduleVariants}
      >
        {topLeft}
      </motion.div>

      {/* Bottom Left Module */}
      <motion.div
        className="module module-bottom-left"
        style={{
          gridArea: '2 / 1',
          display: 'flex',
          flexDirection: 'column',
        }}
        variants={moduleVariants}
      >
        {bottomLeft}
      </motion.div>

      {/* Center Module (spans both rows) */}
      <motion.div
        className="module module-center"
        style={{
          gridColumn: '2',
          gridRow: '1 / -1',
          display: 'flex',
          flexDirection: 'column',
        }}
        variants={moduleVariants}
      >
        {center}
      </motion.div>

      {/* Top Right Module */}
      <motion.div
        className="module module-top-right"
        style={{
          gridArea: '1 / 3',
          display: 'flex',
          flexDirection: 'column',
        }}
        variants={moduleVariants}
      >
        {topRight}
      </motion.div>

      {/* Bottom Right Module */}
      <motion.div
        className="module module-bottom-right"
        style={{
          gridArea: '2 / 3',
          display: 'flex',
          flexDirection: 'column',
        }}
        variants={moduleVariants}
      >
        {bottomRight}
      </motion.div>
    </motion.div>
  );
};

export default ModularGrid;
