import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { colors } from '../../constants/theme';
import ModularGrid from './ModularGrid';
import Sidebar from './Sidebar';

/**
 * MainLayout Component
 * Root layout component that orchestrates the entire dashboard
 * Manages sidebar state and coordinates between all modules
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.topLeft - Top left module
 * @param {React.ReactNode} props.bottomLeft - Bottom left module
 * @param {React.ReactNode} props.center - Center module (main content)
 * @param {React.ReactNode} props.topRight - Top right module
 * @param {React.ReactNode} props.bottomRight - Bottom right module
 * @param {Array} props.videoHistory - Video history data
 * @param {Function} props.onAddVideo - Add video handler
 * @param {Function} props.onSettings - Settings handler
 * @param {Function} props.onVideoSelect - Video selection handler
 */
const MainLayout = ({
  topLeft,
  bottomLeft,
  center,
  topRight,
  bottomRight,
  videoHistory = [],
  onAddVideo,
  onSettings,
  onVideoSelect,
  ...props
}) => {
  const layoutStyle = {
    position: 'relative',
    width: '100vw',
    height: '100vh',
    background: colors.background.primary,
    overflow: 'hidden',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  };

  const contentStyle = {
    marginLeft: '60px', // Account for collapsed sidebar
    height: '100vh',
    transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  };

  return (
    <motion.div
      style={layoutStyle}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      {...props}
    >
      {/* Sidebar */}
      <Sidebar
        videoHistory={videoHistory}
        onAddVideo={onAddVideo}
        onSettings={onSettings}
        onVideoSelect={onVideoSelect}
      />

      {/* Main Content Area */}
      <div style={contentStyle}>
        <ModularGrid
          topLeft={topLeft}
          bottomLeft={bottomLeft}
          center={center}
          topRight={topRight}
          bottomRight={bottomRight}
        />
      </div>

      {/* Background Effects */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(circle at 20% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.05) 0%, transparent 50%)
          `,
          pointerEvents: 'none',
          zIndex: -1,
        }}
      />
    </motion.div>
  );
};

export default MainLayout;
