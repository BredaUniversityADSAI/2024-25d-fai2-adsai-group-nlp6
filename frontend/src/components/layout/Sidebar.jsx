import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu as MenuIcon, 
  History as HistoryIcon, 
  Add as AddIcon, 
  Settings as SettingsIcon, 
  Close as CloseIcon,
  VideoLibrary as VideoLibraryIcon,
} from '@mui/icons-material';
import { colors, glassmorphism, borderRadius, spacing, typography } from '../../constants/theme';
import GlassCard from '../shared/GlassCard';

/**
 * Sidebar Component
 * Expandable navigation sidebar with history, add video, and settings
 * 
 * @param {Object} props - Component props
 * @param {Array} props.videoHistory - Array of video history items
 * @param {Function} props.onAddVideo - Handler for add video action
 * @param {Function} props.onSettings - Handler for settings action
 * @param {Function} props.onVideoSelect - Handler for video selection
 */
const Sidebar = ({
  videoHistory = [],
  onAddVideo,
  onSettings,
  onVideoSelect,
  ...props
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const sidebarStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    height: '100vh',
    zIndex: 1000,
    display: 'flex',
    flexDirection: 'column',
  };

  const collapsedStyle = {
    width: '60px',
    padding: spacing.sm,
  };

  const expandedStyle = {
    width: '320px',
    padding: spacing.md,
  };

  const toggleSidebar = () => setIsExpanded(!isExpanded);

  const sidebarVariants = {
    collapsed: {
      width: 60,
      transition: { duration: 0.3, ease: 'easeInOut' }
    },
    expanded: {
      width: 320,
      transition: { duration: 0.3, ease: 'easeInOut' }
    }
  };

  const contentVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { 
      opacity: 1, 
      x: 0,
      transition: { duration: 0.2, delay: 0.1 }
    }
  };

  return (
    <motion.div
      style={sidebarStyle}
      variants={sidebarVariants}
      animate={isExpanded ? 'expanded' : 'collapsed'}
      {...props}
    >
      <GlassCard
        style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: spacing.md,
        }}
        variant="primary"
      >
        {/* Header with toggle button */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: isExpanded ? 'space-between' : 'center',
          marginBottom: spacing.lg,
          minHeight: '40px',
        }}>
          <AnimatePresence>
            {isExpanded && (
              <motion.h2
                style={{
                  color: colors.text.primary,
                  fontSize: typography.fontSize.lg,
                  fontWeight: typography.fontWeight.semibold,
                  margin: 0,
                }}
                variants={contentVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
              >
                Navigation
              </motion.h2>
            )}
          </AnimatePresence>
          
          <motion.button
            onClick={toggleSidebar}
            style={{
              background: 'rgba(139, 92, 246, 0.2)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: borderRadius.md,
              padding: spacing.sm,
              color: colors.primary[400],
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '40px',
              height: '40px',
            }}
            whileHover={{ 
              background: 'rgba(139, 92, 246, 0.3)',
              scale: 1.05,
            }}
            whileTap={{ scale: 0.95 }}
          >
            {isExpanded ? <CloseIcon /> : <MenuIcon />}
          </motion.button>
        </div>

        {/* Navigation Items */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
              variants={contentVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
            >
              {/* Add Video Button */}
              <motion.button
                onClick={onAddVideo}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing.md,
                  padding: spacing.md,
                  background: `linear-gradient(135deg, ${colors.primary[600]}, ${colors.primary[700]})`,
                  border: 'none',
                  borderRadius: borderRadius.lg,
                  color: colors.text.primary,
                  fontSize: typography.fontSize.base,
                  fontWeight: typography.fontWeight.medium,
                  cursor: 'pointer',
                  marginBottom: spacing.md,
                  boxShadow: '0 4px 15px rgba(139, 92, 246, 0.3)',
                }}
                whileHover={{ 
                  scale: 1.02,
                  boxShadow: '0 6px 20px rgba(139, 92, 246, 0.4)',
                }}
                whileTap={{ scale: 0.98 }}
              >
                <AddIcon />
                Add Video
              </motion.button>

              {/* Video History Section */}
              <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                marginBottom: spacing.lg,
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing.sm,
                  marginBottom: spacing.md,
                  color: colors.text.secondary,
                  fontSize: typography.fontSize.sm,
                  fontWeight: typography.fontWeight.medium,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}>
                  <VideoLibraryIcon style={{ fontSize: '16px' }} />
                  History
                </div>

                {/* History List */}
                <div style={{
                  flex: 1,
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: spacing.sm,
                }}>
                  {videoHistory.length > 0 ? (
                    videoHistory.map((video, index) => (
                      <motion.div
                        key={video.id || index}
                        onClick={() => onVideoSelect?.(video)}
                        style={{
                          padding: spacing.md,
                          background: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: borderRadius.md,
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                        }}
                        whileHover={{
                          background: 'rgba(255, 255, 255, 0.08)',
                          x: 4,
                        }}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ 
                          opacity: 1, 
                          y: 0,
                          transition: { delay: index * 0.05 }
                        }}
                      >
                        <div style={{
                          color: colors.text.primary,
                          fontSize: typography.fontSize.sm,
                          fontWeight: typography.fontWeight.medium,
                          marginBottom: '4px',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                        }}>
                          {video.title || 'Untitled Video'}
                        </div>
                        <div style={{
                          color: colors.text.tertiary,
                          fontSize: typography.fontSize.xs,
                        }}>
                          {video.date ? new Date(video.date).toLocaleDateString() : 'No date'}
                        </div>
                      </motion.div>
                    ))
                  ) : (
                    <div style={{
                      padding: spacing.lg,
                      textAlign: 'center',
                      color: colors.text.tertiary,
                      fontSize: typography.fontSize.sm,
                    }}>
                      No videos yet. Add one to get started!
                    </div>
                  )}
                </div>
              </div>

              {/* Settings Button */}
              <motion.button
                onClick={onSettings}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: spacing.md,
                  padding: spacing.md,
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: borderRadius.md,
                  color: colors.text.secondary,
                  fontSize: typography.fontSize.base,
                  fontWeight: typography.fontWeight.medium,
                  cursor: 'pointer',
                }}
                whileHover={{ 
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: colors.text.primary,
                }}
                whileTap={{ scale: 0.98 }}
              >
                <SettingsIcon />
                Settings
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Collapsed Icons */}
        <AnimatePresence>
          {!isExpanded && (
            <motion.div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: spacing.md,
                alignItems: 'center',
              }}
              variants={contentVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
            >
              <motion.button
                onClick={onAddVideo}
                style={{
                  width: '40px',
                  height: '40px',
                  background: 'rgba(139, 92, 246, 0.2)',
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  borderRadius: borderRadius.md,
                  color: colors.primary[400],
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Add Video"
              >
                <AddIcon />
              </motion.button>

              <motion.button
                style={{
                  width: '40px',
                  height: '40px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: borderRadius.md,
                  color: colors.text.secondary,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="History"
              >
                <HistoryIcon />
              </motion.button>

              <motion.button
                onClick={onSettings}
                style={{
                  width: '40px',
                  height: '40px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: borderRadius.md,
                  color: colors.text.secondary,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Settings"
              >
                <SettingsIcon />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </GlassCard>
    </motion.div>
  );
};

export default Sidebar;
