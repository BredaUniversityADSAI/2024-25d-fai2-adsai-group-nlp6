import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Divider,
  Badge,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Add as AddIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  VideoLibrary as VideoLibraryIcon,
} from '@mui/icons-material';
import theme from '../theme';

// Styled Components
const SidebarContainer = styled(motion.div)(({ isexpanded }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  height: '100vh',
  width: isexpanded === 'true' ? '320px' : '60px',
  zIndex: theme.zIndex.dropdown,
  transition: `width ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
  display: 'flex',
  flexDirection: 'column',
  background: theme.glassmorphism.primary.background,
  backdropFilter: theme.glassmorphism.primary.backdropFilter,
  borderRight: theme.glassmorphism.primary.border,
}));

const SidebarHeader = styled(Box)(() => ({
  padding: theme.spacing.lg,
  borderBottom: `1px solid ${theme.colors.border}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  minHeight: '72px',
}));

const SidebarBody = styled(Box)(() => ({
  flex: 1,
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  padding: theme.spacing.md,
}));

const ActionButton = styled(IconButton)(() => ({
  width: '48px',
  height: '48px',
  borderRadius: theme.borderRadius.lg,
  background: `linear-gradient(135deg, ${theme.colors.primary.main}, ${theme.colors.secondary.main})`,
  color: theme.colors.text.primary,
  boxShadow: theme.shadows.glow,
  '&:hover': {
    background: `linear-gradient(135deg, ${theme.colors.primary.light}, ${theme.colors.secondary.light})`,
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows.xl,
  },
  '&:active': {
    transform: 'translateY(0)',
  },
}));

const MenuButton = styled(IconButton)(({ isactive }) => ({
  width: '48px',
  height: '48px',
  borderRadius: theme.borderRadius.lg,
  color: theme.colors.text.secondary,
  background: isactive === 'true' 
    ? `linear-gradient(135deg, ${theme.colors.primary.main}20, ${theme.colors.secondary.main}10)`
    : 'transparent',
  border: `1px solid ${isactive === 'true' ? theme.colors.borderActive : 'transparent'}`,
  transition: `all ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
  '&:hover': {
    background: isactive === 'true' 
      ? `linear-gradient(135deg, ${theme.colors.primary.main}30, ${theme.colors.secondary.main}15)`
      : theme.colors.surface.glass,
    border: `1px solid ${theme.colors.borderHover}`,
    transform: 'translateY(-1px)',
  },
}));

const HistoryContainer = styled(Box)(() => ({
  flex: 1,
  overflowY: 'auto',
  marginTop: theme.spacing.md,
  '&::-webkit-scrollbar': {
    width: '6px',
  },
  '&::-webkit-scrollbar-track': {
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '3px',
  },
  '&::-webkit-scrollbar-thumb': {
    background: 'rgba(255, 255, 255, 0.3)',
    borderRadius: '3px',
    '&:hover': {
      background: 'rgba(255, 255, 255, 0.5)',
    },
  },
}));

const HistoryItem = styled(motion.div)(({ isactive }) => ({
  marginBottom: theme.spacing.sm,
  borderRadius: theme.borderRadius.lg,
  overflow: 'hidden',
  background: isactive === 'true' 
    ? `linear-gradient(135deg, ${theme.colors.primary.main}20, ${theme.colors.secondary.main}10)`
    : 'transparent',
  border: `1px solid ${isactive === 'true' ? theme.colors.borderActive : 'transparent'}`,
  transition: `all ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
  '&:hover': {
    background: isactive === 'true' 
      ? `linear-gradient(135deg, ${theme.colors.primary.main}30, ${theme.colors.secondary.main}15)`
      : theme.colors.surface.glass,
    border: `1px solid ${theme.colors.borderHover}`,
    transform: 'translateY(-1px)',
  },
}));

/**
 * Enhanced Sidebar Component
 * Implements premium dark theme with glassmorphism effects
 * Provides navigation, history management, and quick actions
 */
const Sidebar = ({ 
  videoHistory = [], 
  onAddVideo, 
  onSettings, 
  onVideoSelect,
  currentVideoId = null 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState('history');

  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  const handleAddVideo = () => {
    if (onAddVideo) onAddVideo();
  };

  const handleSettings = () => {
    if (onSettings) onSettings();
  };

  const handleVideoSelect = (video) => {
    if (onVideoSelect) onVideoSelect(video);
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <>
      <SidebarContainer
        isexpanded={isExpanded.toString()}
        initial={{ x: -60 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      >
        {/* Header */}
        <SidebarHeader>
          <Tooltip title={isExpanded ? '' : 'Toggle Menu'} placement="right">
            <IconButton
              onClick={toggleSidebar}
              sx={{ 
                color: theme.colors.text.primary,
                '&:hover': { 
                  background: theme.colors.surface.glass 
                }
              }}
            >
              {isExpanded ? <CloseIcon /> : <MenuIcon />}
            </IconButton>
          </Tooltip>
          
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                style={{ flex: 1, marginLeft: 16 }}
              >
                <Typography variant="h6" sx={{
                  fontWeight: 700,
                  background: `linear-gradient(135deg, ${theme.colors.primary.main}, ${theme.colors.secondary.main})`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}>
                  Emotion AI
                </Typography>
              </motion.div>
            )}
          </AnimatePresence>
        </SidebarHeader>

        {/* Body */}
        <SidebarBody>
          {/* Add Video Button */}
          <Box sx={{ mb: 2 }}>
            <Tooltip title={isExpanded ? '' : 'Add Video'} placement="right">
              <ActionButton onClick={handleAddVideo}>
                <AddIcon />
              </ActionButton>
            </Tooltip>
            
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ marginTop: 12 }}
                >
                  <Typography variant="body2" sx={{
                    color: theme.colors.text.secondary,
                    fontWeight: 600,
                    textAlign: 'center',
                  }}>
                    Add New Video
                  </Typography>
                </motion.div>
              )}
            </AnimatePresence>
          </Box>

          <Divider sx={{ 
            borderColor: theme.colors.border, 
            mb: 2 
          }} />

          {/* Navigation Menu */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ marginBottom: theme.spacing.lg }}
              >
                <List sx={{ py: 0 }}>
                  <ListItem disablePadding>
                    <ListItemButton
                      onClick={() => setActiveSection('history')}
                      sx={{
                        borderRadius: theme.borderRadius.lg,
                        mb: 1,
                        background: activeSection === 'history' 
                          ? `linear-gradient(135deg, ${theme.colors.primary.main}20, ${theme.colors.secondary.main}10)`
                          : 'transparent',
                        '&:hover': {
                          background: `linear-gradient(135deg, ${theme.colors.primary.main}30, ${theme.colors.secondary.main}15)`,
                        },
                      }}
                    >
                      <ListItemIcon sx={{ color: theme.colors.primary.main }}>
                        <Badge badgeContent={videoHistory.length} color="primary">
                          <HistoryIcon />
                        </Badge>
                      </ListItemIcon>
                      <ListItemText 
                        primary="History" 
                        sx={{ 
                          '& .MuiListItemText-primary': { 
                            color: theme.colors.text.primary,
                            fontWeight: 600,
                          } 
                        }} 
                      />
                    </ListItemButton>
                  </ListItem>
                </List>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Video History */}
          <AnimatePresence>
            {isExpanded && activeSection === 'history' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
              >
                <Typography variant="subtitle2" sx={{
                  color: theme.colors.text.secondary,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontSize: '0.75rem',
                  mb: 2,
                }}>
                  Recent Videos ({videoHistory.length})
                </Typography>

                <HistoryContainer>
                  {videoHistory.length > 0 ? (
                    videoHistory.map((video, index) => (
                      <HistoryItem
                        key={video.id || index}
                        isactive={(currentVideoId === video.id).toString()}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        onClick={() => handleVideoSelect(video)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Box sx={{ p: 2, cursor: 'pointer' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <VideoLibraryIcon sx={{ 
                              fontSize: '1rem', 
                              color: theme.colors.secondary.main,
                              mr: 1 
                            }} />
                            <Typography variant="body2" sx={{
                              color: theme.colors.text.primary,
                              fontWeight: 600,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              flex: 1,
                            }}>
                              {video.title || 'Untitled Video'}
                            </Typography>
                          </Box>
                          
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="caption" sx={{
                              color: theme.colors.text.tertiary,
                              fontSize: '0.7rem',
                            }}>
                              {video.date ? formatDate(video.date) : 'No date'}
                            </Typography>
                            
                            {video.duration && (
                              <Typography variant="caption" sx={{
                                color: theme.colors.text.tertiary,
                                fontSize: '0.7rem',
                                background: theme.colors.surface.glass,
                                px: 1,
                                py: 0.5,
                                borderRadius: theme.borderRadius.sm,
                              }}>
                                {formatDuration(video.duration)}
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      </HistoryItem>
                    ))
                  ) : (
                    <Box sx={{
                      textAlign: 'center',
                      py: 4,
                      color: theme.colors.text.tertiary,
                    }}>
                      <VideoLibraryIcon sx={{ fontSize: '2rem', mb: 1, opacity: 0.5 }} />
                      <Typography variant="body2">
                        No videos yet
                      </Typography>
                      <Typography variant="caption">
                        Add a video to get started
                      </Typography>
                    </Box>
                  )}
                </HistoryContainer>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Settings Button */}
          <Box sx={{ mt: 'auto', pt: 2 }}>
            <Tooltip title={isExpanded ? '' : 'Settings'} placement="right">
              <MenuButton 
                onClick={handleSettings}
                isactive="false"
              >
                <SettingsIcon />
              </MenuButton>
            </Tooltip>
            
            <AnimatePresence>
              {isExpanded && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  style={{ 
                    marginLeft: 8, 
                    fontWeight: 600,
                    fontSize: '0.95rem',
                    color: theme.colors.text.secondary,
                  }}
                >
                  Settings
                </motion.span>
              )}
            </AnimatePresence>
          </Box>
        </SidebarBody>
      </SidebarContainer>

      {/* Backdrop for mobile */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={toggleSidebar}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.5)',
              zIndex: theme.zIndex.dropdown - 1,
              display: 'none',
              '@media (max-width: 768px)': {
                display: 'block',
              },
            }}
          />
        )}
      </AnimatePresence>
    </>
  );
};

export default Sidebar;
