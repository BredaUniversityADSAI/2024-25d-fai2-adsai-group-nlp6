import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  IconButton,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Close as CloseIcon,
  YouTube as YouTubeIcon,
  CloudUpload as UploadIcon,
  Link as LinkIcon,
  VideoFile as VideoFileIcon,
} from '@mui/icons-material';
import theme from '../theme';

// Styled Components
const StyledDialog = styled(Dialog)(() => ({
  '& .MuiDialog-paper': {
    background: theme.glassmorphism.primary.background,
    backdropFilter: theme.glassmorphism.primary.backdropFilter,
    border: theme.glassmorphism.primary.border,
    borderRadius: theme.borderRadius['2xl'],
    boxShadow: theme.shadows['2xl'],
    color: theme.colors.text.primary,
    minWidth: '500px',
    maxWidth: '600px',
  },
  '& .MuiBackdrop-root': {
    background: theme.colors.surface.overlay,
    backdropFilter: 'blur(8px)',
  },
}));

const StyledDialogTitle = styled(DialogTitle)(() => ({
  background: `linear-gradient(135deg, ${theme.colors.primary.main}15, ${theme.colors.secondary.main}08)`,
  borderBottom: `1px solid ${theme.colors.border}`,
  color: theme.colors.text.primary,
  fontWeight: theme.typography.fontWeight.bold,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing.xl,
}));

const StyledTabs = styled(Tabs)(() => ({
  marginBottom: theme.spacing.lg,
  '& .MuiTabs-indicator': {
    background: `linear-gradient(90deg, ${theme.colors.primary.main}, ${theme.colors.secondary.main})`,
    height: 3,
    borderRadius: theme.borderRadius.full,
  },
  '& .MuiTab-root': {
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.fontWeight.medium,
    fontSize: theme.typography.fontSize.sm,
    textTransform: 'none',
    transition: `all ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
    '&:hover': {
      color: theme.colors.text.primary,
    },
    '&.Mui-selected': {
      color: theme.colors.primary.main,
      fontWeight: theme.typography.fontWeight.semibold,
    },
  },
}));

const InputContainer = styled(Box)(() => ({
  padding: theme.spacing.lg,
  borderRadius: theme.borderRadius.xl,
  background: theme.colors.surface.glass,
  border: `1px solid ${theme.colors.border}`,
  marginBottom: theme.spacing.lg,
}));

const StyledTextField = styled(TextField)(() => ({
  '& .MuiOutlinedInput-root': {
    color: theme.colors.text.primary,
    background: theme.colors.surface.card,
    borderRadius: theme.borderRadius.lg,
    '& fieldset': {
      borderColor: theme.colors.border,
    },
    '&:hover fieldset': {
      borderColor: theme.colors.borderHover,
    },
    '&.Mui-focused fieldset': {
      borderColor: theme.colors.primary.main,
      boxShadow: `0 0 0 3px ${theme.colors.primary.main}20`,
    },
  },
  '& .MuiInputLabel-root': {
    color: theme.colors.text.secondary,
    '&.Mui-focused': {
      color: theme.colors.primary.main,
    },
  },
}));

const ActionButton = styled(Button)(({ variant: buttonVariant }) => ({
  borderRadius: theme.borderRadius.lg,
  padding: `${theme.spacing.md} ${theme.spacing.xl}`,
  fontSize: theme.typography.fontSize.sm,
  fontWeight: theme.typography.fontWeight.semibold,
  textTransform: 'none',
  transition: `all ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
  
  ...(buttonVariant === 'contained' ? {
    background: `linear-gradient(135deg, ${theme.colors.primary.main}, ${theme.colors.secondary.main})`,
    color: theme.colors.text.primary,
    boxShadow: theme.shadows.glow,
    '&:hover': {
      background: `linear-gradient(135deg, ${theme.colors.primary.light}, ${theme.colors.secondary.light})`,
      transform: 'translateY(-2px)',
      boxShadow: theme.shadows.xl,
    },
    '&:disabled': {
      background: theme.colors.surface.card,
      color: theme.colors.text.tertiary,
      boxShadow: 'none',
    },
  } : {
    border: `1px solid ${theme.colors.border}`,
    color: theme.colors.text.secondary,
    background: 'transparent',
    '&:hover': {
      borderColor: theme.colors.borderHover,
      background: theme.colors.surface.glass,
      color: theme.colors.text.primary,
    },
  }),
}));

const TabPanel = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && children}
  </div>
);

/**
 * AddVideoModal Component
 * Modal for adding new videos via YouTube URL or file upload
 * Features tabbed interface for different input methods
 */
const AddVideoModal = ({ open, onClose, onSubmit }) => {
  const [tabValue, setTabValue] = useState(0);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [uploadFile, setUploadFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setError('');
  };

  const handleClose = () => {
    setYoutubeUrl('');
    setUploadFile(null);
    setError('');
    setLoading(false);
    setTabValue(0);
    onClose();
  };
  const validateYouTubeUrl = (url) => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
    return youtubeRegex.test(url);
  };

  const handleSubmit = async () => {
    setError('');
    setLoading(true);

    try {
      if (tabValue === 0) {
        // YouTube URL submission
        if (!youtubeUrl.trim()) {
          throw new Error('Please enter a YouTube URL');
        }
        if (!validateYouTubeUrl(youtubeUrl)) {
          throw new Error('Please enter a valid YouTube URL');
        }
        
        await onSubmit({ type: 'youtube', url: youtubeUrl });
      } else {
        // File upload submission
        if (!uploadFile) {
          throw new Error('Please select a video file');
        }
        
        await onSubmit({ type: 'file', file: uploadFile });
      }
      
      handleClose();
    } catch (err) {
      setError(err.message || 'An error occurred while processing your request');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/webm', 'video/ogg'];
      if (!validTypes.includes(file.type)) {
        setError('Please select a valid video file (MP4, WebM, or OGG)');
        return;
      }
      
      // Validate file size (100MB limit)
      const maxSize = 100 * 1024 * 1024; // 100MB
      if (file.size > maxSize) {
        setError('File size must be less than 100MB');
        return;
      }
      
      setUploadFile(file);
      setError('');
    }
  };

  return (
    <StyledDialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
    >
      <StyledDialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box sx={{
            p: 1.5,
            borderRadius: theme.borderRadius.lg,
            background: `linear-gradient(135deg, ${theme.colors.primary.main}20, ${theme.colors.secondary.main}10)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <YouTubeIcon sx={{ color: theme.colors.primary.main }} />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: theme.typography.fontWeight.bold }}>
            Add New Video
          </Typography>
        </Box>
        
        <IconButton
          onClick={handleClose}
          sx={{
            color: theme.colors.text.secondary,
            '&:hover': {
              background: theme.colors.surface.glass,
              color: theme.colors.text.primary,
            },
          }}
        >
          <CloseIcon />
        </IconButton>
      </StyledDialogTitle>

      <DialogContent sx={{ p: theme.spacing.xl }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <StyledTabs value={tabValue} onChange={handleTabChange} variant="fullWidth">
            <Tab
              icon={<LinkIcon fontSize="small" />}
              label="YouTube URL"
              iconPosition="start"
            />
            <Tab
              icon={<UploadIcon fontSize="small" />}
              label="Upload File"
              iconPosition="start"
            />
          </StyledTabs>

          <AnimatePresence mode="wait">
            <motion.div
              key={tabValue}
              initial={{ opacity: 0, x: tabValue === 0 ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: tabValue === 0 ? 20 : -20 }}
              transition={{ duration: 0.3 }}
            >
              <TabPanel value={tabValue} index={0}>
                <InputContainer>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <YouTubeIcon sx={{ color: theme.colors.primary.main }} />
                    <Typography variant="subtitle1" sx={{
                      color: theme.colors.text.primary,
                      fontWeight: theme.typography.fontWeight.semibold,
                    }}>
                      YouTube Video
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" sx={{
                    color: theme.colors.text.secondary,
                    mb: 3,
                  }}>
                    Enter a YouTube URL to analyze the video's emotional content
                  </Typography>
                  
                  <StyledTextField
                    fullWidth
                    label="YouTube URL"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                    disabled={loading}
                    variant="outlined"
                    InputProps={{
                      startAdornment: <LinkIcon sx={{ mr: 1, color: theme.colors.text.tertiary }} />,
                    }}
                  />
                </InputContainer>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <InputContainer>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <VideoFileIcon sx={{ color: theme.colors.secondary.main }} />
                    <Typography variant="subtitle1" sx={{
                      color: theme.colors.text.primary,
                      fontWeight: theme.typography.fontWeight.semibold,
                    }}>
                      Upload Video File
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" sx={{
                    color: theme.colors.text.secondary,
                    mb: 3,
                  }}>
                    Upload a video file from your device (MP4, WebM, OGG - max 100MB)
                  </Typography>
                  
                  <Box sx={{
                    border: `2px dashed ${theme.colors.border}`,
                    borderRadius: theme.borderRadius.xl,
                    p: 4,
                    textAlign: 'center',
                    background: theme.colors.surface.card,
                    transition: `all ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
                    cursor: 'pointer',
                    '&:hover': {
                      borderColor: theme.colors.borderHover,
                      background: theme.colors.surface.glass,
                    },
                  }}>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileChange}
                      style={{ display: 'none' }}
                      id="video-upload"
                      disabled={loading}
                    />
                    <label htmlFor="video-upload" style={{ cursor: 'pointer', display: 'block' }}>
                      <UploadIcon sx={{
                        fontSize: '3rem',
                        color: theme.colors.text.tertiary,
                        mb: 2,
                      }} />
                      
                      {uploadFile ? (
                        <Box>
                          <Typography variant="body1" sx={{
                            color: theme.colors.text.primary,
                            fontWeight: theme.typography.fontWeight.medium,
                            mb: 1,
                          }}>
                            {uploadFile.name}
                          </Typography>
                          <Typography variant="body2" sx={{
                            color: theme.colors.text.secondary,
                          }}>
                            {(uploadFile.size / (1024 * 1024)).toFixed(2)} MB
                          </Typography>
                        </Box>
                      ) : (
                        <Box>
                          <Typography variant="body1" sx={{
                            color: theme.colors.text.primary,
                            fontWeight: theme.typography.fontWeight.medium,
                            mb: 1,
                          }}>
                            Click to select a video file
                          </Typography>
                          <Typography variant="body2" sx={{
                            color: theme.colors.text.secondary,
                          }}>
                            Or drag and drop here
                          </Typography>
                        </Box>
                      )}
                    </label>
                  </Box>
                </InputContainer>
              </TabPanel>
            </motion.div>
          </AnimatePresence>

          {error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Alert 
                severity="error" 
                sx={{
                  background: theme.colors.status.errorBg,
                  color: theme.colors.text.primary,
                  border: `1px solid ${theme.colors.status.error}40`,
                  borderRadius: theme.borderRadius.lg,
                  '& .MuiAlert-icon': {
                    color: theme.colors.status.error,
                  },
                }}
              >
                {error}
              </Alert>
            </motion.div>
          )}
        </motion.div>
      </DialogContent>

      <DialogActions sx={{ 
        p: theme.spacing.xl, 
        pt: 0,
        gap: theme.spacing.md,
      }}>
        <ActionButton
          onClick={handleClose}
          disabled={loading}
        >
          Cancel
        </ActionButton>
        
        <ActionButton
          variant="contained"
          onClick={handleSubmit}
          disabled={loading || (tabValue === 0 ? !youtubeUrl.trim() : !uploadFile)}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : null}
        >
          {loading ? 'Processing...' : 'Add Video'}
        </ActionButton>
      </DialogActions>
    </StyledDialog>
  );
};

export default AddVideoModal;
