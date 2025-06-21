import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Select,
  MenuItem,
  FormControl,
  Typography,
  Box,
  IconButton,
  Chip,
  useTheme,
  useMediaQuery,
  Snackbar,
  Alert,
  CircularProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import SaveIcon from '@mui/icons-material/Save';
import EditIcon from '@mui/icons-material/Edit';
import FeedbackIcon from '@mui/icons-material/Feedback';
import { motion, AnimatePresence } from 'framer-motion';
import { getEmotionColor } from '../utils';
import { saveFeedback } from '../api';

// Styled components for better design
const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialog-paper': {
    borderRadius: 20,
    maxWidth: '95vw',
    maxHeight: '90vh',
    width: '1200px',
    background: 'rgba(255, 255, 255, 0.98)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.15)',
  },
}));

const StyledTableContainer = styled(TableContainer)(({ theme }) => ({
  maxHeight: '60vh',
  borderRadius: 12,
  border: '1px solid rgba(229, 231, 235, 0.8)',
  '&::-webkit-scrollbar': {
    width: '8px',
  },
  '&::-webkit-scrollbar-track': {
    background: 'rgba(0, 0, 0, 0.03)',
    borderRadius: '4px',
  },
  '&::-webkit-scrollbar-thumb': {
    background: 'rgba(99, 102, 241, 0.3)',
    borderRadius: '4px',
    '&:hover': {
      background: 'rgba(99, 102, 241, 0.5)',
    },
  },
}));

const StyledTableHead = styled(TableHead)(({ theme }) => ({
  '& .MuiTableCell-head': {
    backgroundColor: 'rgba(99, 102, 241, 0.08)',
    fontWeight: 700,
    fontSize: '0.95rem',
    color: '#374151',
    borderBottom: '2px solid rgba(99, 102, 241, 0.2)',
    position: 'sticky',
    top: 0,
    zIndex: 1,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(even)': {
    backgroundColor: 'rgba(249, 250, 251, 0.5)',
  },
  '&:hover': {
    backgroundColor: 'rgba(99, 102, 241, 0.04)',
    transform: 'scale(1.002)',
    transition: 'all 0.2s ease',
  },
}));

const StyledSelect = styled(Select)(({ theme }) => ({
  minWidth: 120,
  fontSize: '0.875rem',
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(229, 231, 235, 0.8)',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(99, 102, 241, 0.5)',
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: '#6366F1',
  },
}));

const TextCell = styled(TableCell)(({ theme }) => ({
  maxWidth: '300px',
  wordWrap: 'break-word',
  whiteSpace: 'pre-wrap',
  fontSize: '0.875rem',
  lineHeight: 1.4,
}));

// Available options for dropdowns
const EMOTION_OPTIONS = [
  'happiness',
  'sadness', 
  'anger',
  'fear',
  'disgust',
  'surprise',
  'neutral'
];

const SUB_EMOTION_OPTIONS = [
  'neutral',
  'joy', 'excitement', 'curiosity', 'satisfaction', 'pride', 'relief', 'admiration',
  'amusement', 'approval', 'caring', 'desire', 'gratitude', 'love', 'optimism',
  'grief', 'melancholy', 'disappointment', 'despair', 'sorrow', 'remorse',
  'rage', 'annoyance', 'frustration', 'irritation', 'resentment', 'anger',
  'anxiety', 'panic', 'worry', 'nervousness', 'terror', 'fear', 'confusion',
  'revulsion', 'contempt', 'aversion', 'distaste', 'disgust', 'embarrassment',
  'amazement', 'shock', 'wonder', 'astonishment', 'bewilderment', 'surprise', 'realization'
];

const INTENSITY_OPTIONS = [
  'neutral',
  'mild',
  'moderate', 
  'intense'
];

/**
 * Format time value (number or string) to HH:MM:SS format
 */
const formatTimeValue = (timeValue) => {
  // If it's already a formatted string, return as is
  if (typeof timeValue === 'string' && timeValue.includes(':')) {
    return timeValue;
  }
  
  // Convert number to formatted time string
  const seconds = typeof timeValue === 'number' ? timeValue : parseFloat(timeValue) || 0;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

/**
 * FeedbackModal Component
 * 
 * A comprehensive modal for editing emotion predictions with dropdown selections.
 * Features responsive design, data validation, and Azure integration for saving training data.
 */
const FeedbackModal = ({ open, onClose, transcriptData, videoTitle }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
    // State management for feedback data and UI
  const [feedbackData, setFeedbackData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDataLoading, setIsDataLoading] = useState(true);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');  /**
   * Initialize feedback data when modal opens
   * Converts transcript data to editable format
   */  useEffect(() => {
    if (open && transcriptData) {
      setIsDataLoading(true);
      
      // DEBUG: Log what we're receiving from the API
      console.log("=== FRONTEND DEBUG: Raw transcriptData ===");
      console.log("transcriptData:", transcriptData);
      console.log("transcriptData length:", transcriptData?.length);
      if (transcriptData && transcriptData.length > 0) {
        console.log("First item:", transcriptData[0]);
        console.log("First item keys:", Object.keys(transcriptData[0]));
        console.log("start_time value:", transcriptData[0].start_time, "type:", typeof transcriptData[0].start_time);
        console.log("end_time value:", transcriptData[0].end_time, "type:", typeof transcriptData[0].end_time);
        console.log("sentence value:", transcriptData[0].sentence);
        console.log("emotion value:", transcriptData[0].emotion);
        console.log("sub_emotion value:", transcriptData[0].sub_emotion);
        console.log("intensity value:", transcriptData[0].intensity);
      }
      console.log("=== END FRONTEND DEBUG ===");
        // Simulate processing time for better UX
      const processData = () => {
        const initialData = transcriptData.map((item, index) => ({
          id: index,
          start_time: formatTimeValue(item.start_time),
          end_time: formatTimeValue(item.end_time), 
          text: item.sentence || item.text || '',
          emotion: item.emotion && item.emotion !== 'unknown' ? item.emotion : 'neutral',
          sub_emotion: (item.sub_emotion && item.sub_emotion !== 'unknown' && item.sub_emotion !== 'neutral') 
            ? item.sub_emotion 
            : 'neutral',
          intensity: (item.intensity && item.intensity !== 'unknown') 
            ? item.intensity 
            : 'mild'
        }));
        
        setFeedbackData(initialData);
        setIsDataLoading(false);
      };
      
      // Add small delay to show loading state
      setTimeout(processData, 500);
    } else if (!open) {
      // Reset loading state when modal closes
      setIsDataLoading(true);
      setFeedbackData([]);
    }
  }, [open, transcriptData]);

  /**
   * Handle dropdown value changes
   * Updates specific field for a given row
   */
  const handleValueChange = (id, field, value) => {
    setFeedbackData(prev => 
      prev.map(item => 
        item.id === id ? { ...item, [field]: value } : item
      )
    );
  };

  /**
   * Validate feedback data before submission
   * Ensures all required fields are filled
   */
  const validateData = () => {
    const emptyRows = feedbackData.filter(item => 
      !item.text.trim() || 
      !item.emotion || 
      !item.sub_emotion || 
      !item.intensity
    );
    
    if (emptyRows.length > 0) {
      setSnackbarMessage(`Please fill all fields for ${emptyRows.length} row(s)`);
      setSnackbarSeverity('warning');
      setSnackbarOpen(true);
      return false;
    }
    
    return true;
  };
  /**
   * Submit feedback data to backend
   * Saves as CSV file to Azure training data assets
   */
  const handleSubmit = async () => {
    if (!validateData()) return;
    
    setIsLoading(true);
    
    try {
      const result = await saveFeedback({
        videoTitle: videoTitle || 'Unknown Video',
        feedbackData: feedbackData
      });

      setSnackbarMessage(`Feedback saved successfully as ${result.filename}!`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Close modal after short delay
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (error) {
      console.error('Error saving feedback:', error);
      setSnackbarMessage('Failed to save feedback. Please try again.');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Close modal with confirmation if data has been modified
   */
  const handleClose = () => {
    // Could add dirty check here if needed
    onClose();
  };

  /**
   * Render emotion chip with appropriate color
   */
  const renderEmotionChip = (emotion) => (
    <Chip
      label={emotion}
      size="small"
      sx={{
        backgroundColor: `${getEmotionColor(emotion)}20`,
        color: getEmotionColor(emotion),
        fontWeight: 600,
        fontSize: '0.75rem',
        textTransform: 'capitalize',
      }}
    />
  );

  return (
    <>
      <StyledDialog
        open={open}
        onClose={handleClose}
        fullScreen={isMobile}
        maxWidth={false}
        aria-labelledby="feedback-dialog-title"
      >
        <DialogTitle
          id="feedback-dialog-title"
          sx={{
            pb: 2,
            borderBottom: '1px solid rgba(229, 231, 235, 0.8)',
            background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FeedbackIcon />
            <Typography variant="h6" component="span" sx={{ fontWeight: 700 }}>
              Give Feedback for Predictions
            </Typography>
          </Box>
          
          <IconButton 
            onClick={handleClose}
            size="small"
            sx={{ 
              color: 'rgba(0, 0, 0, 0.5)',
              '&:hover': { 
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                color: '#6366F1' 
              }
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ p: 0 }}>
          <Box sx={{ p: 3, pb: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Review and adjust the emotion predictions below. Your feedback will be saved as training data 
              to improve future predictions.
            </Typography>
              <Typography variant="body2" sx={{ mb: 3, fontWeight: 500 }}>
              Video: <strong>{videoTitle || 'Unknown Video'}</strong> | 
              Total Segments: <strong>
                {isDataLoading ? '...' : feedbackData.length}
              </strong>
            </Typography>
          </Box>          <StyledTableContainer component={Paper} elevation={0}>
            {isDataLoading ? (
              <Box 
                sx={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  py: 8,
                  gap: 2
                }}
              >
                <CircularProgress size={40} sx={{ color: '#6366F1' }} />
                <Typography variant="body1" color="text.secondary">
                  Loading feedback data...
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ opacity: 0.7 }}>
                  Preparing {transcriptData?.length || 0} transcript segments for review
                </Typography>
              </Box>
            ) : (
              <Table stickyHeader>
                <StyledTableHead>
                  <TableRow>
                    <TableCell sx={{ width: '15%' }}>Time Range</TableCell>
                    <TableCell sx={{ width: '35%' }}>Text</TableCell>
                    <TableCell sx={{ width: '15%' }}>Emotion</TableCell>
                    <TableCell sx={{ width: '20%' }}>Sub-Emotion</TableCell>
                    <TableCell sx={{ width: '15%' }}>Intensity</TableCell>
                  </TableRow>
                </StyledTableHead>
                
                <TableBody>
                  <AnimatePresence>
                    {feedbackData.map((row) => (
                    <motion.tr
                      key={row.id}
                      component={StyledTableRow}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2, delay: row.id * 0.02 }}
                    >
                      <TableCell sx={{ fontSize: '0.875rem', fontFamily: 'monospace' }}>
                        <Box>
                          <Typography variant="caption" display="block">
                            {row.start_time}
                          </Typography>
                          <Typography variant="caption" display="block" color="text.secondary">
                            {row.end_time}
                          </Typography>
                        </Box>
                      </TableCell>
                      
                      <TextCell>
                        {row.text || <em style={{ color: '#9CA3AF' }}>No text</em>}
                      </TextCell>
                      
                      <TableCell>
                        <FormControl size="small" fullWidth>
                          <StyledSelect
                            value={row.emotion}
                            onChange={(e) => handleValueChange(row.id, 'emotion', e.target.value)}
                            displayEmpty
                          >
                            {EMOTION_OPTIONS.map(option => (
                              <MenuItem key={option} value={option}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  {renderEmotionChip(option)}
                                </Box>
                              </MenuItem>
                            ))}
                          </StyledSelect>
                        </FormControl>
                      </TableCell>
                      
                      <TableCell>
                        <FormControl size="small" fullWidth>
                          <StyledSelect
                            value={row.sub_emotion}
                            onChange={(e) => handleValueChange(row.id, 'sub_emotion', e.target.value)}
                            displayEmpty
                          >
                            {SUB_EMOTION_OPTIONS.map(option => (
                              <MenuItem key={option} value={option} sx={{ textTransform: 'capitalize' }}>
                                {option}
                              </MenuItem>
                            ))}
                          </StyledSelect>
                        </FormControl>
                      </TableCell>
                      
                      <TableCell>
                        <FormControl size="small" fullWidth>
                          <StyledSelect
                            value={row.intensity}
                            onChange={(e) => handleValueChange(row.id, 'intensity', e.target.value)}
                            displayEmpty
                          >
                            {INTENSITY_OPTIONS.map(option => (
                              <MenuItem key={option} value={option} sx={{ textTransform: 'capitalize' }}>
                                {option}
                              </MenuItem>
                            ))}
                          </StyledSelect>
                        </FormControl>
                      </TableCell>
                    </motion.tr>                  ))}
                </AnimatePresence>
              </TableBody>
            </Table>
            )}
          </StyledTableContainer>
        </DialogContent>

        <DialogActions sx={{ p: 3, borderTop: '1px solid rgba(229, 231, 235, 0.8)' }}>
          <Button
            onClick={handleClose}
            variant="outlined"
            sx={{
              textTransform: 'none',
              borderRadius: 10,
              px: 3,
              color: 'rgba(0, 0, 0, 0.6)',
              borderColor: 'rgba(229, 231, 235, 0.8)',
              '&:hover': {
                borderColor: 'rgba(99, 102, 241, 0.3)',
                backgroundColor: 'rgba(99, 102, 241, 0.02)',
              }
            }}
          >
            Cancel
          </Button>
          
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={isLoading}
            startIcon={isLoading ? <CircularProgress size={16} /> : <SaveIcon />}
            sx={{
              textTransform: 'none',
              borderRadius: 10,
              px: 4,
              background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
              '&:hover': {
                background: 'linear-gradient(90deg, #4F46E5, #7C3AED)',
              },
              '&:disabled': {
                background: 'rgba(0, 0, 0, 0.12)',
              }
            }}
          >
            {isLoading ? 'Saving...' : 'Save Feedback'}
          </Button>
        </DialogActions>
      </StyledDialog>

      {/* Success/Error Snackbar */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setSnackbarOpen(false)} 
          severity={snackbarSeverity}
          sx={{ borderRadius: 2 }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
};

export default FeedbackModal;
