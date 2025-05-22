import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useMediaQuery,
  useTheme,
  IconButton,
  TextField,
  InputAdornment,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import AddIcon from '@mui/icons-material/Add';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import YouTubeIcon from '@mui/icons-material/YouTube';
import { motion } from 'framer-motion';
import { useVideo } from '../VideoContext';
import VideocamIcon from '@mui/icons-material/Videocam';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const SearchContainer = styled(motion.div)(({ theme }) => ({
  position: 'relative',
  borderRadius: 14,
  backgroundColor: 'rgba(255, 255, 255, 0.7)',
  width: '100%',
  marginRight: theme.spacing(1.5),
  marginBottom: 0,
  backdropFilter: 'blur(8px)',
  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.03)',
  border: '1px solid rgba(229, 231, 235, 0.8)',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.85)',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  },
  '&.Mui-focused': {
    backgroundColor: '#fff',
    boxShadow: '0 6px 16px rgba(99, 102, 241, 0.1)',
    border: '1px solid rgba(99, 102, 241, 0.3)',
  }
}));

const SearchInputBase = styled('input')(({ theme }) => ({
  color: theme.palette.text.primary,
  width: '100%',
  height: '38px',
  fontWeight: 500,
  padding: theme.spacing(1, 1, 1, 2.8),
  paddingRight: '30px',
  border: 'none',
  outline: 'none',
  backgroundColor: 'transparent',
  fontSize: '0.9rem',
  fontFamily: 'inherit'
}));

const AddButton = styled(Button)(({ theme }) => ({
  borderRadius: 12,
  minWidth: '38px',
  height: '38px',
  padding: 0,
  boxShadow: '0 2px 8px rgba(99,102,241,0.25)',
  background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
  '&:hover': {
    background: 'linear-gradient(90deg, #4F46E5, #7C3AED)',
    transform: 'translateY(-1px)',
  }
}));

const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialog-paper': {
    borderRadius: 20,
    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
    padding: theme.spacing(2),
    background: 'rgba(255, 255, 255, 0.98)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    maxWidth: '90%',
    width: '550px',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      height: '3px',
      background: 'linear-gradient(90deg, #6366F1, #EC4899)',
      zIndex: 1,
    }
  },
}));

const SearchField = styled(TextField)(({ theme }) => ({
  '& .MuiInputBase-root': {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    backdropFilter: 'blur(8px)',
    borderRadius: '50px',
    padding: theme.spacing(0.5, 1, 0.5, 1.5),
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: 'none',
    border: '1px solid rgba(99, 102, 241, 0.2)',
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      boxShadow: '0 4px 12px rgba(99, 102, 241, 0.15)',
      border: '1px solid rgba(99, 102, 241, 0.4)',
    },
    '&.Mui-focused': {
      boxShadow: '0 4px 16px rgba(99, 102, 241, 0.2)',
      border: '1px solid rgba(99, 102, 241, 0.5)',
    },
  },
  '& .MuiOutlinedInput-notchedOutline': {
    border: 'none',
  },
  '& .MuiInputBase-input': {
    padding: theme.spacing(0.75, 0),
    fontSize: '0.925rem',
    fontWeight: 500,
    '&::placeholder': {
      color: 'rgba(0, 0, 0, 0.4)',
      opacity: 1,
    },
  },
}));

const ClearButton = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  color: 'rgba(0, 0, 0, 0.4)',
  borderRadius: '50%',
  padding: '2px',
  transition: 'all 0.2s',
  '&:hover': {
    color: theme.palette.primary.main,
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
  },
}));

const HeaderTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 600,
  marginBottom: theme.spacing(2),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  color: 'rgba(0, 0, 0, 0.8)',
}));

const SearchChip = styled(motion.div)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  position: 'absolute',
  top: 5,
  right: 12,
  zIndex: 10,
}));

const VideoMemoryHeader = ({ searchValue, onSearchChange, onSearchClear }) => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const { processVideo, isLoading } = useVideo();
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('sm'));

  const validateYoutubeUrl = (url) => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
    return youtubeRegex.test(url);
  };

  const handleOpen = () => {
    setDialogOpen(true);
  };

  const handleClose = () => {
    setDialogOpen(false);
    setUrl('');
    setError('');
  };

  const handleSubmit = () => {
    if (!url.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }

    if (!validateYoutubeUrl(url)) {
      setError('Please enter a valid YouTube URL');
      return;
    }

    processVideo(url);
    handleClose();
  };

  return (
    <>
      <Box>
        <HeaderTitle variant="h6">
          <VideocamIcon fontSize="small" sx={{ color: 'rgba(99, 102, 241, 0.8)' }} />
          Video Memory
          <Chip
            icon={<ExpandMoreIcon style={{ fontSize: '0.9rem' }} />}
            label="Recent"
            size="small"
            sx={{
              height: 24,
              fontSize: '0.75rem',
              backgroundColor: 'rgba(99, 102, 241, 0.08)',
              color: '#6366F1',
              fontWeight: 600,
              ml: 'auto',
              '&:hover': {
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
              }
            }}
          />
        </HeaderTitle>

        <Box sx={{ position: 'relative', mb: 2, display: 'flex' }}>
          <SearchField
            sx={{ flex: 1, mr: 1 }}
            placeholder="Search videos..."
            variant="outlined"
            value={searchValue}
            onChange={onSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ color: 'rgba(99, 102, 241, 0.6)' }} />
                </InputAdornment>
              ),
              endAdornment: searchValue ? (
                <InputAdornment position="end">
                  <ClearButton onClick={onSearchClear}>
                    <ClearIcon fontSize="small" />
                  </ClearButton>
                </InputAdornment>
              ) : null,
            }}
          />

          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <AddButton
              variant="contained"
              onClick={handleOpen}
              aria-label="Add new video"
            >
              <AddIcon />
            </AddButton>
          </motion.div>

          {searchValue && (
            <SearchChip
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
            >
              <Chip
                label={`${searchValue}`}
                size="small"
                sx={{
                  height: 20,
                  fontSize: '0.7rem',
                  backgroundColor: 'rgba(99, 102, 241, 0.08)',
                  color: '#6366F1',
                  fontWeight: 600,
                }}
              />
            </SearchChip>
          )}
        </Box>
      </Box>

      <StyledDialog
        fullScreen={fullScreen}
        open={dialogOpen}
        onClose={handleClose}
        aria-labelledby="add-video-dialog"
      >
        <DialogTitle sx={{
          pb: 1,
          fontWeight: 600,
          background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Add New Video
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Enter a YouTube URL to analyze the emotional content of the video
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            fullWidth
            label="YouTube URL"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            error={!!error}
            helperText={error}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <YouTubeIcon color="error" />
                </InputAdornment>
              ),
              sx: {
                borderRadius: 2,
              }
            }}
            variant="outlined"
            disabled={isLoading}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button
            onClick={handleClose}
            color="inherit"
            sx={{
              textTransform: 'none',
              fontWeight: 500
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={isLoading}
            sx={{
              background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
              textTransform: 'none',
              borderRadius: 2,
              px: 3,
              '&:hover': {
                background: 'linear-gradient(90deg, #4F46E5, #7C3AED)',
              }
            }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze'}
          </Button>
        </DialogActions>
      </StyledDialog>
    </>
  );
};

export default VideoMemoryHeader;
