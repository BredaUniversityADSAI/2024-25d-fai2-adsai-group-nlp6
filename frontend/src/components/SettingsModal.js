import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  IconButton,
  Switch,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  FormControl,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Close as CloseIcon,
  Settings as SettingsIcon,
  Palette as PaletteIcon,
  Notifications as NotificationsIcon,
  Analytics as AnalyticsIcon,
  Security as SecurityIcon,
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
    minWidth: '600px',
    maxWidth: '700px',
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

const SettingsSection = styled(Box)(() => ({
  padding: theme.spacing.lg,
  borderRadius: theme.borderRadius.xl,
  background: theme.colors.surface.glass,
  border: `1px solid ${theme.colors.border}`,
  marginBottom: theme.spacing.lg,
}));

const SettingItem = styled(Box)(() => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: `${theme.spacing.md} 0`,
  borderBottom: `1px solid ${theme.colors.border}`,
  '&:last-child': {
    borderBottom: 'none',
  },
}));

const StyledSwitch = styled(Switch)(() => ({
  '& .MuiSwitch-track': {
    backgroundColor: theme.colors.surface.card,
    border: `1px solid ${theme.colors.border}`,
  },
  '& .MuiSwitch-thumb': {
    backgroundColor: theme.colors.text.primary,
  },
  '& .Mui-checked': {
    '& .MuiSwitch-thumb': {
      backgroundColor: theme.colors.primary.main,
    },
    '& + .MuiSwitch-track': {
      backgroundColor: `${theme.colors.primary.main}40`,
      border: `1px solid ${theme.colors.primary.main}`,
    },
  },
}));

const StyledSlider = styled(Slider)(() => ({
  color: theme.colors.primary.main,
  height: 6,
  '& .MuiSlider-track': {
    background: `linear-gradient(90deg, ${theme.colors.primary.main}, ${theme.colors.secondary.main})`,
    border: 'none',
    borderRadius: theme.borderRadius.full,
  },
  '& .MuiSlider-rail': {
    backgroundColor: theme.colors.surface.card,
    border: `1px solid ${theme.colors.border}`,
  },
  '& .MuiSlider-thumb': {
    backgroundColor: theme.colors.text.primary,
    border: `2px solid ${theme.colors.primary.main}`,
    width: 20,
    height: 20,
    '&:hover': {
      boxShadow: theme.shadows.glow,
    },
  },
}));

const StyledSelect = styled(Select)(() => ({
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.colors.border,
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.colors.borderHover,
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.colors.primary.main,
  },
  '& .MuiSelect-select': {
    color: theme.colors.text.primary,
    background: theme.colors.surface.card,
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

/**
 * SettingsModal Component
 * Modal for configuring application settings
 * Includes theme, analysis, and notification preferences
 */
const SettingsModal = ({ open, onClose, settings = {}, onSave }) => {
  const [localSettings, setLocalSettings] = useState({
    theme: 'dark',
    animations: true,
    notifications: true,
    soundEffects: false,
    analysisSpeed: 50,
    confidenceThreshold: 75,
    autoSave: true,
    language: 'en',
    dataRetention: 30,
    ...settings,
  });

  const handleClose = () => {
    setLocalSettings({
      theme: 'dark',
      animations: true,
      notifications: true,
      soundEffects: false,
      analysisSpeed: 50,
      confidenceThreshold: 75,
      autoSave: true,
      language: 'en',
      dataRetention: 30,
      ...settings,
    });
    onClose();
  };

  const handleSave = () => {
    if (onSave) {
      onSave(localSettings);
    }
    onClose();
  };

  const handleSettingChange = (key, value) => {
    setLocalSettings(prev => ({
      ...prev,
      [key]: value,
    }));
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
            <SettingsIcon sx={{ color: theme.colors.primary.main }} />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: theme.typography.fontWeight.bold }}>
            Settings
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

      <DialogContent sx={{ p: theme.spacing.xl, maxHeight: '70vh', overflowY: 'auto' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Appearance Settings */}
          <SettingsSection>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <PaletteIcon sx={{ color: theme.colors.primary.main }} />
              <Typography variant="h6" sx={{
                color: theme.colors.text.primary,
                fontWeight: theme.typography.fontWeight.semibold,
              }}>
                Appearance
              </Typography>
            </Box>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Theme
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Choose your preferred color scheme
                </Typography>
              </Box>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <StyledSelect
                  value={localSettings.theme}
                  onChange={(e) => handleSettingChange('theme', e.target.value)}
                >
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="auto">Auto</MenuItem>
                </StyledSelect>
              </FormControl>
            </SettingItem>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Animations
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Enable smooth animations and transitions
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <StyledSwitch
                    checked={localSettings.animations}
                    onChange={(e) => handleSettingChange('animations', e.target.checked)}
                  />
                }
                label=""
              />
            </SettingItem>
          </SettingsSection>

          {/* Analysis Settings */}
          <SettingsSection>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <AnalyticsIcon sx={{ color: theme.colors.secondary.main }} />
              <Typography variant="h6" sx={{
                color: theme.colors.text.primary,
                fontWeight: theme.typography.fontWeight.semibold,
              }}>
                Analysis
              </Typography>
            </Box>

            <SettingItem>
              <Box sx={{ flex: 1, mr: 4 }}>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Analysis Speed
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary, mb: 2 }}>
                  Adjust processing speed vs accuracy balance
                </Typography>
                <StyledSlider
                  value={localSettings.analysisSpeed}
                  onChange={(e, value) => handleSettingChange('analysisSpeed', value)}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}%`}
                  min={10}
                  max={100}
                  step={10}
                />
              </Box>
            </SettingItem>

            <SettingItem>
              <Box sx={{ flex: 1, mr: 4 }}>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Confidence Threshold
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary, mb: 2 }}>
                  Minimum confidence level for emotion detection
                </Typography>
                <StyledSlider
                  value={localSettings.confidenceThreshold}
                  onChange={(e, value) => handleSettingChange('confidenceThreshold', value)}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}%`}
                  min={50}
                  max={95}
                  step={5}
                />
              </Box>
            </SettingItem>
          </SettingsSection>

          {/* Notifications & Privacy */}
          <SettingsSection>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <NotificationsIcon sx={{ color: theme.colors.status.info }} />
              <Typography variant="h6" sx={{
                color: theme.colors.text.primary,
                fontWeight: theme.typography.fontWeight.semibold,
              }}>
                Notifications & Privacy
              </Typography>
            </Box>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Push Notifications
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Receive notifications about analysis completion
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <StyledSwitch
                    checked={localSettings.notifications}
                    onChange={(e) => handleSettingChange('notifications', e.target.checked)}
                  />
                }
                label=""
              />
            </SettingItem>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Sound Effects
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Play sounds for important events
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <StyledSwitch
                    checked={localSettings.soundEffects}
                    onChange={(e) => handleSettingChange('soundEffects', e.target.checked)}
                  />
                }
                label=""
              />
            </SettingItem>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Auto-save Results
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Automatically save analysis results
                </Typography>
              </Box>
              <FormControlLabel
                control={
                  <StyledSwitch
                    checked={localSettings.autoSave}
                    onChange={(e) => handleSettingChange('autoSave', e.target.checked)}
                  />
                }
                label=""
              />
            </SettingItem>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Data Retention
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  How long to keep analysis data
                </Typography>
              </Box>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <StyledSelect
                  value={localSettings.dataRetention}
                  onChange={(e) => handleSettingChange('dataRetention', e.target.value)}
                >
                  <MenuItem value={7}>7 days</MenuItem>
                  <MenuItem value={30}>30 days</MenuItem>
                  <MenuItem value={90}>90 days</MenuItem>
                  <MenuItem value={365}>1 year</MenuItem>
                  <MenuItem value={-1}>Forever</MenuItem>
                </StyledSelect>
              </FormControl>
            </SettingItem>
          </SettingsSection>

          {/* Language & Region */}
          <SettingsSection>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              <SecurityIcon sx={{ color: theme.colors.status.warning }} />
              <Typography variant="h6" sx={{
                color: theme.colors.text.primary,
                fontWeight: theme.typography.fontWeight.semibold,
              }}>
                Language & Region
              </Typography>
            </Box>

            <SettingItem>
              <Box>
                <Typography variant="body1" sx={{ color: theme.colors.text.primary, fontWeight: 500 }}>
                  Interface Language
                </Typography>
                <Typography variant="body2" sx={{ color: theme.colors.text.secondary }}>
                  Choose your preferred language
                </Typography>
              </Box>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <StyledSelect
                  value={localSettings.language}
                  onChange={(e) => handleSettingChange('language', e.target.value)}
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Español</MenuItem>
                  <MenuItem value="fr">Français</MenuItem>
                  <MenuItem value="de">Deutsch</MenuItem>
                  <MenuItem value="zh">中文</MenuItem>
                  <MenuItem value="ja">日本語</MenuItem>
                </StyledSelect>
              </FormControl>
            </SettingItem>
          </SettingsSection>
        </motion.div>
      </DialogContent>

      <DialogActions sx={{ 
        p: theme.spacing.xl, 
        pt: 0,
        gap: theme.spacing.md,
        borderTop: `1px solid ${theme.colors.border}`,
      }}>
        <ActionButton onClick={handleClose}>
          Cancel
        </ActionButton>
        
        <ActionButton variant="contained" onClick={handleSave}>
          Save Settings
        </ActionButton>
      </DialogActions>
    </StyledDialog>
  );
};

export default SettingsModal;
