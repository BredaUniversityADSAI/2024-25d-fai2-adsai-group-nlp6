/**
 * Premium Dark Theme Configuration
 * Based on the modular dashboard design concept
 * Implements dark, sophisticated color palette with glassmorphism effects
 */

export const theme = {
  // Universal Color Palette
  colors: {
    // Background Colors - Dark gradients for depth and sophistication
    background: {
      primary: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)', // Dark navy gradient
      secondary: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', // Charcoal gradient
      surface: '#1e293b', // Dark surface
      elevated: '#334155', // Elevated surface
    },

    // Primary Accent - Magenta-Purple for important interactive elements
    primary: {
      main: '#d946ef', // Magenta-Purple
      light: '#e879f9',
      dark: '#c026d3',
      100: '#fae8ff',
      200: '#f3e8ff',
      300: '#e879f9',
      400: '#d946ef',
      500: '#c026d3',
      600: '#a21caf',
      700: '#86198f',
      800: '#701a75',
      900: '#581c87',
    },

    // Secondary Accent - Cyan/Blue for less critical elements
    secondary: {
      main: '#06b6d4', // Cyan
      light: '#22d3ee',
      dark: '#0891b2',
      100: '#cffafe',
      200: '#a5f3fc',
      300: '#67e8f9',
      400: '#22d3ee',
      500: '#06b6d4',
      600: '#0891b2',
      700: '#0e7490',
      800: '#155e75',
      900: '#164e63',
    },

    // Text Colors
    text: {
      primary: '#f8fafc', // Off-white for major headings and body text
      secondary: '#cbd5e1', // Light gray for labels, metadata
      tertiary: '#64748b', // Muted gray for less important text
      disabled: '#475569', // Disabled text
    },

    // Status Colors - Semantic colors
    status: {
      success: '#10b981', // Emerald for success/completion
      successBg: 'rgba(16, 185, 129, 0.1)',
      warning: '#f59e0b', // Amber for warnings
      warningBg: 'rgba(245, 158, 11, 0.1)',
      error: '#ef4444', // Red for errors
      errorBg: 'rgba(239, 68, 68, 0.1)',
      info: '#3b82f6', // Blue for info
      infoBg: 'rgba(59, 130, 246, 0.1)',
    },

    // Emotion Colors - Specific to emotion detection
    emotion: {
      happiness: '#10b981', // Emerald - Joy, positivity
      sadness: '#3b82f6', // Blue - Calm, melancholy
      anger: '#ef4444', // Red - Intensity, passion
      fear: '#8b5cf6', // Purple - Mystery, anxiety
      surprise: '#f59e0b', // Amber - Energy, excitement
      disgust: '#84cc16', // Lime - Natural, aversion
      neutral: '#64748b', // Slate - Balance, neutrality
    },

    // Surface Colors
    surface: {
      glass: 'rgba(30, 41, 59, 0.75)', // Glassmorphism effect
      elevated: 'rgba(51, 65, 85, 0.95)', // Elevated surfaces
      card: 'rgba(30, 41, 59, 0.6)', // Card backgrounds
      overlay: 'rgba(0, 0, 0, 0.5)', // Overlay backgrounds
    },

    // Border Colors
    border: 'rgba(148, 163, 184, 0.2)', // Subtle borders
    borderActive: 'rgba(217, 70, 239, 0.4)', // Active borders
    borderHover: 'rgba(148, 163, 184, 0.3)', // Hover borders
  },

  // Typography Scale
  typography: {
    fontFamily: {
      primary: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
      mono: '"SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace',
    },
    fontSize: {
      xs: '0.75rem',    // 12px
      sm: '0.875rem',   // 14px
      base: '1rem',     // 16px
      lg: '1.125rem',   // 18px
      xl: '1.25rem',    // 20px
      '2xl': '1.5rem',  // 24px
      '3xl': '1.875rem', // 30px
      '4xl': '2.25rem', // 36px
      '5xl': '3rem',    // 48px
    },
    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
    },
    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.625,
    },
  },

  // Spacing Scale
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    '2xl': '3rem',   // 48px
    '3xl': '4rem',   // 64px
  },

  // Border Radius Scale
  borderRadius: {
    sm: '0.375rem',  // 6px
    md: '0.5rem',    // 8px
    lg: '0.75rem',   // 12px
    xl: '1rem',      // 16px
    '2xl': '1.5rem', // 24px
    full: '9999px',  // Fully rounded
  },

  // Shadows for depth and elevation
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.25)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.4)',
    glow: '0 0 20px rgba(217, 70, 239, 0.3)', // Glow effect for primary elements
    glowSecondary: '0 0 20px rgba(6, 182, 212, 0.3)', // Glow effect for secondary elements
  },

  // Glassmorphism Effects
  glassmorphism: {
    primary: {
      background: 'rgba(30, 41, 59, 0.75)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(148, 163, 184, 0.2)',
    },
    secondary: {
      background: 'rgba(51, 65, 85, 0.6)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(148, 163, 184, 0.15)',
    },
    overlay: {
      background: 'rgba(15, 23, 42, 0.8)',
      backdropFilter: 'blur(24px)',
      border: '1px solid rgba(148, 163, 184, 0.1)',
    },
  },

  // Animation and Transitions
  animation: {
    duration: {
      fast: '0.15s',
      normal: '0.3s',
      slow: '0.5s',
    },
    easing: {
      easeOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      bounce: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
  },

  // Breakpoints for responsive design
  breakpoints: {
    xs: '0px',
    sm: '600px',
    md: '960px',
    lg: '1280px',
    xl: '1920px',
  },

  // Z-index scale
  zIndex: {
    base: 0,
    dropdown: 1000,
    overlay: 1200,
    modal: 1300,
    tooltip: 1400,
    max: 9999,
  },
};

export default theme;
