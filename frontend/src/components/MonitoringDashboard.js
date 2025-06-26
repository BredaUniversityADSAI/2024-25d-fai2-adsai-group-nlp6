import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  Fab,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Home as HomeIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  Error as ErrorIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Computer as ComputerIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';

// Import all chart components
import {
  ModelPerformanceChart,
  SystemMetricsChart,
  EmotionDistributionChart,
  SubEmotionDistributionChart,
  LatencyTrendsChart,
  DriftDetectionChart,
  ErrorTrackingChart,
  ApiPerformanceChart
} from './MonitoringCharts';

import realMonitoringService from '../services/realMonitoringService';

// Color scheme
const COLORS = {
  primary: '#4F46E5',
  secondary: '#6366f1',
  accent: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  emotions: [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
    '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'
  ]
};

const MonitoringDashboard = () => {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [analyzedData, setAnalyzedData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [overallHealth, setOverallHealth] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setBackendStatus('checking');
      
      console.log('Fetching comprehensive monitoring data...');
      const monitoringData = await realMonitoringService.getAllMonitoringData();
      
      if (!monitoringData) {
        throw new Error('No monitoring data received');
      }

      console.log('Raw monitoring data:', monitoringData);
      setData(monitoringData);
      setBackendStatus('online');

      // Analyze all data types
      const analyzed = {
        modelPerformance: realMonitoringService.analyzeModelPerformance(monitoringData.modelPerformance),
        systemMetrics: realMonitoringService.analyzeSystemMetrics(monitoringData.systemMetrics),
        apiMetrics: realMonitoringService.analyzeApiMetrics(monitoringData.apiMetrics),
        predictionLogs: realMonitoringService.analyzePredictionLogs(monitoringData.predictionLogs),
        driftDetection: realMonitoringService.analyzeDriftDetection(monitoringData.driftDetection),
        errorTracking: realMonitoringService.analyzeErrorTracking(monitoringData.errorTracking)
      };

      console.log('Analyzed data:', analyzed);
      setAnalyzedData(analyzed);

      // Calculate overall health
      const health = realMonitoringService.calculateOverallHealth(
        analyzed.systemMetrics,
        analyzed.apiMetrics,
        analyzed.driftDetection
      );
      setOverallHealth(health);

      setLastUpdated(new Date().toLocaleTimeString());
      setLoading(false);

    } catch (err) {
      console.error('Error fetching monitoring data:', err);
      setError(`Failed to load monitoring data: ${err.message}`);
      setBackendStatus('offline');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRefresh = () => {
    fetchData();
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'online': return <CheckCircleIcon sx={{ color: '#4CAF50', mr: 1 }} />;
      case 'offline': return <ErrorIcon sx={{ color: '#F44336', mr: 1 }} />;
      default: return <CircularProgress size={16} sx={{ mr: 1 }} />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'online': return 'Backend Online';
      case 'offline': return 'Backend Offline';
      default: return 'Checking...';
    }
  };

  if (loading) {
    return (
      <Box 
        sx={{ 
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white'
        }}
      >
        <Box textAlign="center">
          <CircularProgress size={60} sx={{ color: '#4ECDC4', mb: 2 }} />
          <Typography variant="h6">Loading Comprehensive Monitoring Data...</Typography>
          <Typography variant="body2" color="rgba(255,255,255,0.7)" sx={{ mt: 1 }}>
            Analyzing model performance, system metrics, predictions, and more
          </Typography>
        </Box>
      </Box>
    );
  }

  if (error && backendStatus === 'offline') {
    return (
      <Box 
        sx={{ 
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          p: 3
        }}
      >
        <Card sx={{ 
          background: 'rgba(255,255,255,0.02)', 
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 3,
          p: 4,
          textAlign: 'center'
        }}>
          <ErrorIcon sx={{ fontSize: 60, color: '#F44336', mb: 2 }} />
          <Typography variant="h5" gutterBottom>Backend Server Offline</Typography>
          <Typography variant="body1" color="rgba(255,255,255,0.7)" sx={{ mb: 3 }}>
            Unable to connect to the monitoring backend at localhost:3120
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <IconButton 
              onClick={handleRefresh}
              sx={{ 
                backgroundColor: '#4ECDC4', 
                color: 'white',
                '&:hover': { backgroundColor: '#45B7D1' }
              }}
            >
              <RefreshIcon />
            </IconButton>
            <Fab
              onClick={() => navigate('/')}
              sx={{ 
                backgroundColor: '#FF6B6B',
                color: 'white',
                '&:hover': { backgroundColor: '#FF5252' }
              }}
            >
              <HomeIcon />
            </Fab>
          </Box>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
      color: 'white',
      p: 3
    }}>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 3
      }}>
        <Box>
          <Typography variant="h3" component="h1" gutterBottom>
            🔍 Emotion Classification Pipeline
          </Typography>
          <Typography variant="h5" color="rgba(255,255,255,0.8)">
            Comprehensive Monitoring Dashboard
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Backend Status */}
          <Chip
            icon={getStatusIcon(backendStatus)}
            label={getStatusText(backendStatus)}
            color={backendStatus === 'online' ? 'success' : 'error'}
            variant="outlined"
          />
          
          {/* Overall Health */}
          {overallHealth && (
            <Chip
              label={`Health: ${overallHealth.status.toUpperCase()} (${overallHealth.score}%)`}
              sx={{ 
                backgroundColor: overallHealth.color, 
                color: 'white',
                fontWeight: 'bold'
              }}
            />
          )}
          
          {/* Refresh Button */}
          <Tooltip title="Refresh Data">
            <IconButton 
              onClick={handleRefresh}
              sx={{ 
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'white',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Last Updated */}
      {lastUpdated && (
        <Typography variant="body2" color="rgba(255,255,255,0.6)" sx={{ mb: 3 }}>
          Last updated: {lastUpdated}
        </Typography>
      )}

      {/* Dashboard Content */}
      <Grid container spacing={3}>
        {/* Row 1: Key Metrics */}
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <ModelPerformanceChart data={analyzedData.modelPerformance} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <SystemMetricsChart data={analyzedData.systemMetrics} />
            </CardContent>
          </Card>
        </Grid>

        {/* Row 2: API Performance and Predictions */}
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <ApiPerformanceChart data={analyzedData.apiMetrics} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <EmotionDistributionChart data={analyzedData.predictionLogs} />
            </CardContent>
          </Card>
        </Grid>

        {/* Row 3: Detailed Analysis */}
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <SubEmotionDistributionChart data={analyzedData.predictionLogs} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <LatencyTrendsChart data={analyzedData.predictionLogs} />
            </CardContent>
          </Card>
        </Grid>

        {/* Row 4: Drift Detection and Error Tracking */}
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <DriftDetectionChart data={analyzedData.driftDetection} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(255,255,255,0.02)', 
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 3,
            height: '100%'
          }}>
            <CardContent>
              <ErrorTrackingChart data={analyzedData.errorTracking} />
            </CardContent>
          </Card>
        </Grid>

        {/* Summary Statistics */}
        {analyzedData.predictionLogs && (
          <Grid item xs={12}>
            <Card sx={{ 
              background: 'rgba(255,255,255,0.02)', 
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 3
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>📊 Summary Statistics</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2 }}>
                      <Typography variant="h4" color="#4ECDC4">
                        {analyzedData.predictionLogs.totalPredictions}
                      </Typography>
                      <Typography variant="body2" color="rgba(255,255,255,0.7)">
                        Total Predictions
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2 }}>
                      <Typography variant="h4" color="#FECA57">
                        {analyzedData.predictionLogs.latencyStats.avg.toFixed(3)}s
                      </Typography>
                      <Typography variant="body2" color="rgba(255,255,255,0.7)">
                        Average Latency
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2 }}>
                      <Typography variant="h4" color="#FF9FF3">
                        {analyzedData.predictionLogs.emotionDistribution.length}
                      </Typography>
                      <Typography variant="body2" color="rgba(255,255,255,0.7)">
                        Unique Emotions
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2 }}>
                      <Typography variant="h4" color={analyzedData.errorTracking?.totalErrors > 0 ? "#FF6B6B" : "#4CAF50"}>
                        {analyzedData.errorTracking?.totalErrors || 0}
                      </Typography>
                      <Typography variant="body2" color="rgba(255,255,255,0.7)">
                        Total Errors
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Floating Home Button */}
      <Fab
        onClick={() => navigate('/')}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
          color: 'white',
          '&:hover': {
            background: 'linear-gradient(45deg, #FF5252, #26C6DA)',
            transform: 'scale(1.1)'
          },
          transition: 'all 0.3s ease'
        }}
      >
        <HomeIcon />
      </Fab>
    </Box>
  );
};

export default MonitoringDashboard; 