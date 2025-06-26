import React from 'react';
import { Box, Typography, Card, CardContent, CardHeader, Chip, Alert } from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadialBarChart,
  RadialBar,
  ScatterPlot,
  Scatter
} from 'recharts';

const COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#6366F1',
  success: '#059669',
  emotions: [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'
  ]
};

// Color palettes
const EMOTION_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD',
  '#FF8A80', '#82B1FF', '#B388FF', '#FFD54F'
];

const SYSTEM_COLORS = {
  cpu: '#FF6B6B',
  memory: '#4ECDC4',
  disk: '#FECA57'
};

// Performance Chart Component
export const PerformanceChart = ({ data, title = "Performance Timeline" }) => (
  <Card sx={{ background: 'rgba(255,255,255,0.02)', backdropFilter: 'blur(10px)' }}>
    <CardHeader
      title={title}
      subheader="Latency and throughput over time"
      action={
        <Chip
          label="Real-time"
          color="primary"
          variant="outlined"
          size="small"
        />
      }
    />
    <CardContent>
      <Box height={300}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.7)"
              fontSize={12}
            />
            <YAxis stroke="rgba(255,255,255,0.7)" fontSize={12} />
            <Tooltip
              contentStyle={{
                background: 'rgba(0,0,0,0.8)',
                border: 'none',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="latency"
              stroke={COLORS.primary}
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Latency (s)"
            />
            <Line
              type="monotone"
              dataKey="throughput"
              stroke={COLORS.secondary}
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Throughput (/min)"
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </CardContent>
  </Card>
);

// Model Performance Chart
export const ModelPerformanceChart = ({ data }) => {
  if (!data?.timeSeries) return <Typography>No model performance data</Typography>;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Model Performance Over Time</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.timeSeries}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="timestamp" stroke="#fff" fontSize={10} />
          <YAxis stroke="#fff" fontSize={10} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="emotion_f1"
            stroke="#FF6B6B"
            strokeWidth={2}
            name="Emotion F1"
          />
          <Line
            type="monotone"
            dataKey="sub_emotion_f1"
            stroke="#4ECDC4"
            strokeWidth={2}
            name="Sub-Emotion F1"
          />
          <Line
            type="monotone"
            dataKey="intensity_f1"
            stroke="#FECA57"
            strokeWidth={2}
            name="Intensity F1"
          />
          <Line
            type="monotone"
            dataKey="overall_f1"
            stroke="#96CEB4"
            strokeWidth={3}
            name="Overall F1"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Current Performance Metrics */}
      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip
          label={`Emotion F1: ${(data.current.emotion_f1 * 100).toFixed(1)}%`}
          sx={{ backgroundColor: '#FF6B6B', color: 'white' }}
        />
        <Chip
          label={`Sub-Emotion F1: ${(data.current.sub_emotion_f1 * 100).toFixed(1)}%`}
          sx={{ backgroundColor: '#4ECDC4', color: 'white' }}
        />
        <Chip
          label={`Intensity F1: ${(data.current.intensity_f1 * 100).toFixed(1)}%`}
          sx={{ backgroundColor: '#FECA57', color: 'white' }}
        />
        <Chip
          label={`Trend: ${data.trend}`}
          color={data.trend === 'improving' ? 'success' : data.trend === 'declining' ? 'error' : 'default'}
        />
      </Box>
    </Box>
  );
};

// System Metrics Chart
export const SystemMetricsChart = ({ data }) => {
  if (!data?.timeSeries) return <Typography>No system metrics data</Typography>;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>System Resource Usage</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data.timeSeries}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="timestamp" stroke="#fff" fontSize={10} />
          <YAxis stroke="#fff" fontSize={10} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="cpu"
            stackId="1"
            stroke={SYSTEM_COLORS.cpu}
            fill={SYSTEM_COLORS.cpu}
            fillOpacity={0.6}
            name="CPU %"
          />
          <Area
            type="monotone"
            dataKey="memory"
            stackId="2"
            stroke={SYSTEM_COLORS.memory}
            fill={SYSTEM_COLORS.memory}
            fillOpacity={0.6}
            name="Memory %"
          />
          <Area
            type="monotone"
            dataKey="disk"
            stackId="3"
            stroke={SYSTEM_COLORS.disk}
            fill={SYSTEM_COLORS.disk}
            fillOpacity={0.6}
            name="Disk %"
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Current System Status */}
      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip
          label={`CPU: ${data.current.cpu.toFixed(1)}%`}
          sx={{ backgroundColor: SYSTEM_COLORS.cpu, color: 'white' }}
        />
        <Chip
          label={`Memory: ${data.current.memory.toFixed(1)}%`}
          sx={{ backgroundColor: SYSTEM_COLORS.memory, color: 'white' }}
        />
        <Chip
          label={`Disk: ${data.current.disk.toFixed(1)}%`}
          sx={{ backgroundColor: SYSTEM_COLORS.disk, color: 'white' }}
        />
      </Box>
    </Box>
  );
};

// Emotion Distribution Pie Chart
export const EmotionDistributionChart = ({ data }) => {
  if (!data?.emotionDistribution) return <Typography>No emotion distribution data</Typography>;

  const topEmotions = data.emotionDistribution.slice(0, 8); // Show top 8 emotions

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Emotion Distribution ({data.totalPredictions} predictions)
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={topEmotions}
            cx="50%"
            cy="50%"
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percentage }) => `${name}: ${percentage}%`}
          >
            {topEmotions.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={EMOTION_COLORS[index % EMOTION_COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

// Sub-Emotion Distribution Chart
export const SubEmotionDistributionChart = ({ data }) => {
  if (!data?.subEmotionDistribution) return <Typography>No sub-emotion data</Typography>;

  const topSubEmotions = data.subEmotionDistribution.slice(0, 10);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Top Sub-Emotions</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={topSubEmotions} layout="horizontal">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis type="number" stroke="#fff" fontSize={10} />
          <YAxis type="category" dataKey="name" stroke="#fff" fontSize={10} width={80} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Bar dataKey="value" fill="#4ECDC4" />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

// Latency Trends Chart
export const LatencyTrendsChart = ({ data }) => {
  if (!data?.latencyTrend) return <Typography>No latency data</Typography>;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Recent Prediction Latency</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.latencyTrend}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="prediction" stroke="#fff" fontSize={10} />
          <YAxis stroke="#fff" fontSize={10} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Line
            type="monotone"
            dataKey="latency"
            stroke="#FF9FF3"
            strokeWidth={2}
            dot={{ fill: '#FF9FF3', strokeWidth: 2, r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>

      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip
          label={`Avg: ${data.latencyStats.avg.toFixed(3)}s`}
          sx={{ backgroundColor: '#FF9FF3', color: 'white' }}
        />
        <Chip
          label={`P95: ${data.latencyStats.p95.toFixed(3)}s`}
          sx={{ backgroundColor: '#54A0FF', color: 'white' }}
        />
        <Chip
          label={`Min/Max: ${data.latencyStats.min.toFixed(3)}s / ${data.latencyStats.max.toFixed(3)}s`}
          sx={{ backgroundColor: '#5F27CD', color: 'white' }}
        />
      </Box>
    </Box>
  );
};

// Drift Detection Chart
export const DriftDetectionChart = ({ data }) => {
  if (!data?.timeSeries) return <Typography>No drift detection data</Typography>;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Model Drift Detection</Typography>

      {/* Alerts */}
      {data.current.concept_drift_alert && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Concept Drift Alert: Score {data.current.concept_drift_score.toFixed(3)} exceeds threshold {data.current.threshold}
        </Alert>
      )}
      {data.current.data_drift_alert && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Data Drift Alert: Score {data.current.data_drift_score.toFixed(3)} exceeds threshold {data.current.threshold}
        </Alert>
      )}

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.timeSeries}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="timestamp" stroke="#fff" fontSize={10} />
          <YAxis stroke="#fff" fontSize={10} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="data_drift"
            stroke="#FF6B6B"
            strokeWidth={2}
            name="Data Drift"
          />
          <Line
            type="monotone"
            dataKey="concept_drift"
            stroke="#4ECDC4"
            strokeWidth={2}
            name="Concept Drift"
          />
          <Line
            type="monotone"
            dataKey="threshold"
            stroke="#FFA726"
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Threshold"
          />
        </LineChart>
      </ResponsiveContainer>

      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip
          label={`Concept Drift Alerts: ${data.alertCount.concept_drift}`}
          color={data.alertCount.concept_drift > 0 ? 'warning' : 'success'}
        />
        <Chip
          label={`Data Drift Alerts: ${data.alertCount.data_drift}`}
          color={data.alertCount.data_drift > 0 ? 'error' : 'success'}
        />
      </Box>
    </Box>
  );
};

// Error Tracking Chart
export const ErrorTrackingChart = ({ data }) => {
  if (!data || data.totalErrors === 0) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>Error Tracking</Typography>
        <Alert severity="success">No errors detected!</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Error Tracking ({data.totalErrors} total errors)
      </Typography>

      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        {/* Error Types */}
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle2" gutterBottom>Error Types</Typography>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={data.errorTypes}
                cx="50%"
                cy="50%"
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {data.errorTypes.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={EMOTION_COLORS[index % EMOTION_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Box>

        {/* Affected Endpoints */}
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle2" gutterBottom>Affected Endpoints</Typography>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.affectedEndpoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" stroke="#fff" fontSize={10} />
              <YAxis stroke="#fff" fontSize={10} />
              <Tooltip />
              <Bar dataKey="value" fill="#FF6B6B" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      {/* Recent Errors */}
      <Typography variant="subtitle2" gutterBottom>Recent Errors</Typography>
      <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
        {data.recentErrors.map((error, index) => (
          <Box
            key={index}
            sx={{
              p: 1,
              mb: 1,
              backgroundColor: 'rgba(255,0,0,0.1)',
              borderRadius: 1,
              border: '1px solid rgba(255,0,0,0.2)'
            }}
          >
            <Typography variant="caption" color="error">
              {error.timestamp} - {error.type} on {error.endpoint}
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              {error.details}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

// API Performance Overview
export const ApiPerformanceChart = ({ data }) => {
  if (!data?.current) return <Typography>No API metrics data</Typography>;

  const metrics = [
    { name: 'Total Predictions', value: data.current.total_predictions, color: '#4ECDC4' },
    { name: 'Total Errors', value: data.current.total_errors, color: '#FF6B6B' },
    { name: 'Prediction Rate/min', value: data.current.prediction_rate, color: '#96CEB4' },
    { name: 'Uptime (hours)', value: data.current.uptime_hours, color: '#FECA57' }
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>API Performance Overview</Typography>

      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={metrics}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="name" stroke="#fff" fontSize={10} />
          <YAxis stroke="#fff" fontSize={10} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px'
            }}
          />
          <Bar dataKey="value" fill="#4ECDC4" />
        </BarChart>
      </ResponsiveContainer>

      {/* Latency Metrics */}
      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip
          label={`P50: ${(data.current.latency_p50 * 1000).toFixed(1)}ms`}
          sx={{ backgroundColor: '#54A0FF', color: 'white' }}
        />
        <Chip
          label={`P95: ${(data.current.latency_p95 * 1000).toFixed(1)}ms`}
          sx={{ backgroundColor: '#5F27CD', color: 'white' }}
        />
        <Chip
          label={`P99: ${(data.current.latency_p99 * 1000).toFixed(1)}ms`}
          sx={{ backgroundColor: '#FF9FF3', color: 'white' }}
        />
        <Chip
          label={`Error Rate: ${data.current.error_rate.toFixed(2)}%`}
          color={data.current.error_rate > 1 ? 'error' : 'success'}
        />
      </Box>
    </Box>
  );
};
