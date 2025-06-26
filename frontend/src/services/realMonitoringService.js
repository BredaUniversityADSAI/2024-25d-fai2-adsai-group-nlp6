/**
 * Real Monitoring Data Service
 * Fetches actual monitoring data from backend JSON files - NO SIMULATED DATA
 */

import axios from 'axios';

// Get API base URL
const getApiBaseUrl = () => {
  const { protocol, hostname } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:3120';
  }
  return `${protocol}//${hostname}:3120`;
};

const API_BASE_URL = getApiBaseUrl();

class RealMonitoringService {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 30000; // 30 seconds
  }

  async fetchMonitoringData(fileName) {
    const cacheKey = fileName;
    const now = Date.now();

    if (this.cache.has(cacheKey)) {
      const { data, timestamp } = this.cache.get(cacheKey);
      if (now - timestamp < this.cacheTimeout) {
        console.log(`Using cached data for ${fileName}`);
        return data;
      }
    }

    try {
      console.log(`Fetching ${fileName} from API...`);
      const response = await Promise.race([
        axios.get(`${API_BASE_URL}/monitoring/${fileName}`),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 5000)
        )
      ]);

      console.log(`API response for ${fileName}:`, {
        status: response.status,
        dataType: typeof response.data,
        isArray: Array.isArray(response.data),
        dataLength: response.data?.length,
        firstItem: response.data?.[0]
      });

      const data = response.data;
      this.cache.set(cacheKey, { data, timestamp: now });
      return data;
    } catch (error) {
      console.warn(`Failed to fetch ${fileName}:`, error.message);
      return [];
    }
  }

  async getAllMonitoringData() {
    const files = [
      'model_performance.json',
      'api_metrics.json',
      'system_metrics.json',
      'prediction_logs.json',
      'drift_detection.json',
      'error_tracking.json',
      'daily_summary.json'
    ];

    console.log('Fetching monitoring files:', files);

    const results = await Promise.allSettled(
      files.map(file => this.fetchMonitoringData(file))
    );

    console.log('Fetch results:', results.map((result, i) => ({
      file: files[i],
      status: result.status,
      dataLength: result.status === 'fulfilled' ? result.value?.length : 'N/A',
      hasData: result.status === 'fulfilled' && result.value && result.value.length > 0
    })));

    const monitoringData = {
      modelPerformance: results[0].status === 'fulfilled' ? results[0].value : [],
      apiMetrics: results[1].status === 'fulfilled' ? results[1].value : [],
      systemMetrics: results[2].status === 'fulfilled' ? results[2].value : [],
      predictionLogs: results[3].status === 'fulfilled' ? results[3].value : [],
      driftDetection: results[4].status === 'fulfilled' ? results[4].value : [],
      errorTracking: results[5].status === 'fulfilled' ? results[5].value : [],
      dailySummary: results[6].status === 'fulfilled' ? results[6].value : []
    };

    console.log('Final monitoring data structure:', {
      modelPerformance: monitoringData.modelPerformance?.length || 0,
      apiMetrics: monitoringData.apiMetrics?.length || 0,
      systemMetrics: monitoringData.systemMetrics?.length || 0,
      predictionLogs: monitoringData.predictionLogs?.length || 0,
      driftDetection: monitoringData.driftDetection?.length || 0,
      errorTracking: monitoringData.errorTracking?.length || 0,
      dailySummary: monitoringData.dailySummary?.length || 0
    });

    return monitoringData;
  }

  // Analysis functions for dashboard
  analyzeModelPerformance(data) {
    if (!data || data.length === 0) return null;

    const latest = data[data.length - 1];
    const timeSeries = data.slice(-20).map(item => ({
      timestamp: new Date(item.timestamp).toLocaleTimeString(),
      emotion_f1: item.emotion?.f1 || item.f1_score || 0,
      sub_emotion_f1: item.sub_emotion?.f1 || 0,
      intensity_f1: item.intensity?.f1 || 0,
      overall_f1: item.f1_score || (item.emotion?.f1 + item.sub_emotion?.f1 + item.intensity?.f1) / 3 || 0
    }));

    return {
      current: {
        emotion_f1: latest.emotion?.f1 || latest.f1_score || 0,
        sub_emotion_f1: latest.sub_emotion?.f1 || 0,
        intensity_f1: latest.intensity?.f1 || 0,
        overall_accuracy: latest.accuracy || 0
      },
      timeSeries,
      trend: this.calculateTrend(timeSeries.map(d => d.overall_f1))
    };
  }

  analyzeSystemMetrics(data) {
    console.log('analyzeSystemMetrics called with data:', data);
    console.log('Data type:', typeof data, 'Array?', Array.isArray(data), 'Length:', data?.length);

    if (!data || data.length === 0) {
      console.log('No system metrics data - returning null');
      return null;
    }

    const latest = data[data.length - 1];
    console.log('Latest system metrics entry:', latest);

    const timeSeries = data.slice(-50).map(item => ({
      timestamp: new Date(item.timestamp).toLocaleTimeString(),
      cpu: item.cpu_percent,
      memory: item.memory_percent,
      disk: item.disk_percent
    }));

    console.log('Generated time series:', timeSeries.slice(0, 3)); // Log first 3 entries

    const result = {
      current: {
        cpu: latest.cpu_percent,
        memory: latest.memory_percent,
        disk: latest.disk_percent,
        memory_used_gb: latest.memory_used_gb,
        disk_used_gb: latest.disk_used_gb
      },
      timeSeries,
      averages: {
        cpu: this.calculateAverage(timeSeries.map(d => d.cpu)),
        memory: this.calculateAverage(timeSeries.map(d => d.memory)),
        disk: this.calculateAverage(timeSeries.map(d => d.disk))
      }
    };

    console.log('System metrics analysis result:', result);
    return result;
  }

  analyzeApiMetrics(data) {
    if (!data || data.length === 0) return null;

    const latest = data[data.length - 1];
    return {
      current: {
        total_predictions: latest.total_predictions || 0,
        total_errors: latest.total_errors || 0,
        error_rate: latest.error_rate_percent || 0,
        prediction_rate: latest.prediction_rate_per_minute || 0,
        uptime_hours: Math.round((latest.uptime_seconds || 0) / 3600 * 10) / 10,
        latency_p50: latest.latency_p50 || 0,
        latency_p95: latest.latency_p95 || 0,
        latency_p99: latest.latency_p99 || 0
      }
    };
  }

  analyzePredictionLogs(data) {
    if (!data || data.length === 0) return null;

    // Emotion distribution
    const emotionCounts = {};
    const subEmotionCounts = {};
    const intensityCounts = {};
    const latencies = [];
    const confidences = [];

    data.forEach(log => {
      // Count emotions
      emotionCounts[log.emotion] = (emotionCounts[log.emotion] || 0) + 1;
      subEmotionCounts[log.sub_emotion] = (subEmotionCounts[log.sub_emotion] || 0) + 1;
      intensityCounts[log.intensity] = (intensityCounts[log.intensity] || 0) + 1;

      // Collect metrics
      if (log.latency) latencies.push(log.latency);
      if (log.confidence !== undefined) confidences.push(log.confidence);
    });

    // Convert to chart-friendly format
    const emotionDistribution = Object.entries(emotionCounts).map(([emotion, count]) => ({
      name: emotion,
      value: count,
      percentage: Math.round((count / data.length) * 100)
    }));

    const subEmotionDistribution = Object.entries(subEmotionCounts).map(([emotion, count]) => ({
      name: emotion,
      value: count,
      percentage: Math.round((count / data.length) * 100)
    }));

    // Recent latency trend (last 50 predictions)
    const recentData = data.slice(-50);
    const latencyTrend = recentData.map((log, index) => ({
      prediction: index + 1,
      latency: log.latency || 0,
      timestamp: new Date(log.timestamp).toLocaleTimeString()
    }));

    return {
      totalPredictions: data.length,
      emotionDistribution,
      subEmotionDistribution,
      intensityDistribution: Object.entries(intensityCounts).map(([intensity, count]) => ({
        name: intensity,
        value: count,
        percentage: Math.round((count / data.length) * 100)
      })),
      latencyStats: {
        avg: this.calculateAverage(latencies),
        min: Math.min(...latencies),
        max: Math.max(...latencies),
        p50: this.calculatePercentile(latencies, 50),
        p95: this.calculatePercentile(latencies, 95)
      },
      confidenceStats: {
        avg: this.calculateAverage(confidences),
        min: Math.min(...confidences),
        max: Math.max(...confidences)
      },
      latencyTrend
    };
  }

  analyzeDriftDetection(data) {
    if (!data || data.length === 0) return null;

    const latest = data[data.length - 1];
    const timeSeries = data.map(item => ({
      timestamp: new Date(item.timestamp).toLocaleTimeString(),
      data_drift: item.data_drift_score,
      concept_drift: item.concept_drift_score,
      threshold: item.drift_threshold
    }));

    return {
      current: {
        data_drift_score: latest.data_drift_score,
        concept_drift_score: latest.concept_drift_score,
        data_drift_alert: latest.data_drift_alert,
        concept_drift_alert: latest.concept_drift_alert,
        threshold: latest.drift_threshold
      },
      timeSeries,
      alertCount: {
        data_drift: data.filter(d => d.data_drift_alert).length,
        concept_drift: data.filter(d => d.concept_drift_alert).length
      }
    };
  }

  analyzeErrorTracking(data) {
    if (!data || data.length === 0) return null;

    const errorTypes = {};
    const endpoints = {};

    data.forEach(error => {
      errorTypes[error.error_type] = (errorTypes[error.error_type] || 0) + 1;
      endpoints[error.endpoint] = (endpoints[error.endpoint] || 0) + 1;
    });

    return {
      totalErrors: data.length,
      errorTypes: Object.entries(errorTypes).map(([type, count]) => ({
        name: type,
        value: count
      })),
      affectedEndpoints: Object.entries(endpoints).map(([endpoint, count]) => ({
        name: endpoint,
        value: count
      })),
      recentErrors: data.slice(-10).map(error => ({
        timestamp: new Date(error.timestamp).toLocaleString(),
        type: error.error_type,
        endpoint: error.endpoint,
        details: error.error_details
      }))
    };
  }

  // Utility functions
  calculateAverage(values) {
    if (!values || values.length === 0) return 0;
    return Math.round((values.reduce((sum, val) => sum + val, 0) / values.length) * 100) / 100;
  }

  calculatePercentile(values, percentile) {
    if (!values || values.length === 0) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index] || 0;
  }

  calculateTrend(values) {
    if (!values || values.length < 2) return 'stable';
    const recent = values.slice(-5);
    const avg1 = this.calculateAverage(recent.slice(0, Math.floor(recent.length / 2)));
    const avg2 = this.calculateAverage(recent.slice(Math.floor(recent.length / 2)));
    const diff = avg2 - avg1;

    if (Math.abs(diff) < 0.01) return 'stable';
    return diff > 0 ? 'improving' : 'declining';
  }

  // Health status calculation
  calculateOverallHealth(systemMetrics, apiMetrics, driftDetection) {
    let score = 100;

    // System health
    if (systemMetrics?.current) {
      if (systemMetrics.current.cpu > 80) score -= 20;
      else if (systemMetrics.current.cpu > 60) score -= 10;

      if (systemMetrics.current.memory > 85) score -= 20;
      else if (systemMetrics.current.memory > 70) score -= 10;

      if (systemMetrics.current.disk > 90) score -= 15;
      else if (systemMetrics.current.disk > 80) score -= 5;
    }

    // API health
    if (apiMetrics?.current) {
      if (apiMetrics.current.error_rate > 5) score -= 25;
      else if (apiMetrics.current.error_rate > 1) score -= 10;

      if (apiMetrics.current.latency_p95 > 2) score -= 15;
      else if (apiMetrics.current.latency_p95 > 1) score -= 5;
    }

    // Drift alerts
    if (driftDetection?.current) {
      if (driftDetection.current.concept_drift_alert) score -= 20;
      if (driftDetection.current.data_drift_alert) score -= 15;
    }

    score = Math.max(0, Math.min(100, score));

    if (score >= 90) return { status: 'excellent', score, color: '#4CAF50' };
    if (score >= 75) return { status: 'good', score, color: '#8BC34A' };
    if (score >= 60) return { status: 'fair', score, color: '#FF9800' };
    if (score >= 40) return { status: 'poor', score, color: '#FF5722' };
    return { status: 'critical', score, color: '#F44336' };
  }
}

export default new RealMonitoringService();
