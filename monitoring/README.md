# 📊 Emotion Classification API - Monitoring Stack

## Overview

This monitoring stack provides comprehensive observability for the Emotion Classification API, including model performance tracking, system metrics, and real-time dashboards.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Prometheus    │
│   (React)       │────│   (FastAPI)     │────│   (Metrics)     │
│   Port 3121     │    │   Port 3120     │    │   Port 3122     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │  Node Exporter  │    │   cAdvisor      │
│  (Dashboard)    │────│ (System Metrics)│    │(Container Stats)│
│   Port 3123     │    │   Internal      │    │   Internal      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📈 Metrics Collected

### 🤖 Model & ML Metrics
- **Emotion Predictions**: `emotion_predictions_total{emotion, intensity, sub_emotion}`
- **Prediction Latency**: `emotion_prediction_latency_seconds`
- **Model Confidence**: `model_confidence_distribution`
- **Accuracy Tracking**: `model_accuracy_score`, `model_f1_score`
- **Drift Detection**: `data_drift_score`, `concept_drift_score`

### 🎵 Audio Processing Metrics
- **Transcription Latency**: `transcription_latency_seconds`
- **Audio Quality**: `audio_quality_score`
- **Transcript Confidence**: `transcript_confidence_score`

### 🌐 API Performance Metrics
- **Request Tracking**: `active_requests_count`
- **Error Rates**: `api_errors_total`
- **Response Times**: Built into FastAPI metrics

### 🖥️ System & Infrastructure Metrics
- **CPU Usage**: Via Node Exporter
- **Memory Usage**: Via Node Exporter
- **Container Stats**: Via cAdvisor
- **Python GC**: `python_gc_*`
- **Process Metrics**: `process_*`

## 🚀 Quick Start

### Development Build & Deploy
```bash
# Build images and start all services
docker-compose -f docker-compose.build.yml up -d

# View logs
docker-compose -f docker-compose.build.yml logs -f
```

### Production Deploy (Pre-built Images)
```bash
# Deploy with pre-built images
docker-compose up -d

# Health check
docker-compose ps
```

## 🌐 Service Access

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Frontend** | http://localhost:3121 | None | Video analysis interface |
| **Backend API** | http://localhost:3120 | None | REST API endpoints |
| **Prometheus** | http://localhost:3122 | None | Metrics query interface |
| **Grafana** | http://localhost:3123 | admin/admin123 | Monitoring dashboards |

## 📊 Dashboard Configuration

### Pre-configured Dashboards
- **Emotion Analysis Overview**: Model predictions, latency, accuracy
- **System Performance**: CPU, memory, container health
- **API Monitoring**: Request rates, error tracking, response times
- **Data Quality**: Audio quality, transcript confidence, drift detection

### Custom Queries
```promql
# Average prediction latency over 5 minutes
rate(emotion_prediction_latency_seconds_sum[5m]) / rate(emotion_prediction_latency_seconds_count[5m])

# Top emotions predicted in last hour
topk(5, increase(emotion_predictions_total[1h]))

# System memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# API error rate
rate(api_errors_total[5m]) / rate(http_requests_total[5m]) * 100
```

## 🔧 Configuration Files

### Prometheus Configuration
- **Location**: `./prometheus/prometheus.yml`
- **Scrape Targets**: Backend (:3120/metrics), Node Exporter (:9100), cAdvisor (:8080)
- **Retention**: 200 hours
- **Scrape Interval**: 15 seconds

### Grafana Configuration
- **Provisioning**: `./grafana/provisioning/`
- **Dashboards**: `./grafana/dashboards/`
- **Data Sources**: Auto-configured Prometheus connection
- **Plugins**: Pre-installed visualization plugins

## 🛠️ Development & Customization

### Adding New Metrics
1. **Backend**: Add metrics to `src/emotion_clf_pipeline/monitoring.py`
2. **Prometheus**: Metrics auto-discovered via `/metrics` endpoint
3. **Grafana**: Create panels using PromQL queries

### Scaling Considerations
```yaml
# For high-load environments, consider:
prometheus:
  command:
    - '--storage.tsdb.retention.time=720h'  # Longer retention
    - '--query.max-concurrency=50'         # More concurrent queries

grafana:
  environment:
    - GF_DATABASE_TYPE=postgres           # External database
    - GF_SESSION_PROVIDER=redis           # Session clustering
```

### Alert Rules Example
```yaml
# prometheus/alerts.yml
groups:
  - name: emotion_api_alerts
    rules:
      - alert: HighPredictionLatency
        expr: rate(emotion_prediction_latency_seconds_sum[5m]) / rate(emotion_prediction_latency_seconds_count[5m]) > 30
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"

      - alert: ModelDriftDetected
        expr: data_drift_score > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model data drift detected"
```

## 🐛 Troubleshooting

### Common Issues

**1. Metrics not appearing in Prometheus**
```bash
# Check if backend is exposing metrics
curl http://localhost:3120/metrics

# Verify Prometheus targets
curl http://localhost:3122/api/v1/targets
```

**2. Grafana dashboard not loading**
```bash
# Check Grafana logs
docker-compose logs grafana

# Verify data source connection
# Grafana UI → Configuration → Data Sources → Prometheus → Test
```

**3. Container health issues**
```bash
# Check all container status
docker-compose ps

# Inspect specific container
docker-compose logs [service_name]

# Restart specific service
docker-compose restart [service_name]
```

### Performance Optimization

**For High-Volume Usage:**
```yaml
# Increase Prometheus storage
prometheus:
  volumes:
    - prometheus_data:/prometheus:Z
  command:
    - '--storage.tsdb.max-block-duration=2h'
    - '--storage.tsdb.min-block-duration=5m'

# Optimize Grafana
grafana:
  environment:
    - GF_RENDERING_SERVER_URL=http://renderer:8081/render
    - GF_RENDERING_CALLBACK_URL=http://grafana:3000/
    - GF_FEATURE_TOGGLES_ENABLE=ngalert
```

## 📚 Additional Resources

### Documentation Links
- [Prometheus Query Language (PromQL)](https://prometheus.io/docs/prometheus/latest/querying/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
- [FastAPI Metrics Integration](https://fastapi.tiangolo.com/advanced/sub-applications/)

### Monitoring Best Practices
1. **Use histograms** for latency measurements
2. **Set appropriate retention** policies for metrics
3. **Create meaningful alerts** based on business impact
4. **Monitor the monitoring stack** itself
5. **Regular backup** of Grafana dashboards and Prometheus data

### Support & Maintenance
- **Logs Location**: `/var/lib/docker/containers/*/`
- **Data Persistence**: Docker volumes `prometheus_data`, `grafana_data`
- **Backup Strategy**: Regular volume snapshots recommended
- **Updates**: Monitor upstream image updates for security patches

---

**📞 Need Help?**
Check the main project README or open an issue for support with monitoring setup and configuration.
