"""
Monitoring and metrics collection for the emotion classification pipeline.

This module provides comprehensive monitoring capabilities including:
- API performance metrics (latency, throughput, error rates)
- Model performance tracking (accuracy, F1 scores)
- Data drift detection (feature distribution changes)
- Concept drift detection (prediction distribution changes)
- System resource monitoring (CPU, memory, disk usage)
- Error tracking and logging
"""

import functools
import json
import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Set up logging
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring system.

    Tracks API performance, model accuracy, data/concept drift,
    and system resources. Provides Prometheus-compatible metrics
    and saves monitoring data to JSON files for dashboard consumption.
    """

    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.05):
        """Initialize the metrics collector."""
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.start_time = datetime.now()

        # Initialize counters and histograms
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)

        # Track active requests
        self.active_requests = 0
        self.total_predictions = 0
        self.total_errors = 0

        # History for drift detection
        self.prediction_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)

        # Model performance tracking
        self.model_performance = {}

        # System metrics
        self.system_stats = {}

        # Monitoring data directory
        self.monitoring_dir = Path("results/monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring files
        self._initialize_monitoring_files()

        # Load baseline statistics for drift detection
        self.baseline_stats = self._load_baseline_stats()

        # Start background monitoring
        self._start_background_monitoring()

    def _initialize_monitoring_files(self):
        """Initialize JSON files for monitoring data storage."""
        files_to_init = [
            "api_metrics.json",
            "model_performance.json",
            "drift_detection.json",
            "system_metrics.json",
            "error_tracking.json",
            "prediction_logs.json",
        ]

        for filename in files_to_init:
            filepath = self.monitoring_dir / filename
            if not filepath.exists():
                with open(filepath, "w") as f:
                    json.dump([], f)

    def _load_baseline_stats(self) -> Dict[str, Any]:
        """Load baseline statistics for drift detection."""
        try:
            baseline_path = "models/baseline_stats.pkl"
            if os.path.exists(baseline_path):
                with open(baseline_path, "rb") as f:
                    return pickle.load(f)
            else:
                logger.warning("No baseline stats found for drift detection")
                return {
                    "feature_means": {},
                    "feature_stds": {},
                    "prediction_distribution": {},
                    "performance_baseline": {},
                }
        except Exception as e:
            logger.error(f"Failed to load baseline stats: {e}")
            return {}

    def record_prediction(
        self,
        prediction_data: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
        latency: float = 0.0,
        confidence: float = 0.0,
    ):
        """Record a prediction for monitoring and drift detection."""
        try:
            self.total_predictions += 1

            # Store prediction in history for drift detection
            prediction_record = {
                "timestamp": datetime.now().isoformat(),
                "emotion": prediction_data.get("emotion", "unknown"),
                "sub_emotion": prediction_data.get("sub_emotion", "unknown"),
                "intensity": prediction_data.get("intensity", "unknown"),
                "confidence": confidence,
                "latency": latency,
            }

            self.prediction_history.append(prediction_record)

            # Store features if provided
            if features:
                feature_record = {
                    "timestamp": datetime.now().isoformat(),
                    "features": features,
                }
                self.feature_history.append(feature_record)

            # Record latency
            self.histograms["prediction_latency"].append(latency)

            # Save to prediction log
            self._save_prediction_log(prediction_record)

            # Update API metrics periodically
            if self.total_predictions % 10 == 0:
                self._update_api_metrics()

            # Check for drift periodically
            if self.total_predictions % 50 == 0:
                self._detect_and_save_drift()

        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")

    def record_transcription(
        self,
        transcript_data: Dict[str, Any],
        latency: float = 0.0,
        audio_quality: float = 0.0,
        confidence: float = 0.0,
    ):
        """Record transcription metrics."""
        try:
            self.histograms["transcription_latency"].append(latency)
            self.histograms["audio_quality"].append(audio_quality)
            self.histograms["transcription_confidence"].append(confidence)
        except Exception as e:
            logger.error(f"Failed to record transcription: {e}")

    def record_error(self, error_type: str, endpoint: str, error_details: str = ""):
        """Record an error occurrence."""
        try:
            self.total_errors += 1
            self.counters[f"error_{error_type}"] += 1

            error_record = {
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "endpoint": endpoint,
                "details": error_details,
            }

            self._save_error_log(error_record)

        except Exception as e:
            logger.error(f"Failed to record error: {e}")

    def update_model_performance(self, performance_metrics: Dict[str, float]):
        """Update model performance metrics."""
        try:
            self.model_performance.update(performance_metrics)

            performance_record = {
                "timestamp": datetime.now().isoformat(),
                **performance_metrics,
            }

            self._save_model_performance(performance_record)

        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")

    def start_request(self) -> str:
        """Mark the start of a request."""
        self.active_requests += 1
        return f"req_{datetime.now().timestamp()}"

    def end_request(self, request_id: str):
        """Mark the end of a request."""
        self.active_requests = max(0, self.active_requests - 1)

    def record_system_metrics(self):
        """Record current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            system_record = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
            }

            self.system_stats = system_record
            self._save_system_metrics(system_record)

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def _start_background_monitoring(self):
        """Start background thread for periodic monitoring tasks."""

        def background_monitor():
            while True:
                try:
                    time.sleep(30)  # Run every 30 seconds
                    self.record_system_metrics()
                    self._update_api_metrics()
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")

        monitor_thread = threading.Thread(target=background_monitor, daemon=True)
        monitor_thread.start()

    def _detect_and_save_drift(self):
        """Detect data and concept drift and save results."""
        try:
            data_drift_score = self._calculate_data_drift()
            concept_drift_score = self._calculate_concept_drift()

            drift_record = {
                "timestamp": datetime.now().isoformat(),
                "data_drift_score": data_drift_score,
                "concept_drift_score": concept_drift_score,
                "data_drift_alert": data_drift_score > self.drift_threshold,
                "concept_drift_alert": concept_drift_score > self.drift_threshold,
                "drift_threshold": self.drift_threshold,
            }

            self._save_drift_detection(drift_record)

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")

    def _calculate_data_drift(self) -> float:
        """Calculate data drift score using feature distributions."""
        if len(self.feature_history) < 50:
            return 0.0

        try:
            # Get recent features
            recent_features = list(self.feature_history)[-50:]

            # Calculate feature means
            feature_means = {}
            for record in recent_features:
                for feature, value in record["features"].items():
                    if feature not in feature_means:
                        feature_means[feature] = []
                    feature_means[feature].append(value)

            # Compare with baseline
            drift_scores = []
            baseline_means = self.baseline_stats.get("feature_means", {})

            for feature, values in feature_means.items():
                if feature in baseline_means:
                    recent_mean = np.mean(values)
                    baseline_mean = baseline_means[feature]

                    # Simple drift score based on relative change
                    if baseline_mean != 0:
                        drift_score = abs(recent_mean - baseline_mean) / abs(
                            baseline_mean
                        )
                        drift_scores.append(drift_score)

            return np.mean(drift_scores) if drift_scores else 0.0

        except Exception as e:
            logger.error(f"Data drift calculation failed: {e}")
            return 0.0

    def _calculate_concept_drift(self) -> float:
        """Calculate concept drift score using prediction distributions."""
        if len(self.prediction_history) < 50:
            return 0.0

        try:
            # Get recent predictions
            recent_predictions = list(self.prediction_history)[-50:]

            # Calculate current emotion distribution
            current_dist = {}
            for pred in recent_predictions:
                emotion = pred["emotion"]
                current_dist[emotion] = current_dist.get(emotion, 0) + 1

            # Normalize
            total = sum(current_dist.values())
            current_dist = {k: v / total for k, v in current_dist.items()}

            # Compare with baseline distribution
            baseline_dist = self.baseline_stats.get("prediction_distribution", {})

            if baseline_dist:
                return self._jensen_shannon_divergence(current_dist, baseline_dist)

            return 0.0

        except Exception as e:
            logger.error(f"Concept drift calculation failed: {e}")
            return 0.0

    def _jensen_shannon_divergence(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())

        p = np.array([dist1.get(k, 0) for k in all_keys])
        q = np.array([dist2.get(k, 0) for k in all_keys])

        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
        return js_divergence

    def _save_prediction_log(self, prediction_record: Dict[str, Any]):
        """Save individual prediction to log file."""
        try:
            filepath = self.monitoring_dir / "prediction_logs.json"
            self._append_to_json_file(filepath, prediction_record)
        except Exception as e:
            logger.error(f"Failed to save prediction log: {e}")

    def _save_error_log(self, error_record: Dict[str, Any]):
        """Save error to log file."""
        try:
            filepath = self.monitoring_dir / "error_tracking.json"
            self._append_to_json_file(filepath, error_record)
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")

    def _save_model_performance(self, performance_record: Dict[str, Any]):
        """Save model performance metrics."""
        try:
            filepath = self.monitoring_dir / "model_performance.json"
            self._append_to_json_file(filepath, performance_record)
        except Exception as e:
            logger.error(f"Failed to save model performance: {e}")

    def _save_drift_detection(self, drift_record: Dict[str, Any]):
        """Save drift detection results."""
        try:
            filepath = self.monitoring_dir / "drift_detection.json"
            self._append_to_json_file(filepath, drift_record)
        except Exception as e:
            logger.error(f"Failed to save drift detection: {e}")

    def _save_system_metrics(self, system_record: Dict[str, Any]):
        """Save system metrics."""
        try:
            filepath = self.monitoring_dir / "system_metrics.json"
            self._append_to_json_file(filepath, system_record)
        except Exception as e:
            logger.error(f"Failed to save system metrics: {e}")

    def _update_api_metrics(self):
        """Update and save API performance metrics."""
        try:
            timestamp = datetime.now()
            uptime = (timestamp - self.start_time).total_seconds()

            # Calculate rates
            prediction_rate = self.total_predictions / max(uptime, 1) * 60  # per minute
            error_rate = self.total_errors / max(self.total_predictions, 1) * 100

            # Calculate latency statistics
            latencies = self.histograms.get("prediction_latency", [])
            if latencies:
                p50_latency = np.percentile(
                    latencies[-100:], 50
                )  # Last 100 predictions
                p95_latency = np.percentile(latencies[-100:], 95)
                p99_latency = np.percentile(latencies[-100:], 99)
            else:
                p50_latency = p95_latency = p99_latency = 0.0

            api_record = {
                "timestamp": timestamp.isoformat(),
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "active_requests": self.active_requests,
                "prediction_rate_per_minute": prediction_rate,
                "error_rate_percent": error_rate,
                "uptime_seconds": uptime,
                "latency_p50": p50_latency,
                "latency_p95": p95_latency,
                "latency_p99": p99_latency,
            }

            filepath = self.monitoring_dir / "api_metrics.json"
            self._append_to_json_file(filepath, api_record)

        except Exception as e:
            logger.error(f"Failed to update API metrics: {e}")

    def _append_to_json_file(self, filepath: Path, record: Dict[str, Any]):
        """Append a record to a JSON file."""
        try:
            # Read existing data
            if filepath.exists() and filepath.stat().st_size > 0:
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    # File is corrupted, start fresh
                    logger.warning(f"Corrupted JSON file {filepath}, reinitializing")
                    data = []
            else:
                data = []

            # Add new record
            data.append(record)

            # Keep only last 1000 records to prevent files from growing too large
            if len(data) > 1000:
                data = data[-1000:]

            # Write back to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=self._json_serializer)

        except Exception as e:
            logger.error(f"Failed to append to {filepath}: {e}")
            # Try to reinitialize the file
            try:
                with open(filepath, "w") as f:
                    json.dump([record], f, indent=2, default=self._json_serializer)
                logger.info(f"Reinitialized corrupted file {filepath}")
            except Exception as e2:
                logger.error(f"Failed to reinitialize {filepath}: {e2}")

    def generate_daily_summary(self):
        """Generate a daily summary of monitoring metrics."""
        try:
            summary = {
                "date": datetime.now().date().isoformat(),
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "error_rate_percent": (
                    self.total_errors / max(self.total_predictions, 1)
                )
                * 100,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds()
                / 3600,
                "avg_latency": np.mean(self.histograms.get("prediction_latency", [0])),
                "p95_latency": np.percentile(
                    self.histograms.get("prediction_latency", [0]), 95
                ),
                "model_performance": self.model_performance.copy(),
                "system_stats": self.system_stats.copy(),
            }

            filepath = self.monitoring_dir / "daily_summary.json"

            # Save as single object (not array)
            with open(filepath, "w") as f:
                json.dump(summary, f, indent=2, default=self._json_serializer)

        except Exception as e:
            logger.error(f"Failed to generate daily summary: {e}")

    def _json_serializer(self, obj):
        """JSON serializer for datetime and other objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return str(obj)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all metrics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()

            summary = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "api_metrics": {
                    "total_predictions": self.total_predictions,
                    "total_errors": self.total_errors,
                    "active_requests": self.active_requests,
                    "prediction_rate_per_minute": (
                        self.total_predictions / max(uptime, 1)
                    )
                    * 60,
                    "error_rate_percent": (
                        self.total_errors / max(self.total_predictions, 1)
                    )
                    * 100,
                },
                "latency_metrics": {
                    "avg": np.mean(self.histograms.get("prediction_latency", [0])),
                    "p50": np.percentile(
                        self.histograms.get("prediction_latency", [0]), 50
                    ),
                    "p95": np.percentile(
                        self.histograms.get("prediction_latency", [0]), 95
                    ),
                    "p99": np.percentile(
                        self.histograms.get("prediction_latency", [0]), 99
                    ),
                },
                "model_performance": self.model_performance.copy(),
                "system_metrics": self.system_stats.copy(),
                "drift_scores": {
                    "data_drift": self._calculate_data_drift(),
                    "concept_drift": self._calculate_concept_drift(),
                    "threshold": self.drift_threshold,
                },
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}")
            return {"error": str(e)}

    # Prometheus-compatible metric properties
    @property
    def prediction_counter(self):
        """Mock Prometheus counter for predictions."""
        return MockCounter(self, "predictions_total")

    @property
    def prediction_latency(self):
        """Mock Prometheus histogram for prediction latency."""
        return MockHistogram(self, "prediction_latency")

    @property
    def transcription_latency(self):
        """Mock Prometheus histogram for transcription latency."""
        return MockHistogram(self, "transcription_latency")

    @property
    def model_confidence(self):
        """Mock Prometheus histogram for model confidence."""
        return MockHistogram(self, "model_confidence")


class MockCounter:
    """Mock Prometheus Counter for compatibility."""

    def __init__(self, collector: MetricsCollector, metric_name: str):
        self.collector = collector
        self.metric_name = metric_name

    def labels(self, **kwargs):
        return self

    def inc(self, amount: float = 1):
        self.collector.counters[self.metric_name] += amount


class MockHistogram:
    """Mock Prometheus Histogram for compatibility."""

    def __init__(self, collector: MetricsCollector, metric_name: str):
        self.collector = collector
        self.metric_name = metric_name

    def labels(self, **kwargs):
        return self

    def observe(self, value: float):
        self.collector.histograms[self.metric_name].append(value)


class RequestTracker:
    """Context manager for tracking API requests."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or metrics_collector
        self.request_id = None

    def __enter__(self):
        self.request_id = self.collector.start_request()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.request_id:
            self.collector.end_request(self.request_id)

        # Record error if exception occurred
        if exc_type:
            self.collector.record_error(
                error_type=exc_type.__name__,
                endpoint="unknown",
                error_details=str(exc_val),
            )

    def record_request(self, request_data: Dict[str, Any]):
        """Record request data for monitoring."""
        pass  # Implementation can be added as needed


def monitoring_trace(operation_name: str):
    """Decorator to automatically trace function execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time

                # Record successful operation
                metrics_collector.histograms[f"{operation_name}_latency"].append(
                    latency
                )
                metrics_collector.counters[f"{operation_name}_success"] += 1

                return result
            except Exception as e:
                latency = time.time() - start_time

                # Record failed operation
                metrics_collector.histograms[f"{operation_name}_latency"].append(
                    latency
                )
                metrics_collector.record_error(
                    error_type=type(e).__name__,
                    endpoint=operation_name,
                    error_details=str(e),
                )
                raise

        return wrapper

    return decorator


def _periodic_tasks():
    """Background tasks for periodic monitoring operations."""
    try:
        # Generate daily summary
        metrics_collector.generate_daily_summary()

        # Clean up old monitoring files if needed
        # Could add file rotation logic here

    except Exception as e:
        logger.error(f"Periodic tasks failed: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Start periodic background tasks
import atexit

atexit.register(_periodic_tasks)
