"""
Enhanced monitoring module for emotion classification pipeline.

Provides comprehensive metrics collection for:
- Model performance monitoring    - Data drift detection
    - Concept drift tracking
- Prediction latency and throughput
- System resource utilization

Key Features:
    - Prometheus metrics integration
    - Real-time drift detection using statistical methods
    - Performance benchmarking with historical baselines
    - Automated alerting for anomaly detection
"""

import functools
import logging
import os
import pickle
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from scipy import stats

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collection system for ML pipeline monitoring.

    Tracks performance, drift, and system metrics with Prometheus integration.
    """

    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.05):
        """
        Initialize metrics collector with configurable parameters.

        Args:
            window_size: Number of recent predictions to keep for drift detection
            drift_threshold: Statistical significance threshold for drift detection
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Data storage for drift detection
        self.prediction_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=100)  # Last 100 evaluation cycles

        # Baseline statistics (loaded from training data)
        self.baseline_stats = self._load_baseline_stats()

        # Threading lock for thread-safe operations
        self._lock = threading.Lock()

        logger.info(f"MetricsCollector initialized with window_size={window_size}")

    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metrics."""
        # Prediction metrics
        self.prediction_counter = Counter(
            "emotion_predictions_total",
            "Total number of emotion predictions made",
            ["emotion", "sub_emotion", "intensity"],
        )

        self.prediction_latency = Histogram(
            "emotion_prediction_latency_seconds",
            "Time taken for emotion prediction",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        self.transcription_latency = Histogram(
            "transcription_latency_seconds",
            "Time taken for audio transcription",
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # Model performance metrics
        self.model_accuracy = Gauge(
            "model_accuracy_score", "Current model accuracy score", ["task"]
        )

        self.model_f1_score = Gauge(
            "model_f1_score", "Current model F1 score", ["task"]
        )

        self.model_confidence = Histogram(
            "model_confidence_distribution",
            "Distribution of model confidence scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Drift detection metrics
        self.data_drift_score = Gauge(
            "data_drift_score", "Statistical measure of input data drift"
        )

        self.concept_drift_score = Gauge(
            "concept_drift_score", "Statistical measure of prediction concept drift"
        )

        self.drift_alert = Counter(
            "drift_alerts_total", "Number of drift alerts triggered", ["drift_type"]
        )

        # System metrics
        self.active_requests = Gauge(
            "active_requests_count", "Number of currently active requests"
        )

        self.error_counter = Counter(
            "api_errors_total", "Total number of API errors", ["error_type", "endpoint"]
        )

        # Data quality metrics
        self.audio_quality_score = Histogram(
            "audio_quality_score",
            "Audio quality assessment scores",
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        self.transcript_confidence = Histogram(
            "transcript_confidence_score",
            "Transcription confidence scores",
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

    def _load_baseline_stats(self) -> Dict[str, Any]:
        """Load baseline statistics from training data."""
        baseline_paths = [
            "models/baseline_stats.pkl",
            "/models/baseline_stats.pkl",
            os.path.join(os.path.dirname(__file__), "../../models/baseline_stats.pkl"),
        ]

        for baseline_path in baseline_paths:
            if os.path.exists(baseline_path):
                try:
                    with open(baseline_path, "rb") as f:
                        logger.info(f"Loaded baseline stats from: {baseline_path}")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(
                        f"Failed to load baseline stats from {baseline_path}: {e}"
                    )

        logger.warning("No baseline stats file found - using defaults")
        # Return default baseline if file doesn't exist
        return {
            "feature_means": {},
            "feature_stds": {},
            "prediction_distribution": {},
            "performance_baseline": {"accuracy": 0.85, "f1": 0.83},
        }

    def record_prediction(
        self,
        prediction_data: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
        latency: float = 0.0,
        confidence: float = 0.0,
    ):
        """
        Record a prediction event with comprehensive metrics.

        Args:
            prediction_data: Dictionary containing prediction results
            features: Input features for drift detection
            latency: Prediction latency in seconds
            confidence: Model confidence score
        """
        with self._lock:
            # Record basic metrics
            emotion = prediction_data.get("emotion", "unknown")
            sub_emotion = prediction_data.get("sub_emotion", "unknown")
            intensity = prediction_data.get("intensity", "unknown")

            self.prediction_counter.labels(
                emotion=emotion, sub_emotion=sub_emotion, intensity=intensity
            ).inc()

            self.prediction_latency.observe(latency)
            self.model_confidence.observe(confidence)

            # Store for drift detection
            self.prediction_history.append(
                {
                    "timestamp": datetime.now(),
                    "prediction": prediction_data,
                    "confidence": confidence,
                }
            )

            if features:
                self.feature_history.append(
                    {"timestamp": datetime.now(), "features": features}
                )

            # Trigger drift detection periodically
            if len(self.prediction_history) % 100 == 0:
                self._detect_drift()

    def record_transcription(
        self,
        transcript_data: Dict[str, Any],
        latency: float = 0.0,
        audio_quality: float = 0.0,
    ):
        """Record transcription metrics."""
        self.transcription_latency.observe(latency)
        self.audio_quality_score.observe(audio_quality)

        # Extract transcript confidence if available
        if "confidence" in transcript_data:
            self.transcript_confidence.observe(transcript_data["confidence"])

    def record_error(self, error_type: str, endpoint: str):
        """Record API errors for monitoring."""
        self.error_counter.labels(error_type=error_type, endpoint=endpoint).inc()

    def update_model_performance(self, performance_metrics: Dict[str, float]):
        """Update model performance metrics from evaluation."""
        for task, metrics in performance_metrics.items():
            if "accuracy" in metrics:
                self.model_accuracy.labels(task=task).set(metrics["accuracy"])
            if "f1" in metrics:
                self.model_f1_score.labels(task=task).set(metrics["f1"])

        # Store for trend analysis
        with self._lock:
            self.performance_history.append(
                {"timestamp": datetime.now(), "metrics": performance_metrics}
            )

    def _detect_drift(self):
        """Detect data and concept drift using statistical tests."""
        try:
            # Data drift detection (feature distribution changes)
            if len(self.feature_history) >= 50:
                data_drift = self._detect_data_drift()
                self.data_drift_score.set(data_drift)

                if data_drift > self.drift_threshold:
                    self.drift_alert.labels(drift_type="data").inc()
                    logger.warning(f"Data drift detected: score={data_drift:.4f}")

            # Concept drift detection (prediction distribution changes)
            if len(self.prediction_history) >= 50:
                concept_drift = self._detect_concept_drift()
                self.concept_drift_score.set(concept_drift)

                if concept_drift > self.drift_threshold:
                    self.drift_alert.labels(drift_type="concept").inc()
                    logger.warning(f"Concept drift detected: score={concept_drift:.4f}")

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")

    def _detect_data_drift(self) -> float:
        """Detect drift in input features using KS test."""
        if not self.baseline_stats.get("feature_means"):
            return 0.0

        recent_features = list(self.feature_history)[-50:]
        drift_scores = []

        for feature_name in self.baseline_stats["feature_means"].keys():
            # Extract feature values from recent history
            recent_values = [
                f["features"].get(feature_name, 0.0)
                for f in recent_features
                if feature_name in f["features"]
            ]

            if len(recent_values) < 20:
                continue

            # Compare with baseline using KS test
            baseline_mean = self.baseline_stats["feature_means"][feature_name]
            baseline_std = self.baseline_stats["feature_stds"].get(feature_name, 1.0)

            # Generate baseline sample for comparison
            baseline_sample = np.random.normal(
                baseline_mean, baseline_std, len(recent_values)
            )

            # Perform KS test
            statistic, p_value = stats.ks_2samp(recent_values, baseline_sample)
            drift_scores.append(statistic)

        return np.mean(drift_scores) if drift_scores else 0.0

    def _detect_concept_drift(self) -> float:
        """Detect drift in prediction patterns using distribution comparison."""
        if not self.baseline_stats.get("prediction_distribution"):
            return 0.0

        recent_predictions = list(self.prediction_history)[-50:]

        # Get recent prediction distribution
        recent_emotions = [
            p["prediction"].get("emotion", "unknown") for p in recent_predictions
        ]
        recent_dist = pd.Series(recent_emotions).value_counts(normalize=True).to_dict()

        baseline_dist = self.baseline_stats["prediction_distribution"]

        # Calculate Jensen-Shannon divergence
        return self._jensen_shannon_divergence(recent_dist, baseline_dist)

    def _jensen_shannon_divergence(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        # Get all unique labels
        all_labels = set(list(dist1.keys()) + list(dist2.keys()))

        # Convert to probability arrays
        p = np.array([dist1.get(label, 1e-10) for label in all_labels])
        q = np.array([dist2.get(label, 1e-10) for label in all_labels])

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate JS divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

        return js_div

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for debugging."""
        with self._lock:
            return {
                "total_predictions": len(self.prediction_history),
                "recent_drift_scores": {
                    "data_drift": self.data_drift_score._value._value,
                    "concept_drift": self.concept_drift_score._value._value,
                },
                "performance_trend": len(self.performance_history),
                "active_requests": self.active_requests._value._value,
                "last_update": datetime.now().isoformat(),
            }

    def export_metrics(self) -> str:
        """Export Prometheus metrics in text format."""
        return generate_latest()


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Decorator for timing function execution
def time_function(metric_name: str):
    """Decorator to time function execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time

                if metric_name == "prediction":
                    metrics_collector.prediction_latency.observe(latency)
                elif metric_name == "transcription":
                    metrics_collector.transcription_latency.observe(latency)

                return result
            except Exception as e:
                metrics_collector.record_error(str(type(e).__name__), func.__name__)
                raise

        return wrapper

    return decorator


# Context manager for active request tracking
class RequestTracker:
    """Context manager to track active requests."""

    def __enter__(self):
        metrics_collector.active_requests.inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics_collector.active_requests.dec()
        if exc_type:
            metrics_collector.record_error(exc_type.__name__, "api_request")
