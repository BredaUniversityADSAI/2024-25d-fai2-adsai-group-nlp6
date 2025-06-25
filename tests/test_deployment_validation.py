#!/usr/bin/env python3
"""
Deployment Validation Test Module
Validates all model components, configurations, and shapes before Azure deployment
to catch mismatches early and save debugging time.
"""

import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

from emotion_clf_pipeline.features import FeatureExtractor
from emotion_clf_pipeline.model import DEBERTAClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Stores validation results for a component."""

    component: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class DeploymentValidator:
    """
    Comprehensive validator for all deployment components.
    Simulates Azure ML environment to catch issues before deployment.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize validator with model directory.

        Args:
            model_dir: Path to model artifacts directory
        """
        self.model_dir = Path(model_dir)
        self.results = []

        # Track loaded components
        self.config = None
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.label_encoders = None

        logger.info(f"🔍 Initializing deployment validator for: {self.model_dir}")

    def validate_all(self) -> bool:
        """
        Run all validation tests.

        Returns:
            True if all validations pass, False otherwise
        """
        logger.info("🚀 Starting comprehensive deployment validation...")

        # Core validations
        self._validate_model_directory()
        self._validate_config_loading()
        self._validate_model_weights()
        self._validate_tokenizer()
        self._validate_feature_extractor()
        self._validate_label_encoders()
        self._validate_model_architecture()
        self._validate_inference_pipeline()

        # Summary
        self._print_validation_summary()

        # Return overall result
        return all(result.passed for result in self.results)

    def _validate_model_directory(self) -> None:
        """Validate model directory structure."""
        logger.info("📁 Validating model directory structure...")

        required_files = [
            "weights/baseline_weights.pt",
            "weights/model_config.json",
            "features/EmoLex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
            "encoders/emotion_encoder.pkl",
            "encoders/sub_emotion_encoder.pkl",
            "encoders/intensity_encoder.pkl",
        ]

        missing_files = []
        existing_files = []

        for file_path in required_files:
            full_path = self.model_dir / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)

        if missing_files:
            self.results.append(
                ValidationResult(
                    component="ModelDirectory",
                    passed=False,
                    message=f"Missing required files: {missing_files}",
                    details={"missing": missing_files, "existing": existing_files},
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    component="ModelDirectory",
                    passed=True,
                    message="All required files found",
                    details={"existing": existing_files},
                )
            )

    def _validate_config_loading(self) -> None:
        """Validate configuration file loading."""
        logger.info("⚙️ Validating configuration loading...")

        config_path = self.model_dir / "weights" / "model_config.json"

        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)

            # Validate required config keys
            required_keys = [
                "model_name",
                "feature_dim",
                "num_classes",
                "hidden_dim",
                "dropout",
                "feature_config",
            ]

            missing_keys = [key for key in required_keys if key not in self.config]

            if missing_keys:
                self.results.append(
                    ValidationResult(
                        component="Configuration",
                        passed=False,
                        message=f"Missing config keys: {missing_keys}",
                        details={"config": self.config, "missing_keys": missing_keys},
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="Configuration",
                        passed=True,
                        message=f"Config loaded successfully. hidden_dim={self.config['hidden_dim']}",
                        details={"config": self.config},
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="Configuration",
                    passed=False,
                    message=f"Failed to load config: {e}",
                    details={"error": str(e), "path": str(config_path)},
                )
            )

    def _validate_model_weights(self) -> None:
        """Validate model weights can be loaded."""
        logger.info("🏋️ Validating model weights...")

        weights_path = self.model_dir / "weights" / "baseline_weights.pt"

        try:
            # Load weights to check format
            state_dict = torch.load(weights_path, map_location="cpu")

            # Analyze weight shapes
            weight_info = {}
            for key, tensor in state_dict.items():
                weight_info[key] = list(tensor.shape)

            # Check for key weight dimensions
            feature_proj_weight = state_dict.get("feature_projection.0.weight")
            if feature_proj_weight is not None:
                hidden_dim_from_weights = feature_proj_weight.shape[0]
                feature_dim_from_weights = feature_proj_weight.shape[1]

                self.results.append(
                    ValidationResult(
                        component="ModelWeights",
                        passed=True,
                        message=f"Weights loaded. Detected hidden_dim={hidden_dim_from_weights}, feature_dim={feature_dim_from_weights}",
                        details={
                            "weight_shapes": weight_info,
                            "detected_hidden_dim": hidden_dim_from_weights,
                            "detected_feature_dim": feature_dim_from_weights,
                        },
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="ModelWeights",
                        passed=False,
                        message="feature_projection.0.weight not found in state_dict",
                        details={"available_keys": list(state_dict.keys())},
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="ModelWeights",
                    passed=False,
                    message=f"Failed to load weights: {e}",
                    details={"error": str(e), "path": str(weights_path)},
                )
            )

    def _validate_tokenizer(self) -> None:
        """Validate tokenizer initialization."""
        logger.info("🔤 Validating tokenizer...")

        if not self.config:
            self.results.append(
                ValidationResult(
                    component="Tokenizer",
                    passed=False,
                    message="Cannot validate tokenizer without config",
                )
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

            # Test tokenization
            test_text = "I am feeling happy"
            tokens = self.tokenizer(test_text, return_tensors="pt")

            self.results.append(
                ValidationResult(
                    component="Tokenizer",
                    passed=True,
                    message=f"Tokenizer initialized successfully for {self.config['model_name']}",
                    details={
                        "model_name": self.config["model_name"],
                        "test_tokens_shape": tokens["input_ids"].shape,
                        "vocab_size": self.tokenizer.vocab_size,
                    },
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="Tokenizer",
                    passed=False,
                    message=f"Failed to initialize tokenizer: {e}",
                    details={
                        "error": str(e),
                        "model_name": self.config.get("model_name"),
                    },
                )
            )

    def _validate_feature_extractor(self) -> None:
        """Validate feature extractor initialization."""
        logger.info("🎯 Validating feature extractor...")

        if not self.config:
            self.results.append(
                ValidationResult(
                    component="FeatureExtractor",
                    passed=False,
                    message="Cannot validate feature extractor without config",
                )
            )
            return

        try:
            # Find EmoLex path
            emolex_path = (
                self.model_dir
                / "features"
                / "EmoLex"
                / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
            )

            if not emolex_path.exists():
                emolex_path = None

            self.feature_extractor = FeatureExtractor(
                feature_config=self.config.get("feature_config", {}),
                lexicon_path=str(emolex_path) if emolex_path else None,
            )

            # Test feature extraction
            test_text = "I am feeling happy"
            features = self.feature_extractor.extract_all_features(test_text)

            expected_dim = self.config.get("feature_dim", 121)
            actual_dim = len(features)

            if actual_dim == expected_dim:
                self.results.append(
                    ValidationResult(
                        component="FeatureExtractor",
                        passed=True,
                        message=f"Feature extractor working. Extracted {actual_dim} features",
                        details={
                            "feature_dim": actual_dim,
                            "expected_dim": expected_dim,
                            "emolex_available": emolex_path is not None,
                            "feature_config": self.config.get("feature_config", {}),
                        },
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        component="FeatureExtractor",
                        passed=False,
                        message=f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}",
                        details={
                            "feature_dim": actual_dim,
                            "expected_dim": expected_dim,
                            "emolex_available": emolex_path is not None,
                        },
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="FeatureExtractor",
                    passed=False,
                    message=f"Failed to initialize feature extractor: {e}",
                    details={"error": str(e)},
                )
            )

    def _validate_label_encoders(self) -> None:
        """Validate label encoders loading."""
        logger.info("🏷️ Validating label encoders...")

        encoder_files = {
            "emotion": self.model_dir / "encoders" / "emotion_encoder.pkl",
            "sub_emotion": self.model_dir / "encoders" / "sub_emotion_encoder.pkl",
            "intensity": self.model_dir / "encoders" / "intensity_encoder.pkl",
        }

        self.label_encoders = {}
        encoder_info = {}
        failed_encoders = []

        for task, encoder_path in encoder_files.items():
            try:
                with open(encoder_path, "rb") as f:
                    encoder = pickle.load(f)
                    self.label_encoders[task] = encoder
                    encoder_info[task] = {
                        "classes": list(encoder.classes_),
                        "num_classes": len(encoder.classes_),
                    }
            except Exception as e:
                failed_encoders.append(f"{task}: {e}")

        if failed_encoders:
            self.results.append(
                ValidationResult(
                    component="LabelEncoders",
                    passed=False,
                    message=f"Failed to load encoders: {failed_encoders}",
                    details={"failed": failed_encoders, "loaded": encoder_info},
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    component="LabelEncoders",
                    passed=True,
                    message="All label encoders loaded successfully",
                    details={"encoders": encoder_info},
                )
            )

    def _validate_model_architecture(self) -> None:
        """Validate model architecture matches weights."""
        logger.info("🏗️ Validating model architecture...")

        if not self.config:
            self.results.append(
                ValidationResult(
                    component="ModelArchitecture",
                    passed=False,
                    message="Cannot validate model without config",
                )
            )
            return

        try:
            # Initialize model with config
            self.model = DEBERTAClassifier(
                model_name=self.config["model_name"],
                feature_dim=self.config["feature_dim"],
                num_classes=self.config["num_classes"],
                hidden_dim=self.config["hidden_dim"],
                dropout=self.config["dropout"],
            )

            # Try loading weights
            weights_path = self.model_dir / "weights" / "baseline_weights.pt"
            state_dict = torch.load(weights_path, map_location="cpu")

            # Check if weights match architecture
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys or unexpected_keys:
                self.results.append(
                    ValidationResult(
                        component="ModelArchitecture",
                        passed=False,
                        message="Model architecture mismatch with weights",
                        details={
                            "missing_keys": missing_keys,
                            "unexpected_keys": unexpected_keys,
                            "config_hidden_dim": self.config["hidden_dim"],
                            "config_feature_dim": self.config["feature_dim"],
                        },
                    )
                )
            else:
                self.model.eval()
                self.results.append(
                    ValidationResult(
                        component="ModelArchitecture",
                        passed=True,
                        message="Model architecture matches weights perfectly",
                        details={
                            "config_hidden_dim": self.config["hidden_dim"],
                            "config_feature_dim": self.config["feature_dim"],
                            "num_parameters": sum(
                                p.numel() for p in self.model.parameters()
                            ),
                        },
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="ModelArchitecture",
                    passed=False,
                    message=f"Failed to validate model architecture: {e}",
                    details={"error": str(e)},
                )
            )

    def _validate_inference_pipeline(self) -> None:
        """Validate complete inference pipeline."""
        logger.info("🧠 Validating inference pipeline...")

        if not all([self.model, self.tokenizer, self.feature_extractor]):
            self.results.append(
                ValidationResult(
                    component="InferencePipeline",
                    passed=False,
                    message="Cannot validate inference pipeline - missing components",
                )
            )
            return

        try:
            # Test complete inference
            test_text = "I am feeling really happy and excited!"

            # Extract features
            features = self.feature_extractor.extract_all_features(test_text)
            features_tensor = torch.tensor(features).unsqueeze(0)

            # Tokenize
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    features=features_tensor,
                )

            # Check outputs
            expected_tasks = ["emotion", "sub_emotion", "intensity"]
            output_info = {}

            for task in expected_tasks:
                if task in outputs:
                    output_tensor = outputs[task]
                    output_info[task] = {
                        "shape": list(output_tensor.shape),
                        "num_classes": output_tensor.shape[-1],
                    }

            self.results.append(
                ValidationResult(
                    component="InferencePipeline",
                    passed=True,
                    message="Complete inference pipeline working",
                    details={
                        "input_text": test_text,
                        "feature_dim": len(features),
                        "input_ids_shape": list(inputs["input_ids"].shape),
                        "outputs": output_info,
                    },
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    component="InferencePipeline",
                    passed=False,
                    message=f"Inference pipeline failed: {e}",
                    details={"error": str(e)},
                )
            )

    def _print_validation_summary(self) -> None:
        """Print comprehensive validation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DEPLOYMENT VALIDATION SUMMARY")
        logger.info("=" * 80)

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        logger.info(f"Overall Result: {passed_count}/{total_count} validations passed")

        if passed_count == total_count:
            logger.info("🎉 ALL VALIDATIONS PASSED - Ready for Azure deployment!")
        else:
            logger.error("❌ VALIDATION FAILURES - Fix issues before deployment!")

        logger.info("\nDetailed Results:")
        logger.info("-" * 50)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"{status} {result.component:20} | {result.message}")

            if not result.passed and result.details:
                for key, value in result.details.items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        logger.info(
                            f"     {key}: {type(value).__name__} with {len(value)} items"
                        )
                    else:
                        logger.info(f"     {key}: {value}")

        logger.info("=" * 80)


def main():
    """Main validation script entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate deployment components")
    parser.add_argument(
        "--model-dir", default="models", help="Path to model artifacts directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    validator = DeploymentValidator(args.model_dir)
    success = validator.validate_all()

    # Exit with appropriate code
    exit_code = 0 if success else 1

    if success:
        print("\n🎉 All validations passed! Ready for Azure deployment.")
    else:
        print("\n❌ Validation failures detected. Fix issues before deploying.")
        print("Run with --verbose for more details.")

    return exit_code


if __name__ == "__main__":
    exit(main())
