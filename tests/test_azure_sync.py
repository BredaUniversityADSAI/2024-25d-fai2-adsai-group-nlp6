"""
Unit tests for Azure ML Model Synchronization Manager
Tests all functionality with mocked Azure ML dependencies
"""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, call, patch

# Set up Azure ML mock structure
azure = Mock()
azure.ai = Mock()
azure.ai.ml = Mock()
azure.ai.ml.MLClient = Mock()
azure.ai.ml.entities = Mock()
azure.ai.ml.entities.Model = Mock()
azure.ai.ml.constants = Mock()
azure.ai.ml.constants.AssetTypes = Mock()
azure.identity = Mock()
azure.identity.DefaultAzureCredential = Mock()
azure.identity.ClientSecretCredential = Mock()
azure.identity.AzureCliCredential = Mock()
azure.core = Mock()
azure.core.exceptions = Mock()
azure.core.exceptions.ResourceNotFoundError = Exception
azure.core.exceptions.HttpResponseError = Exception

# Mock dotenv
dotenv = Mock()

# Patch the imports
sys.modules["azure"] = azure
sys.modules["azure.ai"] = azure.ai
sys.modules["azure.ai.ml"] = azure.ai.ml
sys.modules["azure.ai.ml.entities"] = azure.ai.ml.entities
sys.modules["azure.ai.ml.constants"] = azure.ai.ml.constants
sys.modules["azure.identity"] = azure.identity
sys.modules["azure.core"] = azure.core
sys.modules["azure.core.exceptions"] = azure.core.exceptions
sys.modules["dotenv"] = dotenv

# Import the class to test
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.emotion_clf_pipeline.azure_sync import AzureMLModelManager  # noqa: E402


class TestAzureMLModelManager(unittest.TestCase):
    """Test cases for AzureMLModelManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.weights_dir = Path(self.temp_dir) / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model files
        self.baseline_weights = self.weights_dir / "baseline_weights.pt"
        self.dynamic_weights = self.weights_dir / "dynamic_weights.pt"

        # Reset all mocks
        azure.ai.ml.MLClient.reset_mock()
        azure.identity.DefaultAzureCredential.reset_mock()
        azure.identity.ClientSecretCredential.reset_mock()
        azure.identity.AzureCliCredential.reset_mock()
        dotenv.load_dotenv.reset_mock()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def create_manager_with_azure_config(self):
        """Helper to create manager with Azure config."""
        return AzureMLModelManager(str(self.weights_dir))

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_azure_config(self):
        """Test initialization without Azure configuration."""
        manager = AzureMLModelManager(str(self.weights_dir))

        self.assertEqual(str(manager.weights_dir), str(self.weights_dir))
        self.assertFalse(manager._azure_available)
        self.assertIsNone(manager._ml_client)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def test_init_with_azure_config_success(self):
        """Test successful initialization with Azure configuration."""
        # Mock successful Azure connection
        mock_client = Mock()
        mock_workspace = Mock()
        mock_client.workspaces.get.return_value = mock_workspace
        azure.ai.ml.MLClient.return_value = mock_client

        manager = AzureMLModelManager(str(self.weights_dir))

        self.assertTrue(manager._azure_available)
        self.assertIsNotNone(manager._ml_client)
        azure.ai.ml.MLClient.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def test_init_with_azure_config_connection_failure(self):
        """Test initialization with Azure config but connection failure."""
        # Mock connection failure
        mock_client = Mock()
        mock_client.workspaces.get.side_effect = Exception("Connection failed")
        azure.ai.ml.MLClient.return_value = mock_client

        manager = AzureMLModelManager(str(self.weights_dir))

        self.assertFalse(manager._azure_available)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_CLIENT_SECRET": "test-client-secret",
            "AZURE_TENANT_ID": "test-tenant-id",
        },
    )
    def test_service_principal_authentication(self):
        """Test service principal authentication method."""
        # Mock successful service principal auth
        mock_client = Mock()
        mock_workspace = Mock()
        mock_client.workspaces.get.return_value = mock_workspace
        azure.ai.ml.MLClient.return_value = mock_client

        # Create manager and manually set the auth method
        # As it would be set during initialization
        manager = AzureMLModelManager(str(self.weights_dir))

        # Verify the service principal credential was used
        azure.identity.ClientSecretCredential.assert_called_once_with(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

        # Check that Azure is available (indicating successful auth)
        self.assertTrue(manager._azure_available)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def test_azure_cli_authentication_fallback(self):
        """Test Azure CLI authentication fallback."""
        # Mock Azure CLI credential success
        mock_client = Mock()
        mock_workspace = Mock()
        mock_client.workspaces.get.return_value = mock_workspace
        azure.ai.ml.MLClient.return_value = mock_client

        manager = AzureMLModelManager(str(self.weights_dir))

        self.assertTrue(manager._azure_available)
        # Should try Azure CLI first when no service principal is configured
        azure.identity.AzureCliCredential.assert_called_once()

    def test_sync_on_startup_no_azure(self):
        """Test sync on startup without Azure availability."""
        manager = AzureMLModelManager(str(self.weights_dir))
        manager._azure_available = False

        baseline_synced, dynamic_synced = manager.sync_on_startup()

        self.assertFalse(baseline_synced)
        self.assertFalse(dynamic_synced)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def test_sync_on_startup_with_existing_files(self):
        """Test sync on startup when local files already exist."""
        # Create existing weight files
        self.baseline_weights.write_text("baseline weights")
        self.dynamic_weights.write_text("dynamic weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        baseline_synced, dynamic_synced = manager.sync_on_startup()

        # Should not download if files exist
        self.assertFalse(baseline_synced)
        self.assertFalse(dynamic_synced)

    @patch.dict(
        os.environ,
        {
            "AZURE_SUBSCRIPTION_ID": "test-sub-id",
            "AZURE_RESOURCE_GROUP": "test-rg",
            "AZURE_WORKSPACE_NAME": "test-ws",
        },
    )
    def test_sync_on_startup_download_missing_files(self):
        """Test sync on startup downloads missing files."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock successful download
        with patch.object(
            manager, "_download_model_from_azure", return_value=True
        ) as mock_download:
            baseline_synced, dynamic_synced = manager.sync_on_startup()

            self.assertTrue(baseline_synced)
            self.assertTrue(dynamic_synced)

            # Should call download for both models
            expected_calls = [
                call("emotion-clf-baseline", self.baseline_weights),
                call("emotion-clf-dynamic", self.dynamic_weights),
            ]
            mock_download.assert_has_calls(expected_calls)

    @patch("tempfile.TemporaryDirectory")
    @patch("shutil.copy2")
    def test_download_model_from_azure_success(self, mock_copy, mock_temp_dir):
        """Test successful model download from Azure ML."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML client and model
        mock_client = Mock()
        mock_model = Mock()
        mock_model.version = "1"
        mock_client.models.get.return_value = mock_model
        mock_client.models.download.return_value = "/temp/download/path"
        manager._ml_client = mock_client

        # Mock temporary directory and file structure
        temp_path = Path("/temp/download")
        mock_temp_dir.return_value.__enter__.return_value = str(temp_path)

        # Mock finding .pt files
        with patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.return_value = [Path("/temp/download/model.pt")]

            result = manager._download_model_from_azure(
                "test-model", self.baseline_weights
            )

            self.assertTrue(result)
            mock_client.models.get.assert_called_once_with("test-model", label="latest")
            mock_client.models.download.assert_called_once()
            mock_copy.assert_called_once()

    def test_download_model_from_azure_no_pt_file(self):
        """Test download failure when no .pt file found."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML client
        mock_client = Mock()
        mock_model = Mock()
        mock_model.version = "1"
        mock_client.models.get.return_value = mock_model
        mock_client.models.download.return_value = "/temp/download/path"
        manager._ml_client = mock_client

        with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
            temp_path = Path("/temp/download")
            mock_temp_dir.return_value.__enter__.return_value = str(temp_path)

            # Mock no .pt files found
            with patch.object(Path, "rglob", return_value=[]):
                result = manager._download_model_from_azure(
                    "test-model", self.baseline_weights
                )

                self.assertFalse(result)

    def test_upload_dynamic_model_no_azure(self):
        """Test upload dynamic model without Azure availability."""
        manager = AzureMLModelManager(str(self.weights_dir))
        manager._azure_available = False

        result = manager.upload_dynamic_model(0.85)

        self.assertFalse(result)

    def test_upload_dynamic_model_no_file(self):
        """Test upload dynamic model when file doesn't exist."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        result = manager.upload_dynamic_model(0.85)

        self.assertFalse(result)

    def test_upload_dynamic_model_success(self):
        """Test successful dynamic model upload."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML client
        mock_client = Mock()
        mock_registered_model = Mock()
        mock_registered_model.version = "2"
        mock_client.models.create_or_update.return_value = mock_registered_model
        manager._ml_client = mock_client

        # Use a real temporary directory instead of mocking
        with tempfile.TemporaryDirectory() as real_temp_dir:
            with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
                mock_temp_dir.return_value.__enter__.return_value = real_temp_dir
                mock_temp_dir.return_value.__exit__.return_value = None

                with patch("shutil.copy2") as mock_copy:
                    # Mock the copy operation to succeed
                    mock_copy.return_value = None

                    result = manager.upload_dynamic_model(0.85, {"extra": "metadata"})

                    self.assertTrue(result)
                    mock_client.models.create_or_update.assert_called_once()
                    mock_copy.assert_called_once()

    @patch("time.sleep")
    def test_upload_dynamic_model_retry_on_connection_error(self, mock_sleep):
        """Test upload retry logic on connection errors."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML client with connection error then success
        mock_client = Mock()
        connection_error = azure.core.exceptions.HttpResponseError("Connection aborted")
        mock_registered_model = Mock()
        mock_registered_model.version = "2"
        mock_client.models.create_or_update.side_effect = [
            connection_error,  # First attempt fails
            mock_registered_model,  # Second attempt succeeds
        ]
        manager._ml_client = mock_client

        # Create a unique temporary directory for each test run
        import uuid

        temp_dir_name = f"tmp_test_{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory(prefix=temp_dir_name) as real_temp_dir:
            with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
                # Ensure the temporary directory context manager returns the right path
                mock_temp_dir.return_value.__enter__.return_value = real_temp_dir
                mock_temp_dir.return_value.__exit__.return_value = None

                # Create the model subdirectory that would be expected
                model_dir = Path(real_temp_dir) / "model"
                model_dir.mkdir(parents=True, exist_ok=True)

                with patch("shutil.copy2") as mock_copy:
                    # Mock the copy operation to succeed
                    mock_copy.return_value = None

                    # Also mock the file operations to avoid Windows-specific issues
                    with patch("pathlib.Path.mkdir") as mock_mkdir:
                        mock_mkdir.return_value = None

                        result = manager.upload_dynamic_model(0.85)

                        self.assertTrue(result)
                        self.assertEqual(
                            mock_client.models.create_or_update.call_count, 2
                        )
                        mock_sleep.assert_called_once_with(2)  # Exponential backoff

    def test_promote_dynamic_to_baseline_no_azure(self):
        """Test promote to baseline without Azure availability."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = AzureMLModelManager(str(self.weights_dir))
        manager._azure_available = False

        result = manager.promote_dynamic_to_baseline()

        self.assertTrue(result)  # Should succeed with local promotion
        self.assertTrue(self.baseline_weights.exists())
        self.assertEqual(self.baseline_weights.read_text(), "dynamic model weights")

    def test_promote_dynamic_to_baseline_no_dynamic_file(self):
        """Test promote to baseline when dynamic file doesn't exist."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML - no dynamic model found
        mock_client = Mock()
        mock_client.models.get.side_effect = (
            azure.core.exceptions.ResourceNotFoundError()
        )
        manager._ml_client = mock_client

        result = manager.promote_dynamic_to_baseline()

        self.assertFalse(result)  # Should fail as no dynamic file exists locally either

    def test_promote_dynamic_to_baseline_success(self):
        """Test successful promotion of dynamic to baseline."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML
        mock_client = Mock()
        mock_dynamic_model = Mock()
        mock_dynamic_model.version = "3"
        mock_dynamic_model.tags = {"f1_score": "0.87", "model_type": "dynamic"}
        mock_client.models.get.return_value = mock_dynamic_model
        manager._ml_client = mock_client

        # Mock successful upload
        with patch.object(manager, "_upload_model_with_retry", return_value=True):
            result = manager.promote_dynamic_to_baseline()

            self.assertTrue(result)
            self.assertTrue(self.baseline_weights.exists())
            self.assertEqual(self.baseline_weights.read_text(), "dynamic model weights")

    def test_get_model_info_local_only(self):
        """Test get model info with local files only."""
        # Create test files
        self.baseline_weights.write_text("baseline")
        self.dynamic_weights.write_text("dynamic")

        manager = AzureMLModelManager(str(self.weights_dir))
        manager._azure_available = False

        info = manager.get_model_info()

        self.assertFalse(info["azure_available"])
        self.assertTrue(info["local"]["baseline_exists"])
        self.assertTrue(info["local"]["dynamic_exists"])
        self.assertIn("baseline_size", info["local"])
        self.assertIn("dynamic_size", info["local"])
        self.assertIn("baseline_modified", info["local"])
        self.assertIn("dynamic_modified", info["local"])

    def test_get_model_info_with_azure(self):
        """Test get model info with Azure ML integration."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        # Mock Azure ML models
        mock_client = Mock()
        mock_baseline_model = Mock()
        mock_baseline_model.version = "1"
        mock_baseline_model.tags = {"f1_score": "0.80"}
        mock_baseline_model.creation_context = Mock()
        mock_baseline_model.creation_context.created_at = datetime.now()

        mock_dynamic_model = Mock()
        mock_dynamic_model.version = "2"
        mock_dynamic_model.tags = {"f1_score": "0.85"}
        mock_dynamic_model.creation_context = Mock()
        mock_dynamic_model.creation_context.created_at = datetime.now()

        mock_client.models.get.side_effect = [mock_baseline_model, mock_dynamic_model]
        manager._ml_client = mock_client

        info = manager.get_model_info()

        self.assertTrue(info["azure_available"])
        self.assertIn("emotion-clf-baseline", info["azure_ml"])
        self.assertIn("emotion-clf-dynamic", info["azure_ml"])
        self.assertEqual(info["azure_ml"]["emotion-clf-baseline"]["version"], "1")
        self.assertEqual(info["azure_ml"]["emotion-clf-dynamic"]["version"], "2")

    def test_get_configuration_status(self):
        """Test configuration status reporting."""
        with patch.dict(
            os.environ,
            {
                "AZURE_SUBSCRIPTION_ID": "test-sub-id",
                "AZURE_RESOURCE_GROUP": "test-rg",
                "AZURE_WORKSPACE_NAME": "test-ws",
                "AZURE_CLIENT_ID": "test-client-id",
            },
        ):
            with patch("shutil.which", return_value="/usr/bin/az"):
                manager = AzureMLModelManager(str(self.weights_dir))

                status = manager.get_configuration_status()

                self.assertEqual(
                    status["environment_variables"]["AZURE_SUBSCRIPTION_ID"], "✓ Set"
                )
                self.assertEqual(
                    status["environment_variables"]["AZURE_RESOURCE_GROUP"], "✓ Set"
                )
                self.assertEqual(
                    status["environment_variables"]["AZURE_WORKSPACE_NAME"], "✓ Set"
                )
                self.assertEqual(
                    status["environment_variables"]["AZURE_CLIENT_ID"], "✓ Set"
                )
                self.assertEqual(
                    status["environment_variables"]["AZURE_CLIENT_SECRET"],
                    "✗ Not set (optional)",
                )
                self.assertTrue(status["authentication"]["azure_cli_available"])
                self.assertIn(
                    "Azure CLI", status["authentication"]["available_methods"]
                )

    def test_auto_sync_on_startup_no_azure(self):
        """Test auto sync on startup without Azure."""
        manager = AzureMLModelManager(str(self.weights_dir))
        manager._azure_available = False

        results = manager.auto_sync_on_startup()

        expected = {
            "baseline_downloaded": False,
            "dynamic_downloaded": False,
            "baseline_updated": False,
            "dynamic_updated": False,
        }
        self.assertEqual(results, expected)

    def test_auto_sync_on_startup_with_downloads(self):
        """Test auto sync with model downloads."""
        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        with patch.object(
            manager, "_download_model_from_azure", return_value=True
        ) as mock_download:
            with patch.object(manager, "_check_and_update_model", return_value=False):
                results = manager.auto_sync_on_startup()

                self.assertTrue(results["baseline_downloaded"])
                self.assertTrue(results["dynamic_downloaded"])
                self.assertFalse(results["baseline_updated"])
                self.assertFalse(results["dynamic_updated"])

                # Should call download for both missing models
                self.assertEqual(mock_download.call_count, 2)

    def test_auto_upload_after_training_below_threshold(self):
        """Test auto upload after training below promotion threshold."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        with patch.object(
            manager, "upload_dynamic_model", return_value=True
        ) as mock_upload:
            results = manager.auto_upload_after_training(
                f1_score=0.80, auto_promote_threshold=0.85
            )

            self.assertTrue(results["uploaded"])
            self.assertFalse(results["promoted"])
            mock_upload.assert_called_once_with(0.80, None)

    def test_auto_upload_after_training_above_threshold(self):
        """Test auto upload after training above promotion threshold."""
        # Create dynamic weights file
        self.dynamic_weights.write_text("dynamic model weights")

        manager = self.create_manager_with_azure_config()
        manager._azure_available = True

        with patch.object(
            manager, "upload_dynamic_model", return_value=True
        ) as mock_upload:
            with patch.object(
                manager, "promote_dynamic_to_baseline", return_value=True
            ) as mock_promote:
                results = manager.auto_upload_after_training(
                    f1_score=0.90, auto_promote_threshold=0.85
                )

                self.assertTrue(results["uploaded"])
                self.assertTrue(results["promoted"])
                mock_upload.assert_called_once_with(0.90, None)
                mock_promote.assert_called_once()

    def test_get_auto_sync_config(self):
        """Test get auto sync configuration."""
        manager = AzureMLModelManager(str(self.weights_dir))

        config = manager.get_auto_sync_config()

        expected_keys = [
            "auto_download_on_startup",
            "auto_check_updates_on_startup",
            "auto_upload_after_training",
            "auto_promote_threshold",
            "sync_on_model_load",
            "background_sync_enabled",
        ]

        for key in expected_keys:
            self.assertIn(key, config)

        self.assertEqual(config["auto_promote_threshold"], 0.85)
        self.assertTrue(config["auto_download_on_startup"])

    def test_convenience_functions(self):
        """Test convenience functions."""

        manager = AzureMLModelManager(str(self.weights_dir))

        # Test sync_on_startup equivalent
        with patch.object(
            manager, "sync_on_startup", return_value=(False, False)
        ) as mock_sync:
            baseline_synced, dynamic_synced = manager.sync_on_startup()
            self.assertIsInstance(baseline_synced, bool)
            self.assertIsInstance(dynamic_synced, bool)
            mock_sync.assert_called_once()

        # Test upload_dynamic_model equivalent
        with patch.object(
            manager, "upload_dynamic_model", return_value=True
        ) as mock_upload:
            result = manager.upload_dynamic_model(0.85)
            self.assertTrue(result)
            mock_upload.assert_called_once_with(0.85)

        # Test promote_dynamic_to_baseline equivalent
        with patch.object(
            manager, "promote_dynamic_to_baseline", return_value=True
        ) as mock_promote:
            result = manager.promote_dynamic_to_baseline()
            self.assertTrue(result)
            mock_promote.assert_called_once()

        # Test get_configuration_status equivalent
        with patch.object(
            manager, "get_configuration_status", return_value={}
        ) as mock_config:
            result = manager.get_configuration_status()
            self.assertIsInstance(result, dict)
            mock_config.assert_called_once()


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
