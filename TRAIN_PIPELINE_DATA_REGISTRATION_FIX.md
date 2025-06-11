# Train-Pipeline Data Registration Fix

## 🐛 **Problem Identified**

The `train-pipeline` command was not registering processed data assets to Azure ML, while the individual `preprocess` command was working correctly.

## 🔍 **Root Cause Analysis**

**Individual Preprocess Command** (in `add_preprocess_args`):
```python
parser.add_argument(
    "--register-data-assets",
    action="store_true",
    default=True,  # ✅ Defaults to True
    help="Register processed data as Azure ML data assets after completion"
)
```

**Train-Pipeline Command** (in `add_pipeline_args`):
```python
parser.add_argument(
    "--register-data-assets",
    action="store_true",  # ❌ No default=True, so defaults to False
    help="Register processed data as Azure ML data assets."
)
```

## ✅ **Solution Applied**

Updated the `train-pipeline` command arguments to match the behavior of the individual `preprocess` command:

```python
parser.add_argument(
    "--register-data-assets",
    action="store_true",
    default=True,  # ✅ Now defaults to True
    help="Register processed data as Azure ML data assets."
)

parser.add_argument(
    "--no-register-data-assets",
    action="store_false",
    dest="register_data_assets",
    help="Skip registering processed data as Azure ML data assets"
)
```

## 🔧 **How It Works**

1. **Default Behavior**: `train-pipeline` now automatically includes `--register-data-assets` flag
2. **Azure ML Pipeline**: The preprocessing component command becomes:
   ```bash
   python -m src.emotion_clf_pipeline.data \
     --raw-train-path ${inputs.raw_train_data} \
     --raw-test-path ${inputs.raw_test_data} \
     --output-dir ${outputs.processed_data} \
     --encoders-dir ${outputs.encoders} \
     --register-data-assets  # ✅ Now included by default
   ```

3. **Data Registration**: The `data.py` module will now register processed data assets after completion

## 🎯 **Expected Behavior**

After this fix:

✅ **Individual Command** (already working):
```bash
poetry run python -m emotion_clf_pipeline.cli preprocess --azure --register-data-assets --verbose
```

✅ **Full Pipeline** (now fixed):
```bash
poetry run python -m emotion_clf_pipeline.cli train-pipeline --azure --verbose
```

Both commands will now register the processed data as Azure ML data assets.

## 🚫 **Opt-Out Option**

Users can still disable data registration if needed:
```bash
poetry run python -m emotion_clf_pipeline.cli train-pipeline --azure --no-register-data-assets --verbose
```

## 📋 **Testing**

To verify the fix:
1. Run the train-pipeline command
2. Check Azure ML Studio for new data asset versions
3. Confirm "emotion-processed-train" and "emotion-processed-test" assets are created

---

**Status**: ✅ **FIXED** - Train-pipeline now registers data assets by default
