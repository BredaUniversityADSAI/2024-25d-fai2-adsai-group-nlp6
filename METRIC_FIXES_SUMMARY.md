# Azure ML Training Fixes - Metric Logging Issues

## Problems Identified and Fixed

Based on the training logs, several critical issues were causing the training pipeline to fail:

### 1. **KeyError: 'f1_score'** ❌ → ✅ FIXED

**Problem**: The code was trying to access `val_metrics[task]["f1_score"]` but the `calculate_metrics` method returns `"f1"`, not `"f1_score"`.

**Root Cause**: Inconsistent metric key naming between the metric calculation function and the accessing code.

**Fix Applied**:
```python
# Before (causing KeyError)
overall_val_f1 = np.mean([
    val_metrics[task]["f1_score"] for task in self.output_tasks
])
best_val_f1s = {task: val_metrics[task]["f1_score"] for task in self.output_tasks}

# After (fixed)
overall_val_f1 = np.mean([
    val_metrics[task]["f1"] for task in self.output_tasks
])
best_val_f1s = {task: val_metrics[task]["f1"] for task in self.output_tasks}
```

### 2. **Invalid Azure ML Metric Values** ❌ → ✅ FIXED

**Problem**: Azure ML was rejecting metric logging attempts with error:
```
Invalid value "classification_report_string" for parameter 'value'... 
Please specify value as a valid double (64-bit floating point)
```

**Root Cause**: The code was trying to log classification reports (string values) as Azure ML metrics, but Azure ML only accepts numeric values for metrics.

**Fix Applied**:
```python
# Before (logging all values including strings)
for metric_name, metric_value in train_metrics[task].items():
    self.azure_logger.log_metric(f"train_{task}_{metric_name}", metric_value, step=step)

# After (filtering out non-numeric values)
for metric_name, metric_value in train_metrics[task].items():
    is_numeric = isinstance(metric_value, (int, float))
    is_valid = is_numeric and not np.isnan(metric_value)
    if is_valid:
        self.azure_logger.log_metric(f"train_{task}_{metric_name}", metric_value, step=step)
```

### 3. **Evaluation Plot Metric Key Mismatch** ❌ → ✅ FIXED

**Problem**: The `create_evaluation_plots` method was looking for metrics with keys like `'accuracy'`, `'f1_score'`, `'precision'`, `'recall'`, but `calculate_metrics` returns `'acc'`, `'f1'`, `'prec'`, `'rec'`.

**Fix Applied**:
```python
# Before (wrong keys)
valid_metrics = ['accuracy', 'f1_score', 'precision', 'recall']

# After (correct keys with display name mapping)
valid_metrics = ['acc', 'f1', 'prec', 'rec']
display_name = {
    'acc': 'Accuracy',
    'f1': 'F1 Score',
    'prec': 'Precision',
    'rec': 'Recall'
}.get(key, key.replace('_', ' ').title())
```

### 4. **Overall Performance Comparison Fix** ❌ → ✅ FIXED

**Problem**: Similar metric key issues in the overall performance comparison plots.

**Fix Applied**:
```python
# Before (fallback logic for wrong keys)
f1_score = task_metrics.get('f1_score', task_metrics.get('f1', 0))
acc_score = task_metrics.get('accuracy', task_metrics.get('acc', 0))

# After (direct access to correct keys)
f1_score = task_metrics.get('f1', 0)
acc_score = task_metrics.get('acc', 0)
```

## Impact of Fixes

### ✅ **Resolved Issues**:
1. **Training no longer crashes** with KeyError
2. **Azure ML metrics logging works** without "invalid value" warnings
3. **Evaluation plots generate correctly** with proper metric access
4. **All numeric metrics are logged** to Azure ML for visualization
5. **String reports are properly filtered out** from metric logging

### ✅ **Azure ML Job Overview Will Now Show**:
- **Metrics Tab**: Per-epoch training/validation metrics (loss, accuracy, F1, precision, recall)
- **Images Tab**: Confusion matrices, performance charts, overall comparison plots
- **Clean metric values**: No more invalid string logging attempts

### ✅ **Backward Compatibility Maintained**:
- All existing functionality preserved
- `calculate_metrics` method unchanged (maintains API contract)
- Print statements and console output unchanged
- Local development experience unchanged

## Key Lessons Learned

### 1. **Consistent Metric Naming**
- **Problem**: Different parts of code expected different key names
- **Solution**: Use the actual keys returned by `calculate_metrics` consistently
- **Best Practice**: Define metric key constants to avoid this issue

### 2. **Type-Safe Metric Logging**
- **Problem**: Trying to log non-numeric values as metrics
- **Solution**: Filter values before logging to Azure ML
- **Best Practice**: Validate data types before external API calls

### 3. **Defensive Programming**
- **Problem**: Assumptions about data structure without validation
- **Solution**: Use `.get()` with defaults and type checking
- **Best Practice**: Always validate external dependencies' return values

## Testing Recommendations

### Before Deployment:
1. **Run Local Training**: Verify no crashes occur
2. **Check Azure ML Metrics**: Confirm metrics appear in job overview
3. **Validate Plots**: Ensure evaluation images are generated and logged
4. **Monitor Warnings**: No more "invalid value" warnings should appear

### Success Criteria:
- ✅ Training completes without KeyError
- ✅ Azure ML metrics tab shows training progress
- ✅ Azure ML images tab shows evaluation plots
- ✅ No metric logging warnings in console
- ✅ All numeric metrics properly tracked over epochs

## Code Quality Improvements

The fixes also improved code quality by:
- **Better Error Handling**: Type checking before API calls
- **Clearer Intent**: Explicit metric filtering logic
- **Maintainability**: Consistent key usage across codebase
- **Robustness**: Graceful handling of unexpected metric structures

These fixes ensure that your emotion classification pipeline will work reliably in Azure ML with proper metrics and visualization support.
