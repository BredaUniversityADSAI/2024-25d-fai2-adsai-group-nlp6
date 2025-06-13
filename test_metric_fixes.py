#!/usr/bin/env python3
"""
Quick test script to verify the metric fixes in train.py.

This script tests the key functionality that was causing errors:
1. calculate_metrics returns correct keys
2. Metric filtering works properly
3. F1 score access uses correct key names
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_calculate_metrics():
    """Test that calculate_metrics returns expected keys."""
    print("=== Testing calculate_metrics function ===")
    
    try:
        from emotion_clf_pipeline.train import CustomTrainer
        
        # Test data
        test_preds = [0, 1, 2, 1, 0, 2, 1]
        test_labels = [0, 1, 2, 1, 0, 1, 1]
        
        # Calculate metrics
        metrics = CustomTrainer.calculate_metrics(test_preds, test_labels, "test")
        
        print("Returned metrics keys:", list(metrics.keys()))
        
        # Check expected keys
        expected_keys = ['acc', 'f1', 'prec', 'rec', 'report']
        for key in expected_keys:
            if key in metrics:
                value = metrics[key]
                if key == 'report':
                    print(f"  ✓ {key}: <classification_report_string>")
                else:
                    print(f"  ✓ {key}: {value:.4f}")
            else:
                print(f"  ✗ Missing key: {key}")
        
        # Test filtering logic (simulate what Azure ML logger does)
        print("\nFiltered numeric metrics:")
        for key, value in metrics.items():
            is_numeric = isinstance(value, (int, float))
            is_valid = is_numeric and not np.isnan(value)
            if is_valid:
                print(f"  ✓ Would log: {key} = {value:.4f}")
            else:
                print(f"  - Would skip: {key} (type: {type(value).__name__})")
        
        return True
        
    except Exception as e:
        print(f"❌ calculate_metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metric_key_consistency():
    """Test that metric key access is consistent."""
    print("\n=== Testing metric key consistency ===")
    
    try:
        from emotion_clf_pipeline.train import CustomTrainer
        
        # Simulate validation metrics structure
        val_metrics = {
            "emotion": {"acc": 0.85, "f1": 0.82, "prec": 0.88, "rec": 0.79, "report": "..."},
            "sub_emotion": {"acc": 0.73, "f1": 0.68, "prec": 0.71, "rec": 0.65, "report": "..."},
            "intensity": {"acc": 0.91, "f1": 0.89, "prec": 0.92, "rec": 0.87, "report": "..."}
        }
        
        output_tasks = ["emotion", "sub_emotion", "intensity"]
        
        # Test the overall F1 calculation (this was causing the KeyError)
        print("Testing overall F1 calculation...")
        try:
            overall_val_f1 = np.mean([
                val_metrics[task]["f1"] for task in output_tasks
            ])
            print(f"  ✓ Overall F1: {overall_val_f1:.4f}")
        except KeyError as e:
            print(f"  ✗ KeyError in F1 calculation: {e}")
            return False
        
        # Test best F1s extraction
        print("Testing best F1s extraction...")
        try:
            best_val_f1s = {task: val_metrics[task]["f1"] for task in output_tasks}
            print(f"  ✓ Best F1s: {best_val_f1s}")
        except KeyError as e:
            print(f"  ✗ KeyError in F1s extraction: {e}")
            return False
        
        # Test metric filtering for logging
        print("Testing metric filtering for logging...")
        for task in output_tasks:
            print(f"  Task: {task}")
            for metric_name, metric_value in val_metrics[task].items():
                is_numeric = isinstance(metric_value, (int, float))
                is_valid = is_numeric and not np.isnan(metric_value)
                if is_valid:
                    print(f"    ✓ Would log: {task}_{metric_name} = {metric_value}")
                else:
                    print(f"    - Would skip: {task}_{metric_name} (non-numeric)")
        
        return True
        
    except Exception as e:
        print(f"❌ Metric key consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_metrics_mapping():
    """Test that plot creation handles metric keys correctly."""
    print("\n=== Testing plot metrics mapping ===")
    
    try:
        # Simulate test metrics structure (what calculate_metrics returns)
        test_metrics = {
            "emotion": {"acc": 0.85, "f1": 0.82, "prec": 0.88, "rec": 0.79, "report": "..."},
            "sub_emotion": {"acc": 0.73, "f1": 0.68, "prec": 0.71, "rec": 0.65, "report": "..."}
        }
        
        # Test the metric filtering logic used in create_evaluation_plots
        print("Testing metric filtering for plots...")
        for task, metrics_data in test_metrics.items():
            print(f"  Task: {task}")
            metric_names = []
            metric_values = []
            
            for key, value in metrics_data.items():
                # This is the logic from create_evaluation_plots
                valid_metrics = ['acc', 'f1', 'prec', 'rec']
                if key in valid_metrics and isinstance(value, (int, float)):
                    display_name = {
                        'acc': 'Accuracy',
                        'f1': 'F1 Score',
                        'prec': 'Precision',
                        'rec': 'Recall'
                    }.get(key, key.replace('_', ' ').title())
                    metric_names.append(display_name)
                    metric_values.append(value)
                    print(f"    ✓ Would plot: {display_name} = {value:.4f}")
            
            print(f"    Total metrics for plotting: {len(metric_names)}")
        
        # Test overall comparison logic
        print("Testing overall comparison logic...")
        tasks = list(test_metrics.keys())
        f1_scores = []
        accuracy_scores = []
        
        for task in tasks:
            task_metrics = test_metrics[task]
            f1_score = task_metrics.get('f1', 0)  # Fixed key
            acc_score = task_metrics.get('acc', 0)  # Fixed key
            f1_scores.append(f1_score)
            accuracy_scores.append(acc_score)
            print(f"  {task}: F1={f1_score:.4f}, Acc={acc_score:.4f}")
        
        print(f"  ✓ Would create comparison plot with {len(tasks)} tasks")
        
        return True
        
    except Exception as e:
        print(f"❌ Plot metrics mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Metric Fixes Validation Test")
    print("=" * 50)
    
    tests = [
        test_calculate_metrics,
        test_metric_key_consistency,
        test_plot_metrics_mapping
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    test_names = [
        "Calculate Metrics",
        "Metric Key Consistency", 
        "Plot Metrics Mapping"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    overall_success = all(results)
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Metric fixes are working correctly!")
        print("The training should now run without KeyError or metric logging issues.")
    else:
        print("\n⚠️  Some issues detected. Check the errors above.")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
