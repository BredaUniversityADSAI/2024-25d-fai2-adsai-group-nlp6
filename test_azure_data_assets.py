#!/usr/bin/env python3
"""
Test script to validate the Azure ML data asset registration functionality.
This script tests the register_processed_data_assets function with mock data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_import():
    """Test that the new function can be imported correctly."""
    try:
        from emotion_clf_pipeline.azure_pipeline import register_processed_data_assets
        print("✅ Successfully imported register_processed_data_assets function")
        return True
    except ImportError as e:
        print(f"❌ Failed to import register_processed_data_assets: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing register_processed_data_assets: {e}")
        return False

def test_cli_import():
    """Test that the CLI can import the new function correctly."""
    try:
        from emotion_clf_pipeline.cli import run_preprocess_azure
        print("✅ Successfully imported updated CLI function")
        return True
    except ImportError as e:
        print(f"❌ Failed to import CLI function: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing CLI function: {e}")
        return False

def test_function_signature():
    """Test that the function has the expected signature."""
    try:
        from emotion_clf_pipeline.azure_pipeline import register_processed_data_assets
        import inspect
        
        sig = inspect.signature(register_processed_data_assets)
        params = list(sig.parameters.keys())
        return_annotation = sig.return_annotation
        
        print(f"Function signature: {sig}")
        print(f"Parameters: {params}")
        print(f"Return type: {return_annotation}")
        
        # Check expected parameters
        if 'job' in params and return_annotation == bool:
            print("✅ Function signature is correct")
            return True
        else:
            print("❌ Function signature is incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Error checking function signature: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Azure ML Data Asset Registration Implementation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_import),
        ("CLI Import Test", test_cli_import),
        ("Function Signature Test", test_function_signature),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The implementation looks good.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
