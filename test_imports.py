#!/usr/bin/env python3
"""Test script to identify which imports are causing issues."""

import sys
import os

# Add src to path
sys.path.append('src')

print("1. Testing basic Python imports...")
import datetime
import typing
print("✅ Basic imports OK")

print("2. Testing FastAPI imports...")
from fastapi import FastAPI
from pydantic import BaseModel
print("✅ FastAPI imports OK")

print("3. Testing Azure imports...")
try:
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    print("✅ Azure basic imports OK")
except ImportError as e:
    print(f"⚠️ Azure imports failed: {e}")

print("4. Testing package structure...")
try:
    import emotion_clf_pipeline
    print("✅ Package import OK")
except Exception as e:
    print(f"❌ Package import failed: {e}")

print("5. Testing individual modules...")
modules_to_test = [
    'emotion_clf_pipeline.data',
    'emotion_clf_pipeline.model', 
    'emotion_clf_pipeline.features',
    'emotion_clf_pipeline.predict',
    'emotion_clf_pipeline.azure_sync',
    'emotion_clf_pipeline.azure_pipeline'
]

for module_name in modules_to_test:
    try:
        print(f"Testing {module_name}...")
        __import__(module_name)
        print(f"✅ {module_name} OK")
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")

print("6. Testing API imports...")
try:
    from emotion_clf_pipeline.api import app
    print("✅ API import OK")
except Exception as e:
    print(f"❌ API import failed: {e}")

print("All tests completed!")
