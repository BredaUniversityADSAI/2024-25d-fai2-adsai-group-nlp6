#!/usr/bin/env python3
"""Minimal test to check basic imports."""

print("Starting import test...")

print("1. Testing basic imports...")
import sys
import os
print("✅ sys, os OK")

print("2. Testing FastAPI...")
from fastapi import FastAPI
print("✅ FastAPI OK")

print("3. Testing Azure basic imports...")
try:
    from azure.ai.ml.entities import Data
    print("✅ Azure entities OK")
except ImportError as e:
    print(f"⚠️ Azure entities failed: {e}")

print("4. Testing Azure constants...")
try:
    from azure.ai.ml.constants import AssetTypes
    print("✅ Azure constants OK")
except ImportError as e:
    print(f"⚠️ Azure constants failed: {e}")

print("5. Testing Azure identity...")
try:
    from azure.identity import DefaultAzureCredential
    print("✅ Azure identity OK")
except ImportError as e:
    print(f"⚠️ Azure identity failed: {e}")

print("All basic tests completed!")
