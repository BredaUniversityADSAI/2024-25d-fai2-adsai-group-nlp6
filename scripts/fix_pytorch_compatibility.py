#!/usr/bin/env python3
"""
PyTorch 2.6 Compatibility Fix Script

This script helps fix compatibility issues when upgrading from older PyTorch versions
to PyTorch 2.6, which changed the default behavior of torch.load to use weights_only=True.

Key fixes:
1. Detects and removes corrupted pickle files (baseline_stats.pkl)
2. Validates model weight files
3. Provides recommendations for fixing issues

Usage:
    python scripts/fix_pytorch_compatibility.py
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_pickle_file(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a pickle file can be loaded successfully.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            pickle.load(f)
        return True, "OK"
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        return False, f"Pickle error: {str(e)}"
    except Exception as e:
        return False, f"General error: {str(e)}"


def check_torch_model(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a PyTorch model file can be loaded successfully.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try with weights_only=False (compatible mode)
        state_dict = torch.load(file_path, map_location="cpu", weights_only=False)
        
        if not isinstance(state_dict, dict):
            return False, "Not a valid state dictionary"
            
        if len(state_dict) == 0:
            return False, "Empty state dictionary"
            
        return True, f"OK - {len(state_dict)} parameters"
        
    except Exception as e:
        return False, f"Model loading error: {str(e)}"


def find_problematic_files(project_root: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Scan project for problematic files.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary with file types and their issues
    """
    issues = {
        "pickle_files": [],
        "model_files": [],
    }
    
    # Check pickle files
    pickle_patterns = [
        "models/baseline_stats.pkl",
        "models/**/encoders/*.pkl",
        "models/**/*.pkl"
    ]
    
    for pattern in pickle_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                is_valid, error = check_pickle_file(file_path)
                if not is_valid:
                    issues["pickle_files"].append((file_path, error))
    
    # Check PyTorch model files
    model_patterns = [
        "models/weights/*.pt",
        "models/**/*.pt",
        "models/**/*.pth"
    ]
    
    for pattern in model_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                is_valid, error = check_torch_model(file_path)
                if not is_valid:
                    issues["model_files"].append((file_path, error))
    
    return issues


def fix_corrupted_files(issues: Dict[str, List[Tuple[Path, str]]], 
                       backup: bool = True) -> Dict[str, int]:
    """
    Fix corrupted files by removing them.
    
    Args:
        issues: Dictionary of file issues
        backup: Whether to create backups before deletion
        
    Returns:
        Dictionary with counts of fixed files
    """
    fixed_counts = {
        "pickle_files_removed": 0,
        "model_files_removed": 0,
        "backups_created": 0
    }
    
    # Fix pickle files
    for file_path, error in issues["pickle_files"]:
        logger.warning(f"Corrupted pickle file: {file_path} - {error}")
        
        if backup:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            try:
                file_path.rename(backup_path)
                logger.info(f"Backed up to: {backup_path}")
                fixed_counts["backups_created"] += 1
            except Exception as e:
                logger.error(f"Failed to backup {file_path}: {e}")
                continue
        else:
            try:
                file_path.unlink()
                logger.info(f"Removed corrupted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
                continue
                
        fixed_counts["pickle_files_removed"] += 1
    
    # Log model file issues (don't auto-remove model files)
    for file_path, error in issues["model_files"]:
        logger.error(f"Corrupted model file: {file_path} - {error}")
        logger.info(f"Model files require manual investigation: {file_path}")
    
    return fixed_counts


def generate_recommendations(issues: Dict[str, List[Tuple[Path, str]]]) -> List[str]:
    """
    Generate recommendations for fixing issues.
    
    Args:
        issues: Dictionary of file issues
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if issues["pickle_files"]:
        recommendations.append(
            "1. Corrupted pickle files found. These will be regenerated automatically on next run."
        )
        recommendations.append(
            "   Run: python -c 'from src.emotion_clf_pipeline.api import setup_baseline_stats; setup_baseline_stats()'"
        )
    
    if issues["model_files"]:
        recommendations.append(
            "2. Corrupted model files found. These need to be re-downloaded or retrained:"
        )
        for file_path, _ in issues["model_files"]:
            recommendations.append(f"   - {file_path}")
        recommendations.append(
            "   Run: python -m src.emotion_clf_pipeline.azure_sync --download-models"
        )
    
    if not any(issues.values()):
        recommendations.append("✅ No compatibility issues found!")
    
    return recommendations


def main():
    """Main function to check and fix PyTorch compatibility issues."""
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    logger.info(f"🔍 Scanning project: {project_root}")
    logger.info("🔍 Checking for PyTorch 2.6 compatibility issues...")
    
    # Find problematic files
    issues = find_problematic_files(project_root)
    
    # Report findings
    total_issues = sum(len(file_list) for file_list in issues.values())
    
    if total_issues == 0:
        logger.info("✅ No compatibility issues found!")
        return 0
    
    logger.warning(f"⚠️  Found {total_issues} compatibility issues:")
    
    for category, file_list in issues.items():
        if file_list:
            logger.warning(f"  {category}: {len(file_list)} files")
            for file_path, error in file_list[:3]:  # Show first 3
                logger.warning(f"    - {file_path}: {error}")
            if len(file_list) > 3:
                logger.warning(f"    ... and {len(file_list) - 3} more")
    
    # Ask for confirmation to fix
    try:
        user_input = input("\n🛠️  Fix corrupted files? (y/N): ").strip().lower()
        if user_input in ['y', 'yes']:
            logger.info("🛠️  Fixing corrupted files...")
            fixed_counts = fix_corrupted_files(issues, backup=True)
            
            logger.info("✅ Fix results:")
            for category, count in fixed_counts.items():
                if count > 0:
                    logger.info(f"  {category}: {count}")
        else:
            logger.info("⏭️  Skipping automatic fixes")
    except KeyboardInterrupt:
        logger.info("\n⏭️  Skipping automatic fixes")
    
    # Generate recommendations
    recommendations = generate_recommendations(issues)
    
    logger.info("\n📋 Recommendations:")
    for rec in recommendations:
        logger.info(rec)
    
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main()) 