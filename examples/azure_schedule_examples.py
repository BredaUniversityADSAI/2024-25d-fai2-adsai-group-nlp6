#!/usr/bin/env python3
"""
Azure ML Pipeline Scheduling Examples

This script demonstrates how to use the Azure ML pipeline scheduling functionality
for the Emotion Classification Pipeline project.

Examples include:
1. Creating daily schedules
2. Creating weekly schedules  
3. Creating monthly schedules
4. Using cron expressions
5. Managing existing schedules
"""

import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def example_1_daily_schedule():
    """Example 1: Create a daily schedule that runs at midnight"""
    print("=== Example 1: Daily Schedule ===")
    print("Command to create a daily schedule that runs at midnight UTC:")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'daily-training-midnight' \\")
    print("  --daily \\")
    print("  --hour 0 \\")
    print("  --minute 0 \\")
    print("  --timezone 'UTC' \\")
    print("  --description 'Daily emotion classification training at midnight' \\")
    print("  --enabled \\")
    print("  --mode azure")
    print()

def example_2_weekly_schedule():
    """Example 2: Create a weekly schedule that runs on Sundays"""
    print("=== Example 2: Weekly Schedule ===")
    print("Command to create a weekly schedule that runs on Sundays at 2 AM:")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'weekly-training-sunday' \\")
    print("  --weekly 0 \\")  # 0 = Sunday
    print("  --hour 2 \\")
    print("  --minute 0 \\")
    print("  --timezone 'UTC' \\")
    print("  --description 'Weekly emotion classification training on Sundays' \\")
    print("  --enabled \\")
    print("  --mode azure")
    print()

def example_3_monthly_schedule():
    """Example 3: Create a monthly schedule that runs on the 1st of each month"""
    print("=== Example 3: Monthly Schedule ===")
    print("Command to create a monthly schedule that runs on the 1st at 3 AM:")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'monthly-training-first' \\")
    print("  --monthly 1 \\")  # 1st day of month
    print("  --hour 3 \\")
    print("  --minute 0 \\")
    print("  --timezone 'UTC' \\")
    print("  --description 'Monthly emotion classification training on 1st' \\")
    print("  --enabled \\")
    print("  --mode azure")
    print()

def example_4_cron_schedule():
    """Example 4: Create a schedule using cron expression"""
    print("=== Example 4: Cron Expression Schedule ===")
    print("Command to create a schedule using cron expression (every 6 hours):")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'every-6-hours' \\")
    print("  --cron '0 */6 * * *' \\")  # Every 6 hours
    print("  --timezone 'UTC' \\")
    print("  --description 'Emotion classification training every 6 hours' \\")
    print("  --enabled \\")
    print("  --mode azure")
    print()

def example_5_custom_pipeline_config():
    """Example 5: Create a schedule with custom pipeline configuration"""
    print("=== Example 5: Custom Pipeline Configuration ===")
    print("Command to create a schedule with custom pipeline settings:")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'custom-config-daily' \\")
    print("  --daily \\")
    print("  --hour 6 \\")
    print("  --minute 30 \\")
    print("  --timezone 'America/New_York' \\")
    print("  --description 'Custom pipeline configuration example' \\")
    print("  --enabled \\")
    print("  --pipeline-name 'custom-emotion-pipeline' \\")
    print("  --data-path './data/custom_processed' \\")
    print("  --output-path './models/custom_output' \\")
    print("  --experiment-name 'custom-emotion-experiment' \\")
    print("  --compute-target 'gpu-cluster' \\")
    print("  --mode azure")
    print()

def example_6_schedule_management():
    """Example 6: Schedule management commands"""
    print("=== Example 6: Schedule Management ===")
    print("Commands for managing existing schedules:")
    print()
    
    print("1. List all schedules:")
    print("python -m src.emotion_clf_pipeline.cli schedule list --mode azure")
    print()
    
    print("2. Get details of a specific schedule:")
    print("python -m src.emotion_clf_pipeline.cli schedule details \\")
    print("  --schedule-name 'daily-training-midnight' \\")
    print("  --mode azure")
    print()
    
    print("3. Enable a disabled schedule:")
    print("python -m src.emotion_clf_pipeline.cli schedule enable \\")
    print("  --schedule-name 'daily-training-midnight' \\")
    print("  --mode azure")
    print()
    
    print("4. Disable an active schedule:")
    print("python -m src.emotion_clf_pipeline.cli schedule disable \\")
    print("  --schedule-name 'daily-training-midnight' \\")
    print("  --mode azure")
    print()
    
    print("5. Delete a schedule:")
    print("python -m src.emotion_clf_pipeline.cli schedule delete \\")
    print("  --schedule-name 'daily-training-midnight' \\")
    print("  --mode azure")
    print()

def example_7_setup_defaults():
    """Example 7: Setup common schedule patterns"""
    print("=== Example 7: Setup Default Schedule Patterns ===")
    print("Command to setup common schedule patterns automatically:")
    print("python -m src.emotion_clf_pipeline.cli schedule setup-defaults \\")
    print("  --mode azure")
    print()
    print("This will create three schedules:")
    print("- Daily at midnight UTC")
    print("- Weekly on Sundays at 2 AM UTC")
    print("- Monthly on 1st at 3 AM UTC")
    print()

def example_8_timezone_examples():
    """Example 8: Different timezone examples"""
    print("=== Example 8: Timezone Examples ===")
    print("Examples with different timezones:")
    print()
    
    print("1. Eastern Time (New York):")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'daily-eastern' \\")
    print("  --daily --hour 9 --minute 0 \\")
    print("  --timezone 'America/New_York' \\")
    print("  --mode azure")
    print()
    
    print("2. Pacific Time (Los Angeles):")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'daily-pacific' \\")
    print("  --daily --hour 6 --minute 0 \\")
    print("  --timezone 'America/Los_Angeles' \\")
    print("  --mode azure")
    print()
    
    print("3. Central European Time:")
    print("python -m src.emotion_clf_pipeline.cli schedule create \\")
    print("  --schedule-name 'daily-cet' \\")
    print("  --daily --hour 8 --minute 0 \\")
    print("  --timezone 'Europe/Berlin' \\")
    print("  --mode azure")
    print()

def main():
    """Run all examples"""
    print("Azure ML Pipeline Scheduling Examples")
    print("=" * 50)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Prerequisites:")
    print("1. Azure ML workspace configured")
    print("2. Azure CLI authenticated")
    print("3. Required compute targets available")
    print("4. Pipeline dependencies installed")
    print()
    
    example_1_daily_schedule()
    example_2_weekly_schedule()
    example_3_monthly_schedule()
    example_4_cron_schedule()
    example_5_custom_pipeline_config()
    example_6_schedule_management()
    example_7_setup_defaults()
    example_8_timezone_examples()
    
    print("=== Additional Notes ===")
    print("- Always use --mode azure for Azure ML scheduling")
    print("- Schedules are created in disabled state by default unless --enabled is used")
    print("- All schedules appear in Azure ML Studio under 'Schedules' section")
    print("- Cron expressions follow standard cron format (minute hour day month weekday)")
    print("- Timezones support both abbreviations (UTC, EST) and full names (America/New_York)")
    print()
    print("For more information, use:")
    print("python -m src.emotion_clf_pipeline.cli schedule --help")

if __name__ == "__main__":
    main()
