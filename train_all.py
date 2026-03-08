"""
Main training script - Train all tasks sequentially or individually.

Usage:
    python train_all.py --task all              # Train all tasks
    python train_all.py --task a                # Train Task A only
    python train_all.py --task b                # Train Task B only
    python train_all.py --task c                # Train Task C only
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time


def run_training(task: str, config_path: str):
    """
    Run training for a specific task.
    
    Args:
        task: Task name ('a', 'b', or 'c')
        config_path: Path to config file
    """
    print("\n" + "=" * 80)
    print(f"Training Task {task.upper()}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Set PYTHONPATH to include project root
    project_root = str(Path(__file__).parent.absolute())
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Run training script
    cmd = [sys.executable, f"task_{task}/train.py", "--config", config_path]
    result = subprocess.run(cmd, cwd=project_root, env=env)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    if result.returncode == 0:
        print(f"\n✓ Task {task.upper()} training completed successfully")
        print(f"  Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
        return True
    else:
        print(f"\n✗ Task {task.upper()} training failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train SemEval-2026 Task 13 models")
    parser.add_argument(
        '--task',
        type=str,
        choices=['a', 'b', 'c', 'all'],
        required=True,
        help='Which task(s) to train'
    )
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip tasks that already have checkpoints'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SemEval-2026 Task 13 - Model Training")
    print("=" * 80)
    print(f"Mode: {'All tasks' if args.task == 'all' else f'Task {args.task.upper()} only'}")
    print("=" * 80)
    
    results = {}
    overall_start = time.time()
    
    # Task configurations
    tasks = {
        'a': 'task_a/config.yaml',
        'b': 'task_b/config.yaml',
        'c': 'task_c/config.yaml'
    }
    
    # Filter tasks based on argument
    if args.task == 'all':
        tasks_to_run = tasks
    else:
        tasks_to_run = {args.task: tasks[args.task]}
    
    # Check for existing checkpoints
    if args.skip_completed:
        print("\nChecking for existing checkpoints...")
        for task in list(tasks_to_run.keys()):
            checkpoint_dir = Path(f"checkpoints/task_{task}")
            if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
                print(f"  Task {task.upper()}: Found checkpoint, skipping")
                tasks_to_run.pop(task)
            else:
                print(f"  Task {task.upper()}: No checkpoint found, will train")
        
        if not tasks_to_run:
            print("\n✓ All tasks already have checkpoints!")
            return
    
    # Run training for each task
    for task, config in tasks_to_run.items():
        success = run_training(task, config)
        results[task] = success
        
        if not success:
            print(f"\n⚠ Task {task.upper()} failed. Continue with next task? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("\nTraining stopped by user.")
                break
    
    # Print summary
    overall_elapsed = time.time() - overall_start
    hours = int(overall_elapsed // 3600)
    minutes = int((overall_elapsed % 3600) // 60)
    seconds = int(overall_elapsed % 60)
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    for task, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  Task {task.upper()}: {status}")
    
    print(f"\nTotal time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("=" * 80)
    
    # Exit with error if any task failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
