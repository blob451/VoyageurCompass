#!/usr/bin/env python
"""
Fast test runner for development with performance optimizations.
"""
import os
import subprocess  # nosec B404 - Used for test performance benchmarking
import sys
from pathlib import Path


def run_fast_tests():
    """Run tests with maximum performance optimizations."""
    os.environ["TEST_FAST_MODE"] = "true"
    os.environ["DJANGO_SETTINGS_MODULE"] = "VoyageurCompass.test_settings"

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    # Fast test command with parallel execution
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--numprocesses=auto",
        "--dist=worksteal",
        "--tb=short",
        "--durations=10",
        "--reuse-db",
        "--nomigrations",
        "-v",
    ]

    # Add any additional arguments passed to script
    cmd.extend(sys.argv[1:])

    print("🚀 Running fast parallel tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1


if __name__ == "__main__":
    exit_code = run_fast_tests()
    sys.exit(exit_code)
