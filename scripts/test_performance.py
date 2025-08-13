#!/usr/bin/env python
"""
Performance testing and benchmarking script.
"""
import os
import subprocess  # nosec B404 - Used for test performance benchmarking
import sys
import time
from pathlib import Path


def benchmark_tests():
    """Run performance benchmarks on test suite."""
    os.environ["TEST_FAST_MODE"] = "true"
    os.environ["DJANGO_SETTINGS_MODULE"] = "VoyageurCompass.test_settings"

    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    print("📊 Running test performance benchmarks...")

    # Single threaded baseline
    print("\n1️⃣ Single-threaded baseline:")
    start_time = time.time()
    cmd_single = [
        sys.executable,
        "-m",
        "pytest",
        "Core/test/test_health.py",
        "--tb=short",
        "--durations=3",
        "-v",
    ]
    subprocess.run(cmd_single, check=False)
    single_time = time.time() - start_time

    # Parallel execution
    print("\n🔥 Multi-threaded optimized:")
    start_time = time.time()
    cmd_parallel = [
        sys.executable,
        "-m",
        "pytest",
        "Core/test/test_health.py",
        "--numprocesses=2",
        "--dist=worksteal",
        "--tb=short",
        "--durations=3",
        "-v",
    ]
    subprocess.run(cmd_parallel, check=False)
    parallel_time = time.time() - start_time

    # Results
    print(f"\n📈 Performance Results:")
    print(f"   Single-threaded: {single_time:.2f}s")
    print(f"   Multi-threaded:  {parallel_time:.2f}s")
    if single_time > 0:
        speedup = single_time / parallel_time
        print(f"   Speedup:         {speedup:.2f}x")
        improvement = ((single_time - parallel_time) / single_time) * 100
        print(f"   Improvement:     {improvement:.1f}%")


if __name__ == "__main__":
    benchmark_tests()
