#!/usr/bin/env python
"""
Local test runner for VoyageurCompass project.
Runs both backend and frontend tests for local development.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.frontend_dir = self.project_root / "Design" / "frontend"
        
    def run_backend_tests(self, test_type="all", coverage=False, verbose=True):
        """Run Django backend tests."""
        print("🧪 Running Backend Tests...")
        
        os.chdir(self.project_root)
        
        cmd = ["pytest"]
        
        if verbose:
            cmd.append("-v")
            
        if coverage:
            cmd.extend([
                "--cov=Data",
                "--cov=Analytics", 
                "--cov=Core",
                "--cov-report=term-missing"
            ])
        
        # Filter by test type
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "api":
            cmd.extend(["-m", "api"])
        elif test_type == "models":
            cmd.extend(["-m", "models"])
        elif test_type == "fast":
            cmd.extend(["-m", "not slow"])
        
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Backend tests passed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Backend tests failed with exit code {e.returncode}")
            return False
    
    def run_frontend_tests(self, coverage=False, watch=False):
        """Run React frontend tests."""
        print("🧪 Running Frontend Tests...")
        
        os.chdir(self.frontend_dir)
        
        cmd = ["npm", "run"]
        
        if coverage:
            cmd.append("test:coverage")
        elif watch:
            cmd.append("test:watch")
        else:
            cmd.append("test")
        
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Frontend tests passed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Frontend tests failed with exit code {e.returncode}")
            return False
    
    def run_lint_checks(self):
        """Run code quality checks."""
        print("🔍 Running Code Quality Checks...")
        
        # Backend linting (if configured)
        backend_lint_success = True
        try:
            os.chdir(self.project_root)
            subprocess.run(["flake8", "."], check=True)
            print("✅ Backend linting passed!")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Backend linting skipped (flake8 not found or failed)")
            backend_lint_success = False
        
        # Frontend linting
        frontend_lint_success = True
        try:
            os.chdir(self.frontend_dir)
            subprocess.run(["npm", "run", "lint"], check=True)
            print("✅ Frontend linting passed!")
        except subprocess.CalledProcessError:
            print("❌ Frontend linting failed")
            frontend_lint_success = False
        
        return backend_lint_success and frontend_lint_success
    
    
    def run_all_tests(self, args):
        """Run all tests based on arguments."""
        success = True
        
        if args.lint:
            success = self.run_lint_checks() and success
        
        if args.backend or args.all:
            success = self.run_backend_tests(
                test_type=args.test_type,
                coverage=args.coverage,
                verbose=args.verbose
            ) and success
        
        if args.frontend or args.all:
            success = self.run_frontend_tests(
                coverage=args.coverage,
                watch=args.watch
            ) and success
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VoyageurCompass Test Runner")
    
    # Test selection
    parser.add_argument("--all", action="store_true", default=True,
                       help="Run all tests (default)")
    parser.add_argument("--backend", action="store_true",
                       help="Run only backend tests")
    parser.add_argument("--frontend", action="store_true",
                       help="Run only frontend tests")
    
    # Test types
    parser.add_argument("--test-type", choices=["all", "unit", "integration", "api", "models", "fast"],
                       default="all", help="Type of tests to run")
    
    # Options
    parser.add_argument("--coverage", action="store_true",
                       help="Enable coverage reporting")
    parser.add_argument("--watch", action="store_true",
                       help="Run frontend tests in watch mode")
    parser.add_argument("--lint", action="store_true",
                       help="Run code quality checks")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                       help="Quiet output")
    
    args = parser.parse_args()
    
    # If specific test types are selected, don't run all
    if args.backend or args.frontend:
        args.all = False
    
    runner = TestRunner()
    
    print("🚀 Starting VoyageurCompass Test Suite...")
    print("=" * 50)
    
    try:
        success = runner.run_all_tests(args)
        
        print("=" * 50)
        if success:
            print("✅ All tests completed successfully!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()