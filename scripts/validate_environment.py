#!/usr/bin/env python3
"""
Environment Configuration Validator for VoyageurCompass
Validates that all required environment variables are set correctly
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


class EnvironmentValidator:
    """Validates environment configuration for different deployment stages"""

    # Required variables for each environment
    REQUIRED_VARS = {
        "production": [
            "SECRET_KEY",
            "DEBUG",
            "ALLOWED_HOSTS",
            "SITE_URL",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "DB_HOST",
            "REDIS_HOST",
            "REDIS_PASSWORD",
            "EMAIL_HOST",
            "EMAIL_HOST_USER",
            "EMAIL_HOST_PASSWORD",
            "CORS_ALLOWED_ORIGINS",
            "CSRF_TRUSTED_ORIGINS",
        ],
        "staging": [
            "SECRET_KEY",
            "DEBUG",
            "ALLOWED_HOSTS",
            "SITE_URL",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "DB_HOST",
            "REDIS_HOST",
            "CORS_ALLOWED_ORIGINS",
        ],
        "development": [
            "SECRET_KEY",
            "DEBUG",
            "ALLOWED_HOSTS",
            "DB_NAME",
            "DB_USER",
            "DB_HOST",
        ],
    }

    # Security-critical variables that must meet specific criteria
    SECURITY_VARS = {
        "SECRET_KEY": lambda v: len(v) >= 50
        and v != "django-insecure-dev-key-change-this-in-production",
        "DEBUG": lambda v: v.lower() == "false",
        "SECURE_SSL_REDIRECT": lambda v: v.lower() == "true",
        "CSRF_COOKIE_SECURE": lambda v: v.lower() == "true",
        "SESSION_COOKIE_SECURE": lambda v: v.lower() == "true",
    }

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize validator

        Args:
            env_file: Path to .env file to validate
        """
        self.env_file = env_file
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

        if env_file and os.path.exists(env_file):
            self.load_env_file(env_file)

    def load_env_file(self, env_file: str) -> None:
        """Load environment variables from file"""
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    def validate_required(self, environment: str) -> bool:
        """
        Validate that all required variables are present

        Args:
            environment: Environment type (production, staging, development)

        Returns:
            True if all required variables are present
        """
        required = self.REQUIRED_VARS.get(environment, [])
        missing = []

        for var in required:
            value = os.environ.get(var)
            if not value or value.strip() == "":
                missing.append(var)
                self.errors.append(f"Missing required variable: {var}")

        if missing:
            self.errors.append(
                f"Missing {len(missing)} required variables for {environment} environment"
            )
            return False

        self.info.append(f"✅ All {len(required)} required variables present")
        return True

    def validate_security(self, environment: str) -> bool:
        """
        Validate security-critical variables

        Args:
            environment: Environment type

        Returns:
            True if security validation passes
        """
        if environment == "development":
            self.info.append(
                "ℹ️ Skipping security validation for development environment"
            )
            return True

        passed = True

        for var, validator in self.SECURITY_VARS.items():
            value = os.environ.get(var, "")

            if not value:
                if environment == "production":
                    self.errors.append(f"Security variable {var} is not set")
                    passed = False
                else:
                    self.warnings.append(f"Security variable {var} is not set")
            elif not validator(value):
                if environment == "production":
                    self.errors.append(f"Security variable {var} has invalid value")
                    passed = False
                else:
                    self.warnings.append(
                        f"Security variable {var} may have insecure value"
                    )

        if passed:
            self.info.append("✅ Security validation passed")

        return passed

    def validate_database(self) -> bool:
        """Validate database configuration"""
        db_host = os.environ.get("DB_HOST", "")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "")

        if not db_host:
            self.errors.append("Database host not configured")
            return False

        try:
            port = int(db_port)
            if port < 1 or port > 65535:
                self.errors.append(f"Invalid database port: {db_port}")
                return False
        except ValueError:
            self.errors.append(f"Invalid database port: {db_port}")
            return False

        if not db_name:
            self.errors.append("Database name not configured")
            return False

        self.info.append(f"✅ Database configured: {db_name}@{db_host}:{db_port}")
        return True

    def validate_urls(self) -> bool:
        """Validate URL configurations"""
        site_url = os.environ.get("SITE_URL", "")
        cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")

        passed = True

        if site_url:
            try:
                parsed = urlparse(site_url)
                if not parsed.scheme or not parsed.netloc:
                    self.errors.append(f"Invalid SITE_URL: {site_url}")
                    passed = False
                else:
                    self.info.append(f"✅ SITE_URL validated: {site_url}")
            except Exception as e:
                self.errors.append(f"Invalid SITE_URL: {e}")
                passed = False

        if cors_origins:
            for origin in cors_origins.split(","):
                origin = origin.strip()
                try:
                    parsed = urlparse(origin)
                    if not parsed.scheme or not parsed.netloc:
                        self.warnings.append(f"Invalid CORS origin: {origin}")
                except Exception:
                    self.warnings.append(f"Invalid CORS origin: {origin}")

        return passed

    def validate_redis(self) -> bool:
        """Validate Redis configuration"""
        redis_host = os.environ.get("REDIS_HOST", "")
        redis_port = os.environ.get("REDIS_PORT", "6379")

        if not redis_host:
            self.warnings.append("Redis host not configured (optional in development)")
            return True

        try:
            port = int(redis_port)
            if port < 1 or port > 65535:
                self.errors.append(f"Invalid Redis port: {redis_port}")
                return False
        except ValueError:
            self.errors.append(f"Invalid Redis port: {redis_port}")
            return False

        self.info.append(f"✅ Redis configured: {redis_host}:{redis_port}")
        return True

    def validate_performance(self, environment: str) -> bool:
        """Validate performance settings"""
        if environment == "development":
            return True

        workers = os.environ.get("GUNICORN_WORKERS", "1")
        timeout = os.environ.get("GUNICORN_TIMEOUT", "30")

        try:
            worker_count = int(workers)
            if worker_count < 1:
                self.warnings.append(f"Low worker count: {worker_count}")
            elif worker_count > 20:
                self.warnings.append(f"Very high worker count: {worker_count}")

            timeout_val = int(timeout)
            if timeout_val < 10:
                self.warnings.append(f"Very low timeout: {timeout_val}s")
            elif timeout_val > 300:
                self.warnings.append(f"Very high timeout: {timeout_val}s")
        except ValueError:
            self.warnings.append("Invalid performance settings")

        return True

    def validate(self, environment: str = "production") -> bool:
        """
        Run full validation

        Args:
            environment: Environment type to validate for

        Returns:
            True if validation passes
        """
        print(f"\n🔍 Validating {environment} environment configuration...")
        print("=" * 60)

        # Clear previous results
        self.errors = []
        self.warnings = []
        self.info = []

        # Run validations
        checks = [
            ("Required Variables", lambda: self.validate_required(environment)),
            ("Security Settings", lambda: self.validate_security(environment)),
            ("Database Configuration", self.validate_database),
            ("URL Configuration", self.validate_urls),
            ("Redis Configuration", self.validate_redis),
            ("Performance Settings", lambda: self.validate_performance(environment)),
        ]

        all_passed = True

        for check_name, check_func in checks:
            print(f"\nChecking {check_name}...")
            if not check_func():
                all_passed = False

        # Print results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        if self.info:
            print("\n📋 Information:")
            for msg in self.info:
                print(f"  {msg}")

        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  - {msg}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  - {msg}")

        if all_passed and not self.errors:
            print(f"\n✅ Environment validation PASSED for {environment}")
            if self.warnings:
                print(f"   (with {len(self.warnings)} warnings)")
        else:
            print(f"\n❌ Environment validation FAILED for {environment}")
            print(f"   Found {len(self.errors)} errors")

        return all_passed and not self.errors

    def generate_report(self) -> Dict:
        """Generate validation report"""
        return {
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate environment configuration")
    parser.add_argument(
        "--env",
        choices=["production", "staging", "development"],
        default="production",
        help="Environment to validate for",
    )
    parser.add_argument("--file", help="Path to .env file to validate")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )

    args = parser.parse_args()

    # Create validator
    validator = EnvironmentValidator(args.file)

    # Run validation
    passed = validator.validate(args.env)

    # In strict mode, warnings are errors
    if args.strict and validator.warnings:
        passed = False
        print("\n⚠️ Strict mode: Treating warnings as errors")

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
