"""
Django management command for model preloading and warm-up operations.
"""

import asyncio
import json
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from Analytics.services.model_warmup_service import get_model_warmup_service


class Command(BaseCommand):
    help = 'Preload and warm up LLM models for optimal performance'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models',
            nargs='+',
            help='Specific models to warm up (default: all configured models)',
            default=None
        )

        parser.add_argument(
            '--preload-only',
            action='store_true',
            help='Only preload models without warm-up',
        )

        parser.add_argument(
            '--warmup-only',
            action='store_true',
            help='Only warm up models without preloading',
        )

        parser.add_argument(
            '--batch-size',
            type=int,
            default=3,
            help='Number of models to warm up concurrently',
        )

        parser.add_argument(
            '--force',
            action='store_true',
            help='Force warmup even if models were recently warmed up',
        )

        parser.add_argument(
            '--output-format',
            choices=['text', 'json'],
            default='text',
            help='Output format for results',
        )

    def handle(self, *args, **options):
        warmup_service = get_model_warmup_service()

        start_time = timezone.now()
        self.stdout.write(
            self.style.SUCCESS(f"Starting model warmup process at {start_time}")
        )

        results = {
            "start_time": start_time.isoformat(),
            "preload_results": None,
            "warmup_results": None,
            "total_time": 0.0,
            "status": "starting"
        }

        try:
            # Handle preload-only mode
            if options['preload_only']:
                self.stdout.write("Preloading models...")
                preload_results = warmup_service.preload_priority_models()
                results["preload_results"] = preload_results

                if options['output_format'] == 'json':
                    self.stdout.write(json.dumps(preload_results, indent=2))
                else:
                    self._display_preload_results(preload_results)

            # Handle warmup-only mode
            elif options['warmup_only']:
                models = options.get('models')

                self.stdout.write(f"Warming up models: {models or 'default models'}")
                warmup_results = asyncio.run(warmup_service.warmup_models_async(
                    models=models,
                    batch_size=options['batch_size']
                ))
                results["warmup_results"] = warmup_results

                if options['output_format'] == 'json':
                    self.stdout.write(json.dumps(warmup_results, indent=2))
                else:
                    self._display_warmup_results(warmup_results)

            # Handle full initialization (default)
            else:
                self.stdout.write("Running full model initialization...")
                initialization_results = warmup_service.startup_initialization()
                results.update(initialization_results)

                if options['output_format'] == 'json':
                    self.stdout.write(json.dumps(initialization_results, indent=2))
                else:
                    self._display_initialization_results(initialization_results)

            # Calculate total time
            end_time = timezone.now()
            total_time = (end_time - start_time).total_seconds()
            results["total_time"] = total_time
            results["end_time"] = end_time.isoformat()
            results["status"] = "completed"

            self.stdout.write(
                self.style.SUCCESS(
                    f"\nModel warmup completed successfully in {total_time:.2f} seconds"
                )
            )

        except Exception as e:
            error_msg = f"Model warmup failed: {str(e)}"
            results["status"] = "failed"
            results["error"] = str(e)

            if options['output_format'] == 'json':
                self.stdout.write(json.dumps(results, indent=2))
            else:
                self.stdout.write(self.style.ERROR(error_msg))

            raise CommandError(error_msg)

    def _display_preload_results(self, results):
        """Display preload results in human-readable format."""
        self.stdout.write("\n" + "="*50)
        self.stdout.write(self.style.HTTP_INFO("PRELOAD RESULTS"))
        self.stdout.write("="*50)

        self.stdout.write(f"Models preloaded: {results.get('preloaded', 0)}")

        # Display successfully preloaded models
        preloaded_models = results.get('models', [])
        if preloaded_models:
            self.stdout.write("\nSuccessfully preloaded models:")
            for model in preloaded_models:
                self.stdout.write(f"  ✓ {model}")

        # Display failures
        failures = results.get('failures', [])
        if failures:
            self.stdout.write(self.style.WARNING("\nPreload failures:"))
            for failure in failures:
                self.stdout.write(f"  ✗ {failure['model']}: {failure['error']}")

        # Display resource check
        resource_check = results.get('resource_check')
        if resource_check:
            self.stdout.write(f"\nResource Status:")
            self.stdout.write(f"  Available Memory: {resource_check.get('available_memory_gb', 'Unknown')} GB")
            self.stdout.write(f"  GPU Available: {resource_check.get('gpu_available', 'Unknown')}")

    def _display_warmup_results(self, results):
        """Display warmup results in human-readable format."""
        self.stdout.write("\n" + "="*50)
        self.stdout.write(self.style.HTTP_INFO("WARMUP RESULTS"))
        self.stdout.write("="*50)

        total_models = results.get('total_models', 0)
        successful = results.get('successful_warmups', 0)
        failed = results.get('failed_warmups', 0)

        self.stdout.write(f"Total models: {total_models}")
        self.stdout.write(f"Successful warmups: {successful}")
        self.stdout.write(f"Failed warmups: {failed}")

        if total_models > 0:
            success_rate = (successful / total_models) * 100
            self.stdout.write(f"Success rate: {success_rate:.1f}%")

        # Display warmup times
        warmup_times = results.get('warmup_times', {})
        if warmup_times:
            self.stdout.write("\nWarmup times:")
            for model, time_taken in warmup_times.items():
                status = "✓" if time_taken > 0 else "✗"
                self.stdout.write(f"  {status} {model}: {time_taken:.2f}s")

        # Display errors
        errors = results.get('errors', [])
        if errors:
            self.stdout.write(self.style.WARNING("\nWarmup errors:"))
            for error in errors:
                self.stdout.write(f"  ✗ {error}")

    def _display_initialization_results(self, results):
        """Display full initialization results."""
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.HTTP_INFO("MODEL INITIALIZATION RESULTS"))
        self.stdout.write("="*60)

        status = results.get('status', 'unknown')
        total_time = results.get('total_time', 0)

        if status == 'completed':
            self.stdout.write(self.style.SUCCESS(f"Status: {status.upper()}"))
        else:
            self.stdout.write(self.style.ERROR(f"Status: {status.upper()}"))

        self.stdout.write(f"Total time: {total_time:.2f} seconds")

        # Display preload results if available
        preload_results = results.get('preload_results')
        if preload_results:
            self._display_preload_results(preload_results)

        # Display warmup results if available
        warmup_results = results.get('warmup_results')
        if warmup_results:
            self._display_warmup_results(warmup_results)

        # Display error if failed
        error = results.get('error')
        if error:
            self.stdout.write(self.style.ERROR(f"\nError: {error}"))

    def _display_metrics(self, warmup_service):
        """Display current warmup service metrics."""
        metrics = warmup_service.get_performance_metrics()
        status = warmup_service.get_warmup_status()

        self.stdout.write("\n" + "="*50)
        self.stdout.write(self.style.HTTP_INFO("CURRENT METRICS"))
        self.stdout.write("="*50)

        self.stdout.write(f"Total warmups: {metrics.get('total_warmups', 0)}")
        self.stdout.write(f"Successful warmups: {metrics.get('successful_warmups', 0)}")
        self.stdout.write(f"Failed warmups: {metrics.get('failed_warmups', 0)}")
        self.stdout.write(f"Models preloaded: {metrics.get('models_preloaded', 0)}")

        # Display currently preloaded models
        preloaded = status.get('preloaded_models', [])
        if preloaded:
            self.stdout.write(f"\nCurrently preloaded models:")
            for model in preloaded:
                self.stdout.write(f"  • {model}")

        # Display completed warmups
        completed = status.get('completed', [])
        if completed:
            self.stdout.write(f"\nCompleted warmups:")
            for model in completed:
                self.stdout.write(f"  ✓ {model}")