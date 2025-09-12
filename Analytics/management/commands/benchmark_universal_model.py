"""
Universal LSTM Performance Benchmarking Command
==============================================

Comprehensive performance benchmarking for Universal LSTM including:
- Prediction speed and latency analysis
- Memory usage optimization validation
- Throughput testing under load
- Accuracy comparison vs baseline models
"""

import concurrent.futures
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import psutil
from django.core.management.base import BaseCommand, CommandError

from Analytics.services.universal_predictor import UniversalLSTMAnalyticsService
from Data.models import Stock

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Comprehensive Universal LSTM performance benchmarking"

    def add_arguments(self, parser):
        parser.add_argument(
            "--benchmark-type",
            type=str,
            choices=["speed", "memory", "throughput", "accuracy", "all"],
            default="all",
            help="Type of benchmark to run",
        )
        parser.add_argument("--sample-size", type=int, default=50, help="Number of stocks to test")
        parser.add_argument(
            "--concurrent-threads", type=int, default=4, help="Number of concurrent threads for throughput testing"
        )
        parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each benchmark")
        parser.add_argument(
            "--output-file",
            type=str,
            default="Temp/universal_model_benchmarks.json",
            help="Output file for benchmark results",
        )
        parser.add_argument(
            "--target-latency", type=float, default=100.0, help="Target prediction latency in milliseconds"
        )
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    def handle(self, *args, **options):
        """Execute Universal LSTM performance benchmarking."""

        self.stdout.write(self.style.SUCCESS("=== Universal LSTM Performance Benchmarking ==="))

        # Initialize benchmarking components
        try:
            self.service = UniversalLSTMAnalyticsService()
            self.verbose = options["verbose"]

            if not self.service.model:
                raise CommandError("Universal LSTM model not loaded")

            self.stdout.write("Universal LSTM service initialized for benchmarking")

        except Exception as e:
            raise CommandError(f"Failed to initialize benchmarking components: {e}")

        # Run benchmark tests
        results = {}
        benchmark_type = options["benchmark_type"]

        if benchmark_type in ["speed", "all"]:
            self.stdout.write("\n--- Running Speed Benchmarks ---")
            results["speed"] = self.run_speed_benchmarks(
                options["sample_size"], options["iterations"], options["target_latency"]
            )

        if benchmark_type in ["memory", "all"]:
            self.stdout.write("\n--- Running Memory Usage Benchmarks ---")
            results["memory"] = self.run_memory_benchmarks(options["sample_size"])

        if benchmark_type in ["throughput", "all"]:
            self.stdout.write("\n--- Running Throughput Benchmarks ---")
            results["throughput"] = self.run_throughput_benchmarks(
                options["sample_size"], options["concurrent_threads"]
            )

        if benchmark_type in ["accuracy", "all"]:
            self.stdout.write("\n--- Running Accuracy Benchmarks ---")
            results["accuracy"] = self.run_accuracy_benchmarks(options["sample_size"])

        # Save results
        output_file = options["output_file"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        benchmark_report = {
            "timestamp": datetime.now().isoformat(),
            "model_info": self.service.get_model_info(),
            "benchmark_config": {
                "sample_size": options["sample_size"],
                "iterations": options["iterations"],
                "concurrent_threads": options["concurrent_threads"],
                "target_latency_ms": options["target_latency"],
            },
            "benchmark_results": results,
            "summary": self.generate_benchmark_summary(results, options["target_latency"]),
        }

        with open(output_file, "w") as f:
            json.dump(benchmark_report, f, indent=2, default=str)

        self.stdout.write(f"\nBenchmark results saved to: {output_file}")

        # Display summary
        self.display_benchmark_summary(benchmark_report["summary"])

    def run_speed_benchmarks(self, sample_size: int, iterations: int, target_latency: float) -> Dict[str, Any]:
        """Run comprehensive speed benchmarking tests."""

        self.stdout.write("Speed benchmarking started...")

        # Get test stocks
        test_stocks = self.get_test_stocks(sample_size)

        # Single prediction latency test
        single_prediction_times = []

        for iteration in range(iterations):
            if self.verbose:
                self.stdout.write(f"  Speed iteration {iteration + 1}/{iterations}")

            for stock in test_stocks:
                start_time = time.perf_counter()

                try:
                    result = self.service.predict_stock_price(stock, use_cache=False)
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    single_prediction_times.append(
                        {
                            "stock": stock,
                            "iteration": iteration,
                            "latency_ms": latency_ms,
                            "success": result is not None,
                        }
                    )

                except Exception as e:
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000

                    single_prediction_times.append(
                        {
                            "stock": stock,
                            "iteration": iteration,
                            "latency_ms": latency_ms,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Analyze speed results
        successful_predictions = [p for p in single_prediction_times if p["success"]]
        failed_predictions = [p for p in single_prediction_times if not p["success"]]

        if successful_predictions:
            latencies = [p["latency_ms"] for p in successful_predictions]

            speed_results = {
                "total_predictions": len(single_prediction_times),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(failed_predictions),
                "success_rate": len(successful_predictions) / len(single_prediction_times),
                "latency_stats": {
                    "mean_ms": np.mean(latencies),
                    "median_ms": np.median(latencies),
                    "min_ms": np.min(latencies),
                    "max_ms": np.max(latencies),
                    "std_ms": np.std(latencies),
                    "p95_ms": np.percentile(latencies, 95),
                    "p99_ms": np.percentile(latencies, 99),
                },
                "target_compliance": {
                    "target_latency_ms": target_latency,
                    "predictions_under_target": sum(1 for l in latencies if l < target_latency),
                    "compliance_rate": sum(1 for l in latencies if l < target_latency) / len(latencies),
                },
                "detailed_results": single_prediction_times,
            }
        else:
            speed_results = {
                "total_predictions": len(single_prediction_times),
                "successful_predictions": 0,
                "failed_predictions": len(failed_predictions),
                "success_rate": 0.0,
                "error": "No successful predictions to analyze",
                "detailed_results": single_prediction_times,
            }

        return speed_results

    def run_memory_benchmarks(self, sample_size: int) -> Dict[str, Any]:
        """Run memory usage benchmarking tests."""

        self.stdout.write("Memory benchmarking started...")

        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info()

        # Model memory footprint
        if hasattr(self.service, "model") and self.service.model:
            model_params = sum(p.numel() for p in self.service.model.parameters())
            model_memory_bytes = sum(p.numel() * p.element_size() for p in self.service.model.parameters())
            model_memory_mb = model_memory_bytes / (1024**2)
        else:
            model_params = 0
            model_memory_mb = 0

        # Memory usage during predictions
        test_stocks = self.get_test_stocks(min(sample_size, 20))  # Limit for memory testing
        memory_usage_during_predictions = []

        for i, stock in enumerate(test_stocks):
            if self.verbose and i % 5 == 0:
                self.stdout.write(f"  Memory test progress: {i}/{len(test_stocks)}")

            try:
                # Measure memory before prediction
                memory_before = process.memory_info()

                # Make prediction
                result = self.service.predict_stock_price(stock, use_cache=False)

                # Measure memory after prediction
                memory_after = process.memory_info()

                memory_usage_during_predictions.append(
                    {
                        "stock": stock,
                        "memory_before_mb": memory_before.rss / (1024**2),
                        "memory_after_mb": memory_after.rss / (1024**2),
                        "memory_delta_mb": (memory_after.rss - memory_before.rss) / (1024**2),
                        "success": result is not None,
                    }
                )

            except Exception as e:
                memory_usage_during_predictions.append({"stock": stock, "error": str(e), "success": False})

        # Analyze memory results
        successful_memory_tests = [m for m in memory_usage_during_predictions if m["success"]]

        if successful_memory_tests:
            memory_deltas = [m["memory_delta_mb"] for m in successful_memory_tests]
            peak_memory = max([m["memory_after_mb"] for m in successful_memory_tests])

            memory_results = {
                "baseline_memory_mb": baseline_memory.rss / (1024**2),
                "model_memory_mb": model_memory_mb,
                "model_parameters": model_params,
                "peak_memory_mb": peak_memory,
                "memory_efficiency": {
                    "avg_memory_delta_mb": np.mean(memory_deltas),
                    "max_memory_delta_mb": np.max(memory_deltas),
                    "min_memory_delta_mb": np.min(memory_deltas),
                },
                "memory_per_parameter_bytes": model_memory_bytes / model_params if model_params > 0 else 0,
                "detailed_measurements": memory_usage_during_predictions,
            }
        else:
            memory_results = {
                "baseline_memory_mb": baseline_memory.rss / (1024**2),
                "model_memory_mb": model_memory_mb,
                "model_parameters": model_params,
                "error": "No successful memory measurements",
                "detailed_measurements": memory_usage_during_predictions,
            }

        return memory_results

    def run_throughput_benchmarks(self, sample_size: int, concurrent_threads: int) -> Dict[str, Any]:
        """Run throughput benchmarking under concurrent load."""

        self.stdout.write(f"Throughput benchmarking started with {concurrent_threads} threads...")

        test_stocks = self.get_test_stocks(sample_size)

        def make_prediction(stock_symbol):
            """Make a single prediction (for threading)."""
            start_time = time.perf_counter()
            try:
                result = self.service.predict_stock_price(stock_symbol, use_cache=False)
                end_time = time.perf_counter()

                return {
                    "stock": stock_symbol,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": result is not None,
                    "thread_id": threading.current_thread().ident,
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "stock": stock_symbol,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": False,
                    "error": str(e),
                    "thread_id": threading.current_thread().ident,
                }

        # Run concurrent predictions
        throughput_start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            future_to_stock = {executor.submit(make_prediction, stock): stock for stock in test_stocks}

            concurrent_results = []
            for future in concurrent.futures.as_completed(future_to_stock):
                result = future.result()
                concurrent_results.append(result)

        throughput_end_time = time.perf_counter()
        total_throughput_time = throughput_end_time - throughput_start_time

        # Analyze throughput results
        successful_concurrent = [r for r in concurrent_results if r["success"]]

        throughput_results = {
            "concurrent_threads": concurrent_threads,
            "total_predictions": len(test_stocks),
            "successful_predictions": len(successful_concurrent),
            "total_time_seconds": total_throughput_time,
            "throughput_metrics": {
                "predictions_per_second": len(test_stocks) / total_throughput_time,
                "successful_predictions_per_second": len(successful_concurrent) / total_throughput_time,
                "avg_concurrent_latency_ms": (
                    np.mean([r["latency_ms"] for r in successful_concurrent]) if successful_concurrent else 0
                ),
                "max_concurrent_latency_ms": (
                    np.max([r["latency_ms"] for r in successful_concurrent]) if successful_concurrent else 0
                ),
            },
            "thread_distribution": self.analyze_thread_distribution(concurrent_results),
            "detailed_results": concurrent_results,
        }

        return throughput_results

    def run_accuracy_benchmarks(self, sample_size: int) -> Dict[str, Any]:
        """Run accuracy and prediction quality benchmarking."""

        self.stdout.write("Accuracy benchmarking started...")

        test_stocks = self.get_test_stocks(sample_size)
        predictions = []

        for stock in test_stocks:
            try:
                result = self.service.predict_stock_price(stock, use_cache=False)

                if result:
                    predictions.append(
                        {
                            "stock": stock,
                            "predicted_price": result["predicted_price"],
                            "current_price": result["current_price"],
                            "price_change_pct": result["price_change_pct"],
                            "confidence": result["confidence"],
                            "sector": result.get("sector_name", "Unknown"),
                            "success": True,
                        }
                    )
                else:
                    predictions.append({"stock": stock, "success": False, "error": "No prediction returned"})

            except Exception as e:
                predictions.append({"stock": stock, "success": False, "error": str(e)})

        # Analyze prediction quality
        successful_predictions = [p for p in predictions if p["success"]]

        if successful_predictions:
            confidences = [p["confidence"] for p in successful_predictions]
            price_changes = [abs(p["price_change_pct"]) for p in successful_predictions]

            # Check for realistic predictions (not all zeros or extreme values)
            realistic_predictions = [
                p for p in successful_predictions if abs(p["price_change_pct"]) < 50.0 and p["predicted_price"] > 0
            ]

            accuracy_results = {
                "total_predictions": len(predictions),
                "successful_predictions": len(successful_predictions),
                "realistic_predictions": len(realistic_predictions),
                "success_rate": len(successful_predictions) / len(predictions),
                "realistic_rate": len(realistic_predictions) / len(predictions),
                "confidence_stats": {
                    "mean_confidence": np.mean(confidences),
                    "min_confidence": np.min(confidences),
                    "max_confidence": np.max(confidences),
                    "std_confidence": np.std(confidences),
                },
                "prediction_stats": {
                    "mean_abs_price_change": np.mean(price_changes),
                    "median_abs_price_change": np.median(price_changes),
                    "max_abs_price_change": np.max(price_changes),
                    "predictions_with_variation": len(
                        set([round(p["predicted_price"], 2) for p in successful_predictions])
                    ),
                },
                "sector_distribution": self.analyze_sector_distribution(successful_predictions),
                "detailed_predictions": predictions,
            }
        else:
            accuracy_results = {
                "total_predictions": len(predictions),
                "successful_predictions": 0,
                "success_rate": 0.0,
                "error": "No successful predictions to analyze",
                "detailed_predictions": predictions,
            }

        return accuracy_results

    def analyze_thread_distribution(self, concurrent_results: List[Dict]) -> Dict:
        """Analyze how predictions were distributed across threads."""

        thread_stats = {}
        for result in concurrent_results:
            thread_id = result.get("thread_id", "unknown")

            if thread_id not in thread_stats:
                thread_stats[thread_id] = {"total_predictions": 0, "successful_predictions": 0, "latencies": []}

            thread_stats[thread_id]["total_predictions"] += 1
            if result["success"]:
                thread_stats[thread_id]["successful_predictions"] += 1
                thread_stats[thread_id]["latencies"].append(result["latency_ms"])

        # Calculate per-thread averages
        for thread_id in thread_stats:
            stats = thread_stats[thread_id]
            if stats["latencies"]:
                stats["avg_latency_ms"] = np.mean(stats["latencies"])
            else:
                stats["avg_latency_ms"] = 0

        return thread_stats

    def analyze_sector_distribution(self, predictions: List[Dict]) -> Dict:
        """Analyze prediction distribution across sectors."""

        sector_stats = {}
        for prediction in predictions:
            sector = prediction.get("sector", "Unknown")

            if sector not in sector_stats:
                sector_stats[sector] = {"count": 0, "confidences": [], "price_changes": []}

            sector_stats[sector]["count"] += 1
            sector_stats[sector]["confidences"].append(prediction["confidence"])
            sector_stats[sector]["price_changes"].append(abs(prediction["price_change_pct"]))

        # Calculate sector averages
        for sector in sector_stats:
            stats = sector_stats[sector]
            stats["avg_confidence"] = np.mean(stats["confidences"])
            stats["avg_price_change"] = np.mean(stats["price_changes"])

        return sector_stats

    def get_test_stocks(self, sample_size: int) -> List[str]:
        """Get a sample of stocks for testing."""

        try:
            stocks = Stock.objects.values_list("symbol", flat=True)[: sample_size * 2]
            return list(stocks)[:sample_size]
        except Exception as e:
            if self.verbose:
                self.stdout.write(f"Error getting test stocks: {e}")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"][:sample_size]

    def generate_benchmark_summary(self, results: Dict, target_latency: float) -> Dict:
        """Generate overall benchmark summary."""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks_completed": list(results.keys()),
            "overall_performance": "PENDING",
        }

        # Speed summary
        if "speed" in results:
            speed_results = results["speed"]
            if "latency_stats" in speed_results:
                summary["avg_latency_ms"] = speed_results["latency_stats"]["mean_ms"]
                summary["p95_latency_ms"] = speed_results["latency_stats"]["p95_ms"]
                summary["speed_compliance_rate"] = speed_results["target_compliance"]["compliance_rate"]
                summary["meets_latency_target"] = speed_results["latency_stats"]["mean_ms"] < target_latency

        # Memory summary
        if "memory" in results:
            memory_results = results["memory"]
            summary["model_memory_mb"] = memory_results.get("model_memory_mb", 0)
            summary["model_parameters"] = memory_results.get("model_parameters", 0)
            summary["peak_memory_mb"] = memory_results.get("peak_memory_mb", 0)

        # Throughput summary
        if "throughput" in results:
            throughput_results = results["throughput"]
            if "throughput_metrics" in throughput_results:
                summary["predictions_per_second"] = throughput_results["throughput_metrics"]["predictions_per_second"]
                summary["concurrent_latency_ms"] = throughput_results["throughput_metrics"]["avg_concurrent_latency_ms"]

        # Accuracy summary
        if "accuracy" in results:
            accuracy_results = results["accuracy"]
            summary["prediction_success_rate"] = accuracy_results.get("success_rate", 0)
            summary["realistic_prediction_rate"] = accuracy_results.get("realistic_rate", 0)
            if "confidence_stats" in accuracy_results:
                summary["avg_confidence"] = accuracy_results["confidence_stats"]["mean_confidence"]

        # Overall performance assessment
        performance_indicators = []

        if summary.get("meets_latency_target", False):
            performance_indicators.append("SPEED_GOOD")

        if summary.get("prediction_success_rate", 0) >= 0.8:
            performance_indicators.append("SUCCESS_RATE_GOOD")

        if summary.get("model_parameters", 0) < 2_000_000:  # Less than 2M parameters
            performance_indicators.append("MODEL_SIZE_GOOD")

        if len(performance_indicators) >= 2:
            summary["overall_performance"] = "GOOD"
        elif len(performance_indicators) >= 1:
            summary["overall_performance"] = "FAIR"
        else:
            summary["overall_performance"] = "NEEDS_IMPROVEMENT"

        return summary

    def display_benchmark_summary(self, summary: Dict):
        """Display benchmark summary to console."""

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("UNIVERSAL LSTM BENCHMARK SUMMARY"))
        self.stdout.write("=" * 60)

        self.stdout.write(f"Overall Performance: {summary.get('overall_performance', 'UNKNOWN')}")
        self.stdout.write(f"Benchmarks Completed: {', '.join(summary.get('benchmarks_completed', []))}")

        if "avg_latency_ms" in summary:
            latency = summary["avg_latency_ms"]
            target_met = summary.get("meets_latency_target", False)
            status = "PASS" if target_met else "FAIL"
            self.stdout.write(f"Average Latency: {latency:.1f}ms [{status}]")

        if "prediction_success_rate" in summary:
            rate = summary["prediction_success_rate"]
            self.stdout.write(f"Prediction Success Rate: {rate:.1%}")

        if "predictions_per_second" in summary:
            throughput = summary["predictions_per_second"]
            self.stdout.write(f"Throughput: {throughput:.1f} predictions/second")

        if "model_parameters" in summary:
            params = summary["model_parameters"]
            memory = summary.get("model_memory_mb", 0)
            self.stdout.write(f"Model Size: {params:,} parameters ({memory:.1f} MB)")

        self.stdout.write("=" * 60)
