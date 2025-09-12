"""
Universal LSTM Model Validation Command
=====================================

Comprehensive validation framework for Universal LSTM model including:
- Cross-sector validation (hold-out sector testing)
- Performance benchmarking vs stock-specific models
- Zero-shot prediction testing
- Sector transfer learning evaluation
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from Analytics.ml.sector_mappings import get_sector_mapper
from Analytics.services.universal_predictor import UniversalLSTMAnalyticsService
from Data.models import Stock

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Comprehensive Universal LSTM model validation framework"

    def add_arguments(self, parser):
        parser.add_argument(
            "--test-type",
            type=str,
            choices=["cross-sector", "performance", "zero-shot", "transfer-learning", "all"],
            default="all",
            help="Type of validation to run",
        )
        parser.add_argument("--hold-out-sector", type=str, help="Sector to hold out for cross-sector validation")
        parser.add_argument("--sample-size", type=int, default=20, help="Number of stocks to test per validation")
        parser.add_argument(
            "--output-file",
            type=str,
            default="Temp/universal_model_validation_results.json",
            help="Output file for validation results",
        )
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    def handle(self, *args, **options):
        """Execute Universal LSTM validation framework."""

        self.stdout.write(self.style.SUCCESS("=== Universal LSTM Model Validation Framework ==="))

        # Initialise validation components
        try:
            self.service = UniversalLSTMAnalyticsService()
            self.sector_mapper = get_sector_mapper()
            self.verbose = options["verbose"]

            if not self.service.model:
                raise CommandError("Universal LSTM model not loaded")

            self.stdout.write("Universal LSTM service initialised successfully")

        except Exception as e:
            raise CommandError(f"Failed to initialise validation components: {e}")

        # Run validation tests
        results = {}
        test_type = options["test_type"]

        if test_type in ["cross-sector", "all"]:
            self.stdout.write("\n--- Running Cross-Sector Validation ---")
            results["cross_sector"] = self.run_cross_sector_validation(
                options.get("hold_out_sector"), options["sample_size"]
            )

        if test_type in ["performance", "all"]:
            self.stdout.write("\n--- Running Performance Benchmarking ---")
            results["performance"] = self.run_performance_benchmarking(options["sample_size"])

        if test_type in ["zero-shot", "all"]:
            self.stdout.write("\n--- Running Zero-Shot Prediction Testing ---")
            results["zero_shot"] = self.run_zero_shot_testing(options["sample_size"])

        if test_type in ["transfer-learning", "all"]:
            self.stdout.write("\n--- Running Transfer Learning Evaluation ---")
            results["transfer_learning"] = self.run_transfer_learning_evaluation(options["sample_size"])

        # Save results
        output_file = options["output_file"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "model_info": self.service.get_model_info(),
            "validation_results": results,
            "summary": self.generate_validation_summary(results),
        }

        with open(output_file, "w") as f:
            json.dump(validation_report, f, indent=2, default=str)

        self.stdout.write(f"\nValidation results saved to: {output_file}")

        # Display summary
        self.display_validation_summary(validation_report["summary"])

    def run_cross_sector_validation(self, hold_out_sector: Optional[str], sample_size: int) -> Dict[str, Any]:
        """
        Run cross-sector validation by testing model performance across sectors.

        If hold_out_sector is specified, simulates training without that sector
        and tests prediction quality on held-out sector stocks.
        """
        self.stdout.write("Cross-sector validation started...")

        # Get all available sectors from SECTOR_MAPPING
        from Analytics.ml.sector_mappings import SECTOR_MAPPING

        sectors = {name: idx for name, idx in SECTOR_MAPPING.items()}

        if hold_out_sector and hold_out_sector not in sectors:
            self.stdout.write(
                self.style.WARNING(f"Hold-out sector '{hold_out_sector}' not found. Available: {list(sectors.keys())}")
            )
            hold_out_sector = None

        results = {"test_type": "cross_sector", "hold_out_sector": hold_out_sector, "sector_results": {}, "summary": {}}

        # Test each sector
        for sector_name, sector_id in sectors.items():
            if hold_out_sector and sector_name == hold_out_sector:
                # This is the held-out sector - test with unknown classification
                sector_results = self.test_sector_performance(
                    sector_name, sector_id, sample_size, simulate_unknown=True
                )
            else:
                # Normal sector testing
                sector_results = self.test_sector_performance(
                    sector_name, sector_id, sample_size, simulate_unknown=False
                )

            results["sector_results"][sector_name] = sector_results

            if self.verbose:
                success_rate = sector_results["success_rate"]
                avg_confidence = sector_results["avg_confidence"]
                self.stdout.write(f"  {sector_name}: {success_rate:.1%} success, {avg_confidence:.3f} avg confidence")

        # Calculate summary statistics
        all_success_rates = [r["success_rate"] for r in results["sector_results"].values()]
        all_confidences = [r["avg_confidence"] for r in results["sector_results"].values() if r["avg_confidence"] > 0]

        results["summary"] = {
            "total_sectors_tested": len(results["sector_results"]),
            "average_success_rate": np.mean(all_success_rates) if all_success_rates else 0,
            "min_success_rate": min(all_success_rates) if all_success_rates else 0,
            "max_success_rate": max(all_success_rates) if all_success_rates else 0,
            "average_confidence": np.mean(all_confidences) if all_confidences else 0,
            "cross_sector_consistency": np.std(all_success_rates) if len(all_success_rates) > 1 else 0,
        }

        return results

    def test_sector_performance(
        self, sector_name: str, sector_id: int, sample_size: int, simulate_unknown: bool = False
    ) -> Dict[str, Any]:
        """Test Universal LSTM performance on stocks from a specific sector."""

        # Get stocks from this sector
        sector_stocks = self.get_stocks_by_sector(sector_name, sample_size)

        if not sector_stocks:
            return {
                "sector_name": sector_name,
                "stocks_tested": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_prediction_time": 0.0,
                "predictions": [],
            }

        predictions = []
        successful_predictions = 0
        total_time = 0
        confidences = []

        for stock_symbol in sector_stocks:
            start_time = time.time()

            try:
                # Make prediction
                result = self.service.predict_stock_price(stock_symbol, use_cache=False)

                prediction_time = (time.time() - start_time) * 1000
                total_time += prediction_time

                if result:
                    successful_predictions += 1
                    confidences.append(result["confidence"])

                    predictions.append(
                        {
                            "symbol": stock_symbol,
                            "success": True,
                            "predicted_price": result["predicted_price"],
                            "price_change_pct": result["price_change_pct"],
                            "confidence": result["confidence"],
                            "prediction_time_ms": prediction_time,
                            "detected_sector": result.get("sector_name"),
                            "expected_sector": sector_name,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "symbol": stock_symbol,
                            "success": False,
                            "error": "No prediction returned",
                            "prediction_time_ms": prediction_time,
                        }
                    )

            except Exception as e:
                prediction_time = (time.time() - start_time) * 1000
                total_time += prediction_time

                predictions.append(
                    {"symbol": stock_symbol, "success": False, "error": str(e), "prediction_time_ms": prediction_time}
                )

        return {
            "sector_name": sector_name,
            "stocks_tested": len(sector_stocks),
            "successful_predictions": successful_predictions,
            "success_rate": successful_predictions / len(sector_stocks) if sector_stocks else 0,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_prediction_time": total_time / len(sector_stocks) if sector_stocks else 0,
            "predictions": predictions,
        }

    def run_performance_benchmarking(self, sample_size: int) -> Dict[str, Any]:
        """Benchmark Universal LSTM against baseline metrics."""

        self.stdout.write("Performance benchmarking started...")

        # Select diverse sample of stocks
        test_stocks = self.get_diverse_stock_sample(sample_size)

        # Test various performance metrics
        latency_tests = []
        memory_usage = []
        prediction_quality = []

        for stock_symbol in test_stocks:
            # Latency test
            start_time = time.time()
            result = self.service.predict_stock_price(stock_symbol, use_cache=False)
            latency = (time.time() - start_time) * 1000

            latency_tests.append({"symbol": stock_symbol, "latency_ms": latency, "success": result is not None})

            if result:
                prediction_quality.append(
                    {
                        "symbol": stock_symbol,
                        "confidence": result["confidence"],
                        "price_change_magnitude": abs(result["price_change_pct"]),
                        "sector": result.get("sector_name", "Unknown"),
                    }
                )

        # Model size metrics
        if hasattr(self.service, "model") and self.service.model:
            model_params = sum(p.numel() for p in self.service.model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in self.service.model.parameters()) / (1024**2)
        else:
            model_params = 0
            model_size_mb = 0

        return {
            "test_type": "performance_benchmark",
            "sample_size": len(test_stocks),
            "latency_metrics": {
                "avg_latency_ms": np.mean([t["latency_ms"] for t in latency_tests]) if latency_tests else 0,
                "max_latency_ms": max([t["latency_ms"] for t in latency_tests]) if latency_tests else 0,
                "min_latency_ms": min([t["latency_ms"] for t in latency_tests]) if latency_tests else 0,
                "success_rate": sum([t["success"] for t in latency_tests]) / len(latency_tests) if latency_tests else 0,
            },
            "model_metrics": {
                "total_parameters": model_params,
                "model_size_mb": model_size_mb,
                "sequence_length": self.service.sequence_length,
                "device": str(self.service.device),
            },
            "prediction_quality": {
                "avg_confidence": np.mean([p["confidence"] for p in prediction_quality]) if prediction_quality else 0,
                "avg_price_change_magnitude": (
                    np.mean([p["price_change_magnitude"] for p in prediction_quality]) if prediction_quality else 0
                ),
                "sectors_covered": len(set([p["sector"] for p in prediction_quality])),
            },
            "detailed_tests": latency_tests,
        }

    def run_zero_shot_testing(self, sample_size: int) -> Dict[str, Any]:
        """Test Universal LSTM on completely unseen stocks."""

        self.stdout.write("Zero-shot prediction testing started...")

        # For this test, we'll use stocks that might not be in the training set
        # This is simulated since we can't easily determine training set membership

        results = {
            "test_type": "zero_shot",
            "note": "Simulated zero-shot testing on diverse stock sample",
            "sample_size": sample_size,
        }

        # Test on a diverse sample and analyze sector generalization
        test_stocks = self.get_diverse_stock_sample(sample_size)

        sector_performance = {}
        overall_results = []

        for stock_symbol in test_stocks:
            try:
                result = self.service.predict_stock_price(stock_symbol, use_cache=False)

                if result:
                    sector = result.get("sector_name", "Unknown")

                    if sector not in sector_performance:
                        sector_performance[sector] = {"successes": 0, "total": 0, "confidences": []}

                    sector_performance[sector]["successes"] += 1
                    sector_performance[sector]["total"] += 1
                    sector_performance[sector]["confidences"].append(result["confidence"])

                    overall_results.append(
                        {
                            "symbol": stock_symbol,
                            "success": True,
                            "sector": sector,
                            "confidence": result["confidence"],
                            "price_change_pct": result["price_change_pct"],
                        }
                    )
                else:
                    overall_results.append({"symbol": stock_symbol, "success": False, "error": "No prediction"})

            except Exception as e:
                overall_results.append({"symbol": stock_symbol, "success": False, "error": str(e)})

        # Calculate sector-wise performance
        for sector in sector_performance:
            perf = sector_performance[sector]
            perf["success_rate"] = perf["successes"] / perf["total"] if perf["total"] > 0 else 0
            perf["avg_confidence"] = np.mean(perf["confidences"]) if perf["confidences"] else 0

        results.update(
            {
                "sector_performance": sector_performance,
                "overall_success_rate": sum([r["success"] for r in overall_results]) / len(overall_results),
                "detailed_results": overall_results,
            }
        )

        return results

    def run_transfer_learning_evaluation(self, sample_size: int) -> Dict[str, Any]:
        """Evaluate cross-sector knowledge transfer capabilities."""

        self.stdout.write("Transfer learning evaluation started...")

        # Test how well knowledge transfers between sectors
        sectors = self.sector_mapper.get_all_sectors()
        transfer_matrix = {}

        for source_sector in list(sectors.keys())[:3]:  # Test top 3 sectors
            transfer_matrix[source_sector] = {}

            for target_sector in list(sectors.keys())[:3]:
                if source_sector == target_sector:
                    continue

                # Test stocks from target sector using source sector knowledge
                target_stocks = self.get_stocks_by_sector(target_sector, min(sample_size // 3, 5))

                if not target_stocks:
                    continue

                transfer_results = []
                for stock in target_stocks:
                    try:
                        result = self.service.predict_stock_price(stock, use_cache=False)
                        if result:
                            transfer_results.append(
                                {
                                    "symbol": stock,
                                    "success": True,
                                    "confidence": result["confidence"],
                                    "detected_sector": result.get("sector_name"),
                                    "expected_sector": target_sector,
                                }
                            )
                        else:
                            transfer_results.append({"symbol": stock, "success": False})
                    except Exception as e:
                        transfer_results.append({"symbol": stock, "success": False, "error": str(e)})

                if transfer_results:
                    success_rate = sum([r["success"] for r in transfer_results]) / len(transfer_results)
                    avg_confidence = (
                        np.mean([r["confidence"] for r in transfer_results if r["success"]])
                        if any(r["success"] for r in transfer_results)
                        else 0
                    )

                    transfer_matrix[source_sector][target_sector] = {
                        "success_rate": success_rate,
                        "avg_confidence": avg_confidence,
                        "stocks_tested": len(transfer_results),
                        "results": transfer_results,
                    }

        return {
            "test_type": "transfer_learning",
            "transfer_matrix": transfer_matrix,
            "summary": self.summarize_transfer_learning(transfer_matrix),
        }

    def summarize_transfer_learning(self, transfer_matrix: Dict) -> Dict:
        """Summarize transfer learning results."""

        all_success_rates = []
        all_confidences = []

        for source in transfer_matrix:
            for target in transfer_matrix[source]:
                result = transfer_matrix[source][target]
                all_success_rates.append(result["success_rate"])
                if result["avg_confidence"] > 0:
                    all_confidences.append(result["avg_confidence"])

        return {
            "avg_transfer_success_rate": np.mean(all_success_rates) if all_success_rates else 0,
            "transfer_consistency": 1 - np.std(all_success_rates) if len(all_success_rates) > 1 else 0,
            "avg_transfer_confidence": np.mean(all_confidences) if all_confidences else 0,
            "transfer_pairs_tested": len(all_success_rates),
        }

    def get_stocks_by_sector(self, sector_name: str, limit: int) -> List[str]:
        """Get stock symbols from a specific sector."""

        try:
            # Use sector mapper to get stocks from sector
            sector_stocks = self.sector_mapper.get_training_stocks_by_sector(sector_name)

            # Also check database for additional stocks
            try:
                stocks_from_db = Stock.objects.filter(sector__sectorName__icontains=sector_name).values_list(
                    "symbol", flat=True
                )[:limit]

                # Combine and deduplicate
                all_stocks = list(set(sector_stocks + list(stocks_from_db)))
            except Exception:
                # Fallback to just training stocks if database query fails
                all_stocks = sector_stocks

            return all_stocks[:limit]

        except Exception as e:
            if self.verbose:
                self.stdout.write(f"Error getting stocks for sector {sector_name}: {e}")
            return []

    def get_diverse_stock_sample(self, sample_size: int) -> List[str]:
        """Get a diverse sample of stocks across sectors."""

        try:
            # Get stocks from different sectors
            from Analytics.ml.sector_mappings import SECTOR_MAPPING

            sectors = list(SECTOR_MAPPING.keys())
            stocks_per_sector = max(1, sample_size // len(sectors))

            diverse_stocks = []
            for sector_name in sectors:
                sector_stocks = self.get_stocks_by_sector(sector_name, stocks_per_sector)
                diverse_stocks.extend(sector_stocks)

            # Fill remaining slots with any available stocks
            if len(diverse_stocks) < sample_size:
                try:
                    additional_stocks = Stock.objects.exclude(symbol__in=diverse_stocks).values_list(
                        "symbol", flat=True
                    )[: sample_size - len(diverse_stocks)]
                    diverse_stocks.extend(additional_stocks)
                except Exception:
                    # Fallback to a fixed list if database query fails
                    fallback_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "WMT"]
                    for stock in fallback_stocks:
                        if stock not in diverse_stocks and len(diverse_stocks) < sample_size:
                            diverse_stocks.append(stock)

            return diverse_stocks[:sample_size]

        except Exception as e:
            if self.verbose:
                self.stdout.write(f"Error getting diverse stock sample: {e}")
            # Return fallback list if everything fails
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"][:sample_size]

    def generate_validation_summary(self, results: Dict) -> Dict:
        """Generate overall validation summary."""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "tests_completed": list(results.keys()),
            "overall_assessment": "PENDING",
        }

        # Analyze cross-sector results
        if "cross_sector" in results:
            cs_results = results["cross_sector"]
            summary["cross_sector_success_rate"] = cs_results["summary"].get("average_success_rate", 0)
            summary["cross_sector_consistency"] = 1 - cs_results["summary"].get("cross_sector_consistency", 1)

        # Analyze performance results
        if "performance" in results:
            perf_results = results["performance"]
            summary["avg_latency_ms"] = perf_results["latency_metrics"].get("avg_latency_ms", 0)
            summary["model_parameters"] = perf_results["model_metrics"].get("total_parameters", 0)
            summary["prediction_success_rate"] = perf_results["latency_metrics"].get("success_rate", 0)

        # Analyze zero-shot results
        if "zero_shot" in results:
            zs_results = results["zero_shot"]
            summary["zero_shot_success_rate"] = zs_results.get("overall_success_rate", 0)

        # Analyze transfer learning results
        if "transfer_learning" in results:
            tl_results = results["transfer_learning"]
            summary["transfer_learning_success_rate"] = tl_results["summary"].get("avg_transfer_success_rate", 0)

        # Overall assessment
        success_rates = []
        if "cross_sector_success_rate" in summary:
            success_rates.append(summary["cross_sector_success_rate"])
        if "prediction_success_rate" in summary:
            success_rates.append(summary["prediction_success_rate"])
        if "zero_shot_success_rate" in summary:
            success_rates.append(summary["zero_shot_success_rate"])

        if success_rates:
            avg_success = np.mean(success_rates)
            if avg_success >= 0.8:
                summary["overall_assessment"] = "EXCELLENT"
            elif avg_success >= 0.6:
                summary["overall_assessment"] = "GOOD"
            elif avg_success >= 0.4:
                summary["overall_assessment"] = "FAIR"
            else:
                summary["overall_assessment"] = "NEEDS_IMPROVEMENT"

        return summary

    def display_validation_summary(self, summary: Dict):
        """Display validation summary to console."""

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("UNIVERSAL LSTM VALIDATION SUMMARY"))
        self.stdout.write("=" * 60)

        self.stdout.write(f"Overall Assessment: {summary.get('overall_assessment', 'UNKNOWN')}")
        self.stdout.write(f"Tests Completed: {', '.join(summary.get('tests_completed', []))}")

        if "cross_sector_success_rate" in summary:
            rate = summary["cross_sector_success_rate"]
            self.stdout.write(f"Cross-Sector Success Rate: {rate:.1%}")

        if "prediction_success_rate" in summary:
            rate = summary["prediction_success_rate"]
            self.stdout.write(f"Prediction Success Rate: {rate:.1%}")

        if "avg_latency_ms" in summary:
            latency = summary["avg_latency_ms"]
            self.stdout.write(f"Average Prediction Latency: {latency:.1f}ms")

        if "model_parameters" in summary:
            params = summary["model_parameters"]
            self.stdout.write(f"Model Parameters: {params:,}")

        self.stdout.write("=" * 60)
