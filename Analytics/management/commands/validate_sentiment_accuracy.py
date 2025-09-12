"""
Management command to validate sentiment analysis accuracy.
Tests the FinBERT model against labeled financial text samples.
"""

import json
import time
from typing import Dict, List, Tuple

from django.core.management.base import BaseCommand
from django.utils import timezone

from Analytics.services.sentiment_analyzer import (
    get_sentiment_analyzer,
    sentiment_metrics,
)


class Command(BaseCommand):
    """
    Validate sentiment analysis accuracy against labeled test data.
    Tests the 80% accuracy target requirement.
    """

    help = "Validate sentiment analysis accuracy against test samples"

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            "--sample-size", type=int, default=50, help="Number of test samples to evaluate (default: 50)"
        )
        parser.add_argument(
            "--confidence-threshold", type=float, default=0.6, help="Minimum confidence threshold (default: 0.6)"
        )
        parser.add_argument("--output-file", type=str, help="Optional file to save detailed results")
        parser.add_argument("--verbose", action="store_true", help="Show detailed per-sample results")

    def handle(self, *args, **options):
        """Main command handler."""
        sample_size = options["sample_size"]
        confidence_threshold = options["confidence_threshold"]
        output_file = options["output_file"]
        verbose = options["verbose"]

        self.stdout.write(self.style.SUCCESS(f"Starting sentiment analysis accuracy validation..."))
        self.stdout.write(f"Sample size: {sample_size}")
        self.stdout.write(f"Confidence threshold: {confidence_threshold}")

        # Generate test samples
        test_samples = self._generate_financial_test_samples(sample_size)

        # Get sentiment analyzer
        analyzer = get_sentiment_analyzer()

        # Reset metrics
        sentiment_metrics.reset_metrics()

        # Run validation
        results = self._validate_accuracy(analyzer, test_samples, verbose)

        # Display results
        self._display_results(results, confidence_threshold)

        # Save detailed results if requested
        if output_file:
            self._save_results(results, output_file)
            self.stdout.write(f"Detailed results saved to: {output_file}")

        # Check if accuracy target met
        if results["overall_accuracy"] >= 0.8:
            self.stdout.write(
                self.style.SUCCESS(f"✅ PASSED: Accuracy target of 80% achieved ({results['overall_accuracy']:.1%})")
            )
        else:
            self.stdout.write(
                self.style.ERROR(f"❌ FAILED: Accuracy target of 80% not met ({results['overall_accuracy']:.1%})")
            )

    def _generate_financial_test_samples(self, sample_size: int) -> List[Tuple[str, str]]:
        """
        Generate labeled financial text samples for testing.

        Returns:
            List of (text, expected_sentiment) tuples
        """
        # Comprehensive financial sentiment test cases
        positive_samples = [
            "Company reports record quarterly revenue growth of 35% year-over-year",
            "Strong earnings beat analyst expectations by significant margin",
            "Stock reaches new all-time high following merger announcement",
            "Exceptional Q3 performance drives substantial shareholder value",
            "Outstanding quarterly results exceed all market forecasts",
            "Revenue soars 40% as demand for products remains robust",
            "Company announces major breakthrough in core technology",
            "Profit margins expand significantly due to operational efficiency",
            "Strong cash flow generation supports increased dividend payments",
            "Market share gains accelerate in key geographic regions",
            "CEO reports transformational year with record-breaking performance",
            "Successful product launch drives unprecedented customer adoption",
            "Strategic acquisition enhances competitive positioning substantially",
            "Cost reduction initiatives deliver better-than-expected savings",
            "Innovation pipeline strengthens with multiple patent approvals",
            "Customer satisfaction scores reach highest levels in company history",
            "Operating leverage produces exceptional profit growth acceleration",
            "New contract wins provide strong revenue visibility for next year",
            "International expansion delivers impressive early results",
            "Digital transformation drives significant operational improvements",
        ]

        negative_samples = [
            "Company faces SEC investigation over serious accounting irregularities",
            "Quarterly losses widen substantially as revenue continues declining",
            "Stock plummets following disappointing earnings guidance revision",
            "Management warns of significant restructuring costs and layoffs ahead",
            "Credit rating downgraded due to mounting debt and liquidity concerns",
            "Regulatory probe launched into potential compliance violations",
            "Major client cancels contract citing quality and delivery issues",
            "Cybersecurity breach exposes sensitive customer and financial data",
            "Product recall costs escalate amid growing safety concerns",
            "Key executive departures create uncertainty about strategic direction",
            "Market share erosion accelerates in core business segments",
            "Supply chain disruptions cause significant production delays",
            "Legal settlement costs drain cash reserves and impact profitability",
            "Technology platform failure results in widespread service outages",
            "Competitor launches superior product that threatens market position",
            "Environmental violations result in substantial regulatory penalties",
            "Failed merger attempt wastes resources and damages credibility",
            "Analyst downgrades multiply following weak quarterly performance",
            "Customer data breach triggers class-action lawsuit filing",
            "Manufacturing facility closure eliminates hundreds of local jobs",
        ]

        neutral_samples = [
            "Company announces routine quarterly dividend payment to shareholders",
            "Annual shareholder meeting scheduled for next month as planned",
            "Quarterly earnings report will be released next Tuesday morning",
            "Company maintains steady market position during current quarter",
            "Regular business operations continue as planned this quarter",
            "Management team provides standard quarterly business update",
            "Board of directors approves routine stock buyback program",
            "Company files standard regulatory reports with SEC on schedule",
            "Quarterly conference call scheduled to discuss financial results",
            "Standard audit procedures completed by external accounting firm",
            "Routine leadership changes announced in non-critical positions",
            "Company participates in industry conference as scheduled",
            "Standard compliance procedures updated per regulatory requirements",
            "Quarterly investor presentation materials posted to website",
            "Regular maintenance scheduled for primary manufacturing facility",
            "Annual report filed with regulatory authorities on time",
            "Standard employee performance reviews conducted this quarter",
            "Routine facility inspection completed without major findings",
            "Company maintains existing guidance for fiscal year outlook",
            "Regular software updates deployed across enterprise systems",
        ]

        # Balance the samples based on requested size
        samples_per_category = sample_size // 3
        remaining = sample_size % 3

        selected_samples = []

        # Add positive samples
        selected_samples.extend(
            [(text, "positive") for text in positive_samples[: samples_per_category + (1 if remaining > 0 else 0)]]
        )

        # Add negative samples
        selected_samples.extend(
            [(text, "negative") for text in negative_samples[: samples_per_category + (1 if remaining > 1 else 0)]]
        )

        # Add neutral samples
        selected_samples.extend([(text, "neutral") for text in neutral_samples[:samples_per_category]])

        return selected_samples[:sample_size]

    def _validate_accuracy(self, analyzer, test_samples: List[Tuple[str, str]], verbose: bool) -> Dict:
        """
        Run accuracy validation on test samples.

        Args:
            analyzer: SentimentAnalyzer instance
            test_samples: List of (text, expected_sentiment) tuples
            verbose: Whether to show per-sample results

        Returns:
            Dictionary with validation results
        """
        results = {
            "total_samples": len(test_samples),
            "correct_predictions": 0,
            "predictions_by_category": {
                "positive": {"correct": 0, "total": 0},
                "negative": {"correct": 0, "total": 0},
                "neutral": {"correct": 0, "total": 0},
            },
            "confidence_stats": {
                "high_confidence": 0,  # >= 0.8
                "medium_confidence": 0,  # 0.6 - 0.8
                "low_confidence": 0,  # < 0.6
            },
            "processing_time": 0,
            "samples": [],
        }

        start_time = time.time()

        if verbose:
            self.stdout.write("\nDetailed Results:")
            self.stdout.write("-" * 80)

        for i, (text, expected) in enumerate(test_samples):
            try:
                # Analyze sentiment
                result = analyzer.analyzeSentimentSingle(text)

                predicted = result["sentimentLabel"]
                confidence = result["sentimentConfidence"]
                score = result["sentimentScore"]

                # Track overall accuracy
                is_correct = predicted == expected
                if is_correct:
                    results["correct_predictions"] += 1

                # Track per-category accuracy
                results["predictions_by_category"][expected]["total"] += 1
                if is_correct:
                    results["predictions_by_category"][expected]["correct"] += 1

                # Track confidence distribution
                if confidence >= 0.8:
                    results["confidence_stats"]["high_confidence"] += 1
                elif confidence >= 0.6:
                    results["confidence_stats"]["medium_confidence"] += 1
                else:
                    results["confidence_stats"]["low_confidence"] += 1

                # Store sample result
                sample_result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "score": score,
                    "correct": is_correct,
                }
                results["samples"].append(sample_result)

                if verbose:
                    status = "✅" if is_correct else "❌"
                    self.stdout.write(
                        f"{status} [{i+1:2d}] Expected: {expected:8s} | "
                        f"Predicted: {predicted:8s} | "
                        f"Confidence: {confidence:.3f} | "
                        f"Score: {score:6.3f}"
                    )

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing sample {i+1}: {str(e)}"))
                continue

        results["processing_time"] = time.time() - start_time
        results["overall_accuracy"] = results["correct_predictions"] / results["total_samples"]

        return results

    def _display_results(self, results: Dict, confidence_threshold: float):
        """Display validation results summary."""
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("SENTIMENT ANALYSIS ACCURACY VALIDATION RESULTS")
        self.stdout.write("=" * 80)

        # Overall metrics
        accuracy = results["overall_accuracy"]
        self.stdout.write(
            f"Overall Accuracy: {accuracy:.1%} ({results['correct_predictions']}/{results['total_samples']})"
        )

        # Per-category accuracy
        self.stdout.write("\nAccuracy by Sentiment Category:")
        for category, stats in results["predictions_by_category"].items():
            if stats["total"] > 0:
                cat_accuracy = stats["correct"] / stats["total"]
                self.stdout.write(
                    f"  {category.capitalize()}: {cat_accuracy:.1%} ({stats['correct']}/{stats['total']})"
                )

        # Confidence distribution
        self.stdout.write("\nConfidence Distribution:")
        total = results["total_samples"]
        for level, count in results["confidence_stats"].items():
            percentage = (count / total) * 100
            self.stdout.write(f"  {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        # Performance metrics
        self.stdout.write(f"\nProcessing Time: {results['processing_time']:.2f} seconds")
        avg_time = results["processing_time"] / results["total_samples"]
        self.stdout.write(f"Average per Sample: {avg_time:.3f} seconds")

        # Accuracy target assessment
        target_met = accuracy >= 0.8
        status_style = self.style.SUCCESS if target_met else self.style.ERROR
        status_text = "ACHIEVED" if target_met else "NOT MET"

        self.stdout.write(f"\n80% Accuracy Target: {status_style(status_text)}")

        # Additional insights
        high_conf_samples = results["confidence_stats"]["high_confidence"]
        if high_conf_samples > 0:
            # Calculate accuracy for high-confidence predictions
            high_conf_correct = sum(
                1 for sample in results["samples"] if sample["confidence"] >= 0.8 and sample["correct"]
            )
            high_conf_accuracy = high_conf_correct / high_conf_samples
            self.stdout.write(f"High Confidence (≥0.8) Accuracy: {high_conf_accuracy:.1%}")

    def _save_results(self, results: Dict, output_file: str):
        """Save detailed results to JSON file."""
        # Prepare results for JSON serialization
        json_results = {
            "timestamp": timezone.now().isoformat(),
            "summary": {
                "total_samples": results["total_samples"],
                "correct_predictions": results["correct_predictions"],
                "overall_accuracy": results["overall_accuracy"],
                "processing_time": results["processing_time"],
            },
            "category_accuracy": results["predictions_by_category"],
            "confidence_distribution": results["confidence_stats"],
            "samples": results["samples"],
        }

        try:
            with open(output_file, "w") as f:
                json.dump(json_results, f, indent=2)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to save results: {str(e)}"))
