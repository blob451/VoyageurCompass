"""
Management command to generate financial instruction dataset for LLM fine-tuning.
Creates high-quality training data for domain-specific financial explanation generation.
"""

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class FinancialDatasetGenerator:
    """Generator for financial instruction datasets optimized for LLM fine-tuning."""

    def __init__(self):
        self.symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "AMD",
            "CRM",
            "ORCL",
            "ADBE",
            "INTC",
            "CSCO",
            "IBM",
            "PYPL",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "USB",
            "PNC",
            "JNJ",
            "PFE",
            "UNH",
            "ABT",
            "MRK",
            "CVS",
            "AMGN",
            "GILD",
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
            "PXD",
            "KMI",
            "OKE",
            "WMT",
            "HD",
            "MCD",
            "NKE",
            "SBUX",
            "TGT",
            "LOW",
            "COST",
        ]

        self.indicators = {
            "w_sma50vs200": {"name": "SMA 50/200 Crossover", "range": (-0.3, 0.3)},
            "w_price_vs_sma50": {"name": "Price vs SMA50", "range": (-0.2, 0.2)},
            "w_rsi14": {"name": "RSI(14)", "range": (-0.15, 0.15)},
            "w_macd12269": {"name": "MACD Histogram", "range": (-0.15, 0.15)},
            "w_bbpos20": {"name": "Bollinger Bands Position", "range": (-0.12, 0.12)},
            "w_bbwidth20": {"name": "Bollinger Bands Width", "range": (-0.1, 0.15)},
            "w_volsurge": {"name": "Volume Surge", "range": (-0.12, 0.18)},
            "w_obv20": {"name": "On-Balance Volume", "range": (-0.08, 0.08)},
            "w_rel1y": {"name": "Relative Performance 1Y", "range": (-0.1, 0.1)},
            "w_rel2y": {"name": "Relative Performance 2Y", "range": (-0.08, 0.08)},
            "w_support_resistance": {"name": "Support/Resistance", "range": (-0.1, 0.12)},
            "w_candlestick_patterns": {"name": "Candlestick Patterns", "range": (-0.08, 0.1)},
        }

        self.sentiment_profiles = [
            {"label": "positive", "score_range": (0.2, 0.8), "confidence_range": (0.6, 0.95)},
            {"label": "negative", "score_range": (-0.8, -0.2), "confidence_range": (0.6, 0.95)},
            {"label": "neutral", "score_range": (-0.15, 0.15), "confidence_range": (0.4, 0.8)},
        ]

        self.recommendation_templates = {
            "strong_buy": {
                "score_range": (8.0, 10.0),
                "templates": [
                    "**STRONG BUY** recommendation for {symbol} (Score: {score}/10). {technical_analysis} {sentiment_context} {risk_factors}",
                    "{symbol} receives a **STRONG BUY** rating with a {score}/10 score. {technical_analysis} {sentiment_context} Entry recommended with stop-loss at {support_level}.",
                    "**STRONG BUY** - {symbol} shows exceptional technical strength ({score}/10). {technical_analysis} {sentiment_context} Price target: {price_target}.",
                ],
            },
            "buy": {
                "score_range": (6.5, 7.9),
                "templates": [
                    "**BUY** recommendation for {symbol} (Score: {score}/10). {technical_analysis} {sentiment_context} {risk_considerations}",
                    "{symbol} earns a **BUY** rating with solid fundamentals ({score}/10). {technical_analysis} {sentiment_context} Consider entry on pullbacks.",
                    "**BUY** - {symbol} demonstrates good upward momentum ({score}/10). {technical_analysis} {sentiment_context}",
                ],
            },
            "hold": {
                "score_range": (4.0, 6.4),
                "templates": [
                    "**HOLD** recommendation for {symbol} (Score: {score}/10). {technical_analysis} {sentiment_context} Monitor for clearer signals.",
                    "{symbol} receives a **HOLD** rating ({score}/10). {technical_analysis} {sentiment_context} Wait for better entry opportunity.",
                    "**HOLD** - {symbol} shows mixed signals ({score}/10). {technical_analysis} {sentiment_context} Position size carefully.",
                ],
            },
            "sell": {
                "score_range": (2.1, 3.9),
                "templates": [
                    "**SELL** recommendation for {symbol} (Score: {score}/10). {technical_analysis} {sentiment_context} Consider reducing exposure.",
                    "{symbol} receives a **SELL** rating due to weak technicals ({score}/10). {technical_analysis} {sentiment_context} Exit recommended.",
                    "**SELL** - {symbol} shows concerning technical deterioration ({score}/10). {technical_analysis} {sentiment_context}",
                ],
            },
            "strong_sell": {
                "score_range": (0.0, 2.0),
                "templates": [
                    "**STRONG SELL** recommendation for {symbol} (Score: {score}/10). {technical_analysis} {sentiment_context} Immediate exit recommended.",
                    "{symbol} receives a **STRONG SELL** rating ({score}/10). {technical_analysis} {sentiment_context} High downside risk.",
                    "**STRONG SELL** - {symbol} exhibits severe technical weakness ({score}/10). {technical_analysis} {sentiment_context} Avoid or exit position.",
                ],
            },
        }

        self.technical_analysis_patterns = [
            "The {indicator1} shows {strength1} {direction1} momentum ({value1:+.3f}), while {indicator2} indicates {strength2} {direction2} pressure ({value2:+.3f}).",
            "{indicator1} provides a {strength1} {direction1} signal ({value1:+.3f}), complemented by {indicator2}'s {strength2} {direction2} reading ({value2:+.3f}).",
            "Key technical factors include {indicator1} at {value1:+.3f} showing {direction1} bias, and {indicator2} at {value2:+.3f} indicating {direction2} momentum.",
            "Technical analysis reveals {indicator1} with {strength1} {direction1} divergence ({value1:+.3f}) and {indicator2} showing {strength2} {direction2} signals ({value2:+.3f}).",
        ]

        self.sentiment_integration_patterns = [
            "Market sentiment is {sentiment_label} (confidence: {sentiment_confidence:.0%}, score: {sentiment_score:+.2f}), which {alignment} with the technical outlook.",
            "Current {sentiment_label} sentiment (score: {sentiment_score:+.2f}, confidence: {sentiment_confidence:.0%}) {alignment} the technical analysis.",
            "Sentiment analysis shows {sentiment_label} market mood ({sentiment_score:+.2f}) with {sentiment_confidence:.0%} confidence, {alignment} technical indicators.",
            "{sentiment_label.title()} market sentiment ({sentiment_confidence:.0%} confidence, {sentiment_score:+.2f} score) {alignment} the technical picture.",
        ]

    def generate_technical_scenario(self) -> Tuple[Dict[str, Any], float]:
        """Generate realistic technical analysis scenario."""
        # Random symbol
        symbol = random.choice(self.symbols)

        # Generate score first to ensure consistency
        score = round(random.uniform(1.0, 9.5), 1)

        # Generate weighted scores that correlate with overall score
        weighted_scores = {}
        score_bias = (score - 5.0) / 5.0  # Normalize to -1 to 1

        # Select 4-8 random indicators
        selected_indicators = random.sample(list(self.indicators.keys()), random.randint(4, 8))

        for indicator in selected_indicators:
            indicator_range = self.indicators[indicator]["range"]

            # Add bias toward score direction
            if score_bias > 0:  # Bullish score
                # Bias toward positive values
                base_value = random.uniform(indicator_range[0] * 0.3, indicator_range[1])
            else:  # Bearish score
                # Bias toward negative values
                base_value = random.uniform(indicator_range[0], indicator_range[1] * 0.3)

            # Add some randomness
            noise = random.uniform(-0.02, 0.02)
            weighted_scores[indicator] = round(base_value + noise, 4)

        scenario = {"symbol": symbol, "score_0_10": score, "weighted_scores": weighted_scores, "components": {}}

        return scenario, score

    def generate_sentiment_scenario(self) -> Dict[str, Any]:
        """Generate realistic sentiment analysis scenario."""
        profile = random.choice(self.sentiment_profiles)

        sentiment_score = round(random.uniform(*profile["score_range"]), 3)
        sentiment_confidence = round(random.uniform(*profile["confidence_range"]), 3)

        return {
            "sentimentScore": sentiment_score,
            "sentimentLabel": profile["label"],
            "sentimentConfidence": sentiment_confidence,
            "newsCount": random.randint(1, 8),
            "timestamp": datetime.now().isoformat(),
        }

    def get_recommendation_category(self, score: float) -> str:
        """Determine recommendation category based on score."""
        for category, config in self.recommendation_templates.items():
            if config["score_range"][0] <= score <= config["score_range"][1]:
                return category
        return "hold"  # Fallback

    def generate_technical_analysis_text(self, scenario: Dict[str, Any]) -> str:
        """Generate technical analysis description."""
        weighted_scores = scenario["weighted_scores"]

        # Get top 2-3 indicators by absolute value
        sorted_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        if len(sorted_indicators) < 2:
            return "Limited technical data available for comprehensive analysis."

        # Select pattern
        pattern = random.choice(self.technical_analysis_patterns)

        # Get first two indicators
        ind1_key, ind1_value = sorted_indicators[0]
        ind2_key, ind2_value = sorted_indicators[1]

        # Convert to readable names
        ind1_name = self.indicators[ind1_key]["name"]
        ind2_name = self.indicators[ind2_key]["name"]

        # Determine strength and direction
        def get_strength_direction(value):
            abs_value = abs(value)
            if abs_value > 0.15:
                strength = "strong"
            elif abs_value > 0.08:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "bullish" if value > 0 else "bearish" if value < 0 else "neutral"
            return strength, direction

        strength1, direction1 = get_strength_direction(ind1_value)
        strength2, direction2 = get_strength_direction(ind2_value)

        return pattern.format(
            indicator1=ind1_name,
            indicator2=ind2_name,
            strength1=strength1,
            direction1=direction1,
            strength2=strength2,
            direction2=direction2,
            value1=ind1_value,
            value2=ind2_value,
        )

    def generate_sentiment_context_text(self, sentiment_data: Dict[str, Any], score: float) -> str:
        """Generate sentiment integration text."""
        if not sentiment_data:
            return ""

        sentiment_label = sentiment_data["sentimentLabel"]
        sentiment_score = sentiment_data["sentimentScore"]
        sentiment_confidence = sentiment_data["sentimentConfidence"]

        # Determine alignment
        score_direction = "positive" if score > 5.5 else "negative" if score < 4.5 else "neutral"

        if sentiment_label == score_direction or (sentiment_label == "neutral" and 4.5 <= score <= 5.5):
            alignment = "aligns well"
        elif (sentiment_label == "positive" and score > 4.5) or (sentiment_label == "negative" and score < 5.5):
            alignment = "generally supports"
        else:
            alignment = "contrasts"

        pattern = random.choice(self.sentiment_integration_patterns)

        return pattern.format(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            alignment=alignment,
        )

    def generate_risk_and_targets(self, symbol: str, score: float, sentiment_data: Dict[str, Any]) -> str:
        """Generate risk factors and price targets."""
        risk_factors = []

        if score >= 7.5:
            risk_factors.append("Monitor for overbought conditions and potential profit-taking.")
        elif score <= 3.0:
            risk_factors.append("Significant downside risk remains with weak technical foundation.")
        else:
            risk_factors.append("Watch for volatility around key support/resistance levels.")

        if sentiment_data and sentiment_data["sentimentConfidence"] < 0.6:
            risk_factors.append("Uncertain market sentiment adds to volatility risk.")

        # Price targets (illustrative)
        target_phrases = [
            f"Near-term resistance at current levels plus 8-12%.",
            f"Support expected around 5-10% below current price.",
            f"Target price range suggests 10-15% potential movement.",
            f"Key levels to watch for breakout confirmation.",
        ]

        risk_text = " ".join(risk_factors)
        if random.choice([True, False]):  # 50% chance to include targets
            risk_text += " " + random.choice(target_phrases)

        return risk_text

    def generate_instruction_sample(self) -> Dict[str, Any]:
        """Generate a complete instruction sample for fine-tuning."""
        # Generate technical scenario
        technical_scenario, score = self.generate_technical_scenario()

        # Generate sentiment scenario (80% chance)
        sentiment_data = self.generate_sentiment_scenario() if random.random() < 0.8 else None

        # Create input data
        input_data = technical_scenario.copy()
        if sentiment_data:
            input_data["sentiment"] = sentiment_data

        # Generate instruction
        symbol = technical_scenario["symbol"]
        recommendation_category = self.get_recommendation_category(score)

        # Build instruction text
        instruction_parts = [f"Generate a comprehensive investment analysis for {symbol} with score {score}/10"]

        if sentiment_data:
            sentiment_desc = f"{sentiment_data['sentimentLabel']} sentiment (confidence: {sentiment_data['sentimentConfidence']:.2f}, score: {sentiment_data['sentimentScore']:+.2f})"
            instruction_parts.append(f"considering {sentiment_desc}")

        instruction_parts.append("based on the provided technical indicators.")
        if sentiment_data:
            instruction_parts.append("Integrate sentiment analysis with technical findings.")

        instruction = " ".join(instruction_parts)

        # Generate high-quality output
        template = random.choice(self.recommendation_templates[recommendation_category]["templates"])

        # Generate components
        technical_analysis = self.generate_technical_analysis_text(technical_scenario)
        sentiment_context = self.generate_sentiment_context_text(sentiment_data, score) if sentiment_data else ""
        risk_factors = self.generate_risk_and_targets(symbol, score, sentiment_data)

        # Support/resistance levels (illustrative)
        support_level = f"${random.randint(80, 150)}"
        price_target = f"${random.randint(160, 250)}"

        output = template.format(
            symbol=symbol,
            score=score,
            technical_analysis=technical_analysis,
            sentiment_context=sentiment_context,
            risk_factors=risk_factors,
            risk_considerations=risk_factors,
            support_level=support_level,
            price_target=price_target,
        ).strip()

        # Clean up multiple spaces and formatting
        output = " ".join(output.split())

        return {
            "instruction": instruction,
            "input": input_data,
            "output": output,
            "metadata": {
                "recommendation_category": recommendation_category,
                "has_sentiment": sentiment_data is not None,
                "complexity_score": len(technical_scenario["weighted_scores"]) / 12.0,  # Normalized
                "created_at": datetime.now().isoformat(),
            },
        }

    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate complete dataset for fine-tuning."""
        dataset = []

        logger.info(f"Generating {num_samples} financial instruction samples...")

        # Track categories for balanced dataset
        category_counts = {category: 0 for category in self.recommendation_templates.keys()}
        sentiment_counts = {"with_sentiment": 0, "without_sentiment": 0}

        for i in range(num_samples):
            try:
                sample = self.generate_instruction_sample()
                dataset.append(sample)

                # Track statistics
                category = sample["metadata"]["recommendation_category"]
                category_counts[category] += 1

                if sample["metadata"]["has_sentiment"]:
                    sentiment_counts["with_sentiment"] += 1
                else:
                    sentiment_counts["without_sentiment"] += 1

                if (i + 1) % 1000 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} samples...")

            except Exception as e:
                logger.error(f"Error generating sample {i + 1}: {str(e)}")
                continue

        # Log final statistics
        logger.info("Dataset generation complete!")
        logger.info(f"Category distribution: {category_counts}")
        logger.info(f"Sentiment distribution: {sentiment_counts}")

        return dataset


class Command(BaseCommand):
    """Management command to generate financial instruction dataset."""

    help = "Generate financial instruction dataset for LLM fine-tuning"

    def add_arguments(self, parser):
        parser.add_argument(
            "--samples", type=int, default=10000, help="Number of instruction samples to generate (default: 10000)"
        )
        parser.add_argument(
            "--output-dir", type=str, default="Temp/", help="Output directory for dataset files (default: Temp/)"
        )
        parser.add_argument("--split", action="store_true", help="Split dataset into train/validation/test sets")
        parser.add_argument("--format", choices=["json", "jsonl"], default="json", help="Output format (default: json)")

    def handle(self, *args, **options):
        """Handle command execution."""
        num_samples = options["samples"]
        output_dir = options["output_dir"]
        should_split = options["split"]
        output_format = options["format"]

        self.stdout.write(f"Generating {num_samples} financial instruction samples...")

        try:
            # Generate dataset
            generator = FinancialDatasetGenerator()
            dataset = generator.generate_dataset(num_samples)

            if not dataset:
                self.stdout.write(self.style.ERROR("Failed to generate dataset"))
                return

            # Create output directory
            import os

            os.makedirs(output_dir, exist_ok=True)

            if should_split:
                # Split dataset: 80% train, 15% validation, 5% test
                random.shuffle(dataset)

                train_size = int(0.8 * len(dataset))
                val_size = int(0.15 * len(dataset))

                train_data = dataset[:train_size]
                val_data = dataset[train_size : train_size + val_size]
                test_data = dataset[train_size + val_size :]

                splits = {"train": train_data, "validation": val_data, "test": test_data}

                for split_name, split_data in splits.items():
                    filename = f"financial_instruction_dataset_{split_name}.{output_format}"
                    filepath = os.path.join(output_dir, filename)

                    self._save_dataset(split_data, filepath, output_format)

                    self.stdout.write(self.style.SUCCESS(f"Saved {len(split_data)} {split_name} samples to {filepath}"))
            else:
                # Save complete dataset
                filename = f"financial_instruction_dataset.{output_format}"
                filepath = os.path.join(output_dir, filename)

                self._save_dataset(dataset, filepath, output_format)

                self.stdout.write(self.style.SUCCESS(f"Saved {len(dataset)} samples to {filepath}"))

            # Save metadata
            metadata = {
                "total_samples": len(dataset),
                "generation_date": datetime.now().isoformat(),
                "format": output_format,
                "split": should_split,
                "categories": list(generator.recommendation_templates.keys()),
                "indicators": list(generator.indicators.keys()),
                "description": "Financial instruction dataset for LLM fine-tuning on investment analysis",
            }

            metadata_file = os.path.join(output_dir, "dataset_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.stdout.write(self.style.SUCCESS(f"Dataset generation complete! Metadata saved to {metadata_file}"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating dataset: {str(e)}"))
            raise

    def _save_dataset(self, dataset: List[Dict[str, Any]], filepath: str, format_type: str):
        """Save dataset to file."""
        try:
            if format_type == "jsonl":
                # JSONL format - one JSON object per line
                with open(filepath, "w", encoding="utf-8") as f:
                    for sample in dataset:
                        json.dump(sample, f, ensure_ascii=False)
                        f.write("\n")
            else:
                # JSON format - single array
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise Exception(f"Failed to save dataset to {filepath}: {str(e)}")
