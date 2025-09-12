"""
Analytics results writer for storing technical analysis outputs.
Handles upsert operations to ANALYTICS_RESULTS table with PostgreSQL optimization.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from django.db import transaction
from django.db.models import Avg, Count, Max, Min
from django.utils import timezone

from Data.models import AnalyticsResults, Stock


class AnalyticsWriter:
    """
    Repository for writing analytics results with conflict resolution.
    Uses PostgreSQL upsert pattern: ON CONFLICT (stock_id, as_of) DO UPDATE.
    """

    def __init__(self):
        """Initialize the analytics writer."""

    def upsert_analytics_result(
        self,
        symbol: str,
        as_of: datetime,
        weighted_scores: Dict[str, Optional[Decimal]],
        components: Dict[str, Dict[str, Any]],
        composite_raw: Decimal,
        score_0_10: int,
        horizon: str = "blend",
        user=None,
    ) -> AnalyticsResults:
        """
        Upsert analytics result for a stock at specific timestamp.

        Args:
            symbol: Stock ticker symbol
            as_of: Analysis timestamp
            weighted_scores: Dict with all 12 w_* indicator scores
            components: Raw and normalized values per indicator
            composite_raw: Sum of all weighted scores
            score_0_10: Final composite score (0-10)
            horizon: Analysis time horizon
            user: User instance who initiated the analysis

        Returns:
            AnalyticsResults instance (created or updated)

        Raises:
            Stock.DoesNotExist: If stock symbol not found
            ValueError: If required weighted scores missing
        """
        # Validate required weighted scores
        required_indicators = [
            "w_sma50vs200",
            "w_pricevs50",
            "w_rsi14",
            "w_macd12269",
            "w_bbpos20",
            "w_bbwidth20",
            "w_volsurge",
            "w_obv20",
            "w_rel1y",
            "w_rel2y",
            "w_candlerev",
            "w_srcontext",
        ]

        for indicator in required_indicators:
            if indicator not in weighted_scores:
                raise ValueError(f"Missing required weighted score: {indicator}")

        # Get stock instance
        stock = Stock.objects.get(symbol=symbol.upper())

        # Use Django's update_or_create for atomic upsert
        with transaction.atomic():
            lookup_fields = {
                "stock": stock,
                "as_of": as_of,
            }

            # Include user in lookup if provided
            if user:
                lookup_fields["user"] = user

            result, created = AnalyticsResults.objects.update_or_create(
                **lookup_fields,
                defaults={
                    "user": user,
                    "horizon": horizon,
                    "w_sma50vs200": weighted_scores.get("w_sma50vs200"),
                    "w_pricevs50": weighted_scores.get("w_pricevs50"),
                    "w_rsi14": weighted_scores.get("w_rsi14"),
                    "w_macd12269": weighted_scores.get("w_macd12269"),
                    "w_bbpos20": weighted_scores.get("w_bbpos20"),
                    "w_bbwidth20": weighted_scores.get("w_bbwidth20"),
                    "w_volsurge": weighted_scores.get("w_volsurge"),
                    "w_obv20": weighted_scores.get("w_obv20"),
                    "w_rel1y": weighted_scores.get("w_rel1y"),
                    "w_rel2y": weighted_scores.get("w_rel2y"),
                    "w_candlerev": weighted_scores.get("w_candlerev"),
                    "w_srcontext": weighted_scores.get("w_srcontext"),
                    "components": components,
                    "composite_raw": composite_raw,
                    "score_0_10": score_0_10,
                    "updated_at": timezone.now(),
                    # Add sentiment fields if present in components
                    "sentimentScore": components.get("sentiment", {}).get("raw", {}).get("sentiment"),
                    "sentimentLabel": components.get("sentiment", {}).get("raw", {}).get("label"),
                    "sentimentConfidence": components.get("sentiment", {}).get("raw", {}).get("confidence"),
                    "newsCount": components.get("sentiment", {}).get("raw", {}).get("newsCount", 0),
                    "sentimentSources": components.get("sentiment", {}).get("raw", {}).get("sources", {}),
                    # Add LSTM prediction fields if present in components
                    "prediction_1d": components.get("prediction", {}).get("raw", {}).get("predicted_price"),
                    "prediction_7d": None,  # Not implemented yet
                    "prediction_30d": None,  # Not implemented yet
                    "prediction_confidence": components.get("prediction", {}).get("raw", {}).get("confidence"),
                    "model_version": components.get("prediction", {}).get("raw", {}).get("model_version"),
                    "prediction_timestamp": timezone.now() if components.get("prediction", {}).get("raw") else None,
                },
            )

        return result

    def batch_upsert_analytics_results(self, results_data: list[Dict[str, Any]], user=None) -> list[AnalyticsResults]:
        """
        Batch upsert multiple analytics results for performance.

        Args:
            results_data: List of dicts, each containing:
                - symbol: str
                - as_of: datetime
                - weighted_scores: Dict[str, Optional[Decimal]]
                - components: Dict[str, Dict[str, Any]]
                - composite_raw: Decimal
                - score_0_10: int
                - horizon: str (optional, defaults to 'blend')

        Returns:
            List of AnalyticsResults instances

        Raises:
            ValueError: If any result data is invalid
        """
        results = []

        with transaction.atomic():
            for data in results_data:
                result = self.upsert_analytics_result(
                    symbol=data["symbol"],
                    as_of=data["as_of"],
                    weighted_scores=data["weighted_scores"],
                    components=data["components"],
                    composite_raw=data["composite_raw"],
                    score_0_10=data["score_0_10"],
                    horizon=data.get("horizon", "blend"),
                    user=user or data.get("user"),
                )
                results.append(result)

        return results

    def get_latest_analytics_result(self, symbol: str) -> Optional[AnalyticsResults]:
        """
        Get the most recent analytics result for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest AnalyticsResults instance, or None if not found
        """
        try:
            stock = Stock.objects.get(symbol=symbol.upper())
            return stock.analytics_results.order_by("-as_of").first()
        except Stock.DoesNotExist:
            return None

    def get_analytics_results_range(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[AnalyticsResults]:
        """
        Get analytics results for a stock within date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start datetime (inclusive), optional
            end_date: End datetime (inclusive), optional
            limit: Maximum number of results to return

        Returns:
            List of AnalyticsResults instances ordered by as_of descending
        """
        try:
            stock = Stock.objects.get(symbol=symbol.upper())

            queryset = stock.analytics_results.order_by("-as_of")

            if start_date:
                queryset = queryset.filter(as_of__gte=start_date)

            if end_date:
                queryset = queryset.filter(as_of__lte=end_date)

            if limit:
                queryset = queryset[:limit]

            return list(queryset)
        except Stock.DoesNotExist:
            return []

    def delete_old_analytics_results(self, symbol: str, keep_days: int = 90) -> int:
        """
        Delete old analytics results to manage storage.

        Args:
            symbol: Stock ticker symbol
            keep_days: Number of days of results to retain

        Returns:
            Number of results deleted
        """
        try:
            stock = Stock.objects.get(symbol=symbol.upper())

            cutoff_date = timezone.now() - timezone.timedelta(days=keep_days)

            deleted_count, _ = stock.analytics_results.filter(as_of__lt=cutoff_date).delete()

            return deleted_count
        except Stock.DoesNotExist:
            return 0

    def get_analytics_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get summary statistics for a stock's analytics results.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict containing summary statistics
        """
        try:
            stock = Stock.objects.get(symbol=symbol.upper())

            results = stock.analytics_results.all()

            if not results.exists():
                return {
                    "total_results": 0,
                    "earliest_analysis": None,
                    "latest_analysis": None,
                    "avg_score": None,
                    "min_score": None,
                    "max_score": None,
                }

            stats = results.aggregate(
                total_results=Count("id"),
                earliest_analysis=Min("as_of"),
                latest_analysis=Max("as_of"),
                avg_score=Avg("score_0_10"),
                min_score=Min("score_0_10"),
                max_score=Max("score_0_10"),
            )

            return stats

        except Stock.DoesNotExist:
            return {
                "total_results": 0,
                "earliest_analysis": None,
                "latest_analysis": None,
                "avg_score": None,
                "min_score": None,
                "max_score": None,
            }
