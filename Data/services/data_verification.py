"""
Data Verification Service

Provides comprehensive data validation and quality assessment for stock analysis.
Ensures data meets minimum requirements and provides confidence scoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from django.db.models import Count, Min, Max
from Data.models import Stock, StockPrice

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"    # 500+ days
    GOOD = "good"             # 200-499 days  
    FAIR = "fair"             # 50-199 days
    LIMITED = "limited"       # 20-49 days
    INSUFFICIENT = "insufficient"  # <20 days


@dataclass
class DataAvailability:
    """Data availability summary for a stock"""
    symbol: str
    total_days: int
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]
    gaps: int
    gap_percentage: float
    quality_level: DataQualityLevel
    confidence_score: float  # 0-1 scale


@dataclass
class IndicatorRequirement:
    """Requirements for a specific indicator"""
    name: str
    min_days: int
    optimal_days: int
    can_adapt: bool
    fallback_available: bool
    weight: float


@dataclass
class AnalysisReadiness:
    """Assessment of readiness for analysis"""
    overall_confidence: float  # 0-1 scale
    available_indicators: List[str]
    missing_indicators: List[str]
    adaptable_indicators: List[str]
    data_availability: DataAvailability
    recommendations: List[str]


class DataVerificationService:
    """
    Service for comprehensive data verification and analysis readiness assessment.
    """
    
    def __init__(self):
        self.indicator_requirements = {
            'obv20': IndicatorRequirement('OBV 20-day', 20, 50, True, True, 0.08),
            'volsurge': IndicatorRequirement('Volume Surge', 10, 30, True, False, 0.06),
            'srcontext': IndicatorRequirement('Support/Resistance', 50, 100, False, True, 0.12),
            'bbwidth20': IndicatorRequirement('Bollinger Band Width', 20, 50, True, False, 0.08),
            'sma50vs200': IndicatorRequirement('SMA 50 vs 200', 200, 250, True, True, 0.1),
            'pricevs50': IndicatorRequirement('Price vs SMA50', 50, 100, True, True, 0.08),
            'macd12269': IndicatorRequirement('MACD', 26, 50, True, False, 0.08),
            'rsi14': IndicatorRequirement('RSI 14-day', 14, 30, True, False, 0.08),
            'candlerev': IndicatorRequirement('Candle Reversal', 5, 20, True, False, 0.08),
            'bbpos20': IndicatorRequirement('Bollinger Position', 20, 50, True, False, 0.06),
            'rel1y': IndicatorRequirement('1-Year Relative', 252, 300, False, True, 0.08),
            'rel2y': IndicatorRequirement('2-Year Relative', 504, 600, False, True, 0.06),
            'prediction': IndicatorRequirement('LSTM Prediction', 50, 200, False, True, 0.08),
            'sentiment': IndicatorRequirement('Sentiment Analysis', 1, 1, False, False, 0.06)
        }
        
    def assess_data_availability(self, symbol: str) -> DataAvailability:
        """
        Assess data availability for a stock.
        
        Args:
            symbol: Stock symbol to assess
            
        Returns:
            DataAvailability assessment
        """
        try:
            stock = Stock.objects.get(symbol=symbol, is_active=True)
        except Stock.DoesNotExist:
            return DataAvailability(
                symbol=symbol,
                total_days=0,
                date_range_start=None,
                date_range_end=None,
                gaps=0,
                gap_percentage=100.0,
                quality_level=DataQualityLevel.INSUFFICIENT,
                confidence_score=0.0
            )
        
        # Get price data statistics
        price_stats = StockPrice.objects.filter(stock=stock).aggregate(
            count=Count('id'),
            min_date=Min('date'),
            max_date=Max('date')
        )
        
        total_days = price_stats['count'] or 0
        min_date = price_stats['min_date']
        max_date = price_stats['max_date']
        
        # Calculate gaps
        gaps = 0
        gap_percentage = 0.0
        
        if min_date and max_date and total_days > 0:
            # Calculate expected trading days (roughly 252 per year)
            date_range_days = (max_date - min_date).days
            expected_trading_days = int(date_range_days * 0.71)  # Approximate trading days
            gaps = max(0, expected_trading_days - total_days)
            gap_percentage = (gaps / expected_trading_days * 100) if expected_trading_days > 0 else 0
        
        # Determine quality level
        if total_days >= 500:
            quality_level = DataQualityLevel.EXCELLENT
            confidence_score = min(1.0, total_days / 500)
        elif total_days >= 200:
            quality_level = DataQualityLevel.GOOD
            confidence_score = min(0.8, total_days / 250)
        elif total_days >= 50:
            quality_level = DataQualityLevel.FAIR
            confidence_score = min(0.6, total_days / 100)
        elif total_days >= 20:
            quality_level = DataQualityLevel.LIMITED
            confidence_score = min(0.4, total_days / 50)
        else:
            quality_level = DataQualityLevel.INSUFFICIENT
            confidence_score = min(0.2, total_days / 20)
        
        return DataAvailability(
            symbol=symbol,
            total_days=total_days,
            date_range_start=min_date,
            date_range_end=max_date,
            gaps=gaps,
            gap_percentage=gap_percentage,
            quality_level=quality_level,
            confidence_score=confidence_score
        )
        
    def assess_analysis_readiness(self, symbol: str) -> AnalysisReadiness:
        """
        Comprehensive assessment of analysis readiness.
        
        Args:
            symbol: Stock symbol to assess
            
        Returns:
            AnalysisReadiness assessment
        """
        data_availability = self.assess_data_availability(symbol)
        
        available_indicators = []
        missing_indicators = []
        adaptable_indicators = []
        total_weight = 0
        available_weight = 0
        
        # Check each indicator requirement
        for indicator_name, requirement in self.indicator_requirements.items():
            total_weight += requirement.weight
            
            if indicator_name == 'sentiment':
                # Sentiment analysis doesn't depend on historical price data
                available_indicators.append(indicator_name)
                available_weight += requirement.weight
            elif data_availability.total_days >= requirement.min_days:
                available_indicators.append(indicator_name)
                available_weight += requirement.weight
            elif requirement.can_adapt and data_availability.total_days >= (requirement.min_days * 0.5):
                # Can adapt indicator if we have at least 50% of required data
                adaptable_indicators.append(indicator_name)
                available_weight += requirement.weight * 0.7  # Reduced weight for adapted indicators
            else:
                missing_indicators.append(indicator_name)
        
        # Calculate overall confidence
        base_confidence = available_weight / total_weight
        data_quality_bonus = data_availability.confidence_score * 0.2
        overall_confidence = min(1.0, base_confidence + data_quality_bonus)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            data_availability, available_indicators, missing_indicators, adaptable_indicators
        )
        
        return AnalysisReadiness(
            overall_confidence=overall_confidence,
            available_indicators=available_indicators,
            missing_indicators=missing_indicators,
            adaptable_indicators=adaptable_indicators,
            data_availability=data_availability,
            recommendations=recommendations
        )
        
    def get_progressive_data_targets(self, current_days: int) -> List[Dict[str, Any]]:
        """
        Get progressive data collection targets based on current data availability.
        
        Args:
            current_days: Current number of days available
            
        Returns:
            List of target periods to try
        """
        targets = []
        
        if current_days < 30:
            targets.extend([
                {'period': '3mo', 'days': 60, 'priority': 'high'},
                {'period': '6mo', 'days': 120, 'priority': 'medium'},
                {'period': '1y', 'days': 250, 'priority': 'low'}
            ])
        elif current_days < 100:
            targets.extend([
                {'period': '6mo', 'days': 120, 'priority': 'high'},
                {'period': '1y', 'days': 250, 'priority': 'medium'},
                {'period': '2y', 'days': 500, 'priority': 'low'}
            ])
        elif current_days < 250:
            targets.extend([
                {'period': '1y', 'days': 250, 'priority': 'high'},
                {'period': '2y', 'days': 500, 'priority': 'medium'},
                {'period': '5y', 'days': 1250, 'priority': 'low'}
            ])
        else:
            targets.extend([
                {'period': '2y', 'days': 500, 'priority': 'medium'},
                {'period': '5y', 'days': 1250, 'priority': 'low'},
                {'period': 'max', 'days': 2500, 'priority': 'low'}
            ])
            
        return targets
        
    def _generate_recommendations(
        self, 
        data_availability: DataAvailability,
        available_indicators: List[str],
        missing_indicators: List[str],
        adaptable_indicators: List[str]
    ) -> List[str]:
        """Generate actionable recommendations based on data assessment."""
        recommendations = []
        
        if data_availability.quality_level == DataQualityLevel.INSUFFICIENT:
            recommendations.append(
                f"Critical: Only {data_availability.total_days} days available. "
                "Fetch minimum 20 days for basic analysis."
            )
        elif data_availability.quality_level == DataQualityLevel.LIMITED:
            recommendations.append(
                f"Limited data: {data_availability.total_days} days available. "
                "Consider fetching 50+ days for better indicator coverage."
            )
        elif data_availability.quality_level == DataQualityLevel.FAIR:
            recommendations.append(
                f"Fair data coverage: {data_availability.total_days} days. "
                "Fetch 200+ days to enable all long-term indicators."
            )
        
        if data_availability.gap_percentage > 20:
            recommendations.append(
                f"Data gaps detected: {data_availability.gap_percentage:.1f}% missing. "
                "Consider backfilling gaps for improved accuracy."
            )
            
        if missing_indicators:
            missing_count = len(missing_indicators)
            recommendations.append(
                f"{missing_count} indicators unavailable: {', '.join(missing_indicators[:3])}. "
                f"Fetch more historical data to enable these indicators."
            )
            
        if adaptable_indicators:
            adaptable_count = len(adaptable_indicators)
            recommendations.append(
                f"{adaptable_count} indicators can be adapted: {', '.join(adaptable_indicators[:3])}. "
                "Analysis will use modified calculations."
            )
            
        confidence_pct = data_availability.confidence_score * 100
        if confidence_pct < 60:
            recommendations.append(
                f"Low confidence ({confidence_pct:.0f}%). "
                "Results may be less reliable due to limited data."
            )
            
        return recommendations
        
    def get_minimum_requirements(self) -> Dict[str, Any]:
        """Get minimum data requirements for analysis."""
        return {
            'absolute_minimum_days': 20,
            'recommended_minimum_days': 50,
            'optimal_days': 200,
            'excellent_days': 500,
            'critical_indicators': ['rsi14', 'candlerev', 'sentiment'],
            'optional_indicators': ['rel1y', 'rel2y', 'srcontext']
        }


# Global service instance
data_verification_service = DataVerificationService()