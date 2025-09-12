"""
Enhanced Backfill Service

Integrates multi-source data fetching with progressive loading and intelligent fallback.
Provides comprehensive data collection with quality assessment.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal

from django.db import transaction
from django.utils import timezone

from Data.models import Stock, StockPrice
from Data.services.multi_source_fetcher import multi_source_fetcher, DataPoint
from Data.services.data_verification import data_verification_service

logger = logging.getLogger(__name__)


class EnhancedBackfillService:
    """
    Enhanced backfill service with multi-source support and progressive loading.
    """
    
    def __init__(self):
        self.max_attempts = 3
        self.progressive_periods = [
            {'name': '2y', 'days': 500, 'priority': 'high'},
            {'name': '1y', 'days': 250, 'priority': 'medium'},
            {'name': '6mo', 'days': 120, 'priority': 'medium'}, 
            {'name': '3mo', 'days': 60, 'priority': 'low'},
            {'name': '1mo', 'days': 20, 'priority': 'low'}
        ]
        
    def enhanced_backfill_concurrent(
        self,
        symbol: str,
        required_years: int = 2,
        max_attempts: int = 3,
        min_acceptable_days: int = 20
    ) -> Dict[str, Any]:
        """
        Enhanced backfill with multi-source support and progressive loading.
        
        Args:
            symbol: Stock symbol to backfill
            required_years: Target years of data
            max_attempts: Maximum attempts per source
            min_acceptable_days: Minimum days to consider success
            
        Returns:
            Dictionary with enhanced backfill results
        """
        logger.info(f"Starting enhanced backfill for {symbol}")
        start_time = timezone.now()
        
        # Initial data verification
        availability = data_verification_service.assess_data_availability(symbol)
        
        # Determine target data range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=required_years * 365)
        
        # Progressive data loading strategy
        best_result = None
        total_api_calls = 0
        sources_attempted = []
        
        # Try progressive targets based on current data availability
        targets = self._get_progressive_targets(availability.total_days, required_years)
        
        for target in targets:
            logger.info(f"Attempting to fetch {target['name']} of data for {symbol}")
            
            # Calculate target date range
            target_start = end_date - timedelta(days=target['days'])
            
            # Fetch data using multi-source approach
            result = multi_source_fetcher.fetch_historical_data(
                symbol=symbol,
                start_date=target_start,
                end_date=end_date,
                min_days=min_acceptable_days
            )
            
            total_api_calls += result.api_calls
            sources_attempted.append(result.source)
            
            if result.success:
                # Store the data
                prices_stored = self._store_price_data(symbol, result.data)
                
                if prices_stored >= min_acceptable_days:
                    logger.info(f"Successfully backfilled {symbol} with {prices_stored} records from {result.source}")
                    
                    # Skip auto-fitting scalers - handled by analytics engine as needed
                    if prices_stored >= 50:
                        logger.info(f"Sufficient data available for {symbol} - scalers can be fitted during analysis")
                    
                    best_result = result
                    break
                else:
                    logger.warning(f"Insufficient data stored for {symbol}: {prices_stored} < {min_acceptable_days}")
            else:
                logger.warning(f"Failed to fetch data for {symbol} from {result.source}: {result.error}")
        
        # Final assessment
        final_availability = data_verification_service.assess_data_availability(symbol)
        analysis_readiness = data_verification_service.assess_analysis_readiness(symbol)
        
        end_time = timezone.now()
        total_time = (end_time - start_time).total_seconds()
        
        return {
            'success': best_result is not None,
            'stock_backfilled': final_availability.total_days if best_result else 0,
            'sector_backfilled': 0,  # Not implemented in this version
            'industry_backfilled': 0,  # Not implemented in this version
            'attempts_used': len(sources_attempted),
            'sources_attempted': sources_attempted,
            'best_source': best_result.source if best_result else None,
            'data_quality_before': {
                'total_days': availability.total_days,
                'quality_level': availability.quality_level.value,
                'confidence': availability.confidence_score
            },
            'data_quality_after': {
                'total_days': final_availability.total_days,
                'quality_level': final_availability.quality_level.value,
                'confidence': final_availability.confidence_score
            },
            'analysis_readiness': {
                'overall_confidence': analysis_readiness.overall_confidence,
                'available_indicators': len(analysis_readiness.available_indicators),
                'missing_indicators': len(analysis_readiness.missing_indicators),
                'adaptable_indicators': len(analysis_readiness.adaptable_indicators)
            },
            'recommendations': analysis_readiness.recommendations,
            'errors': [],
            'performance': {
                'total_time': total_time,
                'cache_hits': 0,  # Multi-source fetcher handles caching
                'api_calls': total_api_calls,
                'concurrent_fetches': 1
            }
        }
        
    def _get_progressive_targets(self, current_days: int, required_years: int) -> List[Dict[str, Any]]:
        """Get progressive loading targets based on current data availability."""
        target_days = required_years * 250  # Approximate trading days per year
        
        targets = []
        
        if current_days < 30:
            # Very limited data - try to get any reasonable amount
            targets.extend([
                {'name': '2y', 'days': 500, 'priority': 'high'},
                {'name': '1y', 'days': 250, 'priority': 'high'},
                {'name': '6mo', 'days': 120, 'priority': 'medium'},
                {'name': '3mo', 'days': 60, 'priority': 'medium'},
                {'name': '1mo', 'days': 30, 'priority': 'low'}
            ])
        elif current_days < 100:
            # Some data available - prioritize getting to good coverage
            targets.extend([
                {'name': '2y', 'days': 500, 'priority': 'high'},
                {'name': '1y', 'days': 250, 'priority': 'high'},
                {'name': '6mo', 'days': 120, 'priority': 'medium'}
            ])
        elif current_days < target_days:
            # Good amount but not enough - try to get full target
            targets.extend([
                {'name': '2y', 'days': 500, 'priority': 'high'},
                {'name': '1y', 'days': 250, 'priority': 'medium'}
            ])
        else:
            # Already have enough - try to expand if possible
            targets.extend([
                {'name': '5y', 'days': 1250, 'priority': 'low'},
                {'name': '2y', 'days': 500, 'priority': 'low'}
            ])
            
        return targets
        
    def _store_price_data(self, symbol: str, data_points: List[DataPoint]) -> int:
        """
        Store price data points in the database.
        
        Args:
            symbol: Stock symbol
            data_points: List of DataPoint objects
            
        Returns:
            Number of records stored
        """
        if not data_points:
            return 0
            
        try:
            stock = Stock.objects.get(symbol=symbol, is_active=True)
        except Stock.DoesNotExist:
            logger.error(f"Stock {symbol} not found in database")
            return 0
        
        stored_count = 0
        
        with transaction.atomic():
            for data_point in data_points:
                try:
                    # Check if record already exists
                    existing = StockPrice.objects.filter(
                        stock=stock,
                        date=data_point.date.date()
                    ).first()
                    
                    if not existing:
                        StockPrice.objects.create(
                            stock=stock,
                            date=data_point.date.date(),
                            open=Decimal(str(data_point.open)),
                            high=Decimal(str(data_point.high)),
                            low=Decimal(str(data_point.low)),
                            close=Decimal(str(data_point.close)),
                            volume=data_point.volume,
                            data_source=data_point.source
                        )
                        stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to store data point for {symbol} on {data_point.date}: {str(e)}")
                    continue
        
        logger.info(f"Stored {stored_count} new price records for {symbol}")
        return stored_count
        
    def get_data_availability_report(self, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive data availability report.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Comprehensive availability report
        """
        # Data availability assessment
        availability = data_verification_service.assess_data_availability(symbol)
        
        # Analysis readiness assessment
        readiness = data_verification_service.assess_analysis_readiness(symbol)
        
        # Check multi-source availability
        source_availability = multi_source_fetcher.get_available_data_range(symbol)
        
        return {
            'symbol': symbol,
            'current_data': {
                'total_days': availability.total_days,
                'date_range': {
                    'start': availability.date_range_start.isoformat() if availability.date_range_start else None,
                    'end': availability.date_range_end.isoformat() if availability.date_range_end else None
                },
                'gaps': availability.gaps,
                'gap_percentage': availability.gap_percentage,
                'quality_level': availability.quality_level.value,
                'confidence_score': availability.confidence_score
            },
            'analysis_readiness': {
                'overall_confidence': readiness.overall_confidence,
                'available_indicators': readiness.available_indicators,
                'missing_indicators': readiness.missing_indicators,
                'adaptable_indicators': readiness.adaptable_indicators,
                'recommendations': readiness.recommendations
            },
            'source_availability': source_availability,
            'improvement_potential': self._assess_improvement_potential(availability, readiness),
            'generated_at': timezone.now().isoformat()
        }
        
    def _assess_improvement_potential(self, availability, readiness) -> Dict[str, Any]:
        """Assess potential for data quality improvement."""
        potential = {
            'can_improve': False,
            'expected_improvement': {},
            'recommended_actions': []
        }
        
        if availability.quality_level.value in ['insufficient', 'limited', 'fair']:
            potential['can_improve'] = True
            potential['recommended_actions'].append('Fetch additional historical data')
            
            # Estimate improvement potential
            if availability.total_days < 50:
                potential['expected_improvement']['confidence'] = '+30-50%'
                potential['expected_improvement']['indicators'] = '+3-5 indicators'
            elif availability.total_days < 200:
                potential['expected_improvement']['confidence'] = '+20-30%'
                potential['expected_improvement']['indicators'] = '+2-4 indicators'
            else:
                potential['expected_improvement']['confidence'] = '+10-20%'
                potential['expected_improvement']['indicators'] = '+1-2 indicators'
        
        if availability.gap_percentage > 20:
            potential['can_improve'] = True
            potential['recommended_actions'].append('Fill data gaps for better continuity')
            
        if readiness.overall_confidence < 0.7:
            potential['can_improve'] = True
            potential['recommended_actions'].append('Improve data quality for higher confidence scores')
            
        return potential


# Global service instance
enhanced_backfill_service = EnhancedBackfillService()