"""
Enhanced Technical Analysis Engine

Integrates adaptive indicators, confidence scoring, and multi-source data support.
Provides robust analysis with graceful degradation for limited data scenarios.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.engine.adaptive_indicators import adaptive_indicator_engine, IndicatorResult
from Data.services.data_verification import data_verification_service, AnalysisReadiness
from Data.repo.price_reader import PriceReader
from Data.services.enhanced_backfill import enhanced_backfill_service

logger = logging.getLogger(__name__)


class EnhancedTechnicalAnalysisEngine:
    """
    Enhanced TA engine with adaptive indicators and confidence scoring.
    """
    
    def __init__(self):
        self.original_engine = TechnicalAnalysisEngine()
        self.price_reader = PriceReader()
        self.confidence_threshold = 0.6
        
        # Enhanced indicator weights with confidence factors
        self.enhanced_weights = {
            'obv20': {'base_weight': 0.08, 'confidence_factor': 1.0},
            'volsurge': {'base_weight': 0.06, 'confidence_factor': 1.0},
            'srcontext': {'base_weight': 0.12, 'confidence_factor': 0.8},
            'bbwidth20': {'base_weight': 0.08, 'confidence_factor': 1.0},
            'sma50vs200': {'base_weight': 0.10, 'confidence_factor': 0.7},
            'pricevs50': {'base_weight': 0.08, 'confidence_factor': 0.8},
            'macd12269': {'base_weight': 0.08, 'confidence_factor': 0.7},
            'rsi14': {'base_weight': 0.08, 'confidence_factor': 1.0},
            'candlerev': {'base_weight': 0.08, 'confidence_factor': 0.9},
            'bbpos20': {'base_weight': 0.06, 'confidence_factor': 1.0},
            'rel1y': {'base_weight': 0.08, 'confidence_factor': 0.6},
            'rel2y': {'base_weight': 0.06, 'confidence_factor': 0.5},
            'prediction': {'base_weight': 0.08, 'confidence_factor': 0.8},
            'sentiment': {'base_weight': 0.06, 'confidence_factor': 1.0}
        }
        
    def analyze_with_confidence(
        self,
        symbol: str,
        horizon: str = 'blend',
        include_sentiment: bool = True,
        skip_backfill: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced analysis with confidence scoring and adaptive indicators.
        
        Args:
            symbol: Stock symbol to analyze
            horizon: Analysis horizon
            include_sentiment: Whether to include sentiment analysis
            skip_backfill: Skip data backfilling
            
        Returns:
            Enhanced analysis results with confidence metrics
        """
        start_time = timezone.now()
        logger.info(f"[ENHANCED TA ENGINE] Starting enhanced analysis for {symbol}")
        
        # Pre-analysis data assessment
        readiness = data_verification_service.assess_analysis_readiness(symbol)
        
        logger.info(f"Analysis readiness for {symbol}: {readiness.overall_confidence:.2f} confidence")
        logger.info(f"Available indicators: {len(readiness.available_indicators)}")
        logger.info(f"Missing indicators: {len(readiness.missing_indicators)}")
        logger.info(f"Adaptable indicators: {len(readiness.adaptable_indicators)}")
        
        # Enhanced data backfill if needed and not skipped
        if not skip_backfill and readiness.overall_confidence < self.confidence_threshold:
            logger.info(f"Confidence below threshold ({self.confidence_threshold}), attempting enhanced backfill")
            backfill_result = enhanced_backfill_service.enhanced_backfill_concurrent(
                symbol=symbol,
                required_years=2,
                min_acceptable_days=20
            )
            
            if backfill_result['success']:
                logger.info(f"Enhanced backfill improved data quality for {symbol}")
                # Re-assess after backfill
                readiness = data_verification_service.assess_analysis_readiness(symbol)
            else:
                logger.warning(f"Enhanced backfill failed for {symbol}: {backfill_result.get('errors', [])}")
        
        # Get price data  
        try:
            # Default to 2 years of data
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=365 * 2)
            
            price_data = self.price_reader.get_stock_prices(symbol, start_date, end_date)
            
            # Auto-sync fallback if no data found - same as original ta_engine
            if not price_data:
                logger.info(f"No existing data for {symbol}, attempting auto-sync")
                if self._auto_sync_stock_data(symbol):
                    # Retry after sync with multiple attempts
                    for attempt in range(3):
                        try:
                            if attempt > 0:
                                import time
                                time.sleep(2)  # Wait 2 seconds between retries
                            
                            price_data = self.price_reader.get_stock_prices(symbol, start_date, end_date)
                            if price_data:
                                logger.info(f"Successfully retrieved data for {symbol} after auto-sync (attempt {attempt + 1})")
                                break
                        except Exception as retry_e:
                            logger.warning(f"Retry attempt {attempt + 1} failed for {symbol}: {str(retry_e)}")
                    
                    if not price_data:
                        logger.error(f"Auto-sync completed but no data retrieved for {symbol}")
                else:
                    logger.warning(f"Auto-sync failed for {symbol}")
            
            # Get sector and industry data if available
            sector_data = None
            industry_data = None
            try:
                from Data.models import Stock
                stock = Stock.objects.get(symbol=symbol)
                if stock.sector:
                    # Get sector key from sector name
                    from Data.models import DataSector
                    sector_obj = DataSector.objects.get(sectorName=stock.sector)
                    sector_data = self.price_reader.get_sector_prices(sector_obj.sectorKey, start_date, end_date)
                    
                if stock.industry:
                    # Get industry key from industry name  
                    from Data.models import DataIndustry
                    try:
                        industry_obj = DataIndustry.objects.get(industryName=stock.industry)
                        industry_data = self.price_reader.get_industry_prices(industry_obj.industryKey, start_date, end_date)
                    except DataIndustry.DoesNotExist:
                        logger.warning(f"Industry '{stock.industry}' not found for {symbol}")
            except Exception as e:
                logger.warning(f"Could not retrieve sector/industry data for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to retrieve price data for {symbol}: {str(e)}")
            return self._create_error_result(symbol, f"Data retrieval failed: {str(e)}")
        
        if not price_data:
            # Check if this was due to a delisted stock detected during auto-sync
            is_delisted = hasattr(self.original_engine, "_delisted_stock") and self.original_engine._delisted_stock
            if is_delisted:
                error_msg = f"Stock {symbol} appears to be delisted and is no longer available for trading"
            else:
                error_msg = "No price data available"
            return self._create_error_result(symbol, error_msg)
        
        # Convert to DataFrame for analysis
        df = self._convert_to_dataframe(price_data)
        
        if len(df) < 5:
            return self._create_error_result(symbol, f"Insufficient data: only {len(df)} days available")
        
        # Enhanced indicator calculation
        indicator_results = self._calculate_enhanced_indicators(
            symbol, df, sector_data, industry_data, include_sentiment
        )
        
        # Calculate adaptive composite score
        composite_result = self._calculate_adaptive_composite_score(
            indicator_results, readiness
        )
        
        # Store results
        try:
            self._store_enhanced_results(symbol, composite_result, indicator_results, readiness)
        except Exception as e:
            logger.error(f"Failed to store results for {symbol}: {str(e)}")
        
        end_time = timezone.now()
        analysis_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"[ENHANCED TA ENGINE] Enhanced analysis complete for {symbol}: "
                   f"{composite_result['final_score']}/10 in {analysis_duration:.2f}s")
        
        return {
            'symbol': symbol,
            'analysis_timestamp': end_time,
            'final_score': composite_result['final_score'],
            'confidence_score': composite_result['confidence_score'],
            'raw_composite': composite_result['raw_composite'],
            'data_quality': {
                'total_days': len(df),
                'quality_level': readiness.data_availability.quality_level.value,
                'overall_confidence': readiness.overall_confidence
            },
            'indicators': {
                name: {
                    'score': result.value,
                    'confidence': result.confidence,
                    'adapted': result.adapted,
                    'fallback_used': result.fallback_used,
                    'data_points_used': result.data_points_used
                }
                for name, result in indicator_results.items()
            },
            'recommendations': readiness.recommendations,
            'analysis_duration': analysis_duration,
            'enhancement_applied': True
        }
        
    def _calculate_enhanced_indicators(
        self,
        symbol: str,
        df: pd.DataFrame,
        sector_data: Optional[List] = None,
        industry_data: Optional[List] = None,
        include_sentiment: bool = True
    ) -> Dict[str, IndicatorResult]:
        """Calculate indicators using adaptive engine."""
        logger.info(f"Calculating enhanced indicators for {symbol}")
        
        # Convert sector/industry data to DataFrames if available
        sector_df = self._convert_composite_to_dataframe(sector_data) if sector_data else None
        industry_df = self._convert_composite_to_dataframe(industry_data) if industry_data else None
        
        indicator_results = {}
        
        # List of all indicators
        all_indicators = [
            'obv20', 'volsurge', 'srcontext', 'bbwidth20', 'sma50vs200',
            'pricevs50', 'macd12269', 'rsi14', 'candlerev', 'bbpos20',
            'rel1y', 'rel2y', 'prediction', 'sentiment'
        ]
        
        # Calculate indicators with parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_indicator = {}
            
            for indicator_name in all_indicators:
                if indicator_name == 'sentiment' and not include_sentiment:
                    continue
                    
                if indicator_name in ['sentiment', 'prediction']:
                    # These require special handling
                    future = executor.submit(
                        self._calculate_special_indicator, 
                        indicator_name, symbol, df
                    )
                else:
                    # Standard adaptive indicators
                    future = executor.submit(
                        adaptive_indicator_engine.calculate_adaptive_indicator,
                        indicator_name, df, sector_df, industry_df
                    )
                
                future_to_indicator[future] = indicator_name
            
            # Collect results
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result()
                    indicator_results[indicator_name] = result
                    logger.debug(f"  {indicator_name}: {result.value:.3f} (confidence: {result.confidence:.2f})")
                except Exception as e:
                    logger.error(f"Error calculating {indicator_name}: {str(e)}")
                    # Create fallback result
                    indicator_results[indicator_name] = IndicatorResult(
                        value=0.5,
                        confidence=0.0,
                        adapted=False,
                        fallback_used=True,
                        data_points_used=len(df),
                        original_name=indicator_name
                    )
        
        return indicator_results
        
    def _calculate_special_indicator(
        self, 
        indicator_name: str, 
        symbol: str, 
        df: pd.DataFrame
    ) -> IndicatorResult:
        """Calculate special indicators (sentiment, prediction)."""
        if indicator_name == 'sentiment':
            try:
                # Use original engine's sentiment calculation
                sentiment_result = self.original_engine._calculate_sentiment_score(symbol)
                if sentiment_result and 'sentiment' in sentiment_result:
                    normalized_sentiment = (sentiment_result['sentiment'] + 1) / 2  # Convert -1,1 to 0,1
                    return IndicatorResult(
                        value=normalized_sentiment,
                        confidence=0.8,
                        adapted=False,
                        fallback_used=False,
                        data_points_used=sentiment_result.get('newsCount', 0),
                        original_name='sentiment'
                    )
            except Exception as e:
                logger.warning(f"Sentiment calculation failed: {str(e)}")
                
            return IndicatorResult(
                value=0.5,
                confidence=0.0,
                adapted=False,
                fallback_used=True,
                data_points_used=0,
                original_name='sentiment'
            )
            
        elif indicator_name == 'prediction':
            try:
                # Use original engine's prediction calculation
                prediction_result = self.original_engine._calculate_prediction_score(symbol)
                if prediction_result and 'prediction' in prediction_result:
                    return IndicatorResult(
                        value=prediction_result['prediction'],
                        confidence=0.7,
                        adapted=False,
                        fallback_used=False,
                        data_points_used=len(df),
                        original_name='prediction'
                    )
            except Exception as e:
                logger.warning(f"Prediction calculation failed: {str(e)}")
                
            return IndicatorResult(
                value=0.5,
                confidence=0.3,
                adapted=False,
                fallback_used=True,
                data_points_used=len(df),
                original_name='prediction'
            )
            
        return IndicatorResult(
            value=0.5,
            confidence=0.0,
            adapted=False,
            fallback_used=True,
            data_points_used=len(df),
            original_name=indicator_name
        )
        
    def _calculate_adaptive_composite_score(
        self,
        indicator_results: Dict[str, IndicatorResult],
        readiness: AnalysisReadiness
    ) -> Dict[str, Any]:
        """Calculate adaptive composite score with confidence weighting."""
        logger.info("Calculating adaptive composite score")
        
        total_weight = 0
        weighted_sum = 0
        confidence_weighted_sum = 0
        total_confidence_weight = 0
        
        indicator_scores = {}
        
        for indicator_name, result in indicator_results.items():
            if indicator_name not in self.enhanced_weights:
                continue
                
            weight_info = self.enhanced_weights[indicator_name]
            base_weight = weight_info['base_weight']
            confidence_factor = weight_info['confidence_factor']
            
            # Adjust weight based on result confidence
            adjusted_weight = base_weight * (result.confidence * confidence_factor)
            
            # Additional weight reduction for adapted/fallback indicators
            if result.adapted:
                adjusted_weight *= 0.8
            if result.fallback_used:
                adjusted_weight *= 0.6
                
            total_weight += adjusted_weight
            weighted_sum += result.value * adjusted_weight
            
            # Confidence-weighted scoring
            confidence_weight = result.confidence * base_weight
            confidence_weighted_sum += result.value * confidence_weight
            total_confidence_weight += confidence_weight
            
            indicator_scores[indicator_name] = {
                'raw_value': result.value,
                'weight': adjusted_weight,
                'confidence': result.confidence,
                'contribution': result.value * adjusted_weight
            }
        
        # Calculate composite scores
        raw_composite = weighted_sum / total_weight if total_weight > 0 else 0.5
        confidence_composite = confidence_weighted_sum / total_confidence_weight if total_confidence_weight > 0 else 0.5
        
        # Blend the two approaches
        final_composite = (raw_composite * 0.7) + (confidence_composite * 0.3)
        
        # Overall confidence score
        avg_confidence = sum(r.confidence for r in indicator_results.values()) / len(indicator_results)
        overall_confidence = (avg_confidence + readiness.overall_confidence) / 2
        
        # Scale to 0-10
        final_score = int(round(final_composite * 10))
        
        return {
            'raw_composite': final_composite,
            'confidence_score': overall_confidence,
            'final_score': final_score,
            'indicator_scores': indicator_scores,
            'total_weight': total_weight,
            'indicators_used': len(indicator_results),
            'avg_indicator_confidence': avg_confidence
        }
        
    def _convert_to_dataframe(self, price_data: List) -> pd.DataFrame:
        """Convert price data to DataFrame."""
        if not price_data:
            return pd.DataFrame()
            
        data = []
        for price in price_data:
            data.append({
                'date': price.date,
                'open': float(price.open),
                'high': float(price.high),
                'low': float(price.low),
                'close': float(price.close),
                'volume': price.volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
        
    def _convert_composite_to_dataframe(self, composite_data: List) -> pd.DataFrame:
        """Convert composite data to DataFrame."""
        if not composite_data:
            return pd.DataFrame()
            
        data = []
        for item in composite_data:
            # Composite data has different structure than price data
            close_price = float(item.close_index)
            data.append({
                'date': item.date,
                'open': close_price,  # Use close_index as approximation for all OHLC
                'high': close_price,
                'low': close_price,
                'close': close_price,
                'volume': item.volume_agg if hasattr(item, 'volume_agg') else 0
            })
            
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
        
    def _store_enhanced_results(
        self,
        symbol: str,
        composite_result: Dict[str, Any],
        indicator_results: Dict[str, IndicatorResult],
        readiness: AnalysisReadiness
    ):
        """Store enhanced analysis results."""
        try:
            # Use original engine's storage mechanism
            metadata = {
                'confidence_score': composite_result['confidence_score'],
                'data_quality': readiness.data_availability.quality_level.value,
                'indicators_used': composite_result['indicators_used'],
                'enhancement_applied': True
            }
            
            # Store using original engine
            self.original_engine._store_results(
                symbol,
                composite_result['final_score'],
                composite_result['raw_composite'],
                indicator_results,
                metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to store enhanced results: {str(e)}")
            
    def _auto_sync_stock_data(self, symbol: str) -> bool:
        """
        Automatically sync stock data when it's missing from the database.
        Delegates to the original engine's auto-sync method.
        """
        return self.original_engine._auto_sync_stock_data(symbol)
            
    def _create_error_result(self, symbol: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'symbol': symbol,
            'analysis_timestamp': timezone.now(),
            'final_score': 0,
            'confidence_score': 0.0,
            'error': error_message,
            'enhancement_applied': False,
            'success': False
        }
    
    def analyze_stock(self, symbol: str, user=None, logger_instance=None):
        """
        Backward compatibility method that delegates to the original engine.
        This ensures views that call analyze_stock() continue to work.
        """
        return self.original_engine.analyze_stock(
            symbol=symbol,
            user=user,
            logger_instance=logger_instance
        )


# Global service instance
enhanced_ta_engine = EnhancedTechnicalAnalysisEngine()