"""
Data Quality Monitoring Service

Provides comprehensive monitoring of data quality including coverage, freshness,
completeness, and anomaly detection for reliable analysis operations.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from django.core.cache import cache
from django.db.models import Count, Q, Min, Max, Avg
from django.utils import timezone

from Data.models import Stock, StockPrice, DataSectorPrice, DataIndustryPrice

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring service.
    
    Monitors data coverage, freshness, gaps, and anomalies across the system
    to ensure reliable data for financial analysis.
    """
    
    def __init__(self):
        self.cache_prefix = "data_quality"
        self.cache_timeout = 3600  # 1 hour
        
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """
        Run comprehensive data quality check across all data types.
        
        Returns:
            Dict containing quality metrics, issues, and recommendations
        """
        logger.info("Starting comprehensive data quality check")
        
        try:
            # Collect all quality metrics
            stock_quality = self._check_stock_data_quality()
            sector_quality = self._check_sector_data_quality()
            industry_quality = self._check_industry_data_quality()
            freshness_metrics = self._check_data_freshness()
            gap_analysis = self._analyze_data_gaps()
            anomaly_detection = self._detect_data_anomalies()
            coverage_metrics = self._calculate_coverage_metrics()
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality_score({
                'stock_quality': stock_quality,
                'sector_quality': sector_quality, 
                'industry_quality': industry_quality,
                'freshness_metrics': freshness_metrics,
                'gap_analysis': gap_analysis
            })
            
            # Generate recommendations
            recommendations = self._generate_recommendations({
                'stock_quality': stock_quality,
                'sector_quality': sector_quality,
                'industry_quality': industry_quality,
                'gap_analysis': gap_analysis,
                'anomaly_detection': anomaly_detection
            })
            
            result = {
                'timestamp': timezone.now().isoformat(),
                'overall_quality_score': overall_score,
                'stock_data_quality': stock_quality,
                'sector_data_quality': sector_quality,
                'industry_data_quality': industry_quality,
                'data_freshness': freshness_metrics,
                'gap_analysis': gap_analysis,
                'anomaly_detection': anomaly_detection,
                'coverage_metrics': coverage_metrics,
                'recommendations': recommendations,
                'summary': self._generate_summary(overall_score, len(recommendations))
            }
            
            # Cache the result
            cache_key = f"{self.cache_prefix}:comprehensive_check"
            cache.set(cache_key, result, self.cache_timeout)
            
            logger.info(f"Data quality check complete. Overall score: {overall_score:.1f}/10")
            return result
            
        except Exception as e:
            logger.error(f"Error during comprehensive data quality check: {str(e)}")
            return {
                'timestamp': timezone.now().isoformat(),
                'error': str(e),
                'overall_quality_score': 0.0
            }
    
    def _check_stock_data_quality(self) -> Dict[str, Any]:
        """Check quality of stock price data."""
        try:
            total_stocks = Stock.objects.count()
            
            # Check stocks with recent data (last 7 days)
            recent_cutoff = timezone.now().date() - timedelta(days=7)
            stocks_with_recent_data = Stock.objects.filter(
                prices__date__gte=recent_cutoff
            ).distinct().count()
            
            # Check stocks with sufficient historical data (>= 252 days)
            stocks_with_history = Stock.objects.annotate(
                price_count=Count('prices')
            ).filter(price_count__gte=252).count()
            
            # Average data points per stock
            avg_data_points = StockPrice.objects.count() / max(total_stocks, 1)
            
            # Check for gaps in last 30 days
            recent_month = timezone.now().date() - timedelta(days=30)
            stocks_with_gaps = self._count_stocks_with_gaps(recent_month)
            
            freshness_ratio = stocks_with_recent_data / max(total_stocks, 1)
            completeness_ratio = stocks_with_history / max(total_stocks, 1)
            gap_ratio = 1 - (stocks_with_gaps / max(total_stocks, 1))
            
            # Quality score (0-10)
            quality_score = (freshness_ratio * 0.4 + completeness_ratio * 0.4 + gap_ratio * 0.2) * 10
            
            return {
                'total_stocks': total_stocks,
                'stocks_with_recent_data': stocks_with_recent_data,
                'stocks_with_sufficient_history': stocks_with_history,
                'stocks_with_gaps': stocks_with_gaps,
                'average_data_points_per_stock': round(avg_data_points, 1),
                'freshness_ratio': round(freshness_ratio, 3),
                'completeness_ratio': round(completeness_ratio, 3),
                'quality_score': round(quality_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error checking stock data quality: {str(e)}")
            return {'error': str(e), 'quality_score': 0.0}
    
    def _check_sector_data_quality(self) -> Dict[str, Any]:
        """Check quality of sector composite data."""
        try:
            total_sectors = DataSectorPrice.objects.values('sector').distinct().count()
            
            # Check sectors with recent data
            recent_cutoff = timezone.now().date() - timedelta(days=7)
            sectors_with_recent_data = DataSectorPrice.objects.filter(
                date__gte=recent_cutoff
            ).values('sector').distinct().count()
            
            # Average data points per sector
            avg_data_points = DataSectorPrice.objects.count() / max(total_sectors, 1)
            
            # Check coverage
            coverage_ratio = sectors_with_recent_data / max(total_sectors, 1)
            quality_score = coverage_ratio * 10
            
            return {
                'total_sectors': total_sectors,
                'sectors_with_recent_data': sectors_with_recent_data,
                'average_data_points_per_sector': round(avg_data_points, 1),
                'coverage_ratio': round(coverage_ratio, 3),
                'quality_score': round(quality_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error checking sector data quality: {str(e)}")
            return {'error': str(e), 'quality_score': 0.0}
    
    def _check_industry_data_quality(self) -> Dict[str, Any]:
        """Check quality of industry composite data."""
        try:
            total_industries = DataIndustryPrice.objects.values('industry').distinct().count()
            
            # Check industries with recent data
            recent_cutoff = timezone.now().date() - timedelta(days=7)
            industries_with_recent_data = DataIndustryPrice.objects.filter(
                date__gte=recent_cutoff
            ).values('industry').distinct().count()
            
            # Average data points per industry
            avg_data_points = DataIndustryPrice.objects.count() / max(total_industries, 1)
            
            # Check coverage
            coverage_ratio = industries_with_recent_data / max(total_industries, 1)
            quality_score = coverage_ratio * 10
            
            return {
                'total_industries': total_industries,
                'industries_with_recent_data': industries_with_recent_data,
                'average_data_points_per_industry': round(avg_data_points, 1),
                'coverage_ratio': round(coverage_ratio, 3),
                'quality_score': round(quality_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error checking industry data quality: {str(e)}")
            return {'error': str(e), 'quality_score': 0.0}
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check how fresh the data is across different data types."""
        try:
            now = timezone.now().date()
            
            # Stock data freshness
            latest_stock_date = StockPrice.objects.aggregate(
                latest=Max('date')
            )['latest']
            stock_staleness_days = (now - latest_stock_date).days if latest_stock_date else 999
            
            # Sector data freshness
            latest_sector_date = DataSectorPrice.objects.aggregate(
                latest=Max('date')
            )['latest']
            sector_staleness_days = (now - latest_sector_date).days if latest_sector_date else 999
            
            # Industry data freshness
            latest_industry_date = DataIndustryPrice.objects.aggregate(
                latest=Max('date')
            )['latest']
            industry_staleness_days = (now - latest_industry_date).days if latest_industry_date else 999
            
            # Overall freshness score (0-10, where 0 days = 10, 7+ days = 0)
            def freshness_score(days):
                return max(0, 10 - (days * 1.4))
            
            stock_freshness = freshness_score(stock_staleness_days)
            sector_freshness = freshness_score(sector_staleness_days)  
            industry_freshness = freshness_score(industry_staleness_days)
            
            overall_freshness = (stock_freshness + sector_freshness + industry_freshness) / 3
            
            return {
                'stock_data_staleness_days': stock_staleness_days,
                'sector_data_staleness_days': sector_staleness_days,
                'industry_data_staleness_days': industry_staleness_days,
                'stock_freshness_score': round(stock_freshness, 1),
                'sector_freshness_score': round(sector_freshness, 1),
                'industry_freshness_score': round(industry_freshness, 1),
                'overall_freshness_score': round(overall_freshness, 1),
                'latest_stock_date': latest_stock_date.isoformat() if latest_stock_date else None,
                'latest_sector_date': latest_sector_date.isoformat() if latest_sector_date else None,
                'latest_industry_date': latest_industry_date.isoformat() if latest_industry_date else None
            }
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {str(e)}")
            return {'error': str(e), 'overall_freshness_score': 0.0}
    
    def _analyze_data_gaps(self) -> Dict[str, Any]:
        """Analyze data gaps across the system."""
        try:
            # Get stocks with significant gaps (>10% missing in last 3 months)
            three_months_ago = timezone.now().date() - timedelta(days=90)
            expected_trading_days = 65  # ~3 months
            
            stocks_with_gaps = []
            total_stocks_checked = 0
            
            # Sample stocks for gap analysis (limit for performance)
            sample_stocks = Stock.objects.all()[:100]  # Check top 100 stocks
            
            for stock in sample_stocks:
                total_stocks_checked += 1
                actual_days = StockPrice.objects.filter(
                    stock=stock,
                    date__gte=three_months_ago
                ).count()
                
                gap_percentage = max(0, (expected_trading_days - actual_days) / expected_trading_days * 100)
                
                if gap_percentage > 10:  # >10% gaps
                    stocks_with_gaps.append({
                        'symbol': stock.symbol,
                        'expected_days': expected_trading_days,
                        'actual_days': actual_days,
                        'gap_percentage': round(gap_percentage, 1)
                    })
            
            # Gap severity analysis
            high_gap_stocks = [s for s in stocks_with_gaps if s['gap_percentage'] > 30]
            medium_gap_stocks = [s for s in stocks_with_gaps if 10 < s['gap_percentage'] <= 30]
            
            # Calculate gap score (0-10, lower gaps = higher score)
            total_gap_percentage = sum(s['gap_percentage'] for s in stocks_with_gaps)
            avg_gap_percentage = total_gap_percentage / max(total_stocks_checked, 1)
            gap_score = max(0, 10 - (avg_gap_percentage / 5))  # Scale: 0% gaps = 10, 50% gaps = 0
            
            return {
                'total_stocks_analyzed': total_stocks_checked,
                'stocks_with_significant_gaps': len(stocks_with_gaps),
                'high_severity_gaps': len(high_gap_stocks),
                'medium_severity_gaps': len(medium_gap_stocks),
                'average_gap_percentage': round(avg_gap_percentage, 2),
                'gap_score': round(gap_score, 1),
                'top_gap_stocks': stocks_with_gaps[:10]  # Show worst 10
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data gaps: {str(e)}")
            return {'error': str(e), 'gap_score': 0.0}
    
    def _detect_data_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in the data that might indicate quality issues."""
        try:
            anomalies = []
            
            # Check for extreme price movements (>50% daily change)
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT symbol, date, open, close, 
                           ABS((close - open) / open * 100) as price_change
                    FROM data_stockprice 
                    WHERE date >= %s AND ABS((close - open) / open * 100) > 50
                    ORDER BY price_change DESC
                """, [timezone.now().date() - timedelta(days=30)])
                
                extreme_movements = cursor.fetchall()
                for row in extreme_movements:
                    symbol, date, open_price, close_price, change = row
                    anomalies.append({
                        'type': 'extreme_price_movement',
                        'symbol': symbol,
                        'date': date,
                        'details': f'Price changed {change:.1f}% from {open_price} to {close_price}'
                    })
            
            # Check for zero volume days
            zero_volume = StockPrice.objects.filter(
                date__gte=timezone.now().date() - timedelta(days=30),
                volume=0
            )[:10]
            
            for zv in zero_volume:
                anomalies.append({
                    'type': 'zero_volume',
                    'symbol': zv.stock.symbol,
                    'date': zv.date.isoformat(),
                    'severity': 'medium'
                })
            
            # Check for duplicate entries on same date
            duplicates = StockPrice.objects.values('stock', 'date').annotate(
                count=Count('id')
            ).filter(count__gt=1)[:10]
            
            for dup in duplicates:
                stock = Stock.objects.get(id=dup['stock'])
                anomalies.append({
                    'type': 'duplicate_entries',
                    'symbol': stock.symbol,
                    'date': dup['date'].isoformat(),
                    'count': dup['count'],
                    'severity': 'high'
                })
            
            # Anomaly score (0-10, fewer anomalies = higher score)
            anomaly_count = len(anomalies)
            anomaly_score = max(0, 10 - (anomaly_count * 0.5))
            
            return {
                'total_anomalies_detected': anomaly_count,
                'anomalies': anomalies,
                'anomaly_score': round(anomaly_score, 1),
                'anomaly_types': {
                    'extreme_price_movements': len([a for a in anomalies if a['type'] == 'extreme_price_movement']),
                    'zero_volume_days': len([a for a in anomalies if a['type'] == 'zero_volume']),
                    'duplicate_entries': len([a for a in anomalies if a['type'] == 'duplicate_entries'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting data anomalies: {str(e)}")
            return {'error': str(e), 'anomaly_score': 10.0}
    
    def _calculate_coverage_metrics(self) -> Dict[str, Any]:
        """Calculate overall data coverage metrics."""
        try:
            total_possible_symbols = 5000  # Rough estimate of tradeable stocks
            current_symbols = Stock.objects.count()
            
            # Calculate coverage percentage
            symbol_coverage = (current_symbols / total_possible_symbols) * 100
            
            # Check sector/industry coverage
            total_sectors = DataSectorPrice.objects.values('sector').distinct().count()
            total_industries = DataIndustryPrice.objects.values('industry').distinct().count()
            
            return {
                'symbol_coverage_percentage': round(symbol_coverage, 1),
                'total_symbols_tracked': current_symbols,
                'estimated_total_symbols': total_possible_symbols,
                'sectors_covered': total_sectors,
                'industries_covered': total_industries,
                'coverage_score': round(min(10, symbol_coverage / 10), 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating coverage metrics: {str(e)}")
            return {'error': str(e), 'coverage_score': 0.0}
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score from individual metrics."""
        try:
            weights = {
                'stock_quality': 0.3,
                'sector_quality': 0.15,
                'industry_quality': 0.15,
                'freshness_metrics': 0.25,
                'gap_analysis': 0.15
            }
            
            total_score = 0
            total_weight = 0
            
            for metric_name, weight in weights.items():
                if metric_name in metrics and 'quality_score' in metrics[metric_name]:
                    score = metrics[metric_name]['quality_score']
                    total_score += score * weight
                    total_weight += weight
                elif metric_name == 'freshness_metrics' and 'overall_freshness_score' in metrics[metric_name]:
                    score = metrics[metric_name]['overall_freshness_score']
                    total_score += score * weight
                    total_weight += weight
                elif metric_name == 'gap_analysis' and 'gap_score' in metrics[metric_name]:
                    score = metrics[metric_name]['gap_score']
                    total_score += score * weight
                    total_weight += weight
            
            return total_score / max(total_weight, 1) if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        try:
            # Check stock data quality
            if 'stock_quality' in metrics:
                stock_metrics = metrics['stock_quality']
                if stock_metrics.get('quality_score', 0) < 7.0:
                    recommendations.append({
                        'type': 'stock_data_quality',
                        'severity': 'high',
                        'message': f"Stock data quality is below acceptable threshold ({stock_metrics.get('quality_score', 0):.1f}/10)",
                        'action': 'Review and backfill missing stock price data'
                    })
                
                if stock_metrics.get('freshness_ratio', 0) < 0.8:
                    recommendations.append({
                        'type': 'data_freshness',
                        'severity': 'medium',
                        'message': f"Only {stock_metrics.get('stocks_with_recent_data', 0)} of {stock_metrics.get('total_stocks', 0)} stocks have recent data",
                        'action': 'Increase frequency of data synchronization'
                    })
            
            # Check gap analysis
            if 'gap_analysis' in metrics:
                gap_metrics = metrics['gap_analysis']
                if gap_metrics.get('gap_score', 10) < 6.0:
                    recommendations.append({
                        'type': 'data_gaps',
                        'severity': 'high',
                        'message': f"Significant data gaps detected (avg: {gap_metrics.get('average_gap_percentage', 0):.1f}%)",
                        'action': 'Run comprehensive backfill for stocks with >10% gaps'
                    })
            
            # Check freshness
            if 'freshness_metrics' in metrics:
                freshness = metrics['freshness_metrics']
                if freshness.get('overall_freshness_score', 0) < 6.0:
                    recommendations.append({
                        'type': 'data_staleness',
                        'severity': 'medium',
                        'message': f"Data freshness is concerning ({freshness.get('overall_freshness_score', 0):.1f}/10)",
                        'action': 'Investigate data synchronization processes'
                    })
            
            # Check anomalies
            if 'anomaly_detection' in metrics:
                anomalies = metrics['anomaly_detection']
                if anomalies.get('total_anomalies_detected', 0) > 5:
                    recommendations.append({
                        'type': 'data_anomalies',
                        'severity': 'high',
                        'message': f"Multiple data anomalies detected ({anomalies.get('total_anomalies_detected', 0)} issues)",
                        'action': 'Review and clean anomalous data entries'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [{'type': 'error', 'severity': 'high', 'message': f'Error generating recommendations: {str(e)}', 'action': 'Review monitoring system'}]
    
    def _generate_summary(self, overall_score: float, recommendation_count: int) -> Dict[str, Any]:
        """Generate executive summary of data quality status."""
        if overall_score >= 8.0:
            status = 'excellent'
            color = 'green'
        elif overall_score >= 6.0:
            status = 'good'
            color = 'yellow'
        elif overall_score >= 4.0:
            status = 'fair'
            color = 'orange'
        else:
            status = 'poor'
            color = 'red'
        
        return {
            'overall_status': status,
            'overall_score': overall_score,
            'status_color': color,
            'issues_found': recommendation_count,
            'message': f"Data quality is {status} ({overall_score:.1f}/10) with {recommendation_count} issues requiring attention."
        }
    
    def _count_stocks_with_gaps(self, start_date: date) -> int:
        """Count stocks with significant data gaps in the given period."""
        try:
            expected_days = 21  # ~1 month trading days
            stocks_with_gaps = 0
            
            for stock in Stock.objects.all()[:50]:  # Sample for performance
                actual_days = StockPrice.objects.filter(
                    stock=stock,
                    date__gte=start_date
                ).count()
                
                if actual_days < expected_days * 0.8:  # Missing >20% of expected days
                    stocks_with_gaps += 1
            
            return stocks_with_gaps
            
        except Exception as e:
            logger.error(f"Error counting stocks with gaps: {str(e)}")
            return 0
    
    def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get cached quality data for dashboard display."""
        cache_key = f"{self.cache_prefix}:comprehensive_check"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        else:
            # Return basic metrics if full check not cached
            return {
                'timestamp': timezone.now().isoformat(),
                'overall_quality_score': 'Not Available',
                'message': 'Full quality check not performed recently. Run comprehensive check.',
                'recommendations': [
                    {
                        'type': 'monitoring',
                        'severity': 'medium',
                        'message': 'Quality monitoring data is not available',
                        'action': 'Run comprehensive data quality check'
                    }
                ]
            }


# Global instance
data_quality_monitor = DataQualityMonitor()