"""
Sector Data Service for Universal LSTM Infrastructure
Handles sector/industry data collection, validation, and storage for universal training.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from django.db import transaction
from django.core.cache import cache

from Data.models import Stock, DataSector, DataIndustry
from Data.repo.price_reader import PriceReader
from Data.services.yahoo_finance import yahoo_finance_service
from Data.services.synchronizer import data_synchronizer
from Analytics.ml.sector_mappings import get_sector_mapper, TRAINING_STOCK_UNIVERSE

logger = logging.getLogger(__name__)


class SectorDataService:
    """
    Data service for sector-based operations supporting Universal LSTM training.
    Handles data collection, sector classification, and training data preparation.
    """
    
    def __init__(self):
        """Initialize the sector data service."""
        self.price_reader = PriceReader()
        self.sector_mapper = get_sector_mapper()
        self.cache_timeout = 3600  # 1 hour cache
        
    def ensure_training_stocks_in_database(self) -> Dict[str, Any]:
        """
        Ensure all training universe stocks are in the database with proper sector classification.
        
        Returns:
            Dictionary with operation results and statistics
        """
        logger.info("Ensuring training universe stocks are in database...")
        
        results = {
            'existing_stocks': 0,
            'new_stocks': 0,
            'updated_stocks': 0,
            'failed_stocks': 0,
            'sector_distribution': {},
            'total_stocks': 0
        }
        
        all_training_stocks = self.sector_mapper.get_all_training_stocks()
        results['total_stocks'] = len(all_training_stocks)
        
        for symbol in all_training_stocks:
            try:
                with transaction.atomic():
                    # Get sector classification
                    sector_name = self.sector_mapper.classify_stock_sector(symbol)
                    sector_id = self.sector_mapper.get_sector_id(sector_name or 'Unknown')
                    
                    # Ensure sector exists in database
                    sector_obj = self._ensure_sector_in_database(sector_name or 'Unknown')
                    
                    # Get or create stock
                    stock, created = Stock.objects.get_or_create(
                        symbol=symbol,
                        defaults={
                            'short_name': symbol,
                            'long_name': f'{symbol} Corporation',
                            'sector': sector_name or 'Unknown',
                            'sector_id': sector_obj,
                            'is_active': True,
                            'data_source': 'yahoo'
                        }
                    )
                    
                    updated = False
                    if not created:
                        # Update existing stock if sector changed
                        if stock.sector_id != sector_obj:
                            stock.sector = sector_name or 'Unknown'
                            stock.sector_id = sector_obj
                            stock.save()
                            updated = True
                            results['updated_stocks'] += 1
                        else:
                            results['existing_stocks'] += 1
                    else:
                        results['new_stocks'] += 1
                    
                    # Track sector distribution
                    sector_key = sector_name or 'Unknown'
                    results['sector_distribution'][sector_key] = results['sector_distribution'].get(sector_key, 0) + 1
                    
                    logger.debug(f"Stock {symbol} -> {sector_name} ({'created' if created else 'updated' if updated else 'exists'})")
                    
            except Exception as e:
                logger.error(f"Failed to process stock {symbol}: {str(e)}")
                results['failed_stocks'] += 1
        
        logger.info(f"Training stocks database sync complete: "
                   f"{results['new_stocks']} new, {results['updated_stocks']} updated, "
                   f"{results['existing_stocks']} existing, {results['failed_stocks']} failed")
        
        return results
    
    def _ensure_sector_in_database(self, sector_name: str) -> DataSector:
        """
        Ensure sector exists in database.
        
        Args:
            sector_name: Name of the sector
            
        Returns:
            DataSector instance
        """
        # Normalize sector key
        sector_key = sector_name.lower().replace(' ', '_').replace('-', '_')
        
        sector, created = DataSector.objects.get_or_create(
            sectorKey=sector_key,
            defaults={
                'sectorName': sector_name,
                'isActive': True,
                'data_source': 'yahoo'
            }
        )
        
        if created:
            logger.debug(f"Created sector: {sector_name}")
        
        return sector
    
    def collect_sector_training_data(
        self,
        years_back: int = 3,
        min_data_points: int = 500
    ) -> Dict[str, Any]:
        """
        Collect training data for all sectors with validation.
        
        Args:
            years_back: Years of historical data to collect
            min_data_points: Minimum data points required per stock
            
        Returns:
            Dictionary with collected data and statistics
        """
        logger.info(f"Collecting {years_back} years of training data for all sectors...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        collection_results = {
            'successful_stocks': {},
            'failed_stocks': {},
            'sector_coverage': {},
            'total_data_points': 0,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        
        # Process each sector
        for sector_name, stock_symbols in TRAINING_STOCK_UNIVERSE.items():
            logger.info(f"Collecting data for {sector_name} sector ({len(stock_symbols)} stocks)...")
            
            sector_results = {
                'successful': 0,
                'failed': 0,
                'total_data_points': 0,
                'stocks': {}
            }
            
            for symbol in stock_symbols:
                try:
                    # Collect price data
                    price_data = self.price_reader.get_stock_prices(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if price_data and len(price_data) >= min_data_points:
                        sector_results['successful'] += 1
                        sector_results['total_data_points'] += len(price_data)
                        sector_results['stocks'][symbol] = {
                            'data_points': len(price_data),
                            'start_date': min(p.date for p in price_data).isoformat(),
                            'end_date': max(p.date for p in price_data).isoformat()
                        }
                        collection_results['successful_stocks'][symbol] = sector_name
                        logger.debug(f"  {symbol}: {len(price_data)} data points")
                    else:
                        sector_results['failed'] += 1
                        failure_reason = 'no_data' if not price_data else 'insufficient_data'
                        sector_results['stocks'][symbol] = {
                            'error': failure_reason,
                            'data_points': len(price_data) if price_data else 0
                        }
                        collection_results['failed_stocks'][symbol] = failure_reason
                        logger.warning(f"  {symbol}: FAILED ({failure_reason})")
                        
                except Exception as e:
                    sector_results['failed'] += 1
                    sector_results['stocks'][symbol] = {'error': str(e)}
                    collection_results['failed_stocks'][symbol] = str(e)
                    logger.error(f"  {symbol}: ERROR - {str(e)}")
            
            collection_results['sector_coverage'][sector_name] = sector_results
            collection_results['total_data_points'] += sector_results['total_data_points']
            
            logger.info(f"  {sector_name} complete: {sector_results['successful']}/{len(stock_symbols)} stocks, "
                       f"{sector_results['total_data_points']} total data points")
        
        # Generate summary statistics
        total_successful = len(collection_results['successful_stocks'])
        total_failed = len(collection_results['failed_stocks'])
        success_rate = (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
        
        logger.info(f"Data collection complete: {total_successful}/{total_successful + total_failed} stocks "
                   f"({success_rate:.1f}% success rate), {collection_results['total_data_points']} total data points")
        
        return collection_results
    
    def validate_sector_balance(self, min_stocks_per_sector: int = 5) -> Dict[str, Any]:
        """
        Validate that each sector has sufficient stocks for training.
        
        Args:
            min_stocks_per_sector: Minimum stocks required per sector
            
        Returns:
            Validation results with recommendations
        """
        logger.info("Validating sector balance for universal training...")
        
        # Get current database stocks by sector
        sector_counts = {}
        training_stocks = self.sector_mapper.get_all_training_stocks()
        
        for symbol in training_stocks:
            sector_name = self.sector_mapper.classify_stock_sector(symbol)
            sector_key = sector_name or 'Unknown'
            sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
        
        # Validate balance
        validation_results = {
            'balanced': True,
            'sector_counts': sector_counts,
            'underrepresented_sectors': [],
            'recommendations': [],
            'total_stocks': sum(sector_counts.values()),
            'min_threshold': min_stocks_per_sector
        }
        
        for sector_name, count in sector_counts.items():
            if count < min_stocks_per_sector:
                validation_results['balanced'] = False
                validation_results['underrepresented_sectors'].append({
                    'sector': sector_name,
                    'current_count': count,
                    'needed': min_stocks_per_sector - count
                })
        
        # Generate recommendations
        if not validation_results['balanced']:
            validation_results['recommendations'].append(
                "Add more stocks to underrepresented sectors for better balance"
            )
            validation_results['recommendations'].append(
                "Consider sector weighting during training to compensate"
            )
        else:
            validation_results['recommendations'].append(
                "Sector balance is adequate for universal training"
            )
        
        # Calculate balance metrics
        if sector_counts:
            avg_stocks_per_sector = sum(sector_counts.values()) / len(sector_counts)
            max_sector_count = max(sector_counts.values())
            min_sector_count = min(sector_counts.values())
            balance_ratio = min_sector_count / max_sector_count if max_sector_count > 0 else 0
            
            validation_results.update({
                'avg_stocks_per_sector': avg_stocks_per_sector,
                'max_sector_count': max_sector_count,
                'min_sector_count': min_sector_count,
                'balance_ratio': balance_ratio  # Higher is better (1.0 = perfect balance)
            })
        
        logger.info(f"Sector balance validation complete: {'BALANCED' if validation_results['balanced'] else 'UNBALANCED'}")
        
        return validation_results
    
    def sync_missing_stock_data(
        self,
        symbols: List[str],
        period: str = '2y'
    ) -> Dict[str, Any]:
        """
        Sync missing stock data from Yahoo Finance for training.
        
        Args:
            symbols: List of stock symbols to sync
            period: Period to sync ('1y', '2y', '5y')
            
        Returns:
            Sync results and statistics
        """
        logger.info(f"Syncing missing data for {len(symbols)} stocks...")
        
        sync_results = {
            'successful_syncs': [],
            'failed_syncs': {},
            'total_symbols': len(symbols),
            'new_data_points': 0
        }
        
        for symbol in symbols:
            try:
                # Use data synchronizer to sync stock data
                result = data_synchronizer.sync_stock_data(symbol, period=period)
                
                if result.get('success', False):
                    sync_results['successful_syncs'].append(symbol)
                    sync_results['new_data_points'] += result.get('prices_synced', 0)
                    logger.debug(f"  {symbol}: {result.get('prices_synced', 0)} new records")
                else:
                    sync_results['failed_syncs'][symbol] = result.get('error', 'Unknown error')
                    logger.warning(f"  {symbol}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                sync_results['failed_syncs'][symbol] = str(e)
                logger.error(f"  {symbol}: ERROR - {str(e)}")
        
        success_rate = (len(sync_results['successful_syncs']) / len(symbols)) * 100 if symbols else 0
        
        logger.info(f"Data sync complete: {len(sync_results['successful_syncs'])}/{len(symbols)} stocks "
                   f"({success_rate:.1f}% success rate), {sync_results['new_data_points']} new data points")
        
        return sync_results
    
    def get_sector_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sector statistics for monitoring.
        
        Returns:
            Sector statistics and health metrics
        """
        cache_key = "sector_statistics"
        cached_stats = cache.get(cache_key)
        
        if cached_stats:
            logger.debug("Using cached sector statistics")
            return cached_stats
        
        logger.info("Generating sector statistics...")
        
        # Get all training stocks
        training_stocks = self.sector_mapper.get_all_training_stocks()
        
        # Calculate sector distribution
        sector_distribution = {}
        for symbol in training_stocks:
            sector_name = self.sector_mapper.classify_stock_sector(symbol)
            sector_key = sector_name or 'Unknown'
            sector_distribution[sector_key] = sector_distribution.get(sector_key, 0) + 1
        
        # Get database statistics
        total_stocks_in_db = Stock.objects.filter(is_active=True).count()
        training_stocks_in_db = Stock.objects.filter(
            symbol__in=training_stocks,
            is_active=True
        ).count()
        
        # Calculate health metrics
        total_training_stocks = len(training_stocks)
        coverage_percentage = (training_stocks_in_db / total_training_stocks) * 100 if total_training_stocks > 0 else 0
        
        statistics = {
            'total_training_universe': total_training_stocks,
            'stocks_in_database': training_stocks_in_db,
            'total_active_stocks': total_stocks_in_db,
            'coverage_percentage': coverage_percentage,
            'sector_distribution': sector_distribution,
            'sector_count': len(sector_distribution),
            'avg_stocks_per_sector': sum(sector_distribution.values()) / len(sector_distribution) if sector_distribution else 0,
            'generated_at': datetime.now().isoformat()
        }
        
        # Cache for 1 hour
        cache.set(cache_key, statistics, self.cache_timeout)
        
        logger.info(f"Sector statistics generated: {total_training_stocks} training stocks, "
                   f"{len(sector_distribution)} sectors, {coverage_percentage:.1f}% coverage")
        
        return statistics
    
    def prepare_training_data_summary(self) -> Dict[str, Any]:
        """
        Prepare comprehensive summary for universal training readiness.
        
        Returns:
            Training readiness summary
        """
        logger.info("Preparing training data summary...")
        
        # Collect all relevant data
        sector_stats = self.get_sector_statistics()
        balance_validation = self.validate_sector_balance()
        
        # Data quality assessment
        training_stocks = self.sector_mapper.get_all_training_stocks()
        data_quality = {
            'stocks_with_sufficient_data': 0,
            'stocks_with_insufficient_data': 0,
            'total_assessed': 0
        }
        
        # Quick data availability check
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months
        
        for symbol in training_stocks[:20]:  # Sample check for performance
            try:
                price_data = self.price_reader.get_stock_prices(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if price_data and len(price_data) >= 100:  # Sufficient recent data
                    data_quality['stocks_with_sufficient_data'] += 1
                else:
                    data_quality['stocks_with_insufficient_data'] += 1
                    
                data_quality['total_assessed'] += 1
                
            except Exception as e:
                logger.debug(f"Data check failed for {symbol}: {str(e)}")
                data_quality['stocks_with_insufficient_data'] += 1
                data_quality['total_assessed'] += 1
        
        # Training readiness score
        readiness_factors = {
            'sector_balance': 1.0 if balance_validation['balanced'] else 0.5,
            'database_coverage': min(sector_stats['coverage_percentage'] / 100.0, 1.0),
            'data_quality': data_quality['stocks_with_sufficient_data'] / max(data_quality['total_assessed'], 1)
        }
        
        overall_readiness = sum(readiness_factors.values()) / len(readiness_factors)
        
        summary = {
            'training_readiness_score': overall_readiness,
            'readiness_factors': readiness_factors,
            'sector_statistics': sector_stats,
            'balance_validation': balance_validation,
            'data_quality_sample': data_quality,
            'recommendations': [],
            'generated_at': datetime.now().isoformat()
        }
        
        # Generate recommendations
        if overall_readiness >= 0.8:
            summary['recommendations'].append("System ready for universal training")
        elif overall_readiness >= 0.6:
            summary['recommendations'].append("System mostly ready, minor improvements recommended")
        else:
            summary['recommendations'].append("System needs improvements before universal training")
        
        if not balance_validation['balanced']:
            summary['recommendations'].append("Improve sector balance by adding more stocks to underrepresented sectors")
        
        if sector_stats['coverage_percentage'] < 90:
            summary['recommendations'].append("Sync missing stocks to database before training")
        
        logger.info(f"Training data summary complete: {overall_readiness:.2f} readiness score")
        
        return summary


# Global service instance
_sector_data_service = None


def get_sector_data_service() -> SectorDataService:
    """Get or create singleton sector data service instance."""
    global _sector_data_service
    if _sector_data_service is None:
        _sector_data_service = SectorDataService()
    return _sector_data_service
