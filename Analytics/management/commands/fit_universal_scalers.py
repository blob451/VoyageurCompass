"""
Management command to fit Universal LSTM scalers for all stocks.
This ensures accurate predictions by pre-fitting scalers for each stock.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.db.models import Q

from Data.models import Stock, StockPrice
from Analytics.services.universal_predictor import UniversalLSTMAnalyticsService
from Analytics.ml.universal_preprocessor import UniversalLSTMPreprocessor

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Fit Universal LSTM scalers for all stocks to improve prediction accuracy'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Specific symbols to fit scalers for (default: all stocks)'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force refit even if scalers already exist'
        )
        
        parser.add_argument(
            '--top-n',
            type=int,
            default=100,
            help='Fit scalers for top N most active stocks (default: 100)'
        )
        
        parser.add_argument(
            '--cache-ttl',
            type=int,
            default=30,
            help='Cache TTL in days (default: 30)'
        )

    def handle(self, *args, **options):
        """Execute scaler fitting for Universal LSTM model."""
        self.stdout.write(self.style.SUCCESS('=== Universal LSTM Scaler Fitting ==='))
        
        symbols = options.get('symbols')
        force_refit = options.get('force', False)
        top_n = options.get('top_n', 100)
        cache_ttl_days = options.get('cache_ttl', 30)
        
        # Initialize the service
        try:
            service = UniversalLSTMAnalyticsService()
            self.stdout.write(self.style.SUCCESS('Universal LSTM service initialized'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to initialize service: {str(e)}')
            )
            return
        
        # Get stocks to process
        if symbols:
            stocks = Stock.objects.filter(symbol__in=symbols)
        else:
            # Get top N most active stocks by recent price record count
            thirty_days_ago = datetime.now() - timedelta(days=30)
            stocks = Stock.objects.filter(
                stockprice__date__gte=thirty_days_ago
            ).distinct()[:top_n]
        
        if not stocks.exists():
            self.stdout.write(self.style.WARNING('No stocks found to process'))
            return
        
        self.stdout.write(f'Processing {stocks.count()} stocks...\n')
        
        success_count = 0
        failure_count = 0
        skipped_count = 0
        
        for stock in stocks:
            try:
                # Check if scalers already exist in cache
                cache_key = f'lstm_scaler_{stock.symbol}'
                existing_scaler = cache.get(cache_key) if not force_refit else None
                
                if existing_scaler:
                    self.stdout.write(
                        f'{stock.symbol}: Scalers already fitted (cached)'
                    )
                    skipped_count += 1
                    continue
                
                # Get recent price data for the stock
                prices = StockPrice.objects.filter(
                    stock=stock
                ).order_by('-date')[:252]  # 1 year of data
                
                if prices.count() < 60:  # Need at least 60 days
                    self.stdout.write(
                        self.style.WARNING(
                            f'{stock.symbol}: Insufficient data ({prices.count()} days)'
                        )
                    )
                    failure_count += 1
                    continue
                
                # Prepare features for scaler fitting
                self.stdout.write(f'{stock.symbol}: Preparing features...')
                
                # Create preprocessor and extract features
                preprocessor = UniversalLSTMPreprocessor()
                
                # Build feature dataset
                features = []
                for i in range(len(prices) - 1):
                    price_data = prices[i]
                    prev_price = prices[i + 1] if i + 1 < len(prices) else None
                    
                    if prev_price:
                        feature_dict = {
                            'price': float(price_data.close),
                            'volume': float(price_data.volume),
                            'price_change': float(price_data.close - prev_price.close),
                            'volume_ratio': float(price_data.volume / prev_price.volume) if prev_price.volume > 0 else 1.0,
                            'high_low_ratio': float(price_data.high / price_data.low) if price_data.low > 0 else 1.0,
                            'close_open_ratio': float(price_data.close / price_data.open) if price_data.open > 0 else 1.0,
                        }
                        features.append(feature_dict)
                
                if len(features) < 30:
                    self.stdout.write(
                        self.style.WARNING(
                            f'{stock.symbol}: Insufficient valid features'
                        )
                    )
                    failure_count += 1
                    continue
                
                # Fit scalers using our method
                scaler_data = self._fit_scalers_for_stock(
                    stock.symbol, features
                )
                
                if scaler_data:
                    # Save to file first (primary storage)
                    self._save_scalers_to_file(stock.symbol, scaler_data)
                    
                    # Cache serialized version
                    cache_ttl = cache_ttl_days * 24 * 3600  # Convert to seconds
                    try:
                        import base64
                        serialized_data = base64.b64encode(pickle.dumps(scaler_data)).decode('utf-8')
                        cache.set(f'{cache_key}_serialized', serialized_data, cache_ttl)
                    except Exception as e:
                        logger.warning(f"Could not cache scalers for {stock.symbol}: {str(e)}")
                    
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'{stock.symbol}: Scalers fitted and cached successfully'
                        )
                    )
                    success_count += 1
                else:
                    self.stdout.write(
                        self.style.ERROR(f'{stock.symbol}: Failed to fit scalers')
                    )
                    failure_count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'{stock.symbol}: Error - {str(e)}')
                )
                failure_count += 1
                logger.exception(f'Error fitting scalers for {stock.symbol}')
        
        # Summary
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS(f'Successfully fitted: {success_count}'))
        if skipped_count > 0:
            self.stdout.write(f'Skipped (already fitted): {skipped_count}')
        if failure_count > 0:
            self.stdout.write(self.style.WARNING(f'Failed: {failure_count}'))
        
        self.stdout.write(self.style.SUCCESS('\nScaler fitting complete!'))

    def _fit_scalers_for_stock(
        self, symbol: str, features: list
    ) -> Optional[Dict[str, Any]]:
        """Fit scalers for a specific stock."""
        try:
            # Import necessary sklearn components
            from sklearn.preprocessing import StandardScaler, RobustScaler
            import numpy as np
            
            # Prepare feature matrix
            feature_names = list(features[0].keys())
            feature_matrix = np.array([
                [f.get(name, 0) for name in feature_names]
                for f in features
            ])
            
            # Create and fit scalers
            standard_scaler = StandardScaler()
            robust_scaler = RobustScaler()
            
            standard_scaler.fit(feature_matrix)
            robust_scaler.fit(feature_matrix)
            
            # Create scaler data dictionary
            scaler_data = {
                'symbol': symbol,
                'feature_names': feature_names,
                'standard_scaler': standard_scaler,
                'robust_scaler': robust_scaler,
                'fitted_at': datetime.now().isoformat(),
                'n_samples': len(features),
                'statistics': {
                    'mean': standard_scaler.mean_.tolist(),
                    'std': standard_scaler.scale_.tolist(),
                    'median': robust_scaler.center_.tolist(),
                    'scale': robust_scaler.scale_.tolist(),
                }
            }
            
            return scaler_data
            
        except Exception as e:
            logger.error(f'Failed to fit scalers for {symbol}: {str(e)}')
            return None

    def _save_scalers_to_file(self, symbol: str, scaler_data: Dict[str, Any]):
        """Save fitted scalers to file for persistence."""
        try:
            # Create scalers directory if it doesn't exist
            scaler_dir = Path('Data/ml_models/universal_lstm/scalers')
            scaler_dir.mkdir(parents=True, exist_ok=True)
            
            # Save scaler data
            file_path = scaler_dir / f'{symbol}_scalers.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            logger.info(f'Saved scalers for {symbol} to {file_path}')
            
        except Exception as e:
            logger.error(f'Failed to save scalers for {symbol}: {str(e)}')