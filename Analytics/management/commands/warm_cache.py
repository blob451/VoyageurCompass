"""
Cache warming management command for preloading frequently accessed data.
"""

import logging
import time
from datetime import datetime, timedelta

from django.core.cache import cache, caches
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Analytics.services.universal_predictor import get_universal_lstm_service
from Data.models import Stock, StockPrice, DataSectorPrice, DataIndustryPrice
from Data.repo.price_reader import PriceReader
from Data.services.yahoo_finance import yahoo_finance_service
from Core.utils.cache_utils import sanitize_cache_key

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Warm up application caches with frequently accessed data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stocks',
            nargs='+',
            default=['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            help='List of stock symbols to warm up (default: top 10 popular stocks)'
        )
        
        parser.add_argument(
            '--skip-models',
            action='store_true',
            help='Skip ML model warming (faster startup)'
        )
        
        parser.add_argument(
            '--skip-indicators',
            action='store_true',
            help='Skip technical indicators warming'
        )
        
        parser.add_argument(
            '--skip-prices',
            action='store_true',
            help='Skip price data warming'
        )
        
        parser.add_argument(
            '--skip-llm',
            action='store_true',
            help='Skip LLM model warming'
        )
        
        parser.add_argument(
            '--llm-only',
            action='store_true',
            help='Only warm LLM models (skip other warming)'
        )

    def handle(self, *args, **options):
        """Execute cache warming process."""
        start_time = time.time()
        
        self.stdout.write(self.style.SUCCESS('Starting cache warming process...'))
        
        stocks = options['stocks']
        llm_only = options.get('llm_only', False)
        
        # Calculate total tasks based on options
        if llm_only:
            total_tasks = 1  # Only LLM warming
        else:
            total_tasks = len(stocks) * 3 + 5  # Price data + indicators + sentiment per stock + 5 global tasks (including LLM)
        current_task = 0
        
        try:
            # LLM-only mode
            if llm_only:
                current_task += 1
                self.stdout.write(f'[{current_task}/{total_tasks}] Warming LLM models...')
                self._warm_llm_models()
                return
            
            # 1. Warm up LLM models first (if not skipped)
            if not options['skip_llm']:
                current_task += 1
                self.stdout.write(f'[{current_task}/{total_tasks}] Warming LLM models...')
                self._warm_llm_models()
            
            # 2. Warm up ML models
            if not options['skip_models']:
                current_task += 1
                self.stdout.write(f'[{current_task}/{total_tasks}] Warming ML models...')
                self._warm_ml_models()
            
            # 3. Warm up sector/industry data
            current_task += 1
            self.stdout.write(f'[{current_task}/{total_tasks}] Warming sector/industry data...')
            self._warm_sector_industry_data()
            
            # 4. Warm up market status
            current_task += 1
            self.stdout.write(f'[{current_task}/{total_tasks}] Warming market status...')
            self._warm_market_status()
            
            # 5. Warm up frequently accessed stocks
            for symbol in stocks:
                if not options['skip_prices']:
                    current_task += 1
                    self.stdout.write(f'[{current_task}/{total_tasks}] Warming price data for {symbol}...')
                    self._warm_stock_prices(symbol)
                
                if not options['skip_indicators']:
                    current_task += 1
                    self.stdout.write(f'[{current_task}/{total_tasks}] Warming technical indicators for {symbol}...')
                    self._warm_technical_indicators(symbol)
                
                current_task += 1
                self.stdout.write(f'[{current_task}/{total_tasks}] Warming sentiment data for {symbol}...')
                self._warm_sentiment_data(symbol)
            
            # 5. Warm up common analysis queries
            current_task += 1
            self.stdout.write(f'[{current_task}/{total_tasks}] Warming common database queries...')
            self._warm_common_queries()
            
            duration = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(
                    f'Cache warming completed successfully in {duration:.2f}s. '
                    f'Warmed {len(stocks)} stocks with {current_task} tasks.'
                )
            )
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Cache warming failed: {str(e)}'))
            raise CommandError(f'Cache warming failed: {str(e)}')

    def _warm_ml_models(self):
        """Warm up ML models by loading and running test predictions."""
        try:
            # 1. Warm up FinBERT sentiment model
            sentiment_analyzer = get_sentiment_analyzer()
            test_texts = [
                "The market outlook is positive for technology stocks.",
                "Economic indicators suggest moderate growth ahead.",
                "Quarterly earnings beat expectations significantly."
            ]
            sentiment_analyzer.analyzeSentimentBatch(test_texts)
            logger.info("FinBERT model warmed up successfully")
            
            # 2. Warm up Universal LSTM model
            lstm_service = get_universal_lstm_service()
            if lstm_service and hasattr(lstm_service, 'model') and lstm_service.model:
                # Try to warm up with a test prediction
                test_result = lstm_service.predict_stock_price('SPY', horizon='1d')
                if test_result:
                    logger.info("Universal LSTM model warmed up successfully")
                else:
                    logger.warning("Universal LSTM model test prediction failed")
            else:
                logger.warning("Universal LSTM model not available for warming")
                
        except Exception as e:
            logger.warning(f"ML model warming failed: {str(e)}")

    def _warm_llm_models(self):
        """Warm up LLM models using the built-in warm-up method."""
        try:
            llm_service = get_local_llm_service()
            if not llm_service:
                logger.warning("LLM service not available for warming")
                return
                
            # Use the LocalLLMService warm-up method
            warm_up_result = llm_service.warm_up_models()
            
            if warm_up_result.get('success'):
                logger.info(f"LLM warm-up successful: {warm_up_result['models_successful']}/{warm_up_result['models_tested']} models in {warm_up_result['total_time']}s")
                
                # Log individual model results
                for result in warm_up_result.get('results', []):
                    if result['success']:
                        logger.info(f"  [OK] {result['model']}: {result['response_time']}s")
                    else:
                        logger.warning(f"  [FAIL] {result['model']}: {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"LLM warm-up failed: {warm_up_result.get('error', 'Unknown error')}")
            
            # Cache warm-up results for monitoring
            cache_key = "llm_warmup_results"
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                **warm_up_result
            }
            cache.set(cache_key, cache_data, 3600)  # Cache for 1 hour
                
        except Exception as e:
            logger.error(f"LLM model warming failed: {str(e)}")

    def _warm_sector_industry_data(self):
        """Warm up sector and industry composite data."""
        try:
            # Cache recent sector composite data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            # Get unique sectors and industries
            sectors = DataSectorPrice.objects.select_related('sector_id').values_list('sector_id__sectorName', flat=True).distinct()[:10]
            industries = DataIndustryPrice.objects.select_related('industry_id').values_list('industry_id__industryName', flat=True).distinct()[:15]
            
            l2_cache = caches['l2_cache']
            
            for sector in sectors:
                if sector:  # Skip None values
                    cache_key = sanitize_cache_key(f"sector_composite:{sector}:{start_date}:{end_date}")
                    sector_data = list(DataSectorPrice.objects.filter(
                        sector_id__sectorName=sector,
                        date__gte=start_date,
                        date__lte=end_date
                    ).values('date', 'close_index', 'volume_agg'))
                    l2_cache.set(cache_key, sector_data, 3600)  # 1 hour
            
            for industry in industries:
                if industry:  # Skip None values
                    cache_key = sanitize_cache_key(f"industry_composite:{industry}:{start_date}:{end_date}")
                    industry_data = list(DataIndustryPrice.objects.filter(
                        industry_id__industryName=industry,
                        date__gte=start_date,
                        date__lte=end_date
                    ).values('date', 'close_index', 'volume_agg'))
                    l2_cache.set(cache_key, industry_data, 3600)  # 1 hour
                
            logger.info(f"Sector/industry data warmed up: {len(sectors)} sectors, {len(industries)} industries")
            
        except Exception as e:
            logger.warning(f"Sector/industry warming failed: {str(e)}")

    def _warm_market_status(self):
        """Warm up market status data."""
        try:
            market_status = yahoo_finance_service.get_market_status()
            cache.set("market_status", market_status, 300)  # 5 minutes
            logger.info("Market status warmed up successfully")
        except Exception as e:
            logger.warning(f"Market status warming failed: {str(e)}")

    def _warm_stock_prices(self, symbol: str):
        """Warm up stock price data for a symbol."""
        try:
            price_reader = PriceReader()
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365 * 2)  # 2 years
            
            # Get price data
            stock_prices = price_reader.get_stock_prices(symbol, start_date, end_date)
            
            if stock_prices:
                # Cache recent price data
                cache_key = f"stock_prices:{symbol}:{start_date}:{end_date}"
                l2_cache = caches['l2_cache']
                l2_cache.set(cache_key, stock_prices, 1800)  # 30 minutes
                
                # Also cache common timeframes
                for days in [30, 90, 180, 365]:
                    recent_start = end_date - timedelta(days=days)
                    recent_prices = [p for p in stock_prices if p.date >= recent_start]
                    cache_key = f"stock_prices:{symbol}:{recent_start}:{end_date}"
                    cache.set(cache_key, recent_prices, 600)  # 10 minutes
                
                logger.info(f"Stock prices warmed up for {symbol}: {len(stock_prices)} records")
            else:
                logger.warning(f"No price data found for {symbol}")
                
        except Exception as e:
            logger.warning(f"Stock price warming failed for {symbol}: {str(e)}")

    def _warm_technical_indicators(self, symbol: str):
        """Warm up technical indicators for a symbol."""
        try:
            engine = TechnicalAnalysisEngine()
            # Set current symbol for caching context
            engine._current_symbol = symbol
            
            # Get price data
            price_reader = PriceReader()
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)  # 1 year
            
            stock_prices = price_reader.get_stock_prices(symbol, start_date, end_date)
            
            if stock_prices and len(stock_prices) >= 50:
                # Calculate and cache major indicators
                indicators = [
                    ('sma_crossover', engine._calculate_sma_crossover),
                    ('price_vs_50d', engine._calculate_price_vs_50d),
                    ('rsi14', engine._calculate_rsi14),
                    ('macd_histogram', engine._calculate_macd_histogram),
                    ('bollinger_position', engine._calculate_bollinger_position),
                    ('bollinger_bandwidth', engine._calculate_bollinger_bandwidth),
                    ('volume_surge', engine._calculate_volume_surge),
                    ('obv_trend', engine._calculate_obv_trend),
                ]
                
                cached_count = 0
                for indicator_name, indicator_func in indicators:
                    try:
                        result = indicator_func(stock_prices)
                        if result:
                            cached_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to warm {indicator_name} for {symbol}: {str(e)}")
                
                logger.info(f"Technical indicators warmed up for {symbol}: {cached_count}/{len(indicators)} indicators")
            else:
                logger.warning(f"Insufficient price data for technical indicators warming: {symbol}")
                
        except Exception as e:
            logger.warning(f"Technical indicators warming failed for {symbol}: {str(e)}")

    def _warm_sentiment_data(self, symbol: str):
        """Warm up sentiment analysis data for a symbol."""
        try:
            # Fetch recent news for sentiment analysis
            news_items = yahoo_finance_service.fetchNewsForStock(symbol, days=30, max_items=20)
            
            if news_items:
                sentiment_analyzer = get_sentiment_analyzer()
                # Process recent news to warm sentiment cache
                result = sentiment_analyzer.analyzeNewsArticles(news_items[:10], aggregate=True, symbol=symbol)
                
                if result:
                    # Cache sentiment result
                    cache_key = f"sentiment_analysis:{symbol}:30d"
                    cache.set(cache_key, result, 1800)  # 30 minutes
                    logger.info(f"Sentiment data warmed up for {symbol}: {len(news_items)} articles")
                else:
                    logger.warning(f"Sentiment analysis failed for {symbol}")
            else:
                logger.warning(f"No news found for sentiment warming: {symbol}")
                
        except Exception as e:
            logger.warning(f"Sentiment warming failed for {symbol}: {str(e)}")

    def _warm_common_queries(self):
        """Warm up common database queries."""
        try:
            # Cache top stocks by market cap
            top_stocks = list(Stock.objects.filter(
                market_cap__gt=0
            ).order_by('-market_cap')[:50].values(
                'id', 'symbol', 'short_name', 'sector', 'industry', 'market_cap'
            ))
            cache.set("top_stocks_by_market_cap", top_stocks, 3600)  # 1 hour
            
            # Cache sector summaries
            sector_summary = list(Stock.objects.values('sector').distinct()[:20])
            cache.set("sector_list", sector_summary, 7200)  # 2 hours
            
            # Cache industry summaries
            industry_summary = list(Stock.objects.values('industry').distinct()[:50])
            cache.set("industry_list", industry_summary, 7200)  # 2 hours
            
            logger.info(f"Common queries warmed up: {len(top_stocks)} stocks, {len(sector_summary)} sectors")
            
        except Exception as e:
            logger.warning(f"Common queries warming failed: {str(e)}")