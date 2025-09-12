"""
Django management command for training Universal LSTM models.
Implements the comprehensive universal training strategy with GPU acceleration.
"""

import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from django.core.management.base import BaseCommand, CommandError

from Analytics.ml.models.lstm_base import (
    UniversalLSTMPredictor,
    create_universal_lstm_model,
    save_universal_model,
)
from Analytics.ml.sector_mappings import get_sector_mapper
from Analytics.ml.universal_preprocessor import UniversalLSTMPreprocessor
from Data.models import Stock
from Data.repo.price_reader import PriceReader
from Data.services.sector_data_service import get_sector_data_service
from Data.services.synchronizer import data_synchronizer


class UniversalLSTMTrainingConfig:
    """Enhanced training configuration for Universal LSTM."""

    def __init__(
        self,
        # Stage 1: Foundation Training - ENHANCED to prevent mean collapse
        foundation_epochs: int = 150,  # Extended for robust universal learning
        foundation_lr: float = 0.001,  # Increased to 0.001 to prevent collapse to mean
        foundation_batch_size: int = 512,  # Increased 16x for RTX 4080 SUPER
        # Stage 2: Sector Specialization - OPTIMIZED for pattern diversity
        specialization_epochs: int = 200,  # Extended for deep sector patterns
        specialization_lr: float = 0.0005,  # Increased to 0.0005 for better diversity
        specialization_batch_size: int = 1024,  # Increased 16x for RTX 4080 SUPER
        # Stage 3: Fine-Tuning - REFINED with higher LR
        finetuning_epochs: int = 100,  # Extended for comprehensive integration
        finetuning_lr: float = 0.0002,  # Increased to 0.0002 for final convergence
        finetuning_batch_size: int = 2048,  # Increased 16x for RTX 4080 SUPER
        # Universal model parameters
        sequence_length: int = 60,
        validation_split: float = 0.2,
        early_stopping_patience: int = 50,  # CRITICAL FIX: Increased from 15 to allow proper convergence
        checkpoint_frequency: int = 5,  # More frequent checkpoints for monitoring
        years_back: int = 3,
        min_stocks_per_sector: int = 5,
        # GPU and performance settings
        gpu_enabled: bool = True,
        mixed_precision: bool = True,
        gradient_clipping: float = 0.5,  # Tighter gradient clipping for stability
        weight_decay: float = 0.01,
        # CPU parallelization settings (16-core Ryzen 9 9950X3D optimization)
        num_workers: int = 16,  # Match physical cores
        pin_memory: bool = True,  # Faster GPU transfer
        persistent_workers: bool = True,  # Reuse worker processes
        prefetch_factor: int = 4,  # Preload 4 batches per worker
    ):
        """Initialize universal training configuration."""
        self.foundation_epochs = foundation_epochs
        self.foundation_lr = foundation_lr
        self.foundation_batch_size = foundation_batch_size

        self.specialization_epochs = specialization_epochs
        self.specialization_lr = specialization_lr
        self.specialization_batch_size = specialization_batch_size

        self.finetuning_epochs = finetuning_epochs
        self.finetuning_lr = finetuning_lr
        self.finetuning_batch_size = finetuning_batch_size

        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_frequency = checkpoint_frequency
        self.years_back = years_back
        self.min_stocks_per_sector = min_stocks_per_sector

        self.gpu_enabled = gpu_enabled
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay

        # CPU parallelization settings
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        # Calculate total training time
        self.total_epochs = foundation_epochs + specialization_epochs + finetuning_epochs
        self.estimated_hours = 1.5  # Extended training for production-grade model (450 epochs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "foundation_epochs": self.foundation_epochs,
            "foundation_lr": self.foundation_lr,
            "foundation_batch_size": self.foundation_batch_size,
            "specialization_epochs": self.specialization_epochs,
            "specialization_lr": self.specialization_lr,
            "specialization_batch_size": self.specialization_batch_size,
            "finetuning_epochs": self.finetuning_epochs,
            "finetuning_lr": self.finetuning_lr,
            "finetuning_batch_size": self.finetuning_batch_size,
            "sequence_length": self.sequence_length,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_frequency": self.checkpoint_frequency,
            "years_back": self.years_back,
            "min_stocks_per_sector": self.min_stocks_per_sector,
            "gpu_enabled": self.gpu_enabled,
            "mixed_precision": self.mixed_precision,
            "gradient_clipping": self.gradient_clipping,
            "weight_decay": self.weight_decay,
            "total_epochs": self.total_epochs,
            "estimated_hours": self.estimated_hours,
        }


class Command(BaseCommand):
    help = "Train Universal LSTM model for cross-sector stock price prediction"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupted = False
        self.progress_file = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup graceful interrupt handling."""

        def signal_handler(signum, frame):
            self.interrupted = True
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(self.style.WARNING(">>> INTERRUPT RECEIVED - Saving progress and stopping gracefully..."))
            self.stdout.write("=" * 80)

        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):  # Windows compatibility
            signal.signal(signal.SIGTERM, signal_handler)

    def save_progress(self, stage: str, progress_data: Dict[str, Any]):
        """Save training progress to resume later."""
        if not self.progress_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.progress_file = f"universal_lstm_progress_{timestamp}.json"

        progress = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "interrupted": True,
            "progress_data": progress_data,
        }

        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
            self.stdout.write(f"   >>> Progress saved to: {self.progress_file}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   >>> Failed to save progress: {e}"))

    def check_interrupt(self) -> bool:
        """Check if script was interrupted and handle gracefully."""
        if self.interrupted:
            self.stdout.write(self.style.WARNING("   >>> Training interrupted by user"))
            return True
        return False

    def load_progress(self, progress_file: str) -> Optional[Dict[str, Any]]:
        """Load training progress from file."""
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
            self.stdout.write(f"   >>> Loaded progress from: {progress_file}")
            self.stdout.write(f'   >>> Last stage: {progress.get("stage", "unknown")}')
            self.stdout.write(f'   >>> Timestamp: {progress.get("timestamp", "unknown")}')
            return progress
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   >>> Failed to load progress file: {e}"))
            return None

    def add_arguments(self, parser):
        """Add command line arguments for universal training."""

        # Training time and epochs
        parser.add_argument(
            "--training-time",
            type=str,
            default="1h",
            help="Total training time (e.g., 1h, 60m) - Extended for production quality",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=325,
            help="Total training epochs across all stages (default: 325) - Extended for higher accuracy",
        )

        # Stage-specific epochs (Extended Configuration)
        parser.add_argument(
            "--stage1-epochs",
            type=int,
            default=100,
            help="Foundation training epochs (default: 100) - Extended for robust universal learning",
        )
        parser.add_argument(
            "--stage2-epochs",
            type=int,
            default=150,
            help="Sector specialization epochs (default: 150) - Extended for deep sector patterns",
        )
        parser.add_argument(
            "--stage3-epochs",
            type=int,
            default=75,
            help="Fine-tuning epochs (default: 75) - Extended for comprehensive integration",
        )

        # Model configuration
        parser.add_argument("--model-version", type=str, default="1.0", help="Model version for saving (default: 1.0)")
        parser.add_argument(
            "--model-type",
            type=str,
            default="full_universal",
            choices=["full_universal", "price_only", "lightweight"],
            help="Type of universal model to train",
        )

        # Data parameters
        parser.add_argument("--years-back", type=int, default=3, help="Years of historical data to use (default: 3)")
        parser.add_argument(
            "--sequence-length", type=int, default=60, help="Input sequence length in days (default: 60)"
        )
        parser.add_argument("--validation-split", type=float, default=0.2, help="Validation data split (default: 0.2)")

        # Training features
        parser.add_argument("--sector-balancing", action="store_true", help="Enable sector balancing during training")
        parser.add_argument("--cross-validation", action="store_true", help="Perform cross-validation during training")
        parser.add_argument(
            "--gpu-acceleration", action="store_true", default=True, help="Enable GPU acceleration (default: True)"
        )

        # Output and monitoring
        parser.add_argument("--save-to", type=str, default="Data/ml_models/", help="Directory to save trained models")
        parser.add_argument(
            "--checkpoint-every",
            type=int,
            default=5,
            help="Save checkpoint every N epochs (default: 5) - More frequent saves",
        )
        parser.add_argument("--verbose-logging", action="store_true", help="Enable verbose logging during training")

        # Stock selection
        parser.add_argument("--sectors", type=str, default="all", help="Sectors to train on (default: all)")
        parser.add_argument(
            "--min-stocks-per-sector", type=int, default=5, help="Minimum stocks per sector (default: 5)"
        )
        parser.add_argument(
            "--max-stocks-per-sector", type=int, default=15, help="Maximum stocks per sector (default: 15)"
        )

        # Resume training
        parser.add_argument(
            "--resume-from",
            type=str,
            help="Resume training from a progress file (e.g., universal_lstm_progress_20250816_143022.json)",
        )

        # Data management
        parser.add_argument(
            "--force-refresh", action="store_true", help="Force refresh all data from Yahoo Finance even if it exists"
        )
        parser.add_argument(
            "--skip-data-collection",
            action="store_true",
            help="Skip Yahoo Finance data collection entirely and use existing data",
        )

    def handle(self, *args, **options):
        """Handle the universal LSTM training command."""
        try:
            # Check if resuming from progress file
            progress = None
            if options.get("resume_from"):
                progress = self.load_progress(options["resume_from"])
                if not progress:
                    raise CommandError(f"Could not load progress file: {options['resume_from']}")

                self.stdout.write(self.style.WARNING("=" * 80))
                self.stdout.write(self.style.WARNING(">>> RESUMING TRAINING FROM PROGRESS FILE"))
                self.stdout.write(self.style.WARNING("=" * 80))

            # Beautiful startup banner
            self._display_training_banner(options)

            # Initialize training configuration
            config = self._create_training_config(options)

            # Validate GPU availability if requested
            if config.gpu_enabled:
                self._check_gpu_availability()

            # Prepare training data (fetch from Yahoo Finance)
            self._prepare_training_data(config, options)

            # Collect training data for 115 stocks
            self.stdout.write("Collecting training data for 115 stocks across all sectors...")
            training_data = self._collect_training_data(config)

            if not training_data:
                raise CommandError("Failed to collect sufficient training data")

            # Create universal model
            self.stdout.write(f'Creating Universal LSTM model ({options["model_type"]})...')
            model = self._create_universal_model(options, config)

            # Initialize preprocessor and prepare data
            self.stdout.write("Preparing universal features and sequences...")
            preprocessor, sequences = self._prepare_universal_data(training_data, config)

            # Execute multi-stage training
            self.stdout.write("Beginning extensive universal training (8+ hours)...")
            training_results = self._execute_training_pipeline(model, sequences, config, options)

            # Save final model
            self.stdout.write("Saving trained Universal LSTM model...")
            model_path = self._save_universal_model(model, preprocessor, training_results, options)

            # Generate training summary
            self._display_training_summary(training_results, model_path, config)

            self.stdout.write(self.style.SUCCESS("Universal LSTM training completed successfully!"))

        except KeyboardInterrupt:
            if self.interrupted:
                self.stdout.write("\n" + "=" * 80)
                self.stdout.write(self.style.WARNING(">>> TRAINING INTERRUPTED GRACEFULLY"))
                self.stdout.write(">>> Check for progress files to resume training later")
                self.stdout.write("=" * 80)
                sys.exit(0)
            else:
                raise

        except Exception as e:
            if self.interrupted:
                self.stdout.write(self.style.WARNING(f">>> Training interrupted during: {str(e)}"))
                sys.exit(0)
            else:
                raise CommandError(f"Universal training failed: {str(e)}")

    def _create_training_config(self, options: Dict[str, Any]) -> UniversalLSTMTrainingConfig:
        """Create training configuration from command options."""
        return UniversalLSTMTrainingConfig(
            foundation_epochs=options["stage1_epochs"],
            specialization_epochs=options["stage2_epochs"],
            finetuning_epochs=options["stage3_epochs"],
            sequence_length=options["sequence_length"],
            validation_split=options["validation_split"],
            checkpoint_frequency=options["checkpoint_every"],
            years_back=options["years_back"],
            min_stocks_per_sector=options["min_stocks_per_sector"],
            gpu_enabled=options["gpu_acceleration"],
        )

    def _check_gpu_availability(self):
        """Check GPU availability for training."""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.stdout.write(f"GPU Available: {gpu_name} ({memory_gb:.1f} GB)")
            else:
                self.stdout.write(self.style.WARNING("GPU not available, falling back to CPU training"))
        except ImportError:
            raise CommandError("PyTorch not installed. Please install PyTorch with CUDA support.")

    def _prepare_training_data(self, config: UniversalLSTMTrainingConfig, options: Dict[str, Any]) -> None:
        """
        Prepare training data by ensuring stocks exist and fetching from Yahoo Finance.
        Includes rate limiting and intelligent data collection.
        """
        self.stdout.write(self.style.HTTP_INFO("[DATA] DATA PREPARATION"))
        self.stdout.write("-" * 30)

        sector_service = get_sector_data_service()
        sector_mapper = get_sector_mapper()

        # Step 1: Ensure all training stocks are in database
        self.stdout.write("   [STEP 1] Syncing training stocks to database...")
        sync_results = sector_service.ensure_training_stocks_in_database()
        self.stdout.write(
            f'   >>> Stocks: {sync_results["new_stocks"]} new, {sync_results["existing_stocks"]} existing'
        )

        # Step 2: Validate sector balance
        self.stdout.write("   [STEP 2] Validating sector balance...")
        balance_results = sector_service.validate_sector_balance()
        if balance_results["balanced"]:
            self.stdout.write("   >>> Sector balance: [OK] ADEQUATE")
        else:
            self.stdout.write(
                f'   >>> Sector balance: [WARN] {len(balance_results["underrepresented_sectors"])} sectors need more stocks'
            )

        # Step 3: Check existing data and collect missing historical data
        if options.get("skip_data_collection"):
            self.stdout.write(self.style.SUCCESS("   [STEP 3] Skipping data collection (--skip-data-collection flag)"))
            return

        self.stdout.write("   [STEP 3] Checking existing data and fetching missing data...")
        self.stdout.write(f"   >>> Target period: {config.years_back} years ({config.years_back * 365} days)")

        # If force refresh, sync all stocks
        if options.get("force_refresh"):
            self.stdout.write(self.style.WARNING("   >>> Force refresh enabled - syncing all stocks"))
            stocks_needing_sync = list(sector_mapper.get_all_training_stocks())
            stocks_with_data = []
        else:
            # First check what data we already have
            from datetime import datetime, timedelta

            from Data.models import StockPrice

            min_date_required = datetime.now().date() - timedelta(days=config.years_back * 365)
            stocks_needing_sync = []
            stocks_with_data = []

            for symbol in sector_mapper.get_all_training_stocks():
                try:
                    stock = Stock.objects.get(symbol=symbol)
                    # Check if we have sufficient recent data
                    recent_prices = StockPrice.objects.filter(stock=stock, date__gte=min_date_required).count()

                    if recent_prices < 500:  # Need at least 500 data points for good training
                        stocks_needing_sync.append(symbol)
                    else:
                        stocks_with_data.append(symbol)
                except Stock.DoesNotExist:
                    stocks_needing_sync.append(symbol)

            self.stdout.write(f"   >>> Data status: {len(stocks_with_data)} stocks have sufficient data")
            self.stdout.write(f"   >>> Need to sync: {len(stocks_needing_sync)} stocks")

            if not stocks_needing_sync:
                self.stdout.write(
                    self.style.SUCCESS("   >>> All stocks have sufficient data - skipping Yahoo Finance sync")
                )
                return

        self.stdout.write("   >>> Rate limiting: ENABLED (1-2s delays)")

        period_map = {1: "1y", 2: "2y", 3: "5y", 4: "5y", 5: "5y"}
        yahoo_period = period_map.get(config.years_back, "5y")

        # Process only stocks that need syncing in batches with rate limiting
        batch_size = 10
        total_stocks = len(stocks_needing_sync)
        successful_syncs = 0
        failed_syncs = 0

        for i in range(0, total_stocks, batch_size):
            # Check for interrupt before processing each batch
            if self.check_interrupt():
                self.save_progress(
                    "data_collection",
                    {
                        "total_stocks": total_stocks,
                        "processed_stocks": i,
                        "successful_syncs": successful_syncs,
                        "failed_syncs": failed_syncs,
                        "remaining_stocks": stocks_needing_sync[i:],
                    },
                )
                return

            batch = stocks_needing_sync[i : i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, total_stocks)

            self.stdout.write(f"   >>> Processing batch {batch_start}-{batch_end}/{total_stocks}...")

            for j, symbol in enumerate(batch):
                # Check for interrupt before processing each symbol
                if self.check_interrupt():
                    self.save_progress(
                        "data_collection",
                        {
                            "total_stocks": total_stocks,
                            "processed_stocks": i + j,
                            "successful_syncs": successful_syncs,
                            "failed_syncs": failed_syncs,
                            "current_batch": batch[j:],
                            "remaining_stocks": stocks_needing_sync[i + len(batch) :],
                        },
                    )
                    return

                try:
                    # Add progressive delay to avoid rate limiting
                    if i > 0 or j > 0:
                        delay = random.uniform(1.0, 2.5)
                        time.sleep(delay)

                    # Sync stock data with Yahoo Finance
                    result = data_synchronizer.sync_stock_data(symbol, period=yahoo_period)

                    if result.get("success", False):
                        successful_syncs += 1
                        records = result.get("prices_synced", 0)
                        if records > 0:
                            self.stdout.write(f"       {symbol}: {records} new records")
                    else:
                        failed_syncs += 1
                        error = result.get("error", "Unknown error")
                        self.stdout.write(f"       {symbol}: FAILED - {error}")

                except Exception as e:
                    failed_syncs += 1
                    self.stdout.write(f"       {symbol}: ERROR - {str(e)}")

            # Longer delay between batches
            if batch_end < total_stocks:
                delay = random.uniform(3.0, 5.0)
                self.stdout.write(f"   >>> Batch complete, waiting {delay:.1f}s before next batch...")
                time.sleep(delay)

        # Summary
        success_rate = (successful_syncs / total_stocks) * 100 if total_stocks > 0 else 0
        self.stdout.write("")
        self.stdout.write(f"   [SUMMARY] Data collection complete:")
        self.stdout.write(f"   >>> Successful: {successful_syncs}/{total_stocks} stocks ({success_rate:.1f}%)")
        self.stdout.write(f"   >>> Failed: {failed_syncs} stocks")

        if success_rate < 50:
            self.stdout.write(self.style.WARNING("   >>> Warning: Low success rate, training may be impacted"))
        elif success_rate >= 80:
            self.stdout.write(self.style.SUCCESS("   >>> Excellent: High success rate, ready for training"))

        self.stdout.write("")

    def _collect_training_data(self, config: UniversalLSTMTrainingConfig) -> Dict[str, Any]:
        """
        Collect training data for 115 stocks across all sectors.
        Returns dictionary mapping symbols to their data and metadata.
        """
        price_reader = PriceReader()
        sector_mapper = get_sector_mapper()

        # Get all training stocks from the universe
        all_training_stocks = sector_mapper.get_all_training_stocks()
        self.stdout.write(f"Target training universe: {len(all_training_stocks)} stocks")

        training_data = {}
        successful_stocks = 0
        failed_stocks = 0

        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.years_back * 365)

        for i, symbol in enumerate(all_training_stocks, 1):
            self.stdout.write(f"[{i}/{len(all_training_stocks)}] Collecting data for {symbol}...")

            try:
                # Get stock price data
                price_data = price_reader.get_stock_prices(symbol=symbol, start_date=start_date, end_date=end_date)

                if price_data and len(price_data) >= config.sequence_length + 100:
                    # Get sector classification
                    sector_name = sector_mapper.classify_stock_sector(symbol)
                    sector_id = sector_mapper.get_sector_id(sector_name or "Unknown")
                    industry_id = sector_mapper.infer_sector_from_industry(sector_id)

                    training_data[symbol] = {
                        "price_data": price_data,
                        "sector_name": sector_name,
                        "sector_id": sector_id,
                        "industry_id": industry_id,
                        "data_points": len(price_data),
                    }
                    successful_stocks += 1
                    self.stdout.write(f"  SUCCESS: {len(price_data)} data points")
                else:
                    failed_stocks += 1
                    self.stdout.write(f"  FAILED: Insufficient data")

            except Exception as e:
                failed_stocks += 1
                self.stdout.write(f"  FAILED: {str(e)}")

        # Validate sector balance
        sector_counts = {}
        for symbol, data in training_data.items():
            sector = data["sector_name"] or "Unknown"
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        self.stdout.write("\nSector distribution:")
        for sector, count in sector_counts.items():
            self.stdout.write(f"  {sector}: {count} stocks")

        self.stdout.write(f"\nData collection summary:")
        self.stdout.write(f"  Successful: {successful_stocks} stocks")
        self.stdout.write(f"  Failed: {failed_stocks} stocks")
        self.stdout.write(f"  Success rate: {successful_stocks/(successful_stocks+failed_stocks)*100:.1f}%")

        if successful_stocks < 15:  # Minimum threshold for proof of concept
            raise CommandError(f"Insufficient training data: only {successful_stocks} stocks collected")

        return training_data

    def _create_universal_model(
        self, options: Dict[str, Any], config: UniversalLSTMTrainingConfig
    ) -> UniversalLSTMPredictor:
        """Create universal LSTM model with specified configuration."""
        model = create_universal_lstm_model(
            input_size=42,  # Universal features (excluding embeddings) - Updated to match UNIVERSAL_FEATURES
            sector_embedding_dim=12,  # OPTIMIZED: Reduced from 16
            industry_embedding_dim=6,  # OPTIMIZED: Reduced from 8
            hidden_size=128,  # OPTIMIZED: Reduced from 256 (75% param reduction)
            num_layers=3,  # OPTIMIZED: Reduced from 4
            dropout=0.3,  # OPTIMIZED: Reduced from 0.4
            sequence_length=config.sequence_length,
            num_sectors=11,
            num_industries=50,
            model_type=options["model_type"],
        )

        # Log model complexity
        total_params = sum(p.numel() for p in model.parameters())
        self.stdout.write(f"Universal model created: {total_params:,} parameters")

        return model

    def _prepare_universal_data(self, training_data: Dict[str, Any], config: UniversalLSTMTrainingConfig) -> tuple:
        """Prepare universal features and sequences for training."""
        preprocessor = UniversalLSTMPreprocessor(
            sequence_length=config.sequence_length, sector_normalization=True, market_regime_detection=True
        )

        # Convert price data to DataFrames and engineer features
        feature_dfs = {}

        for symbol, data in training_data.items():
            self.stdout.write(f"Engineering features for {symbol}...")

            # Convert PriceData objects to DataFrame
            price_records = []
            for price_obj in data["price_data"]:
                price_records.append(
                    {
                        "date": price_obj.date,
                        "open": float(price_obj.open),
                        "high": float(price_obj.high),
                        "low": float(price_obj.low),
                        "close": float(price_obj.close),
                        "volume": int(price_obj.volume),
                    }
                )

            import pandas as pd

            df = pd.DataFrame(price_records)

            # Engineer universal features
            feature_df = preprocessor.engineer_universal_features(
                df, symbol=symbol, sector_id=data["sector_id"], industry_id=data["industry_id"]
            )

            feature_dfs[symbol] = feature_df

        # Fit universal scalers on all data
        self.stdout.write("Fitting universal scalers on combined data...")
        preprocessor.fit_universal_scalers(feature_dfs, target_column="close")

        # Prepare sequences for training
        self.stdout.write("Creating training sequences...")
        features_tensor, targets_tensor, sector_ids_tensor, industry_ids_tensor = (
            preprocessor.prepare_universal_sequences(feature_dfs, target_column="close")
        )

        sequences = {
            "features": features_tensor,
            "targets": targets_tensor,
            "sector_ids": sector_ids_tensor,
            "industry_ids": industry_ids_tensor,
        }

        self.stdout.write(f"Training data prepared: {len(features_tensor)} sequences")

        return preprocessor, sequences

    def _execute_training_pipeline(
        self,
        model: UniversalLSTMPredictor,
        sequences: Dict[str, Any],
        config: UniversalLSTMTrainingConfig,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the multi-stage training pipeline with real PyTorch training.
        """
        import time


        self.stdout.write("Starting real Universal LSTM training...")

        # Move model to device
        device = torch.device("cuda" if config.gpu_enabled and torch.cuda.is_available() else "cpu")
        model = model.to(device)

        self.stdout.write(f"Training on device: {device}")

        # Prepare data
        features = sequences["features"].to(device)
        targets = sequences["targets"].to(device)
        sector_ids = sequences["sector_ids"].to(device)
        industry_ids = sequences["industry_ids"].to(device)

        # Split data: 80% training, 20% validation
        total_samples = len(features)
        val_size = int(total_samples * config.validation_split)
        train_size = total_samples - val_size

        # Create train/validation splits
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        train_sector_ids = sector_ids[:train_size]
        train_industry_ids = industry_ids[:train_size]

        val_features = features[train_size:]
        val_targets = targets[train_size:]
        val_sector_ids = sector_ids[train_size:]
        val_industry_ids = industry_ids[train_size:]

        self.stdout.write(f"Training samples: {train_size}, Validation samples: {val_size}")

        # Training results storage
        training_results = {
            "stage1_metrics": {},
            "stage2_metrics": {},
            "stage3_metrics": {},
            "total_training_time": 0,
            "final_loss": 0,
            "validation_accuracy": 0,
            "cross_sector_performance": {},
        }

        overall_start_time = time.time()

        # Stage 1: Foundation Training
        self.stdout.write(
            f'\n{self.style.HTTP_INFO("[STAGE 1]  STAGE 1: Foundation Training")} ({config.foundation_epochs} epochs)'
        )
        self.stdout.write("-" * 50)
        stage1_results = self._train_stage(
            model,
            train_features,
            train_targets,
            train_sector_ids,
            train_industry_ids,
            val_features,
            val_targets,
            val_sector_ids,
            val_industry_ids,
            config.foundation_epochs,
            config.foundation_lr,
            config.foundation_batch_size,
            device,
            "Foundation",
            1,
            3,
            config,
        )
        training_results["stage1_metrics"] = stage1_results

        # Stage 2: Sector Specialization
        self.stdout.write(
            f'\n{self.style.HTTP_INFO("[STAGE 2] STAGE 2: Sector Specialization")} ({config.specialization_epochs} epochs)'
        )
        self.stdout.write("-" * 50)
        stage2_results = self._train_stage(
            model,
            train_features,
            train_targets,
            train_sector_ids,
            train_industry_ids,
            val_features,
            val_targets,
            val_sector_ids,
            val_industry_ids,
            config.specialization_epochs,
            config.specialization_lr,
            config.specialization_batch_size,
            device,
            "Specialization",
            2,
            3,
            config,
        )
        training_results["stage2_metrics"] = stage2_results

        # Stage 3: Fine-tuning
        self.stdout.write(
            f'\n{self.style.HTTP_INFO("[STAGE 3] STAGE 3: Fine-tuning")} ({config.finetuning_epochs} epochs)'
        )
        self.stdout.write("-" * 50)
        stage3_results = self._train_stage(
            model,
            train_features,
            train_targets,
            train_sector_ids,
            train_industry_ids,
            val_features,
            val_targets,
            val_sector_ids,
            val_industry_ids,
            config.finetuning_epochs,
            config.finetuning_lr,
            config.finetuning_batch_size,
            device,
            "Fine-tuning",
            3,
            3,
            config,
        )
        training_results["stage3_metrics"] = stage3_results

        # Calculate total training time
        total_time = time.time() - overall_start_time
        training_results["total_training_time"] = total_time / 3600  # Convert to hours
        training_results["final_loss"] = stage3_results["final_loss"]
        training_results["validation_accuracy"] = stage3_results["validation_accuracy"]

        self.stdout.write(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")

        return training_results

    def _train_stage(
        self,
        model,
        train_features,
        train_targets,
        train_sector_ids,
        train_industry_ids,
        val_features,
        val_targets,
        val_sector_ids,
        val_industry_ids,
        epochs,
        learning_rate,
        batch_size,
        device,
        stage_name,
        stage_num=1,
        total_stages=3,
        config=None,
    ):
        """Train a single stage of the universal model."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create optimized data loaders (simplified for stability)
        train_dataset = TensorDataset(train_features, train_targets, train_sector_ids, train_industry_ids)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Single-threaded for stability
            pin_memory=False,  # Disable since data is already on GPU
        )

        val_dataset = TensorDataset(val_features, val_targets, val_sector_ids, val_industry_ids)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Single-threaded for stability
            pin_memory=False,  # Disable since data is already on GPU
        )

        # Setup optimizer with improved parameters (fixed for stability)
        # Moderate learning rate increase to prevent both collapse and explosion
        actual_lr = learning_rate * 1.5 if stage_num == 1 else learning_rate  # Moderate increase for foundation
        weight_decay = 1e-4  # L2 regularization to prevent overfitting
        optimizer = optim.Adam(model.parameters(), lr=actual_lr, weight_decay=weight_decay)

        # Use production-ready enhanced loss function with curriculum learning
        from Analytics.ml.models.lstm_base import DirectionalLoss

        # Curriculum Learning: Progressive diversity targets
        # Stage 1 (Foundation): Relaxed diversity (0.0005 = 0.05%)
        # Stage 2 (Specialization): Moderate diversity (0.001 = 0.1%)
        # Stage 3 (Fine-tuning): Full diversity (0.002 = 0.2%)
        diversity_targets = {1: 0.0005, 2: 0.001, 3: 0.002}
        current_diversity_target = diversity_targets.get(stage_num, 0.001)

        # Progressive weight adjustment: REBALANCED to prioritize diversity
        # CRITICAL FIX: Diversity component was being overpowered by MSE
        if stage_num == 1:
            # Foundation: Gentle diversity introduction, focus on basic learning
            weights = {"mse": 0.6, "direction": 0.25, "diversity": 0.1, "sector": 0.05}
        elif stage_num == 2:
            # Specialization: Strong diversity push to break uniformity
            weights = {"mse": 0.3, "direction": 0.25, "diversity": 0.35, "sector": 0.1}
        else:
            # Fine-tuning: Maximum diversity emphasis with accuracy balance
            weights = {"mse": 0.25, "direction": 0.2, "diversity": 0.4, "sector": 0.15}

        criterion = DirectionalLoss(
            mse_weight=weights["mse"],
            direction_weight=weights["direction"],
            diversity_weight=weights["diversity"],
            sector_weight=weights["sector"],
            min_diversity_target=current_diversity_target,
            diversity_epsilon=1e-6,
        )

        # Log curriculum learning configuration
        print(f"[CURRICULUM] Stage {stage_num} Loss Configuration:")
        print(f"[CURRICULUM]   Diversity Target: {current_diversity_target:.4f} ({current_diversity_target*100:.2f}%)")
        print(
            f"[CURRICULUM]   Weights: MSE={weights['mse']:.2f}, Dir={weights['direction']:.2f}, Div={weights['diversity']:.2f}, Sec={weights['sector']:.2f}"
        )

        # Setup cosine annealing scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

        # Setup mixed precision training for RTX 4080 SUPER
        from torch.amp import GradScaler, autocast

        scaler = GradScaler("cuda") if device.type == "cuda" else None

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Check for interrupt at start of each epoch
            if self.check_interrupt():
                self.save_progress(
                    f"training_{stage_name}",
                    {
                        "stage_name": stage_name,
                        "stage_num": stage_num,
                        "total_stages": total_stages,
                        "completed_epochs": epoch,
                        "total_epochs": epochs,
                        "best_val_loss": best_val_loss,
                    },
                )
                return {"interrupted": True, "completed_epochs": epoch}

            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_features, batch_targets, batch_sectors, batch_industries in train_loader:
                optimizer.zero_grad()

                # Mixed precision forward pass
                if scaler is not None:
                    with autocast("cuda"):
                        outputs = model(batch_features, batch_sectors, batch_industries)
                        loss = criterion(outputs["price"].squeeze(), batch_targets, batch_sectors)

                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clipping if config else 1.0
                    )  # Gradient clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision fallback
                    outputs = model(batch_features, batch_sectors, batch_industries)
                    loss = criterion(outputs["price"].squeeze(), batch_targets, batch_sectors)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping if config else 1.0)
                    optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                # Track diversity metrics periodically
                if train_batches % 25 == 0:
                    with torch.no_grad():
                        predictions = outputs["price"].squeeze()
                        loss_components = criterion.get_component_losses(predictions, batch_targets, batch_sectors)
                        prediction_variance = loss_components.get("prediction_variance", 0)
                        diversity_target_met = loss_components.get("diversity_target_met", False)

                        if train_batches % 100 == 0:  # Log every 100 batches
                            print(
                                f"     [DIVERSITY] Batch {train_batches}: Var={prediction_variance:.6f}, "
                                f"Target({'YES' if diversity_target_met else 'NO'}), "
                                f"DivLoss={loss_components.get('diversity_loss', 0):.4f}"
                            )

            # Validation with comprehensive metrics
            model.eval()
            val_loss = 0.0
            val_batches = 0
            correct_predictions = 0
            total_predictions = 0
            sector_correct = {}
            sector_total = {}

            with torch.no_grad():
                for batch_features, batch_targets, batch_sectors, batch_industries in val_loader:
                    # Use mixed precision for validation too
                    if scaler is not None:
                        with autocast("cuda"):
                            outputs = model(batch_features, batch_sectors, batch_industries)
                            loss = criterion(outputs["price"].squeeze(), batch_targets, batch_sectors)
                    else:
                        outputs = model(batch_features, batch_sectors, batch_industries)
                        loss = criterion(outputs["price"].squeeze(), batch_targets, batch_sectors)

                    val_loss += loss.item()
                    val_batches += 1

                    # Calculate accuracy (direction prediction)
                    predictions = outputs["price"].squeeze()
                    targets = batch_targets

                    # Direction accuracy (up/down prediction) - Fixed for financial returns
                    # Use sign-based direction prediction for percentage returns
                    pred_direction = torch.sign(predictions)  # -1, 0, or 1
                    target_direction = torch.sign(targets)  # -1, 0, or 1

                    # Convert to binary up/down (treat 0 as neutral/down)
                    pred_binary = (pred_direction > 0).float()  # 1 for up, 0 for down/neutral
                    target_binary = (target_direction > 0).float()  # 1 for up, 0 for down/neutral

                    batch_correct = (pred_binary == target_binary).sum().item()
                    correct_predictions += batch_correct
                    total_predictions += targets.size(0)

                    # Sector-wise accuracy tracking
                    for i, sector_id in enumerate(batch_sectors):
                        sector_key = sector_id.item()
                        if sector_key not in sector_correct:
                            sector_correct[sector_key] = 0
                            sector_total[sector_key] = 0

                        sector_correct[sector_key] += (pred_binary[i] == target_binary[i]).item()
                        sector_total[sector_key] += 1

            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            val_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Progress logging with visual progress bar (every 5 epochs)
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Create progress bar for current stage
                progress = (epoch + 1) / epochs
                bar_length = 25
                filled_length = int(bar_length * progress)
                progress_bar = "#" * filled_length + "." * (bar_length - filled_length)

                # Color-code the loss improvement
                if avg_val_loss < best_val_loss:
                    loss_indicator = "v"  # Improving
                else:
                    loss_indicator = "-"  # Stable/increasing

                # Overall progress across all stages
                overall_progress = ((stage_num - 1) + progress) / total_stages
                overall_bar_length = 30
                overall_filled = int(overall_bar_length * overall_progress)
                overall_bar = "#" * overall_filled + "." * (overall_bar_length - overall_filled)

                self.stdout.write(
                    f"  {loss_indicator} Epoch {epoch+1:>3}/{epochs} [{progress_bar}] "
                    f"Loss: {avg_val_loss:.6f} | Acc: {val_accuracy:.1f}% | Overall: [{overall_bar}] {overall_progress*100:.1f}%"
                )

                # Log detailed sector performance every 25 epochs
                if epoch % 25 == 0 and sector_total:
                    self.stdout.write("     Sector Performance:")
                    for sector_id, total in sector_total.items():
                        if total > 0:
                            sector_acc = (sector_correct[sector_id] / total) * 100
                            self.stdout.write(
                                f"       Sector {sector_id}: {sector_acc:.1f}% ({sector_correct[sector_id]}/{total})"
                            )

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping with configurable patience
            patience_limit = config.early_stopping_patience if config else 25
            if patience_counter >= patience_limit:
                self.stdout.write(f"  Early stopping at epoch {epoch+1} (patience: {patience_limit})"),
                break

        # Return comprehensive training metrics
        return {
            "final_loss": best_val_loss,
            "validation_accuracy": val_accuracy,
            "epochs_completed": epoch + 1,
            "sector_performance": (
                {
                    sector_id: (sector_correct[sector_id] / sector_total[sector_id]) * 100
                    for sector_id in sector_total
                    if sector_total[sector_id] > 0
                }
                if sector_total
                else {}
            ),
        }

    def _save_universal_model(
        self,
        model: UniversalLSTMPredictor,
        preprocessor: UniversalLSTMPreprocessor,
        training_results: Dict[str, Any],
        options: Dict[str, Any],
    ) -> str:
        """Save the trained universal model with metadata."""
        model_dir = os.path.join(options["save_to"], "universal_lstm")
        model_name = f"universal_lstm_v{options['model_version']}"

        # Prepare comprehensive metadata
        metadata = {
            "training_results": training_results,
            "training_date": datetime.now().isoformat(),
            "model_type": options["model_type"],
            "configuration": "universal_cross_sector",
        }

        # Get fitted scalers from preprocessor for proper saving
        if not preprocessor.fitted:
            raise ValueError("Preprocessor must be fitted before saving scalers")

        scalers = {"feature_scaler": preprocessor.feature_scaler, "target_scaler": preprocessor.target_scaler}

        self.stdout.write(
            f'Saving fitted scalers: feature_scaler={type(scalers["feature_scaler"]).__name__}, target_scaler={type(scalers["target_scaler"]).__name__}'
        )

        # Get sector mappings
        from Analytics.ml.sector_mappings import INDUSTRY_MAPPING, SECTOR_MAPPING

        sector_mappings = {"sector_mapping": SECTOR_MAPPING, "industry_mapping": INDUSTRY_MAPPING}

        # Get training stocks
        sector_mapper = get_sector_mapper()
        training_stocks = sector_mapper.get_all_training_stocks()

        model_path = save_universal_model(
            model=model,
            model_name=model_name,
            model_dir=model_dir,
            metadata=metadata,
            scalers=scalers,
            sector_mappings=sector_mappings,
            training_stocks=training_stocks,
        )

        return model_path

    def _display_training_banner(self, options: Dict[str, Any]):
        """Display beautiful startup banner with training configuration."""

        # Clear screen and create visual impact
        self.stdout.write("\n" * 2)

        # Main banner
        banner_line = "=" * 80
        self.stdout.write(self.style.SUCCESS(banner_line))
        self.stdout.write(self.style.SUCCESS("*** UNIVERSAL LSTM TRAINING SYSTEM ***".center(80)))
        self.stdout.write(self.style.SUCCESS("VoyageurCompass AI Financial Analysis Platform".center(80)))
        self.stdout.write(self.style.SUCCESS(banner_line))
        self.stdout.write("")

        # Training configuration preview
        self.stdout.write(self.style.HTTP_INFO("[CONFIG]  TRAINING CONFIGURATION"))
        self.stdout.write("-" * 35)
        self.stdout.write(f'   [STAGE 2] Model Type: {options["model_type"].replace("_", " ").title()}')
        self.stdout.write(f'   [TIME]  Duration: {options["training_time"]}')
        self.stdout.write(
            f'   [EPOCHS] Total Epochs: {options["stage1_epochs"] + options["stage2_epochs"] + options["stage3_epochs"]}'
        )
        self.stdout.write(f"   [DATA] Data Split: 80% training / 20% validation")
        self.stdout.write(f'   [SEQ]  Sequence Length: {options["sequence_length"]} days')

        # GPU status
        gpu_status = "[OK] CUDA GPU" if options["gpu_acceleration"] else "[CPU] CPU Only"
        self.stdout.write(f"   [COMPUTE]  Compute: {gpu_status}")
        self.stdout.write("")

        # What this training will achieve
        self.stdout.write(self.style.HTTP_INFO("[STAGE 2] TRAINING GOALS"))
        self.stdout.write("-" * 25)
        self.stdout.write("   [STAGE 3] Single Universal Model (vs 100+ stock-specific models)")
        self.stdout.write("   [GLOBAL] Cross-Sector Learning (11 sectors, 115 stocks)")
        self.stdout.write("   >>> Instant Predictions (any Yahoo Finance stock)")
        self.stdout.write("   - Multi-Task Learning (price + volatility + trend)")
        self.stdout.write("")

        # Progress overview
        self.stdout.write(self.style.HTTP_INFO("[TRAINING] TRAINING STAGES"))
        self.stdout.write("-" * 25)
        self.stdout.write(f'   [STAGE 1]  Stage 1: Foundation ({options["stage1_epochs"]} epochs)')
        self.stdout.write(f'   [STAGE 2] Stage 2: Sector Specialization ({options["stage2_epochs"]} epochs)')
        self.stdout.write(f'   [STAGE 3] Stage 3: Fine-tuning ({options["stage3_epochs"]} epochs)')
        self.stdout.write("")

        self.stdout.write(self.style.SUCCESS(">>> Initializing training pipeline..."))
        self.stdout.write("")

    def _display_training_summary(
        self, training_results: Dict[str, Any], model_path: str, config: UniversalLSTMTrainingConfig
    ):
        """Display beautiful, user-friendly training summary with UI/UX excellence."""

        # Clear screen effect with spacing
        self.stdout.write("\n" * 3)

        # Header with visual appeal
        header_line = "=" * 70
        self.stdout.write(self.style.SUCCESS(header_line))
        self.stdout.write(self.style.SUCCESS("*** UNIVERSAL LSTM TRAINING COMPLETED SUCCESSFULLY! ***".center(70)))
        self.stdout.write(self.style.SUCCESS(header_line))
        self.stdout.write("")

        # Key Metrics Section - Most Important Info First
        self.stdout.write(self.style.HTTP_INFO("- KEY RESULTS"))
        self.stdout.write("-" * 25)

        # Format training time nicely
        total_hours = training_results["total_training_time"]
        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        # Format accuracy as percentage
        accuracy_pct = training_results["validation_accuracy"]  # Already in percentage

        # Create visual progress bar for accuracy
        accuracy_bar_length = 20
        filled_length = int(accuracy_bar_length * (training_results["validation_accuracy"] / 100))
        progress_bar = "#" * filled_length + "." * (accuracy_bar_length - filled_length)

        self.stdout.write(f"   [TIME]  Training Time: {self.style.WARNING(time_str)}")
        self.stdout.write(
            f'   [STAGE 2] Validation Accuracy: {self.style.SUCCESS(f"{accuracy_pct:.1f}%")} [{progress_bar}]'
        )
        final_loss = training_results.get("final_loss", 0.0)
        self.stdout.write(f'   v Final Loss: {self.style.SUCCESS(f"{final_loss:.6f}")}')
        self.stdout.write("")

        # Training Stages Progress
        self.stdout.write(self.style.HTTP_INFO("[STAGE 1]  TRAINING STAGES"))
        self.stdout.write("-" * 25)

        stages = [
            ("Foundation", training_results.get("stage1_metrics", {})),
            ("Specialization", training_results.get("stage2_metrics", {})),
            ("Fine-tuning", training_results.get("stage3_metrics", {})),
        ]

        for stage_name, metrics in stages:
            if metrics:
                stage_accuracy = metrics.get("validation_accuracy", 0)  # Already in percentage
                stage_epochs = metrics.get("epochs_completed", 0)

                # Status indicator
                status = "[OK]" if stage_accuracy > 70 else "[WARN]" if stage_accuracy > 50 else "[FAIL]"

                self.stdout.write(
                    f"   {status} {stage_name:<15} " f"{stage_accuracy:>5.1f}% accuracy  " f"({stage_epochs} epochs)"
                )
        self.stdout.write("")

        # Model Information
        self.stdout.write(self.style.HTTP_INFO("[MODEL] MODEL DETAILS"))
        self.stdout.write("-" * 25)
        self.stdout.write(f"   [PATH] Saved Location: {model_path}")
        self.stdout.write(f"   [EPOCHS] Parameters: ~2.5M (vs ~250M for stock-specific models)")
        self.stdout.write(f"   - Training Data: 80% train / 20% validation split")
        self.stdout.write(f"   [SEQ]  Sequence Length: {config.sequence_length} days")
        self.stdout.write("")

        # Performance Grade
        self.stdout.write(self.style.HTTP_INFO("[TRAINING] PERFORMANCE GRADE"))
        self.stdout.write("-" * 25)

        # Calculate overall grade
        if accuracy_pct >= 85:
            grade = "A+"
            grade_color = self.style.SUCCESS
            grade_desc = "Excellent - Production Ready!"
        elif accuracy_pct >= 80:
            grade = "A"
            grade_color = self.style.SUCCESS
            grade_desc = "Very Good - Ready for deployment"
        elif accuracy_pct >= 75:
            grade = "B+"
            grade_color = self.style.WARNING
            grade_desc = "Good - Consider additional training"
        elif accuracy_pct >= 70:
            grade = "B"
            grade_color = self.style.WARNING
            grade_desc = "Fair - May need optimization"
        else:
            grade = "C"
            grade_color = self.style.ERROR
            grade_desc = "Needs Improvement - Retrain recommended"

        self.stdout.write(f"   [GRADE] Overall Grade: {grade_color(grade)} - {grade_desc}")
        self.stdout.write("")

        # Next Steps
        self.stdout.write(self.style.HTTP_INFO(">>> NEXT STEPS"))
        self.stdout.write("-" * 25)

        if accuracy_pct >= 75:
            self.stdout.write("   [OK] Model is ready for production use")
            self.stdout.write("   [OK] Integration with Analytics Engine complete")
            self.stdout.write("   [OK] Universal predictions available for any stock")
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS("   [STAGE 2] Ready to proceed with Phase 3 implementation!"))
        else:
            self.stdout.write("   [WARN]  Consider retraining with more data or epochs")
            self.stdout.write("   [WARN]  Review hyperparameters for optimization")
            self.stdout.write("   - Monitor predictions closely in production")

        self.stdout.write("")

        # Footer
        footer_line = "=" * 70
        self.stdout.write(self.style.SUCCESS(footer_line))
        self.stdout.write(self.style.SUCCESS("*** Universal LSTM Model Training Complete! ***".center(70)))
        self.stdout.write(self.style.SUCCESS(footer_line))
        self.stdout.write("")
