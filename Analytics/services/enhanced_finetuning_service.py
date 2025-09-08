"""
Enhanced Fine-Tuning Service for VoyageurCompass.
Provides dataset generation, fine-tuning management, and evaluation capabilities.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import csv

from django.conf import settings
from django.core.cache import cache
from django.db import models
from Analytics.services.financial_fine_tuner import FinancialDomainFineTuner, FINE_TUNING_AVAILABLE
from Analytics.services.hybrid_analysis_coordinator import get_hybrid_analysis_coordinator
from Analytics.engine.ta_engine import TechnicalAnalysisEngine

logger = logging.getLogger(__name__)


class FineTuningManager:
    """Manages fine-tuning lifecycle and dataset generation."""

    def __init__(self):
        self.dataset_dir = Path(getattr(settings, 'FINE_TUNING_DATASET_DIR', './finetuning_datasets'))
        self.model_dir = Path(getattr(settings, 'FINE_TUNING_MODEL_DIR', './finetuned_models'))

        # Ensure directories exist
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training status tracking
        self.training_jobs = {}

        logger.info("Enhanced Fine-Tuning Service initialized")

    def generate_enhanced_dataset(self, 
                                 num_samples: int = 5000,
                                 include_sentiment: bool = True,
                                 quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Generate enhanced training dataset with sentiment integration.

        Args:
            num_samples: Number of training samples to generate
            include_sentiment: Whether to include sentiment analysis
            quality_threshold: Minimum quality threshold for samples

        Returns:
            Dataset generation results
        """
        try:
            logger.info(f"Generating enhanced dataset with {num_samples} samples")

            # Use hybrid coordinator for enhanced data generation
            hybrid_coordinator = get_hybrid_analysis_coordinator()
            ta_engine = TechnicalAnalysisEngine()

            # Sample stock symbols for diverse training data
            stock_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SHOP', 'SQ',
                'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'CVS', 'TMO', 'DHR',
                'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC'
            ]

            training_samples = []
            generation_start = time.time()

            for i in range(num_samples):
                try:
                    # Select random symbol and parameters
                    import random
                    symbol = random.choice(stock_symbols)
                    months = random.choice([3, 6, 9, 12])
                    detail_level = random.choice(['summary', 'standard', 'detailed'])

                    # Generate analysis data
                    analysis_data = self._generate_synthetic_analysis_data(symbol)

                    if include_sentiment:
                        # Generate enhanced explanation with sentiment
                        explanation_result = hybrid_coordinator.generate_enhanced_explanation(
                            analysis_data=analysis_data,
                            detail_level=detail_level
                        )
                    else:
                        # Generate standard explanation
                        from Analytics.services.local_llm_service import get_local_llm_service
                        llm_service = get_local_llm_service()
                        explanation_result = llm_service.generate_explanation(
                            analysis_data=analysis_data,
                            detail_level=detail_level
                        )

                    if explanation_result and 'content' in explanation_result:
                        # Create training sample
                        instruction = self._create_instruction_prompt(detail_level)

                        training_sample = {
                            'instruction': instruction,
                            'input': {
                                'symbol': symbol,
                                'score_0_10': analysis_data['score_0_10'],
                                'weighted_scores': analysis_data.get('weighted_scores', {}),
                                'detail_level': detail_level,
                                'analysis_context': self._create_analysis_context(analysis_data)
                            },
                            'output': explanation_result['content'],
                            'quality_score': self._calculate_sample_quality(explanation_result['content']),
                            'metadata': {
                                'generated_at': datetime.now().isoformat(),
                                'generation_time': explanation_result.get('generation_time', 0),
                                'model_used': explanation_result.get('model_used', 'unknown'),
                                'sentiment_enhanced': explanation_result.get('sentiment_enhanced', False),
                                'sample_id': f"{symbol}_{detail_level}_{i}"
                            }
                        }

                        # Filter by quality
                        if training_sample['quality_score'] >= quality_threshold:
                            training_samples.append(training_sample)

                    # Progress logging
                    if (i + 1) % 100 == 0:
                        logger.info(f"Generated {len(training_samples)} high-quality samples from {i + 1} attempts")

                except Exception as e:
                    logger.warning(f"Failed to generate sample {i}: {str(e)}")
                    continue

            generation_time = time.time() - generation_start

            # Save dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_filename = f"enhanced_financial_dataset_{timestamp}.json"
            dataset_path = self.dataset_dir / dataset_filename

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(training_samples, f, indent=2, ensure_ascii=False)

            # Create metadata
            metadata = {
                'dataset_path': str(dataset_path),
                'total_samples': len(training_samples),
                'generation_time': generation_time,
                'quality_threshold': quality_threshold,
                'include_sentiment': include_sentiment,
                'average_quality_score': sum(s['quality_score'] for s in training_samples) / len(training_samples) if training_samples else 0,
                'detail_level_distribution': self._analyze_detail_distribution(training_samples),
                'sentiment_enhanced_count': sum(1 for s in training_samples if s['metadata'].get('sentiment_enhanced', False)),
                'created_at': datetime.now().isoformat()
            }

            # Save metadata
            metadata_path = dataset_path.with_suffix('.meta.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Enhanced dataset generated: {len(training_samples)} samples in {generation_time:.2f}s")
            return metadata

        except Exception as e:
            logger.error(f"Enhanced dataset generation failed: {str(e)}")
            raise

    def _generate_synthetic_analysis_data(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic but realistic analysis data."""
        import random

        # Generate realistic score
        score = random.uniform(1.0, 10.0)

        # Generate weighted scores based on the main score
        base_strength = (score - 5.5) / 4.5  # Normalize to [-1, 1]

        weighted_scores = {
            'w_sma50vs200': base_strength * random.uniform(0.8, 1.2) * random.uniform(0.1, 0.3),
            'w_rsi14': (base_strength + random.uniform(-0.3, 0.3)) * random.uniform(0.05, 0.15),
            'w_macd12269': (base_strength + random.uniform(-0.4, 0.4)) * random.uniform(0.08, 0.18),
            'w_bbpos20': (base_strength + random.uniform(-0.2, 0.2)) * random.uniform(0.03, 0.12),
            'w_volsurge': random.uniform(-0.15, 0.15),
            'w_obv20': (base_strength + random.uniform(-0.5, 0.5)) * random.uniform(0.02, 0.10),
            'w_rel1y': (base_strength + random.uniform(-0.3, 0.3)) * random.uniform(0.05, 0.15),
            'w_candlerev': random.uniform(-0.10, 0.10)
        }

        return {
            'symbol': symbol,
            'score_0_10': round(score, 2),
            'weighted_scores': weighted_scores,
            'components': {},  # Would contain raw indicator values
            'analysis_date': datetime.now().isoformat(),
            'synthetic': True
        }

    def _create_instruction_prompt(self, detail_level: str) -> str:
        """Create instruction prompt for fine-tuning."""
        if detail_level == 'summary':
            return "Provide a concise investment recommendation (BUY/SELL/HOLD) for the given stock analysis data with the primary supporting reason."
        elif detail_level == 'detailed':
            return "Generate a comprehensive investment analysis including a clear recommendation, specific technical indicators, risk factors, and confidence level based on the provided analysis data."
        else:  # standard
            return "Analyze the stock data and provide an investment recommendation with key supporting technical factors and main risk considerations."

    def _create_analysis_context(self, analysis_data: Dict[str, Any]) -> str:
        """Create human-readable context from analysis data."""
        symbol = analysis_data['symbol']
        score = analysis_data['score_0_10']

        # Interpret score
        if score >= 8:
            score_desc = "very strong bullish signals"
        elif score >= 6.5:
            score_desc = "bullish momentum"
        elif score >= 5.5:
            score_desc = "slightly positive outlook"
        elif score >= 4:
            score_desc = "mixed signals"
        elif score >= 2:
            score_desc = "bearish indicators"
        else:
            score_desc = "very weak performance"

        context = f"{symbol} shows {score_desc} with a technical score of {score}/10."

        # Add top indicators
        weighted_scores = analysis_data.get('weighted_scores', {})
        if weighted_scores:
            top_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            if top_indicators:
                context += f" Key indicators: {', '.join([ind.replace('w_', '') for ind, _ in top_indicators])}"

        return context

    def _calculate_sample_quality(self, content: str) -> float:
        """Calculate quality score for training sample."""
        if not content:
            return 0.0

        quality_score = 0.0

        # Length check
        if 50 <= len(content) <= 500:
            quality_score += 0.3
        elif len(content) > 500:
            quality_score += 0.1

        # Recommendation check
        recommendations = ['BUY', 'SELL', 'HOLD', 'buy', 'sell', 'hold']
        if any(rec in content for rec in recommendations):
            quality_score += 0.3

        # Technical terms
        technical_terms = ['RSI', 'MACD', 'SMA', 'technical', 'analysis', 'momentum', 'trend']
        term_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        quality_score += min(0.3, term_count * 0.1)

        # Financial language
        financial_terms = ['investment', 'recommendation', 'risk', 'performance', 'signals']
        finance_count = sum(1 for term in financial_terms if term.lower() in content.lower())
        quality_score += min(0.1, finance_count * 0.02)

        return min(1.0, quality_score)

    def _analyze_detail_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of detail levels in dataset."""
        distribution = {'summary': 0, 'standard': 0, 'detailed': 0}
        for sample in samples:
            detail_level = sample['input'].get('detail_level', 'standard')
            distribution[detail_level] = distribution.get(detail_level, 0) + 1
        return distribution

    def start_fine_tuning_job(self, 
                             dataset_path: str,
                             job_name: str = None,
                             config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Start a fine-tuning job (simulated if dependencies unavailable).

        Args:
            dataset_path: Path to training dataset
            job_name: Name for the fine-tuning job
            config_overrides: Configuration overrides

        Returns:
            Job status and details
        """
        try:
            if not job_name:
                job_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            job_id = hashlib.md5(f"{job_name}_{time.time()}".encode(), usedforsecurity=False).hexdigest()[:12]

            if not FINE_TUNING_AVAILABLE:
                # Simulate fine-tuning job for demonstration
                logger.warning("Fine-tuning dependencies not available, creating simulated job")

                job_info = {
                    'job_id': job_id,
                    'job_name': job_name,
                    'status': 'simulated',
                    'dataset_path': dataset_path,
                    'started_at': datetime.now().isoformat(),
                    'estimated_completion': (datetime.now() + timedelta(hours=2)).isoformat(),
                    'config': config_overrides or {},
                    'simulated': True,
                    'message': 'Install transformers, peft, trl, datasets for actual fine-tuning'
                }

                self.training_jobs[job_id] = job_info
                return job_info

            # Real fine-tuning job
            logger.info(f"Starting fine-tuning job: {job_name}")

            # Initialize fine-tuner
            fine_tuner = FinancialDomainFineTuner()

            # Apply config overrides
            if config_overrides:
                fine_tuner.training_config.update(config_overrides)

            # Create job record
            job_info = {
                'job_id': job_id,
                'job_name': job_name,
                'status': 'starting',
                'dataset_path': dataset_path,
                'started_at': datetime.now().isoformat(),
                'config': fine_tuner.training_config.copy(),
                'simulated': False
            }

            self.training_jobs[job_id] = job_info

            # Start training in background (in real implementation, use Celery or similar)
            def run_training():
                try:
                    job_info['status'] = 'running'

                    # Run fine-tuning
                    results = fine_tuner.start_fine_tuning(
                        dataset_path=dataset_path,
                        output_dir=str(self.model_dir / job_name)
                    )

                    # Update job status
                    job_info.update({
                        'status': 'completed',
                        'completed_at': datetime.now().isoformat(),
                        'results': results,
                        'model_path': results.get('final_model_path')
                    })

                    logger.info(f"Fine-tuning job {job_name} completed successfully")

                except Exception as e:
                    logger.error(f"Fine-tuning job {job_name} failed: {str(e)}")
                    job_info.update({
                        'status': 'failed',
                        'error': str(e),
                        'failed_at': datetime.now().isoformat()
                    })

            # Note: In production, this should be submitted to a task queue
            import threading
            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()

            logger.info(f"Fine-tuning job {job_name} started with ID: {job_id}")
            return job_info

        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {str(e)}")
            raise

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a fine-tuning job."""
        return self.training_jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Any]:
        """List all fine-tuning jobs."""
        return {
            'jobs': list(self.training_jobs.values()),
            'total_jobs': len(self.training_jobs),
            'active_jobs': sum(1 for job in self.training_jobs.values() if job['status'] in ['running', 'starting']),
            'completed_jobs': sum(1 for job in self.training_jobs.values() if job['status'] == 'completed'),
            'failed_jobs': sum(1 for job in self.training_jobs.values() if job['status'] == 'failed')
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets."""
        datasets = []

        for dataset_file in self.dataset_dir.glob("*.json"):
            if dataset_file.name.endswith('.meta.json'):
                continue

            # Try to load metadata
            metadata_file = dataset_file.with_suffix('.meta.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Basic metadata from file
                metadata = {
                    'dataset_path': str(dataset_file),
                    'created_at': datetime.fromtimestamp(dataset_file.stat().st_mtime).isoformat(),
                    'file_size': dataset_file.stat().st_size
                }

            datasets.append(metadata)

        return sorted(datasets, key=lambda x: x.get('created_at', ''), reverse=True)

    def list_models(self) -> List[Dict[str, Any]]:
        """List fine-tuned models."""
        models = []

        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                # Look for adapter files or model files
                has_adapter = any(model_dir.glob("adapter_*.bin"))
                has_model = any(model_dir.glob("*.safetensors"))

                if has_adapter or has_model:
                    models.append({
                        'model_name': model_dir.name,
                        'model_path': str(model_dir),
                        'has_adapter': has_adapter,
                        'has_model': has_model,
                        'created_at': datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                        'size_mb': sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                    })

        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)

    def export_dataset_for_external_training(self, dataset_path: str, format: str = 'jsonl') -> str:
        """
        Export dataset in format suitable for external fine-tuning platforms.

        Args:
            dataset_path: Path to source dataset
            format: Export format ('jsonl', 'csv', 'huggingface')

        Returns:
            Path to exported dataset
        """
        try:
            logger.info(f"Exporting dataset to {format} format")

            # Load source dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create export filename
            base_name = Path(dataset_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == 'jsonl':
                export_path = self.dataset_dir / f"{base_name}_export_{timestamp}.jsonl"
                with open(export_path, 'w', encoding='utf-8') as f:
                    for sample in data:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            elif format == 'csv':
                export_path = self.dataset_dir / f"{base_name}_export_{timestamp}.csv"
                with open(export_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['instruction', 'input', 'output', 'quality_score'])
                    for sample in data:
                        writer.writerow([
                            sample['instruction'],
                            json.dumps(sample['input']),
                            sample['output'],
                            sample['quality_score']
                        ])

            elif format == 'huggingface':
                export_path = self.dataset_dir / f"{base_name}_hf_{timestamp}.json"
                hf_format = []
                for sample in data:
                    hf_format.append({
                        'text': f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{json.dumps(sample['input'])}\n\n### Response:\n{sample['output']}"
                    })

                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(hf_format, f, indent=2, ensure_ascii=False)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Dataset exported to: {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Failed to export dataset: {str(e)}")
            raise


# Singleton instance
_finetuning_manager = None


def get_finetuning_manager() -> FineTuningManager:
    """Get singleton instance of FineTuningManager."""
    global _finetuning_manager
    if _finetuning_manager is None:
        _finetuning_manager = FineTuningManager()
    return _finetuning_manager
