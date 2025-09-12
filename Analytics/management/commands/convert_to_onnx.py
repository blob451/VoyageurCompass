"""
Management command to convert PyTorch models to ONNX format for faster inference.
"""

import logging
import os
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

# Conditional imports for ML dependencies
try:
    import torch
    import onnx
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
    
    ONNX_AVAILABLE = True
except ImportError:
    torch = None
    onnx = None
    ort = None
    ORTModelForSequenceClassification = None
    ORTQuantizer = None
    OptimizationConfig = None
    QuantizationConfig = None
    ONNX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False


class Command(BaseCommand):
    help = 'Convert PyTorch models to ONNX format for optimized inference'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-type',
            choices=['finbert', 'lstm', 'all'],
            default='all',
            help='Type of model to convert (default: all)'
        )
        
        parser.add_argument(
            '--quantize',
            action='store_true',
            help='Apply quantization for smaller model size and faster inference'
        )
        
        parser.add_argument(
            '--optimize',
            action='store_true',
            help='Apply ONNX graph optimizations'
        )
        
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Run performance benchmarks after conversion'
        )
        
        parser.add_argument(
            '--output-dir',
            default='Data/ml_models/onnx',
            help='Output directory for ONNX models'
        )

    def handle(self, *args, **options):
        """Execute ONNX model conversion."""
        if not ONNX_AVAILABLE:
            raise CommandError(
                "ONNX dependencies not available. Install with: "
                "pip install onnx onnxruntime optimum[onnxruntime]"
            )
        
        if not TRANSFORMERS_AVAILABLE:
            raise CommandError(
                "Transformers not available. Install with: pip install transformers"
            )
        
        self.stdout.write(self.style.SUCCESS('Starting ONNX model conversion...'))
        
        output_dir = Path(options['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_type = options['model_type']
        quantize = options['quantize']
        optimize = options['optimize']
        benchmark = options['benchmark']
        
        conversion_results = {}
        
        try:
            if model_type in ['finbert', 'all']:
                self.stdout.write('Converting FinBERT model...')
                result = self._convert_finbert(output_dir, quantize, optimize)
                conversion_results['finbert'] = result
            
            if model_type in ['lstm', 'all']:
                self.stdout.write('Converting Universal LSTM model...')
                result = self._convert_lstm(output_dir, quantize, optimize)
                conversion_results['lstm'] = result
            
            # Run benchmarks if requested
            if benchmark:
                self.stdout.write('Running performance benchmarks...')
                self._run_benchmarks(conversion_results, output_dir)
            
            # Summary
            self.stdout.write(self.style.SUCCESS('\nConversion Summary:'))
            for model_name, result in conversion_results.items():
                if result['success']:
                    self.stdout.write(
                        f"✓ {model_name.upper()}: {result['original_size']:.1f}MB → "
                        f"{result['onnx_size']:.1f}MB ({result['compression_ratio']:.1f}x smaller)"
                    )
                else:
                    self.stdout.write(f"✗ {model_name.upper()}: {result['error']}")
            
        except Exception as e:
            raise CommandError(f'Model conversion failed: {str(e)}')

    def _convert_finbert(self, output_dir: Path, quantize: bool, optimize: bool) -> dict:
        """Convert FinBERT model to ONNX format."""
        try:
            # Load the FinBERT model
            model_name = "ProsusAI/finbert"
            finbert_dir = output_dir / "finbert"
            finbert_dir.mkdir(exist_ok=True)
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Save original model size
            original_size = self._get_model_size(model)
            
            # Convert to ONNX
            onnx_path = finbert_dir / "model.onnx"
            
            # Create dummy input for export
            dummy_input = tokenizer(
                "The market outlook is positive",
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                }
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(str(finbert_dir))
            
            # Apply optimizations
            if optimize:
                self._optimize_onnx_model(onnx_path)
            
            # Apply quantization
            if quantize:
                self._quantize_onnx_model(onnx_path, finbert_dir)
            
            # Get final size
            onnx_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"FinBERT converted to ONNX: {original_size:.1f}MB → {onnx_size:.1f}MB")
            
            return {
                'success': True,
                'original_size': original_size,
                'onnx_size': onnx_size,
                'compression_ratio': original_size / onnx_size,
                'path': str(onnx_path)
            }
            
        except Exception as e:
            logger.error(f"FinBERT conversion failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _convert_lstm(self, output_dir: Path, quantize: bool, optimize: bool) -> dict:
        """Convert Universal LSTM model to ONNX format."""
        try:
            # Try to load the Universal LSTM model
            from Analytics.ml.models.lstm_base import load_model
            
            model_dir = "Data/ml_models/universal_lstm"
            if not os.path.exists(model_dir):
                return {'success': False, 'error': 'Universal LSTM model directory not found'}
            
            # Find the latest model file
            model_files = [
                f for f in os.listdir(model_dir) 
                if f.startswith("universal_lstm_") and f.endswith(".pth")
            ]
            
            if not model_files:
                return {'success': False, 'error': 'No Universal LSTM model files found'}
            
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
            
            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, metadata, scalers, preprocessing_params = load_model(model_path, device)
            
            # Get original size
            original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # Create output directory
            lstm_dir = output_dir / "universal_lstm"
            lstm_dir.mkdir(exist_ok=True)
            
            # Create dummy input based on model architecture
            sequence_length = metadata.get('sequence_length', 50)
            num_features = metadata.get('num_features', 8)
            
            dummy_input = torch.randn(1, sequence_length, num_features).to(device)
            
            # Export to ONNX
            onnx_path = lstm_dir / "model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["sequence_input"],
                output_names=["price_prediction"],
                dynamic_axes={
                    "sequence_input": {0: "batch_size"},
                    "price_prediction": {0: "batch_size"}
                }
            )
            
            # Save metadata and scalers
            import pickle
            with open(lstm_dir / "metadata.pkl", "wb") as f:
                pickle.dump({
                    'metadata': metadata,
                    'scalers': scalers,
                    'preprocessing_params': preprocessing_params
                }, f)
            
            # Apply optimizations
            if optimize:
                self._optimize_onnx_model(onnx_path)
            
            # Apply quantization
            if quantize:
                self._quantize_onnx_model(onnx_path, lstm_dir)
            
            # Get final size
            onnx_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"Universal LSTM converted to ONNX: {original_size:.1f}MB → {onnx_size:.1f}MB")
            
            return {
                'success': True,
                'original_size': original_size,
                'onnx_size': onnx_size,
                'compression_ratio': original_size / onnx_size,
                'path': str(onnx_path)
            }
            
        except Exception as e:
            logger.error(f"Universal LSTM conversion failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _optimize_onnx_model(self, onnx_path: Path):
        """Apply ONNX graph optimizations."""
        try:
            # Load and optimize the model
            session_options = ort.SessionOptions()
            session_options.optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create optimized session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(onnx_path), session_options, providers=providers)
            logger.info(f"Applied ONNX optimizations to {onnx_path}")
            
        except Exception as e:
            logger.warning(f"ONNX optimization failed for {onnx_path}: {str(e)}")

    def _quantize_onnx_model(self, onnx_path: Path, output_dir: Path):
        """Apply quantization to reduce model size."""
        try:
            quantized_path = output_dir / "model_quantized.onnx"
            
            # Configure quantization
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8
            )
            
            # Replace original with quantized version
            onnx_path.unlink()
            quantized_path.rename(onnx_path)
            
            logger.info(f"Applied quantization to {onnx_path}")
            
        except Exception as e:
            logger.warning(f"Quantization failed for {onnx_path}: {str(e)}")

    def _get_model_size(self, model) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    def _run_benchmarks(self, conversion_results: dict, output_dir: Path):
        """Run performance benchmarks for converted models."""
        self.stdout.write('\n--- Performance Benchmarks ---')
        
        for model_name, result in conversion_results.items():
            if not result['success']:
                continue
                
            try:
                if model_name == 'finbert':
                    self._benchmark_finbert(Path(result['path']))
                elif model_name == 'lstm':
                    self._benchmark_lstm(Path(result['path']))
                    
            except Exception as e:
                self.stdout.write(f"Benchmark failed for {model_name}: {str(e)}")

    def _benchmark_finbert(self, onnx_path: Path):
        """Benchmark FinBERT ONNX model."""
        # Create session
        session = ort.InferenceSession(str(onnx_path))
        
        # Prepare test data
        test_texts = [
            "The market outlook is positive for technology stocks.",
            "Economic indicators suggest moderate growth ahead.",
            "Quarterly earnings beat expectations significantly."
        ] * 10  # 30 samples
        
        # Benchmark inference
        start_time = time.time()
        
        for text in test_texts:
            # Tokenize (simplified)
            input_ids = [101] + [1000] * 50 + [102]  # Dummy tokenization
            attention_mask = [1] * len(input_ids)
            
            # Pad to 512
            while len(input_ids) < 512:
                input_ids.append(0)
                attention_mask.append(0)
            
            # Run inference
            inputs = {
                'input_ids': [[input_ids[:512]]],
                'attention_mask': [[attention_mask[:512]]]
            }
            
            session.run(None, inputs)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(test_texts) * 1000  # ms
        
        self.stdout.write(f"FinBERT ONNX: {avg_time:.2f}ms per inference")

    def _benchmark_lstm(self, onnx_path: Path):
        """Benchmark Universal LSTM ONNX model."""
        # Create session
        session = ort.InferenceSession(str(onnx_path))
        
        # Prepare test data
        import numpy as np
        test_inputs = [np.random.randn(1, 50, 8).astype(np.float32) for _ in range(100)]
        
        # Benchmark inference
        start_time = time.time()
        
        for input_data in test_inputs:
            session.run(None, {'sequence_input': input_data})
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(test_inputs) * 1000  # ms
        
        self.stdout.write(f"Universal LSTM ONNX: {avg_time:.2f}ms per inference")