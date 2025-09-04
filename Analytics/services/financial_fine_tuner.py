"""
Financial Domain Fine-Tuner for LLaMA models using LoRA.
Implements Parameter-Efficient Fine-Tuning for financial explanation generation.
"""

import json
import logging
import os
import time
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# Fine-tuning dependencies (these would need to be installed)
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from trl import SFTTrainer
    from datasets import Dataset, load_dataset
    import wandb
    FINE_TUNING_AVAILABLE = True
except ImportError:
    # Create dummy classes for type hints when dependencies unavailable
    class Dataset:
        pass
    FINE_TUNING_AVAILABLE = False

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


class FinancialDomainFineTuner:
    """
    Financial domain fine-tuner using LoRA (Low-Rank Adaptation).
    Optimized for efficient fine-tuning of LLaMA models for financial explanation generation.
    """
    
    def __init__(self, base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the financial domain fine-tuner.
        
        Args:
            base_model_path: Path to the base LLaMA model
        """
        if not FINE_TUNING_AVAILABLE:
            raise ImportError(
                "Fine-tuning dependencies not available. "
                "Install: pip install transformers peft trl datasets wandb accelerate bitsandbytes"
            )
        
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # LoRA configuration optimized for financial domain
        self.lora_config = LoraConfig(
            r=16,                              # Low-rank dimension
            lora_alpha=32,                     # Scaling parameter
            target_modules=[                   # Target attention modules
                "q_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,                  # Dropout for regularization
            bias="none",                       # No bias adaptation
            task_type=TaskType.CAUSAL_LM,      # Causal language modeling
            inference_mode=False               # Training mode
        )
        
        # Training configuration
        self.training_config = {
            'output_dir': './financial-llama-lora',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'warmup_steps': 100,
            'learning_rate': 2e-4,
            'fp16': torch.cuda.is_available(),
            'logging_steps': 10,
            'evaluation_strategy': "steps",
            'eval_steps': 500,
            'save_strategy': "steps",
            'save_steps': 500,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'dataloader_num_workers': 4,
            'remove_unused_columns': False,
            'gradient_accumulation_steps': 2,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'adam_epsilon': 1e-8,
            'lr_scheduler_type': 'cosine',
            'save_total_limit': 3,
            'prediction_loss_only': True
        }
        
        # Quality validation metrics
        self.quality_metrics = {
            'recommendation_keywords': ['BUY', 'SELL', 'HOLD', 'buy', 'sell', 'hold'],
            'technical_indicators': ['RSI', 'MACD', 'SMA', 'Bollinger', 'Volume', 'Support', 'Resistance'],
            'financial_terms': ['analysis', 'recommendation', 'technical', 'momentum', 'trend', 'signals'],
            'min_explanation_length': 50,
            'max_explanation_length': 500
        }
        
        logger.info("Financial Domain Fine-Tuner initialized")
    
    def load_base_model(self):
        """Load the base LLaMA model and tokenizer."""
        try:
            logger.info(f"Loading base model: {self.base_model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right"  # Important for training
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization for training
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'device_map': 'auto' if torch.cuda.is_available() else None,
            }
            
            # Use 4-bit quantization if available for memory efficiency
            try:
                from transformers import BitsAndBytesConfig
                if torch.cuda.is_available():
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs['quantization_config'] = bnb_config
                    logger.info("Using 4-bit quantization for memory efficiency")
            except ImportError:
                logger.info("BitsAndBytes not available, using standard precision")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            logger.info("Base model loaded successfully with LoRA adaptation")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset_path: str, validation_split: float = 0.15) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets from instruction data.
        
        Args:
            dataset_path: Path to the instruction dataset
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            
            # Load dataset
            if dataset_path.endswith('.jsonl'):
                raw_dataset = load_dataset('json', data_files=dataset_path)['train']
            else:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                raw_dataset = Dataset.from_list(data)
            
            logger.info(f"Loaded {len(raw_dataset)} instruction samples")
            
            # Filter and validate samples
            valid_samples = []
            for sample in raw_dataset:
                if self._validate_instruction_sample(sample):
                    valid_samples.append(sample)
            
            logger.info(f"Validated {len(valid_samples)} high-quality samples")
            
            # Convert to Hugging Face dataset
            dataset = Dataset.from_list(valid_samples)
            
            # Format samples for training
            def format_instruction_sample(sample):
                """Format instruction sample for fine-tuning."""
                instruction = sample['instruction']
                input_data = json.dumps(sample['input'], separators=(',', ':'))
                output = sample['output']
                
                # Create formatted text for training
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Response:\n{output}"
                
                return {'text': text}
            
            formatted_dataset = dataset.map(format_instruction_sample, remove_columns=dataset.column_names)
            
            # Split into train and validation
            if validation_split > 0:
                split_dataset = formatted_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = split_dataset['train']
                eval_dataset = split_dataset['test']
            else:
                train_dataset = formatted_dataset
                eval_dataset = None
            
            logger.info(f"Prepared {len(train_dataset)} training samples")
            if eval_dataset:
                logger.info(f"Prepared {len(eval_dataset)} validation samples")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {str(e)}")
            raise
    
    def _validate_instruction_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate instruction sample quality.
        
        Args:
            sample: Instruction sample to validate
            
        Returns:
            True if sample meets quality criteria
        """
        try:
            # Check required fields
            required_fields = ['instruction', 'input', 'output']
            if not all(field in sample for field in required_fields):
                return False
            
            output = sample['output']
            
            # Length validation
            if len(output) < self.quality_metrics['min_explanation_length']:
                return False
            if len(output) > self.quality_metrics['max_explanation_length']:
                return False
            
            # Check for recommendation
            has_recommendation = any(
                keyword in output 
                for keyword in self.quality_metrics['recommendation_keywords']
            )
            
            # Check for technical content
            has_technical_content = any(
                term.lower() in output.lower() 
                for term in self.quality_metrics['technical_indicators'] + self.quality_metrics['financial_terms']
            )
            
            return has_recommendation and has_technical_content
            
        except Exception:
            return False
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Create SFT trainer for financial domain fine-tuning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional validation dataset
        """
        try:
            # Training arguments
            training_args = TrainingArguments(
                **self.training_config,
                run_name=f"financial-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=self.lora_config,
                dataset_text_field="text",
                max_seq_length=1024,
                tokenizer=self.tokenizer,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False  # Causal language modeling
                ),
                packing=False,  # Don't pack sequences
            )
            
            logger.info("SFT Trainer created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {str(e)}")
            raise
    
    def start_fine_tuning(self, 
                         dataset_path: str,
                         output_dir: Optional[str] = None,
                         use_wandb: bool = True) -> Dict[str, Any]:
        """
        Start the fine-tuning process.
        
        Args:
            dataset_path: Path to instruction dataset
            output_dir: Output directory for checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting financial domain fine-tuning process")
            
            # Initialize Weights & Biases if requested
            if use_wandb:
                try:
                    wandb.init(
                        project="financial-llama-finetune",
                        name=f"financial-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        config={
                            'base_model': self.base_model_path,
                            'lora_config': self.lora_config.__dict__,
                            'training_config': self.training_config
                        }
                    )
                    logger.info("Weights & Biases initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize W&B: {str(e)}")
                    use_wandb = False
            
            # Load base model if not loaded
            if self.model is None:
                self.load_base_model()
            
            # Prepare datasets
            train_dataset, eval_dataset = self.prepare_dataset(dataset_path)
            
            # Update output directory if provided
            if output_dir:
                self.training_config['output_dir'] = output_dir
            
            # Create trainer
            self.create_trainer(train_dataset, eval_dataset)
            
            # Start training
            logger.info("Beginning fine-tuning process...")
            start_time = time.time()
            
            training_result = self.trainer.train()
            
            training_time = time.time() - start_time
            logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(self.training_config['output_dir'], 'final_model')
            self.trainer.save_model(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
            
            # Evaluate if validation dataset available
            eval_results = {}
            if eval_dataset:
                logger.info("Running final evaluation...")
                eval_results = self.trainer.evaluate()
                logger.info(f"Final evaluation results: {eval_results}")
            
            # Compile results
            results = {
                'training_completed': True,
                'training_time_seconds': training_time,
                'final_model_path': final_model_path,
                'training_history': training_result.log_history if hasattr(training_result, 'log_history') else [],
                'evaluation_results': eval_results,
                'training_config': self.training_config,
                'lora_config': self.lora_config.__dict__,
                'dataset_size': len(train_dataset),
                'validation_size': len(eval_dataset) if eval_dataset else 0,
                'completion_time': datetime.now().isoformat()
            }
            
            # Close W&B run
            if use_wandb:
                wandb.finish()
            
            logger.info("Financial domain fine-tuning process completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Fine-tuning process failed: {str(e)}")
            if use_wandb:
                wandb.finish()
            raise
    
    def load_fine_tuned_model(self, adapter_path: str):
        """
        Load a fine-tuned LoRA adapter.
        
        Args:
            adapter_path: Path to the LoRA adapter
        """
        try:
            logger.info(f"Loading fine-tuned adapter from: {adapter_path}")
            
            # Load base model if not loaded
            if self.model is None:
                self.load_base_model()
            
            # Load adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False  # For inference
            )
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {str(e)}")
            raise
    
    def generate_explanation(self, 
                           analysis_data: Dict[str, Any],
                           detail_level: str = 'standard',
                           max_new_tokens: int = 200) -> Optional[str]:
        """
        Generate explanation using fine-tuned model.
        
        Args:
            analysis_data: Technical analysis data
            detail_level: Level of detail for explanation
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated explanation or None if failed
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Call load_base_model() or load_fine_tuned_model() first.")
            return None
        
        try:
            # Format input
            symbol = analysis_data.get('symbol', 'UNKNOWN')
            score = analysis_data.get('score_0_10', 0)
            
            instruction = f"Generate a comprehensive investment analysis for {symbol} with score {score}/10 based on the provided technical indicators."
            input_data = json.dumps(analysis_data, separators=(',', ':'))
            
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Response:\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate with fine-tuned model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            response_start = generated_text.find("### Response:\n") + len("### Response:\n")
            if response_start > len("### Response:\n"):
                explanation = generated_text[response_start:].strip()
            else:
                explanation = generated_text.strip()
            
            logger.info(f"Generated explanation for {symbol} using fine-tuned model")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation with fine-tuned model: {str(e)}")
            return None
    
    def evaluate_model_quality(self, test_dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate fine-tuned model quality on test dataset.
        
        Args:
            test_dataset_path: Path to test dataset
            
        Returns:
            Quality evaluation metrics
        """
        try:
            logger.info("Evaluating fine-tuned model quality...")
            
            # Load test dataset
            with open(test_dataset_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # Sample subset for evaluation (to save time)
            test_samples = test_data[:min(100, len(test_data))]
            
            quality_scores = {
                'recommendation_accuracy': 0,
                'technical_coverage': 0,
                'content_quality': 0,
                'total_samples': len(test_samples),
                'successful_generations': 0
            }
            
            for i, sample in enumerate(test_samples):
                try:
                    # Generate explanation
                    generated_explanation = self.generate_explanation(
                        sample['input'],
                        max_new_tokens=300
                    )
                    
                    if not generated_explanation:
                        continue
                    
                    quality_scores['successful_generations'] += 1
                    
                    # Evaluate recommendation accuracy
                    expected_output = sample['output']
                    if self._has_consistent_recommendation(generated_explanation, expected_output):
                        quality_scores['recommendation_accuracy'] += 1
                    
                    # Evaluate technical coverage
                    if self._has_technical_coverage(generated_explanation, sample['input']):
                        quality_scores['technical_coverage'] += 1
                    
                    # Evaluate content quality
                    if self._assess_content_quality(generated_explanation):
                        quality_scores['content_quality'] += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Evaluated {i + 1}/{len(test_samples)} samples...")
                        
                except Exception as e:
                    logger.error(f"Error evaluating sample {i}: {str(e)}")
                    continue
            
            # Calculate final scores
            if quality_scores['successful_generations'] > 0:
                for metric in ['recommendation_accuracy', 'technical_coverage', 'content_quality']:
                    quality_scores[f'{metric}_rate'] = (
                        quality_scores[metric] / quality_scores['successful_generations']
                    )
                
                quality_scores['overall_quality_score'] = (
                    quality_scores['recommendation_accuracy_rate'] * 0.4 +
                    quality_scores['technical_coverage_rate'] * 0.3 +
                    quality_scores['content_quality_rate'] * 0.3
                )
            
            logger.info(f"Quality evaluation complete: {quality_scores}")
            return quality_scores
            
        except Exception as e:
            logger.error(f"Error during quality evaluation: {str(e)}")
            raise
    
    def _has_consistent_recommendation(self, generated: str, expected: str) -> bool:
        """Check if generated text has consistent recommendation with expected."""
        gen_rec = self._extract_recommendation(generated)
        exp_rec = self._extract_recommendation(expected)
        return gen_rec == exp_rec if gen_rec and exp_rec else False
    
    def _extract_recommendation(self, text: str) -> Optional[str]:
        """Extract recommendation from text."""
        text_upper = text.upper()
        if 'STRONG BUY' in text_upper:
            return 'STRONG BUY'
        elif 'STRONG SELL' in text_upper:
            return 'STRONG SELL'
        elif 'BUY' in text_upper:
            return 'BUY'
        elif 'SELL' in text_upper:
            return 'SELL'
        elif 'HOLD' in text_upper:
            return 'HOLD'
        return None
    
    def _has_technical_coverage(self, generated: str, input_data: Dict[str, Any]) -> bool:
        """Check if generated text covers technical indicators from input."""
        weighted_scores = input_data.get('weighted_scores', {})
        if not weighted_scores:
            return True
        
        # Get top indicators
        top_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        coverage_count = 0
        for indicator, _ in top_indicators:
            indicator_name = indicator.replace('w_', '').replace('_', ' ')
            if any(term in generated.lower() for term in [
                indicator_name.lower(), 
                'rsi' if 'rsi' in indicator_name.lower() else '',
                'sma' if 'sma' in indicator_name.lower() else '',
                'macd' if 'macd' in indicator_name.lower() else ''
            ] if term):
                coverage_count += 1
        
        return coverage_count >= 1  # At least one indicator mentioned
    
    def _assess_content_quality(self, text: str) -> bool:
        """Assess overall content quality."""
        if len(text) < 50:
            return False
        
        # Check for professional financial language
        financial_terms_present = sum(1 for term in self.quality_metrics['financial_terms'] 
                                    if term.lower() in text.lower())
        
        return financial_terms_present >= 2


# Factory function for easy instantiation
def create_financial_fine_tuner(base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct") -> FinancialDomainFineTuner:
    """
    Create a financial domain fine-tuner instance.
    
    Args:
        base_model_path: Path to base LLaMA model
        
    Returns:
        FinancialDomainFineTuner instance
    """
    return FinancialDomainFineTuner(base_model_path)