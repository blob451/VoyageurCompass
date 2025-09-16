"""
Fine-tuning Integration for Domain-specific FinBERT Models
Provides framework for training custom models on company-specific data
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Conditional imports for ML dependencies
try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = None
    DataLoader = None
    TORCH_AVAILABLE = False
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning process."""

    base_model: str = "ProsusAI/finbert"
    output_dir: str = "./models/finetuned"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 500
    max_length: int = 512
    early_stopping_patience: int = 3


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis fine-tuning."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            texts: List of input texts
            labels: List of labels (0=negative, 1=neutral, 2=positive)
            tokenizer: Pre-trained tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class FineTuningPipeline:
    """
    Pipeline for fine-tuning FinBERT models on domain-specific data.
    """

    def __init__(self, config: FineTuningConfig = None):
        """
        Initialize fine-tuning pipeline.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config or FineTuningConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # Label mappings
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Training history
        self.training_history = []
        self.validation_history = []

    def prepare_data(
        self, texts: List[str], labels: List[str], validation_split: float = 0.2
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.

        Args:
            texts: List of input texts
            labels: List of sentiment labels ('positive', 'negative', 'neutral')
            validation_split: Fraction of data for validation

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Convert labels to integers
        label_ids = [self.label_map[label.lower()] for label in labels]

        # Split data
        n_samples = len(texts)
        n_val = int(n_samples * validation_split)

        # Simple random split (in production, use stratified split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_texts = [texts[i] for i in train_indices]
        train_labels = [label_ids[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [label_ids[i] for i in val_indices]

        # Load tokenizer
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)

        logger.info(f"Prepared datasets: {len(train_dataset)} train, {len(val_dataset)} validation")

        return train_dataset, val_dataset

    def load_base_model(self):
        """Load base FinBERT model for fine-tuning."""
        try:
            logger.info(f"Loading base model: {self.config.base_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model,
                num_labels=3,  # negative, neutral, positive
                problem_type="single_label_classification",
            )

            logger.info("Base model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None) -> Dict[str, Any]:
        """
        Train the model on provided dataset.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Training results
        """
        if not self.model:
            self.load_base_model()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=50,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            greater_is_better=True,
            save_total_limit=3,
            push_to_hub=False,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics if val_dataset else None,
        )

        # Start training
        logger.info("Starting fine-tuning...")
        start_time = time.time()

        try:
            train_result = self.trainer.train()
            training_time = time.time() - start_time

            # Save model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)

            # Log results
            logger.info(
                f"Fine-tuning completed in {training_time:.2f} seconds",
                extra={
                    "event_type": "finetuning_complete",
                    "training_time": training_time,
                    "train_loss": train_result.training_loss,
                    "steps": train_result.global_step,
                },
            )

            # Evaluate on validation set if available
            eval_results = {}
            if val_dataset:
                eval_results = self.trainer.evaluate()
                logger.info(f"Validation results: {eval_results}")

            return {
                "training_loss": train_result.training_loss,
                "training_time": training_time,
                "eval_results": eval_results,
                "model_path": self.config.output_dir,
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise

    def evaluate_model(self, test_dataset: Dataset, model_path: str = None) -> Dict[str, Any]:
        """
        Evaluate trained model on test dataset.

        Args:
            test_dataset: Test dataset
            model_path: Path to trained model (optional)

        Returns:
            Evaluation results
        """
        if model_path and model_path != self.config.output_dir:
            # Load specific model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if not self.trainer:
            # Create trainer for evaluation
            self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer, compute_metrics=self.compute_metrics)

        # Run evaluation
        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)

        logger.info(f"Model evaluation results: {eval_results}")
        return eval_results

    def predict(self, texts: List[str], model_path: str = None) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to analyze
            model_path: Path to trained model (optional)

        Returns:
            List of prediction results
        """
        if model_path:
            # Load specific model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if not self.model or not self.tokenizer:
            raise ValueError("No model loaded for prediction")

        # Prepare data
        inputs = self.tokenizer(
            texts, truncation=True, padding=True, max_length=self.config.max_length, return_tensors="pt"
        )

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to results
        results = []
        for i, text in enumerate(texts):
            probs = predictions[i].cpu().numpy()
            predicted_label_id = np.argmax(probs)
            predicted_label = self.reverse_label_map[predicted_label_id]
            confidence = float(probs[predicted_label_id])

            # Convert to sentiment score (-1 to 1)
            if predicted_label == "positive":
                score = confidence
            elif predicted_label == "negative":
                score = -confidence
            else:
                score = 0.0

            results.append(
                {
                    "text": text,
                    "sentimentLabel": predicted_label,
                    "sentimentScore": score,
                    "sentimentConfidence": confidence,
                    "probabilities": {
                        "negative": float(probs[0]),
                        "neutral": float(probs[1]),
                        "positive": float(probs[2]),
                    },
                }
            )

        return results

    def export_model_info(self, output_path: str):
        """
        Export model information and metrics.

        Args:
            output_path: Path to save model info
        """
        model_info = {
            "base_model": self.config.base_model,
            "fine_tuned_model": self.config.output_dir,
            "configuration": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "max_length": self.config.max_length,
            },
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "label_mapping": self.label_map,
            "created_at": time.time(),
        }

        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model info exported to {output_path}")


class FineTuningDataCollector:
    """
    Utility class for collecting and preparing fine-tuning data.
    """

    @staticmethod
    def load_from_csv(file_path: str, text_column: str, label_column: str) -> Tuple[List[str], List[str]]:
        """
        Load training data from CSV file.

        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Tuple of (texts, labels)
        """
        import pandas as pd

        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        return texts, labels

    @staticmethod
    def create_synthetic_data(symbol: str, count: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Create synthetic financial sentiment data for testing.

        Args:
            symbol: Stock symbol to generate data for
            count: Number of samples to generate

        Returns:
            Tuple of (texts, labels)
        """
        import random

        # Template sentences with placeholders
        positive_templates = [
            f"{symbol} reports strong quarterly earnings with revenue growth",
            f"{symbol} stock surges after positive analyst upgrade",
            f"{symbol} announces strategic partnership that boosts investor confidence",
            f"{symbol} beats earnings expectations for the third consecutive quarter",
            f"{symbol} sees increased market share in key segments",
        ]

        negative_templates = [
            f"{symbol} faces regulatory challenges that impact operations",
            f"{symbol} stock declines following disappointing earnings",
            f"{symbol} warns of potential headwinds in upcoming quarters",
            f"{symbol} experiences supply chain disruptions affecting production",
            f"{symbol} misses analyst expectations amid market volatility",
        ]

        neutral_templates = [
            f"{symbol} announces routine quarterly dividend payment",
            f"{symbol} schedules annual shareholder meeting",
            f"{symbol} files standard regulatory reports with SEC",
            f"{symbol} maintains current guidance for fiscal year",
            f"{symbol} participates in industry conference",
        ]

        texts = []
        labels = []

        samples_per_class = count // 3

        # Generate positive samples
        for _ in range(samples_per_class):
            texts.append(random.choice(positive_templates))
            labels.append("positive")

        # Generate negative samples
        for _ in range(samples_per_class):
            texts.append(random.choice(negative_templates))
            labels.append("negative")

        # Generate neutral samples
        for _ in range(count - 2 * samples_per_class):
            texts.append(random.choice(neutral_templates))
            labels.append("neutral")

        # Shuffle data
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        return list(texts), list(labels)
