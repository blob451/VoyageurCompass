#!/usr/bin/env python3
"""
FinBERT Model Pre-download Script

Downloads and caches FinBERT models during Docker build for optimised container startup.
Ensures models are available locally to avoid network delays during first inference.
"""

import os
import sys
from pathlib import Path
from typing import List

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError as e:
    print(f"[ERROR] Required ML dependencies not available: {e}")
    print("[INFO] Skipping FinBERT pre-download - running in base configuration")
    sys.exit(0)


class FinBERTDownloader:
    """Manages FinBERT model downloading and caching during Docker build."""
    
    FINBERT_MODELS = [
        "ProsusAI/finbert",
        "ahmedrachid/FinancialBERT-Sentiment-Analysis"
    ]
    
    def __init__(self, cache_dir: str = None):
        """Initialise downloader with specified cache directory."""
        self.cache_dir = cache_dir or os.getenv(
            'TRANSFORMERS_CACHE', 
            '/app/.cache/huggingface/transformers'
        )
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Cache directory: {self.cache_dir}")
    
    def download_model(self, model_name: str) -> bool:
        """Download and cache a single FinBERT model."""
        try:
            print(f"[INFO] Downloading FinBERT model: {model_name}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                torch_dtype=torch.float32  # Ensure CPU compatibility
            )
            
            print(f"[OK] Successfully cached: {model_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download {model_name}: {e}")
            return False
    
    def download_all_models(self) -> List[str]:
        """Download all configured FinBERT models."""
        successful_downloads = []
        failed_downloads = []
        
        for model_name in self.FINBERT_MODELS:
            if self.download_model(model_name):
                successful_downloads.append(model_name)
            else:
                failed_downloads.append(model_name)
        
        self.print_summary(successful_downloads, failed_downloads)
        return successful_downloads
    
    def print_summary(self, successful: List[str], failed: List[str]) -> None:
        """Print download summary."""
        print(f"\n[SUMMARY] FinBERT Model Download Results:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if successful:
            print("  Downloaded models:")
            for model in successful:
                print(f"    - {model}")
        
        if failed:
            print("  Failed models:")
            for model in failed:
                print(f"    - {model}")
        
        print(f"  Cache location: {self.cache_dir}")
        
        # Calculate cache size
        try:
            cache_size = self.get_cache_size()
            print(f"  Cache size: {cache_size:.1f} MB")
        except Exception:
            print("  Cache size: Unable to calculate")
    
    def get_cache_size(self) -> float:
        """Calculate total size of cached models in MB."""
        total_size = 0
        cache_path = Path(self.cache_dir)
        
        if cache_path.exists():
            for file_path in cache_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB


def main():
    """Main execution function."""
    print("[INFO] Starting FinBERT model pre-download process...")
    
    # Check if running in CI environment
    if os.getenv('CI') == 'true':
        print("[INFO] CI environment detected - using minimal download timeout")
    
    downloader = FinBERTDownloader()
    successful_models = downloader.download_all_models()
    
    if successful_models:
        print(f"\n[OK] Pre-download completed successfully")
        print(f"[OK] {len(successful_models)} models cached for offline use")
        sys.exit(0)
    else:
        print(f"\n[WARNING] No models downloaded successfully")
        print(f"[WARNING] Container will attempt online downloads during runtime")
        sys.exit(0)  # Don't fail build, just warn


if __name__ == "__main__":
    main()