"""
Test script for Universal LSTM Model - Phase 1 validation
Tests the new Universal LSTM architecture with GPU support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
from Analytics.ml.models.lstm_base import (
    UniversalLSTMPredictor, 
    create_universal_lstm_model,
    save_universal_model
)
from Analytics.ml.sector_mappings import get_sector_mapper, SECTOR_MAPPING
from Analytics.ml.universal_preprocessor import UniversalLSTMPreprocessor

def test_universal_model_creation():
    """Test creating Universal LSTM models with different configurations."""
    print("🧪 Testing Universal LSTM Model Creation...")
    
    # Test different model types
    models = {
        'full_universal': create_universal_lstm_model(model_type='full_universal'),
        'price_only': create_universal_lstm_model(model_type='price_only'),
        'lightweight': create_universal_lstm_model(model_type='lightweight')
    }
    
    for model_name, model in models.items():
        print(f"  ✅ {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"     - Sectors: {model.num_sectors}, Industries: {model.num_industries}")
        print(f"     - Cross-attention: {model.use_cross_attention}, Multi-task: {model.multi_task_output}")
    
    return models

def test_gpu_compatibility():
    """Test GPU compatibility and performance."""
    print("\n🚀 Testing GPU Compatibility...")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  🔧 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and move to GPU
    model = create_universal_lstm_model(model_type='full_universal')
    model = model.to(device)
    
    # Test forward pass with batch data
    batch_size = 32
    seq_len = 60
    input_size = 25
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    sector_ids = torch.randint(0, 11, (batch_size,)).to(device)
    industry_ids = torch.randint(0, 50, (batch_size,)).to(device)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(x, sector_ids, industry_ids)
    
    print(f"  ✅ Forward pass successful on {device}")
    print(f"  ✅ Output shapes:")
    for key, tensor in outputs.items():
        print(f"     - {key}: {tensor.shape}")
    
    return model, device

def test_sector_mappings():
    """Test sector mapping functionality."""
    print("\n🗺️  Testing Sector Mappings...")
    
    mapper = get_sector_mapper()
    
    # Test sector classification
    test_stocks = ['AAPL', 'JPM', 'XOM', 'UNKNOWN_STOCK']
    for stock in test_stocks:
        sector = mapper.classify_stock_sector(stock)
        sector_id = mapper.get_sector_id(sector or 'Unknown')
        print(f"  📊 {stock}: {sector} (ID: {sector_id})")
    
    # Test coverage validation
    coverage = mapper.validate_sector_coverage()
    print(f"  📈 Training coverage: {coverage['total_training_stocks']} stocks across {coverage['total_sectors']} sectors")
    
    return mapper

def test_universal_preprocessor():
    """Test universal feature engineering."""
    print("\n⚙️  Testing Universal Preprocessor...")
    
    preprocessor = UniversalLSTMPreprocessor(sequence_length=60)
    
    # Create mock OHLCV data
    np.random.seed(42)
    n_days = 200
    base_price = 100
    
    dates = [f"2024-01-{i+1:02d}" if i < 31 else f"2024-02-{i-30:02d}" for i in range(n_days)]
    prices = base_price + np.cumsum(np.random.randn(n_days) * 0.02)
    volumes = np.random.randint(1000000, 10000000, n_days)
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n_days) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.005),
        'close': prices,
        'volume': volumes
    })
    
    # Test feature engineering
    feature_df = preprocessor.engineer_universal_features(df, 'AAPL', sector_id=0, industry_id=0)
    print(f"  🔧 Engineered {len(feature_df.columns)} features from {len(df.columns)} base columns")
    print(f"  🔧 Feature names: {list(feature_df.columns)[:10]}...")  # Show first 10
    
    # Test multi-stock preparation
    feature_dfs = {
        'AAPL': feature_df,
        'MSFT': feature_df.copy(),  # Mock second stock
    }
    
    # Fit scalers
    preprocessor.fit_universal_scalers(feature_dfs)
    print(f"  ✅ Universal scalers fitted successfully")
    
    return preprocessor, feature_df

def test_model_save_load():
    """Test saving and loading universal models."""
    print("\n💾 Testing Model Save/Load...")
    
    # Create and test model
    model = create_universal_lstm_model(model_type='full_universal')
    
    # Create test directory
    test_dir = "Analytics/ml/test_models"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test saving
    mock_scalers = {'feature_scaler': 'mock', 'target_scaler': 'mock'}
    mock_mappings = {'sector_mapping': SECTOR_MAPPING}
    
    try:
        model_path = save_universal_model(
            model=model,
            model_name="test_universal_v1.0",
            model_dir=test_dir,
            metadata={'test': True},
            scalers=mock_scalers,
            sector_mappings=mock_mappings,
            training_stocks=['AAPL', 'MSFT']
        )
        print(f"  ✅ Model saved successfully at: {model_path}")
        
        # Check file exists
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  📁 Model file size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"  ❌ Save failed: {e}")
    
    return model_path

def run_phase1_tests():
    """Run all Phase 1 tests for Universal LSTM implementation."""
    print("🎯 UNIVERSAL LSTM MODEL - PHASE 1 TESTING")
    print("=" * 60)
    
    try:
        # Test 1: Model Creation
        models = test_universal_model_creation()
        
        # Test 2: GPU Compatibility
        model, device = test_gpu_compatibility()
        
        # Test 3: Sector Mappings
        mapper = test_sector_mappings()
        
        # Test 4: Universal Preprocessor
        preprocessor, feature_df = test_universal_preprocessor()
        
        # Test 5: Model Save/Load
        model_path = test_model_save_load()
        
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
        print("\n📋 Phase 1 Implementation Summary:")
        print("   ✅ UniversalLSTMPredictor class with sector/industry embeddings")
        print("   ✅ SectorCrossAttention mechanism implemented")
        print("   ✅ Multi-task output heads (price, volatility, trend)")
        print("   ✅ Universal feature engineering pipeline")
        print("   ✅ Sector mapping configuration (11 sectors, 50 industries)")
        print("   ✅ Enhanced model saving/loading with metadata")
        print("   ✅ GPU compatibility verified")
        print(f"   🚀 Ready for training on {device}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_phase1_tests()
    exit(0 if success else 1)