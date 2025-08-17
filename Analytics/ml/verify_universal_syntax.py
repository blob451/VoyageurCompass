"""
Syntax verification for Universal LSTM Model - Phase 1
Checks that all new files have correct Python syntax without requiring PyTorch.
"""

import ast
import os
import sys

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def verify_universal_files():
    """Verify syntax of all Universal LSTM files."""
    print("VERIFYING UNIVERSAL LSTM SYNTAX - PHASE 1")
    print("=" * 60)
    
    # Files to check
    files_to_check = [
        "Analytics/ml/models/lstm_base.py",
        "Analytics/ml/sector_mappings.py", 
        "Analytics/ml/universal_preprocessor.py",
        "Analytics/ml/test_universal_model.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            is_valid, error = check_python_syntax(file_path)
            status = "VALID" if is_valid else "INVALID"
            print(f"  {status}: {file_path}")
            
            if not is_valid:
                print(f"    Error: {error}")
                all_valid = False
        else:
            print(f"  NOT FOUND: {file_path}")
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("ALL FILES HAVE VALID PYTHON SYNTAX!")
        print("\nPhase 1 Files Created:")
        print("   - UniversalLSTMPredictor class in lstm_base.py")
        print("   - SectorCrossAttention mechanism")
        print("   - Sector mappings configuration")
        print("   - Universal preprocessor with enhanced features")
        print("   - Model save/load functions for universal architecture")
        print("   - Test script for validation")
        return True
    else:
        print("SYNTAX ERRORS FOUND - PLEASE FIX BEFORE PROCEEDING")
        return False

def check_class_definitions():
    """Check that all expected classes are defined."""
    print("\nCHECKING CLASS DEFINITIONS...")
    
    expected_classes = [
        ("Analytics/ml/models/lstm_base.py", ["UniversalLSTMPredictor", "SectorCrossAttention"]),
        ("Analytics/ml/sector_mappings.py", ["SectorMapper"]),
        ("Analytics/ml/universal_preprocessor.py", ["UniversalLSTMPreprocessor"])
    ]
    
    for file_path, class_names in expected_classes:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for class_name in class_names:
                if f"class {class_name}" in content:
                    print(f"  FOUND: {class_name} in {file_path}")
                else:
                    print(f"  MISSING: {class_name} NOT found in {file_path}")
        else:
            print(f"  File not found: {file_path}")

def check_imports():
    """Check that key imports are properly structured."""
    print("\nCHECKING IMPORT STATEMENTS...")
    
    lstm_base_file = "Analytics/ml/models/lstm_base.py"
    if os.path.exists(lstm_base_file):
        with open(lstm_base_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            "import torch",
            "import torch.nn as nn", 
            "import torch.nn.functional as F",
            "from typing import"
        ]
        
        for import_stmt in required_imports:
            if import_stmt in content:
                print(f"  FOUND: {import_stmt}")
            else:
                print(f"  MISSING: {import_stmt}")

if __name__ == "__main__":
    success = verify_universal_files()
    check_class_definitions()
    check_imports()
    
    if success:
        print("\nPHASE 1 IMPLEMENTATION READY!")
        print("   Next steps: Install PyTorch and run full tests")
    
    exit(0 if success else 1)