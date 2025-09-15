#!/usr/bin/env python3
"""
GPU Detection and Configuration Management for VoyageurCompass
Automatically detects GPU availability and configures appropriate Docker profiles.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class GPUDetector:
    """Detects and manages GPU availability for optimal LLM performance."""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_info = {}
        self.cuda_available = False
        self.docker_nvidia_available = False
        
    def detect_nvidia_gpu(self) -> Tuple[bool, Dict]:
        """
        Detect NVIDIA GPU availability and specifications.
        
        Returns:
            Tuple[bool, Dict]: (gpu_available, gpu_info)
        """
        try:
            # Check nvidia-smi availability
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = {}
                
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 3:
                            gpu_info[f'gpu_{i}'] = {
                                'name': parts[0],
                                'memory_total_mb': int(parts[1]),
                                'memory_free_mb': int(parts[2])
                            }
                
                return len(gpu_info) > 0, gpu_info
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"NVIDIA GPU detection failed: {e}")
            return False, {}
        
        return False, {}
    
    def check_docker_nvidia_support(self) -> bool:
        """Check if Docker has NVIDIA runtime support."""
        try:
            # Check docker info for nvidia runtime
            result = subprocess.run(['docker', 'info', '--format', '{{json .Runtimes}}'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                runtimes = json.loads(result.stdout)
                return 'nvidia' in runtimes
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Docker NVIDIA runtime check failed: {e}")
            
        return False
    
    def check_cuda_availability(self) -> bool:
        """Check if CUDA is available on the system."""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def run_full_detection(self) -> Dict:
        """
        Run comprehensive GPU detection.
        
        Returns:
            Dict: Complete detection results
        """
        print("Detecting GPU configuration...")
        
        # Check NVIDIA GPU
        self.gpu_available, self.gpu_info = self.detect_nvidia_gpu()
        
        # Check CUDA support
        self.cuda_available = self.check_cuda_availability()
        
        # Check Docker NVIDIA support
        self.docker_nvidia_available = self.check_docker_nvidia_support()
        
        detection_result = {
            'gpu_available': self.gpu_available,
            'gpu_info': self.gpu_info,
            'cuda_available': self.cuda_available,
            'docker_nvidia_available': self.docker_nvidia_available,
            'recommended_profile': self._get_recommended_profile(),
            'configuration': self._get_optimal_configuration()
        }
        
        self._print_detection_summary(detection_result)
        return detection_result
    
    def _get_recommended_profile(self) -> str:
        """Determine the optimal Docker Compose profile based on detection."""
        if self.gpu_available and self.docker_nvidia_available:
            return 'gpu'
        else:
            return 'cpu'
    
    def _get_optimal_configuration(self) -> Dict:
        """Get optimal configuration based on available hardware."""
        config = {
            'OLLAMA_NUM_PARALLEL': 1,
            'OLLAMA_MAX_LOADED_MODELS': 1,
        }
        
        if self.gpu_available and self.gpu_info:
            # Get first GPU memory info
            first_gpu = list(self.gpu_info.values())[0]
            gpu_memory_mb = first_gpu.get('memory_total_mb', 0)
            
            if gpu_memory_mb >= 8000:  # 8GB+ GPU
                config.update({
                    'OLLAMA_GPU_MEMORY_FRACTION': '0.9',
                    'OLLAMA_MAX_LOADED_MODELS': 2,
                    'OLLAMA_NUM_PARALLEL': 2
                })
            elif gpu_memory_mb >= 6000:  # 6GB+ GPU (like RTX 2060)
                config.update({
                    'OLLAMA_GPU_MEMORY_FRACTION': '0.85',
                    'OLLAMA_MAX_LOADED_MODELS': 1,
                    'OLLAMA_NUM_PARALLEL': 1
                })
            else:  # Lower memory GPU
                config.update({
                    'OLLAMA_GPU_MEMORY_FRACTION': '0.8',
                    'OLLAMA_MAX_LOADED_MODELS': 1,
                    'OLLAMA_NUM_PARALLEL': 1
                })
        else:
            # CPU-only configuration
            config.update({
                'OLLAMA_GPU_MEMORY_FRACTION': '0.0',
                'OLLAMA_MAX_LOADED_MODELS': 1,
                'OLLAMA_NUM_PARALLEL': 1
            })
        
        return config
    
    def _print_detection_summary(self, result: Dict):
        """Print a formatted summary of the detection results."""
        print("\n" + "="*60)
        print("VoyageurCompass GPU Detection Summary")
        print("="*60)
        
        print(f"GPU Available: {'[YES]' if result['gpu_available'] else '[NO]'}")
        
        if result['gpu_info']:
            for gpu_id, info in result['gpu_info'].items():
                print(f"  -> {info['name']}: {info['memory_total_mb']}MB total, {info['memory_free_mb']}MB free")
        
        print(f"CUDA Available: {'[YES]' if result['cuda_available'] else '[NO]'}")
        print(f"Docker NVIDIA Runtime: {'[YES]' if result['docker_nvidia_available'] else '[NO]'}")
        
        print(f"\nRecommended Profile: {result['recommended_profile'].upper()}")
        print(f"Optimal Configuration:")
        for key, value in result['configuration'].items():
            print(f"  -> {key}={value}")
        
        if result['recommended_profile'] == 'gpu':
            print(f"\nExpected Performance Improvement: 10-20x faster inference!")
        else:
            print(f"\nTip: Install NVIDIA Container Toolkit for GPU acceleration")
        
        print("="*60)

def update_env_file(config: Dict, env_file_path: str = ".env"):
    """Update .env file with optimal configuration."""
    env_path = Path(env_file_path)
    
    if not env_path.exists():
        print(f"⚠️  Environment file {env_file_path} not found")
        return False
    
    # Read current .env content
    lines = []
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update or add configuration values
    updated_lines = []
    keys_updated = set()
    
    for line in lines:
        line = line.rstrip()
        if '=' in line and not line.strip().startswith('#'):
            key = line.split('=')[0].strip()
            if key in config:
                updated_lines.append(f"{key}={config[key]}\n")
                keys_updated.add(key)
            else:
                updated_lines.append(line + '\n')
        else:
            updated_lines.append(line + '\n')
    
    # Add any new keys that weren't in the original file
    for key, value in config.items():
        if key not in keys_updated:
            updated_lines.append(f"{key}={value}\n")
    
    # Write updated content
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"[OK] Updated {env_file_path} with optimal configuration")
    return True

def generate_docker_command(profile: str) -> str:
    """Generate the appropriate docker-compose command."""
    base_services = "db redis"
    
    if profile == 'gpu':
        return f"docker-compose --profile {profile} up -d {base_services} ollama"
    else:
        return f"docker-compose --profile {profile} up -d {base_services}"

def main():
    """Main entry point for GPU detection and configuration."""
    detector = GPUDetector()
    
    # Run detection
    result = detector.run_full_detection()
    
    # Update environment file if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--update-env':
        env_file = sys.argv[2] if len(sys.argv) > 2 else '../.env'
        update_env_file(result['configuration'], env_file)
    
    # Generate Docker command
    docker_command = generate_docker_command(result['recommended_profile'])
    print(f"\n🐳 Recommended Docker command:")
    print(f"   cd infrastructure && {docker_command}")
    
    # Export results as JSON for scripting
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(json.dumps(result, indent=2))
    
    return 0 if result['gpu_available'] else 1

if __name__ == "__main__":
    sys.exit(main())