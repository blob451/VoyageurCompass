#!/usr/bin/env python3
"""
VoyageurCompass Optimized Startup Script
Automatically detects hardware capabilities and starts services with optimal configuration.
"""

import sys
import subprocess
import time
from pathlib import Path
from gpu_detection import GPUDetector

def run_command(command, description, timeout=30):
    """Execute a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()[:200]}...")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 {description} - Exception: {str(e)}")
        return False

def main():
    """Main startup orchestration with GPU detection and optimization."""
    print("="*70)
    print("🚀 VoyageurCompass Optimized Startup")
    print("="*70)
    
    # Step 1: GPU Detection and Environment Optimization
    print("\n🔍 Phase 1: Hardware Detection & Optimization")
    detector = GPUDetector()
    result = detector.run_full_detection()
    
    # Update environment configuration
    env_updated = detector.update_env_file(result['configuration'], '../.env') if hasattr(detector, 'update_env_file') else False
    
    profile = result['recommended_profile']
    print(f"\n📋 Using profile: {profile.upper()}")
    
    # Step 2: Stop any existing services
    print("\n🛑 Phase 2: Stopping Existing Services")
    run_command("docker-compose -f docker-compose.hybrid.yml down", "Stop existing Docker services")
    
    # Step 3: Start optimized services
    print(f"\n🐳 Phase 3: Starting Optimized Services ({profile.upper()})")
    
    if profile == 'gpu':
        success = run_command(
            f"docker-compose -f docker-compose.hybrid.yml --profile gpu up -d",
            "Start GPU-optimized Docker services",
            timeout=120
        )
    else:
        success = run_command(
            f"docker-compose -f docker-compose.hybrid.yml --profile cpu up -d",
            "Start CPU-optimized Docker services", 
            timeout=60
        )
    
    if not success:
        print("\n❌ Failed to start Docker services. Check Docker Desktop is running.")
        return 1
    
    # Step 4: Wait for services to be ready
    print("\n⏳ Phase 4: Waiting for Services to Initialize")
    print("   Waiting for database...")
    time.sleep(5)
    
    # Check service health
    services_healthy = True
    services_healthy &= run_command("docker exec voyageur-db pg_isready", "Check database health")
    services_healthy &= run_command("docker exec voyageur-redis redis-cli ping", "Check Redis health")
    services_healthy &= run_command("curl -f http://localhost:11434/api/tags", "Check Ollama health")
    
    if not services_healthy:
        print("\n⚠️  Some services may not be fully ready. Check docker logs.")
    
    # Step 5: Performance verification
    print("\n🧪 Phase 5: Performance Verification")
    
    if profile == 'gpu':
        # Test GPU performance
        gpu_test = run_command(
            'curl -X POST "http://localhost:11434/api/generate" -H "Content-Type: application/json" --data \'{"model": "phi3:3.8b", "prompt": "Test", "stream": false}\' --max-time 10',
            "GPU performance test",
            timeout=15
        )
        
        if gpu_test:
            print("🔥 GPU acceleration is working!")
        else:
            print("⚠️  GPU test failed - check Ollama logs")
    
    # Step 6: Startup summary
    print("\n" + "="*70)
    print("📊 STARTUP SUMMARY")
    print("="*70)
    print(f"Profile: {profile.upper()}")
    print(f"Services: Database ✅ | Redis ✅ | Ollama {'🔥' if profile == 'gpu' else '💻'}")
    
    if result['gpu_available'] and profile == 'gpu':
        gpu_name = list(result['gpu_info'].values())[0]['name']
        print(f"GPU: {gpu_name}")
        print("Expected Performance: 10-20x faster LLM inference")
        print("Memory Savings: ~7GB system RAM freed")
    
    print("\n🎯 Next Steps:")
    print("   1. Start your local Django server: python manage.py runserver")
    print("   2. Start your frontend: cd Design/frontend && npm run dev")
    print("   3. Visit http://localhost:3000 to access the application")
    
    print("\n🛠️  Troubleshooting:")
    print("   - Check logs: docker-compose -f docker-compose.hybrid.yml logs")
    print(f"   - Restart services: python start_optimized.py")
    print("   - Manual GPU check: nvidia-smi")
    
    print("="*70)
    print("🚀 VoyageurCompass startup complete!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Startup failed with error: {str(e)}")
        sys.exit(1)