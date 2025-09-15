#!/usr/bin/env python
"""
Test script to validate Premium explanation generation.
Tests both content volume and formatting consistency.
"""

import os
import sys
import django
import time
import re
from typing import Dict, Any, List

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')
django.setup()

from Analytics.services.explanation_service import get_explanation_service
from Data.models import Stock, AnalyticsResults

class PremiumExplanationTester:
    """Comprehensive test suite for Premium explanations."""
    
    def __init__(self):
        self.explanation_service = get_explanation_service()
        self.test_results = []
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all Premium explanation tests."""
        print("STARTING Premium Explanation Validation Tests")
        print("=" * 60)
        
        # Test 1: Word Count Validation
        print("\nTest 1: Word Count Validation")
        word_count_results = self.test_word_count_requirements()
        
        # Test 2: Section Structure Validation
        print("\nTest 2: Section Structure Validation")
        structure_results = self.test_section_structure()
        
        # Test 3: Format Consistency Test
        print("\nTest 3: Format Consistency Test")
        format_results = self.test_format_consistency()
        
        # Test 4: Performance Comparison
        print("\nTest 4: Performance Comparison")
        performance_results = self.test_performance_comparison()
        
        # Compile overall results
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "word_count_test": word_count_results,
            "structure_test": structure_results,
            "format_test": format_results,
            "performance_test": performance_results,
            "overall_status": "PASSED" if all([
                word_count_results.get("passed", False),
                structure_results.get("passed", False),
                format_results.get("passed", False)
            ]) else "FAILED"
        }
        
        self.print_summary(overall_results)
        return overall_results
    
    def test_word_count_requirements(self) -> Dict[str, Any]:
        """Test that Premium explanations meet minimum word count (600+)."""
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        results = {"passed": True, "details": []}
        
        for symbol in test_symbols:
            print(f"  Testing {symbol}...")
            
            # Get test analysis data
            analysis_data = self.get_test_analysis_data(symbol)
            
            # Generate Premium explanation
            start_time = time.time()
            explanation = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level="detailed", force_regenerate=True
            )
            generation_time = time.time() - start_time
            
            if not explanation:
                results["passed"] = False
                results["details"].append({
                    "symbol": symbol,
                    "status": "FAILED",
                    "issue": "No explanation generated",
                    "word_count": 0
                })
                continue
            
            content = explanation.get("content", "")
            word_count = len(content.split())
            
            test_result = {
                "symbol": symbol,
                "word_count": word_count,
                "generation_time": round(generation_time, 2),
                "status": "PASSED" if word_count >= 600 else "FAILED",
                "target": 600,
                "method": explanation.get("method", "unknown")
            }
            
            if word_count < 600:
                results["passed"] = False
                test_result["issue"] = f"Below minimum word count ({word_count} < 600)"
            
            results["details"].append(test_result)
            print(f"    {symbol}: {word_count} words ({'PASS' if word_count >= 600 else 'FAIL'})")
        
        return results
    
    def test_section_structure(self) -> Dict[str, Any]:
        """Test that Premium explanations have proper section structure."""
        test_symbols = ["TSLA", "NVDA"]
        results = {"passed": True, "details": []}
        
        expected_sections = [
            "Investment Summary",
            "Technical Analysis", 
            "Risk Assessment",
            "Entry Strategy",
            "Market Outlook"
        ]
        
        for symbol in test_symbols:
            print(f"  Testing {symbol} section structure...")
            
            analysis_data = self.get_test_analysis_data(symbol)
            explanation = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level="detailed", force_regenerate=True
            )
            
            if not explanation:
                results["passed"] = False
                results["details"].append({
                    "symbol": symbol,
                    "status": "FAILED",
                    "issue": "No explanation generated"
                })
                continue
            
            content = explanation.get("content", "")
            sections_found = []
            
            for section in expected_sections:
                # Check for section headers (case insensitive, flexible matching)
                pattern = rf'\*\*\s*{re.escape(section)}\s*:\s*\*\*'
                if re.search(pattern, content, re.IGNORECASE):
                    sections_found.append(section)
            
            test_result = {
                "symbol": symbol,
                "sections_found": len(sections_found),
                "sections_expected": len(expected_sections),
                "found_sections": sections_found,
                "missing_sections": list(set(expected_sections) - set(sections_found)),
                "status": "PASSED" if len(sections_found) >= 4 else "FAILED"
            }
            
            if len(sections_found) < 4:
                results["passed"] = False
                test_result["issue"] = f"Missing sections: {test_result['missing_sections']}"
            
            results["details"].append(test_result)
            print(f"    {symbol}: {len(sections_found)}/5 sections ({'PASS' if len(sections_found) >= 4 else 'FAIL'})")
        
        return results
    
    def test_format_consistency(self) -> Dict[str, Any]:
        """Test format consistency between Enhanced and Premium explanations."""
        test_symbol = "META"
        results = {"passed": True, "details": []}
        
        print(f"  Testing format consistency for {test_symbol}...")
        
        analysis_data = self.get_test_analysis_data(test_symbol)
        
        # Generate both Enhanced and Premium explanations
        enhanced_explanation = self.explanation_service.explain_prediction_single(
            analysis_data, detail_level="standard", force_regenerate=True
        )
        premium_explanation = self.explanation_service.explain_prediction_single(
            analysis_data, detail_level="detailed", force_regenerate=True
        )
        
        if not enhanced_explanation or not premium_explanation:
            results["passed"] = False
            results["details"].append({
                "symbol": test_symbol,
                "status": "FAILED",
                "issue": "Could not generate both explanation types"
            })
            return results
        
        # Check formatting consistency
        enhanced_content = enhanced_explanation.get("content", "")
        premium_content = premium_explanation.get("content", "")
        
        # Count structured sections in both
        enhanced_sections = len(re.findall(r'\*\*[^*]+:\*\*', enhanced_content))
        premium_sections = len(re.findall(r'\*\*[^*]+:\*\*', premium_content))
        
        # Check for malformed headers
        malformed_enhanced = len(re.findall(r'\*\*[^*]+:\*\*+\s*[^*]', enhanced_content))
        malformed_premium = len(re.findall(r'\*\*[^*]+:\*\*+\s*[^*]', premium_content))
        
        consistency_score = 0
        issues = []
        
        if premium_sections >= enhanced_sections:
            consistency_score += 1
        else:
            issues.append("Premium has fewer sections than Enhanced")
        
        if malformed_premium <= malformed_enhanced:
            consistency_score += 1
        else:
            issues.append("Premium has more malformed headers than Enhanced")
        
        test_result = {
            "symbol": test_symbol,
            "enhanced_sections": enhanced_sections,
            "premium_sections": premium_sections,
            "malformed_enhanced": malformed_enhanced,
            "malformed_premium": malformed_premium,
            "consistency_score": consistency_score,
            "max_score": 2,
            "status": "PASSED" if consistency_score >= 2 else "FAILED",
            "issues": issues
        }
        
        if consistency_score < 2:
            results["passed"] = False
        
        results["details"].append(test_result)
        print(f"    Consistency: {consistency_score}/2 ({'PASS' if consistency_score >= 2 else 'FAIL'})")
        
        return results
    
    def test_performance_comparison(self) -> Dict[str, Any]:
        """Test generation time performance across detail levels."""
        test_symbol = "AMZN"
        results = {"details": []}
        
        print(f"  Testing generation performance for {test_symbol}...")
        
        analysis_data = self.get_test_analysis_data(test_symbol)
        detail_levels = [("summary", "Standard"), ("standard", "Enhanced"), ("detailed", "Premium")]
        
        for level, name in detail_levels:
            start_time = time.time()
            explanation = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level=level, force_regenerate=True
            )
            generation_time = time.time() - start_time
            
            if explanation:
                word_count = len(explanation.get("content", "").split())
                results["details"].append({
                    "level": name,
                    "generation_time": round(generation_time, 2),
                    "word_count": word_count,
                    "words_per_second": round(word_count / generation_time if generation_time > 0 else 0, 1)
                })
                print(f"    {name}: {generation_time:.2f}s, {word_count} words")
        
        return results
    
    def get_test_analysis_data(self, symbol: str) -> Dict[str, Any]:
        """Generate test analysis data for a symbol."""
        import random
        
        # Generate realistic test data
        score = random.uniform(3.5, 8.5)
        
        return {
            "symbol": symbol,
            "score_0_10": score,
            "composite_raw": score * 0.1,
            "analysis_date": "2025-09-14T12:00:00",
            "horizon": "short_term",
            "components": {
                "rsi14": random.uniform(30, 70),
                "macd12269": random.uniform(-0.5, 0.5),
                "sma50": random.uniform(100, 200),
                "sma200": random.uniform(90, 210),
                "current_price": random.uniform(150, 300),
                "srcontext": {
                    "nearest_support": random.uniform(140, 160),
                    "nearest_resistance": random.uniform(180, 220)
                }
            },
            "weighted_scores": {
                "w_sma50vs200": random.uniform(-0.3, 0.3),
                "w_pricevs50": random.uniform(-0.2, 0.2),
                "w_rsi14": random.uniform(-0.25, 0.25),
                "w_macd12269": random.uniform(-0.2, 0.2),
                "w_bbpos20": random.uniform(-0.15, 0.15),
                "w_bbwidth20": random.uniform(-0.1, 0.1),
                "w_volsurge": random.uniform(-0.1, 0.1),
                "w_obv20": random.uniform(-0.1, 0.1),
                "w_rel1y": random.uniform(-0.1, 0.1),
                "w_rel2y": random.uniform(-0.1, 0.1),
                "w_candlerev": random.uniform(-0.05, 0.05),
                "w_srcontext": random.uniform(-0.1, 0.1),
            }
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("PREMIUM EXPLANATION TEST SUMMARY")
        print("=" * 60)
        
        status_text = "PASSED" if results["overall_status"] == "PASSED" else "FAILED"
        print(f"\nOverall Status: {status_text}")
        print(f"Test Completed: {results['timestamp']}")
        
        # Word Count Summary
        wc_test = results["word_count_test"]
        print(f"\nWord Count Test: {'PASSED' if wc_test['passed'] else 'FAILED'}")
        if wc_test["details"]:
            avg_words = sum(d["word_count"] for d in wc_test["details"]) / len(wc_test["details"])
            print(f"   Average words: {avg_words:.0f}")
            failures = [d for d in wc_test["details"] if d["status"] == "FAILED"]
            if failures:
                print(f"   Failed symbols: {[f['symbol'] for f in failures]}")
        
        # Structure Test Summary
        struct_test = results["structure_test"]
        print(f"\nStructure Test: {'PASSED' if struct_test['passed'] else 'FAILED'}")
        if struct_test["details"]:
            avg_sections = sum(d["sections_found"] for d in struct_test["details"]) / len(struct_test["details"])
            print(f"   Average sections: {avg_sections:.1f}/5")
        
        # Format Test Summary  
        format_test = results["format_test"]
        print(f"\nFormat Test: {'PASSED' if format_test['passed'] else 'FAILED'}")
        
        # Performance Summary
        perf_test = results["performance_test"]
        if perf_test["details"]:
            print(f"\nPerformance Summary:")
            for detail in perf_test["details"]:
                print(f"   {detail['level']}: {detail['generation_time']}s ({detail['word_count']} words)")
        
        print("\n" + "=" * 60)


def main():
    """Run the Premium explanation tests."""
    tester = PremiumExplanationTester()
    results = tester.run_comprehensive_tests()
    
    # Return appropriate exit code
    sys.exit(0 if results["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()