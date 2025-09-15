#!/usr/bin/env python
"""
Enhanced Premium Explanation Validation Test Suite
Tests the comprehensive fixes for Premium explanation issues with TEVA.
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

class PremiumFixValidator:
    """Comprehensive validator for Premium explanation fixes."""
    
    def __init__(self):
        self.explanation_service = get_explanation_service()
        self.test_results = []
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests for Premium explanation fixes."""
        print("PREMIUM EXPLANATION FIX VALIDATION")
        print("=" * 60)
        
        # Test 1: Volume Enhancement Validation
        print("\nTest 1: Word Count Volume Enhancement")
        volume_results = self.test_volume_enhancement()
        
        # Test 2: Formatting Consistency Validation
        print("\nTest 2: Formatting Consistency Validation")
        format_results = self.test_formatting_consistency()
        
        # Test 3: Section Structure Validation
        print("\nTest 3: Section Structure Validation")
        structure_results = self.test_section_structure()
        
        # Test 4: TEVA-Specific Validation
        print("\nTest 4: TEVA-Specific Validation")
        teva_results = self.test_teva_specific()
        
        # Test 5: Comparison with Enhanced Mode
        print("\nTest 5: Enhanced vs Premium Comparison")
        comparison_results = self.test_enhanced_vs_premium()
        
        # Compile overall results
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "volume_enhancement": volume_results,
            "formatting_consistency": format_results,
            "section_structure": structure_results,
            "teva_specific": teva_results,
            "enhanced_premium_comparison": comparison_results,
            "overall_status": "PASSED" if all([
                volume_results.get("passed", False),
                format_results.get("passed", False),
                structure_results.get("passed", False),
                teva_results.get("passed", False)
            ]) else "FAILED"
        }
        
        self.print_comprehensive_summary(overall_results)
        return overall_results
    
    def test_volume_enhancement(self) -> Dict[str, Any]:
        """Test that Premium explanations now meet enhanced word count targets."""
        test_symbols = ["TEVA", "AAPL", "MSFT"]
        results = {"passed": True, "details": []}
        
        for symbol in test_symbols:
            print(f"  Testing volume for {symbol}...")
            
            analysis_data = self.get_teva_analysis_data() if symbol == "TEVA" else self.get_test_analysis_data(symbol)
            
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
            
            # Enhanced targets: 700-800 words for Premium
            target_min = 600  # Minimum acceptable
            target_ideal = 700  # Ideal target
            
            test_result = {
                "symbol": symbol,
                "word_count": word_count,
                "generation_time": round(generation_time, 2),
                "target_min": target_min,
                "target_ideal": target_ideal,
                "status": "PASSED" if word_count >= target_min else "FAILED",
                "improvement_level": "EXCELLENT" if word_count >= target_ideal else "GOOD" if word_count >= target_min else "INSUFFICIENT",
                "method": explanation.get("method", "unknown")
            }
            
            if word_count < target_min:
                results["passed"] = False
                test_result["issue"] = f"Below minimum target ({word_count} < {target_min})"
            
            results["details"].append(test_result)
            print(f"    {symbol}: {word_count} words ({test_result['improvement_level']}) - {'PASS' if word_count >= target_min else 'FAIL'}")
        
        return results
    
    def test_formatting_consistency(self) -> Dict[str, Any]:
        """Test that Premium explanations have consistent formatting."""
        test_symbols = ["TEVA", "NVDA"]
        results = {"passed": True, "details": []}
        
        for symbol in test_symbols:
            print(f"  Testing formatting for {symbol}...")
            
            analysis_data = self.get_teva_analysis_data() if symbol == "TEVA" else self.get_test_analysis_data(symbol)
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
            
            # Check for proper section headers
            proper_headers = len(re.findall(r'\*\*[^*]+:\*\*\s+[A-Z]', content))
            malformed_headers = len(re.findall(r'\*\*[^*]+:\*\*+[A-Z]', content))  # Missing space
            extra_asterisks = len(re.findall(r'\*{3,}', content))
            
            test_result = {
                "symbol": symbol,
                "proper_headers": proper_headers,
                "malformed_headers": malformed_headers,
                "extra_asterisks": extra_asterisks,
                "formatting_score": 10 - (malformed_headers * 2) - extra_asterisks,
                "status": "PASSED" if malformed_headers == 0 and extra_asterisks <= 1 else "FAILED"
            }
            
            if test_result["status"] == "FAILED":
                results["passed"] = False
                test_result["issue"] = f"Formatting issues: {malformed_headers} malformed headers, {extra_asterisks} extra asterisks"
            
            results["details"].append(test_result)
            print(f"    {symbol}: {proper_headers} proper headers, formatting score: {test_result['formatting_score']}/10 - {'PASS' if test_result['status'] == 'PASSED' else 'FAIL'}")
        
        return results
    
    def test_section_structure(self) -> Dict[str, Any]:
        """Test that Premium explanations have proper 5-section structure."""
        test_symbols = ["TEVA"]
        results = {"passed": True, "details": []}
        
        expected_sections = [
            "Investment Summary",
            "Technical Analysis", 
            "Risk Assessment",
            "Entry Strategy",
            "Market Outlook"
        ]
        
        for symbol in test_symbols:
            print(f"  Testing section structure for {symbol}...")
            
            analysis_data = self.get_teva_analysis_data()
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
                # Flexible section matching
                patterns = [
                    rf'\*\*\s*{re.escape(section)}\s*:\s*\*\*',
                    rf'\*\*\s*{section.upper()}\s*:\s*\*\*',
                    rf'\*\*\s*{section.lower()}\s*:\s*\*\*'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        sections_found.append(section)
                        break
            
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
            print(f"    {symbol}: {len(sections_found)}/5 sections found - {'PASS' if len(sections_found) >= 4 else 'FAIL'}")
        
        return results
    
    def test_teva_specific(self) -> Dict[str, Any]:
        """Test specific TEVA analysis to ensure fixes work with real data."""
        results = {"passed": True, "details": []}
        
        print("  Testing with real TEVA analysis data...")
        
        try:
            analysis_data = self.get_teva_analysis_data()
            
            # Test Enhanced (standard) explanation first
            enhanced = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level="standard", force_regenerate=True
            )
            
            # Test Premium (detailed) explanation  
            premium = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level="detailed", force_regenerate=True
            )
            
            if not enhanced or not premium:
                results["passed"] = False
                results["details"].append({
                    "symbol": "TEVA",
                    "status": "FAILED",
                    "issue": "Failed to generate both Enhanced and Premium explanations"
                })
                return results
            
            enhanced_words = len(enhanced.get("content", "").split())
            premium_words = len(premium.get("content", "").split())
            
            test_result = {
                "symbol": "TEVA",
                "enhanced_words": enhanced_words,
                "premium_words": premium_words,
                "volume_improvement": round(premium_words / enhanced_words, 2) if enhanced_words > 0 else 0,
                "target_ratio": 2.0,  # Premium should be ~2x Enhanced
                "actual_improvement": "EXCELLENT" if premium_words >= enhanced_words * 1.8 else "GOOD" if premium_words >= enhanced_words * 1.3 else "INSUFFICIENT",
                "status": "PASSED" if premium_words >= enhanced_words * 1.3 else "FAILED"
            }
            
            if test_result["status"] == "FAILED":
                results["passed"] = False
                test_result["issue"] = f"Insufficient volume improvement: {test_result['volume_improvement']}x (target: 1.5x+)"
            
            results["details"].append(test_result)
            print(f"    TEVA: Enhanced={enhanced_words}w, Premium={premium_words}w, Improvement={test_result['volume_improvement']}x - {'PASS' if test_result['status'] == 'PASSED' else 'FAIL'}")
            
        except Exception as e:
            results["passed"] = False
            results["details"].append({
                "symbol": "TEVA",
                "status": "FAILED",
                "issue": f"Exception during testing: {str(e)}"
            })
        
        return results
    
    def test_enhanced_vs_premium(self) -> Dict[str, Any]:
        """Compare Enhanced vs Premium to ensure Premium provides more value."""
        test_symbol = "TEVA"
        results = {"details": []}
        
        print(f"  Comparing Enhanced vs Premium for {test_symbol}...")
        
        analysis_data = self.get_teva_analysis_data()
        
        # Generate both explanations
        enhanced = self.explanation_service.explain_prediction_single(
            analysis_data, detail_level="standard", force_regenerate=True
        )
        premium = self.explanation_service.explain_prediction_single(
            analysis_data, detail_level="detailed", force_regenerate=True
        )
        
        if enhanced and premium:
            enhanced_content = enhanced.get("content", "")
            premium_content = premium.get("content", "")
            
            comparison = {
                "symbol": test_symbol,
                "enhanced_analysis": {
                    "word_count": len(enhanced_content.split()),
                    "sections": len(re.findall(r'\*\*[^*]+:\*\*', enhanced_content)),
                    "generation_time": enhanced.get("generation_time", 0)
                },
                "premium_analysis": {
                    "word_count": len(premium_content.split()),
                    "sections": len(re.findall(r'\*\*[^*]+:\*\*', premium_content)),
                    "generation_time": premium.get("generation_time", 0)
                }
            }
            
            # Calculate value metrics
            comparison["value_metrics"] = {
                "word_ratio": round(comparison["premium_analysis"]["word_count"] / comparison["enhanced_analysis"]["word_count"], 2),
                "section_ratio": round(comparison["premium_analysis"]["sections"] / comparison["enhanced_analysis"]["sections"], 2) if comparison["enhanced_analysis"]["sections"] > 0 else 0,
                "time_ratio": round(comparison["premium_analysis"]["generation_time"] / comparison["enhanced_analysis"]["generation_time"], 2) if comparison["enhanced_analysis"]["generation_time"] > 0 else 0
            }
            
            results["details"].append(comparison)
            print(f"    Enhanced: {comparison['enhanced_analysis']['word_count']}w, {comparison['enhanced_analysis']['sections']} sections")
            print(f"    Premium: {comparison['premium_analysis']['word_count']}w, {comparison['premium_analysis']['sections']} sections")
            print(f"    Ratios: Words={comparison['value_metrics']['word_ratio']}x, Sections={comparison['value_metrics']['section_ratio']}x")
        
        return results
    
    def get_teva_analysis_data(self) -> Dict[str, Any]:
        """Get real TEVA analysis data from the database."""
        try:
            teva_stock = Stock.objects.get(symbol="TEVA")
            latest_analysis = AnalyticsResults.objects.filter(stock=teva_stock).order_by('-created_at').first()
            
            if not latest_analysis:
                print("  Warning: No TEVA analysis found, using mock data")
                return self.get_test_analysis_data("TEVA")
            
            return {
                "symbol": "TEVA",
                "score_0_10": latest_analysis.score_0_10,
                "composite_raw": latest_analysis.composite_raw,
                "analysis_date": latest_analysis.created_at.isoformat(),
                "horizon": latest_analysis.horizon,
                "components": latest_analysis.components,
                "weighted_scores": {
                    "w_sma50vs200": latest_analysis.w_sma50vs200,
                    "w_pricevs50": latest_analysis.w_pricevs50,
                    "w_rsi14": latest_analysis.w_rsi14,
                    "w_macd12269": latest_analysis.w_macd12269,
                    "w_bbpos20": latest_analysis.w_bbpos20,
                    "w_bbwidth20": latest_analysis.w_bbwidth20,
                    "w_volsurge": latest_analysis.w_volsurge,
                    "w_obv20": latest_analysis.w_obv20,
                    "w_rel1y": latest_analysis.w_rel1y,
                    "w_rel2y": latest_analysis.w_rel2y,
                    "w_candlerev": latest_analysis.w_candlerev,
                    "w_srcontext": latest_analysis.w_srcontext,
                }
            }
        except Exception as e:
            print(f"  Warning: Error fetching TEVA data: {e}, using mock data")
            return self.get_test_analysis_data("TEVA")
    
    def get_test_analysis_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic test analysis data for a symbol."""
        import random
        
        # Generate realistic test data based on symbol
        if symbol == "TEVA":
            score = 5.0  # HOLD-range score
        else:
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
    
    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 60)
        print("PREMIUM EXPLANATION FIX VALIDATION SUMMARY")
        print("=" * 60)
        
        status_text = "PASSED" if results["overall_status"] == "PASSED" else "FAILED"
        print(f"\nOverall Status: {status_text}")
        print(f"Validation Completed: {results['timestamp']}")
        
        # Volume Enhancement Summary
        volume_test = results["volume_enhancement"]
        print(f"\nVolume Enhancement: {'PASSED' if volume_test['passed'] else 'FAILED'}")
        if volume_test["details"]:
            avg_words = sum(d["word_count"] for d in volume_test["details"]) / len(volume_test["details"])
            print(f"   Average Premium words: {avg_words:.0f} (target: 700+)")
            excellent = [d for d in volume_test["details"] if d.get("improvement_level") == "EXCELLENT"]
            print(f"   Excellent performance: {len(excellent)}/{len(volume_test['details'])} symbols")
        
        # Formatting Consistency Summary
        format_test = results["formatting_consistency"]
        print(f"\nFormatting Consistency: {'PASSED' if format_test['passed'] else 'FAILED'}")
        if format_test["details"]:
            avg_score = sum(d["formatting_score"] for d in format_test["details"]) / len(format_test["details"])
            print(f"   Average formatting score: {avg_score:.1f}/10")
        
        # Section Structure Summary
        structure_test = results["section_structure"]
        print(f"\nSection Structure: {'PASSED' if structure_test['passed'] else 'FAILED'}")
        if structure_test["details"]:
            avg_sections = sum(d["sections_found"] for d in structure_test["details"]) / len(structure_test["details"])
            print(f"   Average sections found: {avg_sections:.1f}/5")
        
        # TEVA-Specific Summary
        teva_test = results["teva_specific"]
        print(f"\nTEVA-Specific Validation: {'PASSED' if teva_test['passed'] else 'FAILED'}")
        if teva_test["details"]:
            teva_detail = teva_test["details"][0]
            if "volume_improvement" in teva_detail:
                print(f"   TEVA improvement: {teva_detail['volume_improvement']}x ({teva_detail['actual_improvement']})")
        
        # Enhanced vs Premium Comparison
        comparison_test = results["enhanced_premium_comparison"]
        if comparison_test["details"]:
            comp_detail = comparison_test["details"][0]
            print(f"\nEnhanced vs Premium Comparison:")
            print(f"   Word ratio: {comp_detail['value_metrics']['word_ratio']}x")
            print(f"   Section ratio: {comp_detail['value_metrics']['section_ratio']}x")
        
        print("\n" + "=" * 60)


def main():
    """Run the Premium explanation fix validation."""
    validator = PremiumFixValidator()
    results = validator.run_comprehensive_validation()
    
    # Return appropriate exit code
    sys.exit(0 if results["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()