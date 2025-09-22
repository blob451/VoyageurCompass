#!/usr/bin/env python3
"""
Pre-populate stock analyses for demo purposes
Creates analyses for 5 stocks across 5 sectors in EN/FR/ES with all explanation types
"""

import time
import json
import requests
from django.core.management.base import BaseCommand
from django.core.cache import cache
from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.explanation_service import ExplanationService
from Analytics.services.translation_service import TranslationService
from Analytics.services.local_llm_service import LocalLLMService
from Data.models import Stock
from Data.repo.price_reader import PriceReader
from Data.services.yahoo_finance import yahoo_finance_service

class Command(BaseCommand):
    help = 'Pre-populate stock analyses for demo purposes'

    DEMO_STOCKS = {
        'Technology': 'AAPL',      # Apple Inc.
        'Healthcare': 'JNJ',        # Johnson & Johnson
        'Financial': 'JPM',         # JPMorgan Chase
        'Energy': 'XOM',            # Exxon Mobil
        'Consumer': 'DIS',          # Walt Disney
    }

    LANGUAGES = ['en', 'fr', 'es']
    EXPLANATION_TYPES = ['simple', 'technical', 'financial_advisor']

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting pre-population of demo stock analyses...'))

        base_url = 'http://localhost:8000'
        total_analyses = 0
        successful = 0

        for sector, symbol in self.DEMO_STOCKS.items():
            self.stdout.write(f'\n{self.style.WARNING(f"Processing {sector} sector: {symbol}")}')

            try:
                # Step 1: Run base analysis via API
                self.stdout.write(f'  Running analysis for {symbol}...')

                analysis_url = f'{base_url}/api/v1/analytics/analyze/{symbol}/'

                response = requests.get(analysis_url, timeout=60)

                if response.status_code != 200:
                    self.stdout.write(self.style.ERROR(f'  Failed to analyze {symbol}: {response.status_code}'))
                    continue

                analysis_data = response.json()
                if 'error' in analysis_data:
                    self.stdout.write(self.style.ERROR(f'  Analysis error for {symbol}: {analysis_data["error"]}'))
                    continue

                self.stdout.write(self.style.SUCCESS(f'  ✓ Analysis completed for {symbol}'))

                # Step 2: Generate explanations for each type and language
                for lang in self.LANGUAGES:
                    self.stdout.write(f'  Processing {lang.upper()} language...')

                    for exp_type in self.EXPLANATION_TYPES:
                        total_analyses += 1

                        try:
                            # Generate explanation via API
                            self.stdout.write(f'    Generating {exp_type} explanation...')

                            explanation_url = f'{base_url}/api/v1/analytics/explanation/'
                            explanation_payload = {
                                'symbol': symbol,
                                'type': exp_type,
                                'language': lang
                            }

                            exp_response = requests.post(explanation_url, json=explanation_payload, timeout=45)

                            if exp_response.status_code == 200:
                                explanation_data = exp_response.json()
                                if 'error' not in explanation_data:
                                    successful += 1
                                    self.stdout.write(self.style.SUCCESS(f'      + {exp_type} ({lang})'))
                                else:
                                    self.stdout.write(self.style.ERROR(f'      - {exp_type} ({lang}): {explanation_data["error"]}'))
                            else:
                                self.stdout.write(self.style.ERROR(f'      - {exp_type} ({lang}): HTTP {exp_response.status_code}'))

                            # Small delay to avoid overwhelming the system
                            time.sleep(1.0)

                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f'      - {exp_type} ({lang}): {str(e)}'))

                self.stdout.write(self.style.SUCCESS(f'  Completed {symbol}'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  Error processing {symbol}: {str(e)}'))

        # Summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write(self.style.SUCCESS(f'Pre-population complete!'))
        self.stdout.write(f'Total analyses attempted: {total_analyses}')
        self.stdout.write(f'Successful: {successful}')
        self.stdout.write(f'Success rate: {(successful/total_analyses*100):.1f}%' if total_analyses > 0 else 'N/A')

        # Verification step
        self.stdout.write('\nVerifying cached data...')
        verification_successful = 0
        for sector, symbol in self.DEMO_STOCKS.items():
            try:
                check_url = f'{base_url}/api/v1/analytics/analyze/{symbol}/'
                check_response = requests.get(check_url, timeout=30)

                if check_response.status_code == 200:
                    verification_successful += 1
                    self.stdout.write(f'  + {symbol} ({sector}) - cached and ready')
                else:
                    self.stdout.write(f'  - {symbol} ({sector}) - verification failed')
            except:
                self.stdout.write(f'  - {symbol} ({sector}) - verification error')

        self.stdout.write(f'\nVerification: {verification_successful}/{len(self.DEMO_STOCKS)} stocks ready for demo')