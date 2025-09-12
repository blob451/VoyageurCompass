"""
Generate Composite Data Command

Generates daily sector and industry composite data from constituent stocks.
Creates DataSectorPrice and DataIndustryPrice entries for historical analysis.
"""

import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Avg, Sum, Count, Min, Max
from django.utils import timezone

from Data.models import Stock, StockPrice, DataSector, DataIndustry, DataSectorPrice, DataIndustryPrice

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate sector and industry composite data from constituent stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days to generate composite data for (default: 30)'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be generated without creating data'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Regenerate all composite data, even existing records'
        )

    def handle(self, *args, **options):
        """Execute composite data generation."""
        days = options['days']
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write(self.style.SUCCESS('=== COMPOSITE DATA GENERATION ==='))
        self.stdout.write(f'Days to process: {days}')
        self.stdout.write(f'Dry run: {dry_run}')
        self.stdout.write(f'Force regeneration: {force}')
        
        # Calculate date range
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        self.stdout.write(f'Date range: {start_date} to {end_date}')
        
        # Generate sector composites
        sector_results = self._generate_sector_composites(start_date, end_date, dry_run, force)
        
        # Generate industry composites
        industry_results = self._generate_industry_composites(start_date, end_date, dry_run, force)
        
        # Summary
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('COMPOSITE DATA GENERATION COMPLETED'))
        self.stdout.write(f'Sector composites: {sector_results["created"]} created, {sector_results["skipped"]} skipped')
        self.stdout.write(f'Industry composites: {industry_results["created"]} created, {industry_results["skipped"]} skipped')
        self.stdout.write(f'Total records: {sector_results["created"] + industry_results["created"]}')
        self.stdout.write('='*60)

    def _generate_sector_composites(self, start_date, end_date, dry_run, force):
        """Generate sector composite data."""
        self.stdout.write('\nGenerating sector composites...')
        
        results = {'created': 0, 'skipped': 0}
        
        # Get all active sectors with stocks
        sectors = DataSector.objects.filter(
            isActive=True,
            stocks__is_active=True
        ).distinct()
        
        for sector in sectors:
            self.stdout.write(f'Processing sector: {sector.sectorName}')
            
            # Get stocks in this sector
            sector_stocks = Stock.objects.filter(
                sector_id=sector,
                is_active=True
            )
            
            if not sector_stocks.exists():
                self.stdout.write(f'  No active stocks in {sector.sectorName}')
                continue
            
            # Process each date in range
            current_date = start_date
            while current_date <= end_date:
                # Check if composite already exists
                if not force and DataSectorPrice.objects.filter(
                    sector=sector, date=current_date
                ).exists():
                    results['skipped'] += 1
                    current_date += timedelta(days=1)
                    continue
                
                # Calculate composite values for this date
                composite_data = self._calculate_sector_composite(sector_stocks, current_date)
                
                if composite_data and not dry_run:
                    # Create or update composite record
                    DataSectorPrice.objects.update_or_create(
                        sector=sector,
                        date=current_date,
                        defaults=composite_data
                    )
                    results['created'] += 1
                elif composite_data and dry_run:
                    self.stdout.write(f'  Would create: {current_date} - {composite_data["close"]:.2f}')
                    results['created'] += 1
                else:
                    results['skipped'] += 1
                
                current_date += timedelta(days=1)
        
        return results

    def _generate_industry_composites(self, start_date, end_date, dry_run, force):
        """Generate industry composite data."""
        self.stdout.write('\nGenerating industry composites...')
        
        results = {'created': 0, 'skipped': 0}
        
        # Get all active industries with stocks
        industries = DataIndustry.objects.filter(
            isActive=True,
            stocks__is_active=True
        ).distinct()
        
        for industry in industries:
            self.stdout.write(f'Processing industry: {industry.industryName}')
            
            # Get stocks in this industry
            industry_stocks = Stock.objects.filter(
                industry_id=industry,
                is_active=True
            )
            
            if not industry_stocks.exists():
                self.stdout.write(f'  No active stocks in {industry.industryName}')
                continue
            
            # Process each date in range
            current_date = start_date
            while current_date <= end_date:
                # Check if composite already exists
                if not force and DataIndustryPrice.objects.filter(
                    industry=industry, date=current_date
                ).exists():
                    results['skipped'] += 1
                    current_date += timedelta(days=1)
                    continue
                
                # Calculate composite values for this date
                composite_data = self._calculate_industry_composite(industry_stocks, current_date)
                
                if composite_data and not dry_run:
                    # Create or update composite record
                    DataIndustryPrice.objects.update_or_create(
                        industry=industry,
                        date=current_date,
                        defaults=composite_data
                    )
                    results['created'] += 1
                elif composite_data and dry_run:
                    self.stdout.write(f'  Would create: {current_date} - {composite_data["close"]:.2f}')
                    results['created'] += 1
                else:
                    results['skipped'] += 1
                
                current_date += timedelta(days=1)
        
        return results

    def _calculate_sector_composite(self, sector_stocks, date):
        """Calculate weighted composite values for a sector on a specific date."""
        # Get price data for all stocks in sector for this date
        prices = StockPrice.objects.filter(
            stock__in=sector_stocks,
            date=date
        ).select_related('stock')
        
        if not prices.exists():
            return None
        
        # Calculate market cap weighted averages
        total_market_cap = Decimal('0')
        weighted_open = Decimal('0')
        weighted_high = Decimal('0')
        weighted_low = Decimal('0')
        weighted_close = Decimal('0')
        total_volume = 0
        
        for price in prices:
            # Use shares outstanding as weight (fallback to equal weighting)
            weight = price.stock.shares_outstanding or Decimal('1')
            market_cap = weight * price.close
            
            total_market_cap += market_cap
            weighted_open += price.open * market_cap
            weighted_high += price.high * market_cap
            weighted_low += price.low * market_cap
            weighted_close += price.close * market_cap
            total_volume += price.volume
        
        if total_market_cap == 0:
            return None
        
        return {
            'open': weighted_open / total_market_cap,
            'high': weighted_high / total_market_cap,
            'low': weighted_low / total_market_cap,
            'close': weighted_close / total_market_cap,
            'volume': total_volume,
            'stock_count': prices.count()
        }

    def _calculate_industry_composite(self, industry_stocks, date):
        """Calculate weighted composite values for an industry on a specific date."""
        # Get price data for all stocks in industry for this date
        prices = StockPrice.objects.filter(
            stock__in=industry_stocks,
            date=date
        ).select_related('stock')
        
        if not prices.exists():
            return None
        
        # Calculate market cap weighted averages
        total_market_cap = Decimal('0')
        weighted_open = Decimal('0')
        weighted_high = Decimal('0')
        weighted_low = Decimal('0')
        weighted_close = Decimal('0')
        total_volume = 0
        
        for price in prices:
            # Use shares outstanding as weight (fallback to equal weighting)
            weight = price.stock.shares_outstanding or Decimal('1')
            market_cap = weight * price.close
            
            total_market_cap += market_cap
            weighted_open += price.open * market_cap
            weighted_high += price.high * market_cap
            weighted_low += price.low * market_cap
            weighted_close += price.close * market_cap
            total_volume += price.volume
        
        if total_market_cap == 0:
            return None
        
        return {
            'open': weighted_open / total_market_cap,
            'high': weighted_high / total_market_cap,
            'low': weighted_low / total_market_cap,
            'close': weighted_close / total_market_cap,
            'volume': total_volume,
            'stock_count': prices.count()
        }