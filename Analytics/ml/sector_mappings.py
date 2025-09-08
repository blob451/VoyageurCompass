"""
Sector and Industry Mapping Configuration for Universal LSTM Model
Defines the sector-differentiation framework with standardized mappings.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Sector Mapping as defined in the Universal Model plan
SECTOR_MAPPING = {
    'Technology': 0,
    'Healthcare': 1,
    'Financial Services': 2,
    'Energy': 3,
    'Consumer Cyclical': 4,
    'Industrials': 5,
    'Communication Services': 6,
    'Consumer Defensive': 7,
    'Real Estate': 8,
    'Utilities': 9,
    'Unknown': 10
}

# Reverse mapping for sector ID to name
SECTOR_ID_TO_NAME = {v: k for k, v in SECTOR_MAPPING.items()}

# Industry Sub-Classification Mapping (50 major industries)
INDUSTRY_MAPPING = {
    # Technology (0-9)
    'Software—Application': 0,
    'Software—Infrastructure': 1,
    'Semiconductors': 2,
    'Computer Hardware': 3,
    'Consumer Electronics': 4,
    'Internet Content & Information': 5,
    'Electronic Gaming & Multimedia': 6,
    'Information Technology Services': 7,
    'Scientific & Technical Instruments': 8,
    'Solar': 9,

    # Healthcare (10-14)
    'Drug Manufacturers—General': 10,
    'Drug Manufacturers—Specialty & Generic': 11,
    'Biotechnology': 12,
    'Medical Devices': 13,
    'Healthcare Plans': 14,

    # Financial Services (15-19)
    'Banks—Diversified': 15,
    'Banks—Regional': 16,
    'Insurance—Life': 17,
    'Insurance—Property & Casualty': 18,
    'Asset Management': 19,

    # Energy (20-24)
    'Oil & Gas Integrated': 20,
    'Oil & Gas E&P': 21,
    'Oil & Gas Refining & Marketing': 22,
    'Oil & Gas Midstream': 23,
    'Renewable Energy': 24,

    # Consumer Cyclical (25-29)
    'Auto Manufacturers': 25,
    'Auto Parts': 26,
    'Restaurants': 27,
    'Specialty Retail': 28,
    'Internet Retail': 29,

    # Industrials (30-34)
    'Aerospace & Defense': 30,
    'Industrial Machinery': 31,
    'Building Materials': 32,
    'Transportation & Logistics': 33,
    'Electrical Equipment & Parts': 34,

    # Communication Services (35-39)
    'Telecom Services': 35,
    'Entertainment': 36,
    'Interactive Media & Services': 37,
    'Broadcasting': 38,
    'Advertising Agencies': 39,

    # Consumer Defensive (40-44)
    'Beverages—Non-Alcoholic': 40,
    'Food Distribution': 41,
    'Packaged Foods': 42,
    'Household & Personal Products': 43,
    'Discount Stores': 44,

    # Real Estate (45-47)
    'REIT—Retail': 45,
    'REIT—Residential': 46,
    'REIT—Industrial': 47,

    # Utilities (48-49)
    'Utilities—Regulated Electric': 48,
    'Utilities—Renewable': 49
}

# Reverse mapping for industry ID to name
INDUSTRY_ID_TO_NAME = {v: k for k, v in INDUSTRY_MAPPING.items()}

# Sector to Industry groupings
SECTOR_TO_INDUSTRIES = {
    'Technology': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Healthcare': [10, 11, 12, 13, 14],
    'Financial Services': [15, 16, 17, 18, 19],
    'Energy': [20, 21, 22, 23, 24],
    'Consumer Cyclical': [25, 26, 27, 28, 29],
    'Industrials': [30, 31, 32, 33, 34],
    'Communication Services': [35, 36, 37, 38, 39],
    'Consumer Defensive': [40, 41, 42, 43, 44],
    'Real Estate': [45, 46, 47],
    'Utilities': [48, 49],
    'Unknown': []  # No specific industries mapped
}

# Training Stock Universe as defined in the plan
TRAINING_STOCK_UNIVERSE = {
    'Technology': [
        # Mega-cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        # Cloud/Software  
        'CRM', 'ORCL', 'ADBE', 'INTU', 'WDAY',
        # Semiconductors
        'NVDA', 'INTC', 'AMD', 'QCOM', 'TSM'
    ],
    'Healthcare': [
        # Pharmaceuticals
        'JNJ', 'PFE', 'MRK', 'BMY', 'LLY',
        # Biotechnology
        'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX',
        # Medical Devices
        'TMO', 'DHR', 'ABT', 'SYK', 'ISRG'
    ],
    'Financial Services': [
        # Banks
        'JPM', 'BAC', 'WFC', 'C', 'GS',
        # Insurance
        'BRK-A', 'UNH', 'AIG', 'PGR', 'ALL',  # Replaced BRK.B with BRK-A (Berkshire Hathaway Class A)
        # Asset Management
        'BLK', 'MS', 'SCHW', 'TROW', 'AMG'
    ],
    'Energy': [
        # Integrated
        'XOM', 'CVX', 'COP', 'BP', 'SLB',  # Replaced TOT with SLB (Schlumberger - oilfield services)
        # E&P
        'EOG', 'PSX', 'VLO', 'MPC', 'OKE'
    ],
    'Consumer Cyclical': [
        # Retail
        'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
        # Automotive
        'TSLA', 'GM', 'F', 'NIO', 'RIVN',
        # E-commerce
        'AMZN', 'BABA', 'EBAY', 'ETSY', 'SE'
    ],
    'Industrials': [
        # Aerospace
        'BA', 'LMT', 'RTX', 'GD', 'NOC',
        # Machinery
        'CAT', 'DE', 'CMI', 'ITW', 'ROK',
        # Logistics
        'UPS', 'FDX', 'XPO', 'ODFL', 'CHRW'
    ],
    'Communication Services': [
        # Media
        'DIS', 'NFLX', 'WBD', 'PARA', 'FOXA',
        # Telecom
        'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA'
    ],
    'Consumer Defensive': [
        # Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST',
        # Food
        'GIS', 'K', 'CPB', 'CAG', 'SJM'
    ],
    'Real Estate': [
        # REITs
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA'
    ],
    'Utilities': [
        # Electric
        'NEE', 'SO', 'DUK', 'AEP', 'EXC'
    ]
}

# Universal Feature Set (25 normalized features + sector/industry embeddings)
UNIVERSAL_FEATURES = [
    # Price dynamics (normalized)
    'price_change_pct',
    'price_range_pct', 
    'volume_change_pct',

    # Moving averages (relative)
    'price_vs_sma10',
    'price_vs_sma20', 
    'price_vs_sma50',

    # Technical indicators (sector-normalized)
    'rsi_sector_relative',
    'macd_normalized',
    'bollinger_position',

    # Volume patterns
    'volume_ratio',
    'volume_price_trend',

    # Momentum (time-normalized)
    'momentum_5d',
    'momentum_20d', 
    'roc_10d',

    # Volatility (regime-aware)
    'volatility_10d',
    'volatility_ratio',

    # Market microstructure
    'bid_ask_spread_proxy',
    'close_position',

    # Support/resistance
    'resistance_touch',
    'support_touch',

    # Cross-stock correlations
    'sector_correlation',
    'market_beta',

    # Regime indicators
    'volatility_regime',
    'trend_strength'
]


class SectorMapper:
    """
    Utility class for sector and industry mapping operations.
    """

    def __init__(self):
        self.sector_mapping = SECTOR_MAPPING
        self.industry_mapping = INDUSTRY_MAPPING
        self.sector_to_industries = SECTOR_TO_INDUSTRIES

    def get_sector_id(self, sector_name: str) -> int:
        """
        Get sector ID from sector name.

        Args:
            sector_name: Name of the sector

        Returns:
            Sector ID (0-10), defaults to 10 (Unknown) if not found
        """
        return self.sector_mapping.get(sector_name, 10)  # 10 = Unknown

    def get_industry_id(self, industry_name: str) -> int:
        """
        Get industry ID from industry name.

        Args:
            industry_name: Name of the industry

        Returns:
            Industry ID (0-49), defaults to 0 if not found
        """
        return self.industry_mapping.get(industry_name, 0)  # Default to first industry

    def get_sector_name(self, sector_id: int) -> str:
        """Get sector name from sector ID."""
        return SECTOR_ID_TO_NAME.get(sector_id, 'Unknown')

    def get_industry_name(self, industry_id: int) -> str:
        """Get industry name from industry ID."""
        return INDUSTRY_ID_TO_NAME.get(industry_id, 'Unknown')

    def infer_sector_from_industry(self, industry_id: int) -> int:
        """
        Infer sector ID from industry ID.

        Args:
            industry_id: Industry ID

        Returns:
            Corresponding sector ID
        """
        for sector_name, industry_ids in self.sector_to_industries.items():
            if industry_id in industry_ids:
                return self.get_sector_id(sector_name)
        return 10  # Unknown sector

    def get_sector_industries(self, sector_name: str) -> List[int]:
        """
        Get all industry IDs for a given sector.

        Args:
            sector_name: Name of the sector

        Returns:
            List of industry IDs in that sector
        """
        return self.sector_to_industries.get(sector_name, [])

    def get_training_stocks_by_sector(self, sector_name: str) -> List[str]:
        """
        Get training stock symbols for a specific sector.

        Args:
            sector_name: Name of the sector

        Returns:
            List of stock symbols in that sector
        """
        return TRAINING_STOCK_UNIVERSE.get(sector_name, [])

    def get_all_training_stocks(self) -> List[str]:
        """
        Get all training stock symbols across all sectors.

        Returns:
            List of all training stock symbols
        """
        all_stocks = []
        for stocks in TRAINING_STOCK_UNIVERSE.values():
            all_stocks.extend(stocks)
        return list(set(all_stocks))  # Remove duplicates

    def classify_stock_sector(self, symbol: str) -> Optional[str]:
        """
        Classify a stock symbol into its sector based on training universe.

        Args:
            symbol: Stock symbol

        Returns:
            Sector name if found, None otherwise
        """
        for sector_name, stocks in TRAINING_STOCK_UNIVERSE.items():
            if symbol in stocks:
                return sector_name
        return None

    def validate_sector_coverage(self) -> Dict[str, Any]:
        """
        Validate sector and industry coverage for training.

        Returns:
            Dictionary with coverage statistics
        """
        total_stocks = self.get_all_training_stocks()

        coverage_stats = {
            'total_sectors': len(SECTOR_MAPPING),
            'total_industries': len(INDUSTRY_MAPPING),
            'total_training_stocks': len(total_stocks),
            'stocks_per_sector': {},
            'sector_balance': {}
        }

        for sector_name, stocks in TRAINING_STOCK_UNIVERSE.items():
            coverage_stats['stocks_per_sector'][sector_name] = len(stocks)
            coverage_stats['sector_balance'][sector_name] = len(stocks) / len(total_stocks)

        return coverage_stats


# Global mapper instance
sector_mapper = SectorMapper()


def get_sector_mapper() -> SectorMapper:
    """Get the global sector mapper instance."""
    return sector_mapper


# Export key mappings for easy access
__all__ = [
    'SECTOR_MAPPING',
    'INDUSTRY_MAPPING', 
    'SECTOR_ID_TO_NAME',
    'INDUSTRY_ID_TO_NAME',
    'SECTOR_TO_INDUSTRIES',
    'TRAINING_STOCK_UNIVERSE',
    'UNIVERSAL_FEATURES',
    'SectorMapper',
    'sector_mapper',
    'get_sector_mapper'
]
