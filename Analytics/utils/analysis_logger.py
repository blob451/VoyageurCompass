"""
Analysis Logger for Web-based Stock Analysis Requests
Generates detailed log files for user-specific analysis requests with naming convention:
username-stock_ticker-YYYYMMDD_HHMMSS.txt
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

from django.conf import settings

from Analytics.engine.ta_engine import IndicatorResult


class AnalysisLogger:
    """
    Logger for user-specific stock analysis requests.
    Creates detailed log files for web-based analyses.
    """

    def __init__(self, username: str, symbol: str, log_dir: Optional[str] = None):
        """
        Initialise the analysis logger.

        Args:
            username: Username of the user requesting analysis
            symbol: Stock ticker symbol being analysed
            log_dir: Directory to store log files (optional)
        """
        self.username = username
        self.symbol = symbol.upper()
        self.timestamp = datetime.now()

        # Set up log directory
        if log_dir:
            self.log_dir = log_dir
        else:
            base_dir = getattr(settings, "BASE_DIR", os.getcwd())
            self.log_dir = os.path.join(base_dir, "Temp", "logs", "web_analysis", "current")

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate log filename: username-symbol-YYYYMMDD_HHMMSS.txt
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{self.username}-{self.symbol}-{timestamp_str}.txt"
        self.log_path = os.path.join(self.log_dir, self.log_filename)

        # Initialize the log file
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the log file with header information."""
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("VOYAGEUR COMPASS - WEB ANALYSIS LOG\n")
                f.write("=" * 80 + "\n")
                f.write(f"User: {self.username}\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Analysis Started: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log File: {self.log_filename}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            # If we can't create the log file, log to Django logger
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize analysis log file {self.log_path}: {e}")

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message to the analysis log file.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_line = f"[{timestamp}] {level}: {message}\n"

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            # Fallback to Django logger if file logging fails
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to write to analysis log {self.log_path}: {e}")

    def log_info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(message, "WARNING")

    def log_error(self, message: str):
        """Log an error message."""
        self.log(message, "ERROR")

    def log_debug(self, message: str):
        """Log a debug message."""
        self.log(message, "DEBUG")

    def log_analysis_start(self, analysis_date: datetime, horizon: str):
        """Log the start of technical analysis."""
        self.log_info("=" * 50)
        self.log_info("TECHNICAL ANALYSIS START")
        self.log_info("=" * 50)
        self.log_info(f"Analysis Date: {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info(f"Horizon: {horizon}")
        self.log_info("")

    def log_data_retrieval(self, stock_data_count: int, auto_sync: bool = False):
        """Log data retrieval information."""
        self.log_info(f"Stock Data Retrieved: {stock_data_count} price records")
        if auto_sync:
            self.log_info("Note: Data was automatically synchronized from external source")
        self.log_info("")

    def log_indicator_calculation(
        self, indicator_name: str, result: Optional[Union[Dict[str, Any], IndicatorResult]]
    ):
        """Log individual indicator calculation results."""
        if result:
            # Handle both dictionary and IndicatorResult objects
            if hasattr(result, "score"):  # IndicatorResult object
                score = result.score
                weighted_score = result.weighted_score
                raw_data = result.raw if hasattr(result, "raw") else {}
            else:  # Dictionary
                score = result.get("score", "N/A")
                weighted_score = result.get("weighted_score", "N/A")
                raw_data = result.get("raw", {})

            self.log_info(f"INDICATOR: {indicator_name.upper()}")
            self.log_info(f"  Score: {score}")
            self.log_info(f"  Weighted Score: {weighted_score}")
            if raw_data:
                for key, value in raw_data.items():
                    self.log_info(f"  {key}: {value}")
            self.log_info("")
        else:
            self.log_warning(f"INDICATOR: {indicator_name.upper()}")
            self.log_warning("  Result: FAILED (insufficient data or calculation error)")
            self.log_info("")

    def log_analysis_complete(self, composite_score: float, final_score: int):
        """Log analysis completion with final results."""
        self.log_info("=" * 50)
        self.log_info("TECHNICAL ANALYSIS COMPLETE")
        self.log_info("=" * 50)
        self.log_info(f"Composite Raw Score: {composite_score:.6f}")
        self.log_info(f"Final Score (0-10): {final_score}")
        self.log_info(f"Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info("=" * 50)

    def log_analysis_error(self, error_message: str):
        """Log analysis failure."""
        self.log_error("=" * 50)
        self.log_error("TECHNICAL ANALYSIS FAILED")
        self.log_error("=" * 50)
        self.log_error(f"Error: {error_message}")
        self.log_error(f"Failed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_error("=" * 50)

    def get_log_path(self) -> str:
        """Get the full path to the log file."""
        return self.log_path

    def get_log_filename(self) -> str:
        """Get the log filename."""
        return self.log_filename

    def finalize(self):
        """Finalize the log file with footer."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("LOG END\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to finalize analysis log {self.log_path}: {e}")
