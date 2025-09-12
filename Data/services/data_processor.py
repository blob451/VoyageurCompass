"""
Data Processing Module
Handles data transformation and processing for VoyageurCompass.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Service class for data processing and transformation.
    """

    def __init__(self):
        """Initialize the data processor."""
        logger.info("Data Processor initialized")

    def process_price_data(self, raw_data: List[Dict]) -> Dict:
        """
        Process raw price data into a structured format.

        Args:
            raw_data: List of raw price data points

        Returns:
            Processed price data dictionary
        """
        try:
            if not raw_data:
                return {
                    "prices": [],
                    "dates": [],
                    "volumes": [],
                    "high": None,
                    "low": None,
                    "average": None,
                    "total_volume": 0,
                }

            prices = []
            dates = []
            volumes = []

            for data_point in raw_data:
                if "close" in data_point and data_point["close"] is not None:
                    try:
                        prices.append(float(data_point["close"]))
                    except (ValueError, TypeError):
                        # Skip invalid close prices
                        continue
                if "date" in data_point and data_point["date"] is not None:
                    dates.append(data_point["date"])
                if "volume" in data_point and data_point["volume"] is not None:
                    try:
                        volumes.append(int(data_point["volume"]))
                    except (ValueError, TypeError):
                        # Skip invalid volumes
                        continue

            processed = {
                "prices": prices,
                "dates": dates,
                "volumes": volumes,
                "high": max(prices) if prices else None,
                "low": min(prices) if prices else None,
                "average": sum(prices) / len(prices) if prices else None,
                "total_volume": sum(volumes) if volumes else 0,
                "data_points": len(prices),
                "processed_at": datetime.now().isoformat(),
            }

            return processed

        except Exception as e:
            logger.error(f"Error processing price data: {str(e)}")
            return {"error": str(e)}

    def normalize_data(self, data: List[float], method: str = "minmax") -> List[float]:
        """
        Normalize a list of numerical data.

        Args:
            data: List of numerical values
            method: Normalization method ('minmax' or 'zscore')

        Returns:
            Normalized data list
        """
        try:
            if not data:
                return []

            if method == "minmax":
                min_val = min(data)
                max_val = max(data)
                range_val = max_val - min_val

                if range_val == 0:
                    return [0.5] * len(data)

                normalized = [(x - min_val) / range_val for x in data]

            elif method == "zscore":
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                std_dev = variance**0.5

                if std_dev == 0:
                    return [0] * len(data)

                normalized = [(x - mean) / std_dev for x in data]

            else:
                raise ValueError(f"Unknown normalization method: {method}")

            return normalized

        except ValueError:
            # Re-raise ValueError for invalid method parameter
            raise
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return []

    def aggregate_data(self, data: List[Dict], key: str, aggregation: str = "sum") -> float:
        """
        Aggregate data based on a key and aggregation method.

        Args:
            data: List of dictionaries
            key: Key to aggregate on
            aggregation: Aggregation method ('sum', 'avg', 'min', 'max')

        Returns:
            Aggregated value
        """
        try:
            values = [float(item.get(key, 0)) for item in data if key in item]

            if not values:
                return 0

            if aggregation == "sum":
                return sum(values)
            elif aggregation == "avg":
                return sum(values) / len(values)
            elif aggregation == "min":
                return min(values)
            elif aggregation == "max":
                return max(values)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return 0

    def filter_outliers(self, data: List[float], threshold: float = 3.0) -> List[float]:
        """
        Filter outliers from data using z-score method.

        Args:
            data: List of numerical values
            threshold: Z-score threshold for outlier detection

        Returns:
            Filtered data list
        """
        try:
            if len(data) < 3:
                return data

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std_dev = variance**0.5

            if std_dev == 0:
                return data

            filtered = []
            for value in data:
                z_score = abs((value - mean) / std_dev)
                if z_score <= threshold:
                    filtered.append(value)

            return filtered if filtered else data

        except Exception as e:
            logger.error(f"Error filtering outliers: {str(e)}")
            return data

    def resample_data(self, data: List[Dict], date_key: str, value_key: str, frequency: str = "daily") -> List[Dict]:
        """
        Resample time series data to a different frequency.

        Args:
            data: List of dictionaries with date and value
            date_key: Key for date field
            value_key: Key for value field
            frequency: Target frequency ('daily', 'weekly', 'monthly')

        Returns:
            Resampled data list
        """
        try:
            if not data:
                return []

            # Sort data by date
            sorted_data = sorted(data, key=lambda x: x.get(date_key, ""))

            # Group data by period
            grouped = {}

            for item in sorted_data:
                if date_key not in item or value_key not in item:
                    continue

                date_str = item[date_key]
                value = float(item[value_key])

                # Parse date and determine period key
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

                if frequency == "daily":
                    period_key = date.date().isoformat()
                elif frequency == "weekly":
                    week_start = date - timedelta(days=date.weekday())
                    period_key = week_start.date().isoformat()
                elif frequency == "monthly":
                    period_key = f"{date.year}-{date.month:02d}"
                else:
                    period_key = date.date().isoformat()

                if period_key not in grouped:
                    grouped[period_key] = []
                grouped[period_key].append(value)

            # Calculate average for each period
            resampled = []
            for period, values in grouped.items():
                resampled.append({"period": period, "value": sum(values) / len(values), "count": len(values)})

            return resampled

        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            return []

    def calculate_correlations(self, data1: List[float], data2: List[float]) -> float:
        """
        Calculate correlation between two data series.

        Args:
            data1: First data series
            data2: Second data series

        Returns:
            Correlation coefficient
        """
        try:
            if len(data1) != len(data2) or len(data1) < 2:
                return 0

            n = len(data1)

            # Calculate means
            mean1 = sum(data1) / n
            mean2 = sum(data2) / n

            # Calculate correlation components
            numerator = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(n))

            sum_sq1 = sum((x - mean1) ** 2 for x in data1)
            sum_sq2 = sum((x - mean2) ** 2 for x in data2)

            denominator = (sum_sq1 * sum_sq2) ** 0.5

            if denominator == 0:
                return 0

            correlation = numerator / denominator
            return correlation

        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0

    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean data by removing nulls and invalid values.

        Args:
            data: List of dictionaries to clean

        Returns:
            Cleaned data list
        """
        try:
            cleaned = []

            for item in data:
                # Remove None values
                cleaned_item = {k: v for k, v in item.items() if v is not None}

                # Validate numerical fields
                for key, value in cleaned_item.items():
                    if isinstance(value, str):
                        # Try to convert string numbers
                        try:
                            if "." in value:
                                cleaned_item[key] = float(value)
                            else:
                                cleaned_item[key] = int(value)
                        except ValueError:
                            pass

                if cleaned_item:
                    cleaned.append(cleaned_item)

            return cleaned

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return data

    def export_to_json(self, data: Any, filename: str) -> bool:
        """
        Export data to JSON file.

        Args:
            data: Data to export
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            from Core.services.utils import sanitize_filename

            safe_filename = sanitize_filename(filename)
            if not safe_filename.endswith(".json"):
                safe_filename += ".json"

            filepath = f"Temp/{safe_filename}"

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Data exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data to JSON: {str(e)}")
            return False


# Singleton instance
data_processor = DataProcessor()
