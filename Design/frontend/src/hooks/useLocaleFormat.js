import { useTranslation } from 'react-i18next';
import { useCallback } from 'react';
import {
  formatNumber,
  formatCurrency,
  formatPercentage,
  formatDate,
  formatTime,
  formatLargeNumber,
  parseNumber
} from '../utils/localeUtils';

/**
 * Custom hook for locale-aware formatting
 * Provides functions that automatically use the current i18n language and formatting config
 */
export const useLocaleFormat = () => {
  const { i18n, t } = useTranslation('common');

  // Get formatting configuration from translations
  const getFormattingConfig = useCallback(() => {
    try {
      return {
        currency: t('formatting.currency'),
        currencyPosition: t('formatting.currencyPosition'),
        decimalSeparator: t('formatting.decimalSeparator'),
        thousandsSeparator: t('formatting.thousandsSeparator'),
        dateFormat: t('formatting.dateFormat'),
        timeFormat: t('formatting.timeFormat')
      };
    } catch {
      // Fallback to default English formatting if translation fails
      return {
        currency: '$',
        currencyPosition: 'before',
        decimalSeparator: '.',
        thousandsSeparator: ',',
        dateFormat: 'MM/DD/YYYY',
        timeFormat: 'h:mm A'
      };
    }
  }, [t]);

  // Number formatting hook
  const formatNumberLocale = useCallback((value) => {
    return formatNumber(value, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Currency formatting hook
  const formatCurrencyLocale = useCallback((value) => {
    return formatCurrency(value, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Percentage formatting hook
  const formatPercentageLocale = useCallback((value) => {
    return formatPercentage(value, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Date formatting hook
  const formatDateLocale = useCallback((date) => {
    return formatDate(date, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Time formatting hook
  const formatTimeLocale = useCallback((date) => {
    return formatTime(date, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Large number formatting hook
  const formatLargeNumberLocale = useCallback((value) => {
    return formatLargeNumber(value, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  // Number parsing hook
  const parseNumberLocale = useCallback((formattedValue) => {
    return parseNumber(formattedValue, i18n.language, getFormattingConfig());
  }, [i18n.language, getFormattingConfig]);

  return {
    formatNumber: formatNumberLocale,
    formatCurrency: formatCurrencyLocale,
    formatPercentage: formatPercentageLocale,
    formatDate: formatDateLocale,
    formatTime: formatTimeLocale,
    formatLargeNumber: formatLargeNumberLocale,
    parseNumber: parseNumberLocale,
    locale: i18n.language,
    config: getFormattingConfig()
  };
};

/**
 * Hook specifically for financial data formatting
 * Provides commonly used financial formatting functions
 */
export const useFinancialFormat = () => {
  const { formatCurrency, formatPercentage, formatLargeNumber, formatNumber } = useLocaleFormat();

  // Format stock price
  const formatStockPrice = useCallback((price) => {
    return formatCurrency(price);
  }, [formatCurrency]);

  // Format market cap
  const formatMarketCap = useCallback((value) => {
    return formatLargeNumber(value);
  }, [formatLargeNumber]);

  // Format percentage change
  const formatPercentageChange = useCallback((change) => {
    const formatted = formatPercentage(Math.abs(change));
    return change >= 0 ? `+${formatted}` : `-${formatted}`;
  }, [formatPercentage]);

  // Format volume
  const formatVolume = useCallback((volume) => {
    return formatLargeNumber(volume);
  }, [formatLargeNumber]);

  // Format ratio (P/E, etc.)
  const formatRatio = useCallback((ratio, decimals = 2) => {
    if (ratio === null || ratio === undefined || isNaN(ratio)) {
      return '-';
    }
    return formatNumber(parseFloat(ratio.toFixed(decimals)));
  }, [formatNumber]);

  // Format score (0-10 scale)
  const formatScore = useCallback((score) => {
    if (score === null || score === undefined || isNaN(score)) {
      return '-';
    }
    return formatNumber(parseFloat(score.toFixed(1)));
  }, [formatNumber]);

  return {
    formatStockPrice,
    formatMarketCap,
    formatPercentageChange,
    formatVolume,
    formatRatio,
    formatScore
  };
};

export default useLocaleFormat;