/**
 * Locale-aware formatting utilities for international financial data display
 */

/**
 * Format number according to locale-specific formatting rules
 * @param {number} value - The number to format
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted number
 */
export const formatNumber = (value, locale = 'en', formatting = {}) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '-';
  }

  const config = {
    en: { decimalSeparator: '.', thousandsSeparator: ',' },
    fr: { decimalSeparator: ',', thousandsSeparator: ' ' },
    es: { decimalSeparator: ',', thousandsSeparator: '.' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;

  // Split number into integer and decimal parts
  const [integerPart, decimalPart] = value.toFixed(2).split('.');

  // Add thousands separators
  const formattedInteger = integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, localeConfig.thousandsSeparator);

  // Combine with decimal part using locale separator
  if (decimalPart && decimalPart !== '00') {
    return `${formattedInteger}${localeConfig.decimalSeparator}${decimalPart}`;
  }

  return formattedInteger;
};

/**
 * Format currency according to locale-specific rules
 * @param {number} value - The currency value to format
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value, locale = 'en', formatting = {}) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '-';
  }

  const config = {
    en: { currency: '$', currencyPosition: 'before', decimalSeparator: '.', thousandsSeparator: ',' },
    fr: { currency: '€', currencyPosition: 'after', decimalSeparator: ',', thousandsSeparator: ' ' },
    es: { currency: '€', currencyPosition: 'after', decimalSeparator: ',', thousandsSeparator: '.' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;
  const formattedNumber = formatNumber(value, locale, localeConfig);

  if (localeConfig.currencyPosition === 'before') {
    return `${localeConfig.currency}${formattedNumber}`;
  } else {
    return `${formattedNumber} ${localeConfig.currency}`;
  }
};

/**
 * Format percentage according to locale-specific rules
 * @param {number} value - The percentage value (0.15 for 15%)
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value, locale = 'en', formatting = {}) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '-';
  }

  const config = {
    en: { decimalSeparator: '.', thousandsSeparator: ',' },
    fr: { decimalSeparator: ',', thousandsSeparator: ' ' },
    es: { decimalSeparator: ',', thousandsSeparator: '.' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;
  const percentValue = value * 100;
  const formattedNumber = formatNumber(percentValue, locale, localeConfig);

  return `${formattedNumber}%`;
};

/**
 * Format date according to locale-specific rules
 * @param {Date|string} date - The date to format
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted date string
 */
export const formatDate = (date, locale = 'en', formatting = {}) => {
  if (!date) {
    return '-';
  }

  const dateObj = new Date(date);
  if (isNaN(dateObj.getTime())) {
    return '-';
  }

  const config = {
    en: { dateFormat: 'MM/DD/YYYY' },
    fr: { dateFormat: 'DD/MM/YYYY' },
    es: { dateFormat: 'DD/MM/YYYY' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;

  const day = String(dateObj.getDate()).padStart(2, '0');
  const month = String(dateObj.getMonth() + 1).padStart(2, '0');
  const year = dateObj.getFullYear();

  switch (localeConfig.dateFormat) {
    case 'DD/MM/YYYY':
      return `${day}/${month}/${year}`;
    case 'MM/DD/YYYY':
    default:
      return `${month}/${day}/${year}`;
  }
};

/**
 * Format time according to locale-specific rules
 * @param {Date|string} date - The date/time to format
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted time string
 */
export const formatTime = (date, locale = 'en', formatting = {}) => {
  if (!date) {
    return '-';
  }

  const dateObj = new Date(date);
  if (isNaN(dateObj.getTime())) {
    return '-';
  }

  const config = {
    en: { timeFormat: 'h:mm A' },
    fr: { timeFormat: 'HH:mm' },
    es: { timeFormat: 'HH:mm' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;

  if (localeConfig.timeFormat === 'HH:mm') {
    // 24-hour format
    const hours = String(dateObj.getHours()).padStart(2, '0');
    const minutes = String(dateObj.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  } else {
    // 12-hour format with AM/PM
    let hours = dateObj.getHours();
    const minutes = String(dateObj.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // 0 should be 12
    return `${hours}:${minutes} ${ampm}`;
  }
};

/**
 * Format large numbers with appropriate suffixes (K, M, B)
 * @param {number} value - The number to format
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {string} Formatted number with suffix
 */
export const formatLargeNumber = (value, locale = 'en', formatting = {}) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '-';
  }

  const suffixes = {
    en: { thousand: 'K', million: 'M', billion: 'B' },
    fr: { thousand: 'k', million: 'M', billion: 'Md' },
    es: { thousand: 'K', million: 'M', billion: 'B' }
  };

  const localeSuffixes = suffixes[locale] || suffixes.en;

  if (Math.abs(value) >= 1000000000) {
    return formatNumber(value / 1000000000, locale, formatting) + localeSuffixes.billion;
  } else if (Math.abs(value) >= 1000000) {
    return formatNumber(value / 1000000, locale, formatting) + localeSuffixes.million;
  } else if (Math.abs(value) >= 1000) {
    return formatNumber(value / 1000, locale, formatting) + localeSuffixes.thousand;
  } else {
    return formatNumber(value, locale, formatting);
  }
};

/**
 * Parse locale-formatted number back to JavaScript number
 * @param {string} formattedValue - The formatted number string
 * @param {string} locale - Locale code (en, fr, es)
 * @param {object} formatting - Formatting configuration from translations
 * @returns {number} Parsed number
 */
export const parseNumber = (formattedValue, locale = 'en', formatting = {}) => {
  if (!formattedValue || formattedValue === '-') {
    return null;
  }

  const config = {
    en: { decimalSeparator: '.', thousandsSeparator: ',' },
    fr: { decimalSeparator: ',', thousandsSeparator: ' ' },
    es: { decimalSeparator: ',', thousandsSeparator: '.' },
    ...formatting
  };

  const localeConfig = config[locale] || config.en;

  // Remove thousands separators and replace decimal separator
  let cleanValue = formattedValue.replace(new RegExp(`\\${localeConfig.thousandsSeparator}`, 'g'), '');
  cleanValue = cleanValue.replace(localeConfig.decimalSeparator, '.');

  return parseFloat(cleanValue);
};

// Default export with all utilities
export default {
  formatNumber,
  formatCurrency,
  formatPercentage,
  formatDate,
  formatTime,
  formatLargeNumber,
  parseNumber
};