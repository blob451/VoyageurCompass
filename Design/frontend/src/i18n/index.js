import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Translation resources
import enTranslations from './locales/en/common.json';
import frTranslations from './locales/fr/common.json';
import esTranslations from './locales/es/common.json';

const resources = {
  en: {
    common: enTranslations
  },
  fr: {
    common: frTranslations
  },
  es: {
    common: esTranslations
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,

    // Fallback language if detection fails
    fallbackLng: 'en',

    // Debug mode for development
    debug: import.meta.env.DEV,

    // Default namespace
    defaultNS: 'common',

    // Interpolation settings
    interpolation: {
      escapeValue: false, // React already escapes by default
    },

    // Language detection options
    detection: {
      // Order of language detection methods
      order: ['localStorage', 'navigator', 'htmlTag'],

      // Cache user language
      caches: ['localStorage'],

      // localStorage key
      lookupLocalStorage: 'voyageur-language',

      // Fallback if no language is detected
      checkWhitelist: true
    },

    // Supported languages whitelist
    whitelist: ['en', 'fr', 'es'],

    // Key separator for nested translations
    keySeparator: '.',

    // Enable namespace separation
    nsSeparator: ':',

    // Return null for missing keys instead of key name
    returnNull: false,

    // Return empty string for empty values
    returnEmptyString: false,

    // React specific options
    react: {
      // Use React Suspense
      useSuspense: false,

      // Bind i18n instance to component
      bindI18n: 'languageChanged loaded',

      // Bind i18n store to component
      bindI18nStore: 'added removed',

      // Translate component render on language change
      transEmptyNodeValue: '',

      // Trans component wrapping
      transSupportBasicHtmlNodes: true,
      transKeepBasicHtmlNodesFor: ['br', 'strong', 'i', 'em']
    }
  });

export default i18n;