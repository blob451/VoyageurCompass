import React from 'react';
import { useTranslation } from 'react-i18next';
import { useFinancialFormat, useLocaleFormat } from '../hooks/useLocaleFormat';
import {
  Paper,
  Typography,
  Grid,
  Box,
  Card,
  CardContent,
  Divider
} from '@mui/material';

/**
 * Demo component to showcase multilingual formatting capabilities
 * This component demonstrates how numbers, currencies, dates, and text
 * change based on the selected language.
 */
const LocaleTestDemo = () => {
  const { t, i18n } = useTranslation('common');
  const { formatCurrency, formatDate, formatTime, formatNumber } = useLocaleFormat();
  const { formatStockPrice, formatScore, formatPercentageChange } = useFinancialFormat();

  // Demo data
  const demoData = {
    stockPrice: 1234.56,
    score: 8.7,
    percentageChange: 0.035,
    marketCap: 1250000000,
    volume: 15420000,
    date: new Date(),
    amount: 98765.43
  };

  return (
    <Paper sx={{ p: 3, m: 2 }}>
      <Typography variant="h5" gutterBottom>
        🌍 Multilingual Formatting Demo
      </Typography>

      <Typography variant="body2" color="text.secondary" paragraph>
        Current Language: <strong>{t('languages.' + i18n.language)}</strong> ({i18n.language})
      </Typography>

      <Grid container spacing={3}>
        {/* Financial Formatting */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                💰 {t('dashboard.statistics')}
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Stock Price:
                </Typography>
                <Typography variant="h6">
                  {formatStockPrice(demoData.stockPrice)}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  {t('analysis.score')}:
                </Typography>
                <Typography variant="h6">
                  {formatScore(demoData.score)}/10
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Percentage Change:
                </Typography>
                <Typography variant="h6" color="success.main">
                  {formatPercentageChange(demoData.percentageChange)}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Amount:
                </Typography>
                <Typography variant="h6">
                  {formatCurrency(demoData.amount)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Date & Time Formatting */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                📅 Date & Time Formatting
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Current Date:
                </Typography>
                <Typography variant="h6">
                  {formatDate(demoData.date)}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Current Time:
                </Typography>
                <Typography variant="h6">
                  {formatTime(demoData.date)}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Large Number:
                </Typography>
                <Typography variant="h6">
                  {formatNumber(demoData.marketCap)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Translation Examples */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                🗣️ Translation Examples
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    Navigation:
                  </Typography>
                  <Typography>• {t('navigation.home')}</Typography>
                  <Typography>• {t('navigation.search')}</Typography>
                  <Typography>• {t('navigation.settings')}</Typography>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    Recommendations:
                  </Typography>
                  <Typography>• {t('recommendations.buy')}</Typography>
                  <Typography>• {t('recommendations.hold')}</Typography>
                  <Typography>• {t('recommendations.sell')}</Typography>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    Dashboard:
                  </Typography>
                  <Typography>• {t('dashboard.quickActions')}</Typography>
                  <Typography>• {t('dashboard.recentAnalyses')}</Typography>
                  <Typography>• {t('dashboard.popularStocks')}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Divider sx={{ my: 3 }} />

      <Typography variant="body2" color="text.secondary" align="center">
        Change language in Settings to see all formatting update automatically!
      </Typography>
    </Paper>
  );
};

export default LocaleTestDemo;