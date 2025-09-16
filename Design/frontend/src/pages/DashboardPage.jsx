import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocaleFormat, useFinancialFormat } from '../hooks/useLocaleFormat';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Button,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  AccountBalanceWallet,
  TrendingUp,
  Analytics,
  Search,
  History,
  Star,
  ShoppingCart,
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  NewspaperOutlined,
  // Add // Not used currently
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { selectCurrentUser } from '../features/auth/authSlice';
import { useGetUserAnalysisHistoryQuery } from '../features/api/apiSlice';

const DashboardPage = () => {
  const user = useSelector(selectCurrentUser);
  const navigate = useNavigate();
  const { t } = useTranslation('common');
  const { formatLargeNumber } = useLocaleFormat();
  const { formatScore } = useFinancialFormat();
  
  // API queries
  const { data: analysisHistoryData, isLoading: analysisLoading } = useGetUserAnalysisHistoryQuery({ limit: 5 });
  
  // Mock user credit balance and quick search
  const [userCredits] = useState(25);
  const [quickSearch, setQuickSearch] = useState('');
  
  // Get analysis history from API
  const recentAnalyses = analysisHistoryData?.analyses || [];
  
  // Calculate dashboard stats from real data
  const totalAnalyses = analysisHistoryData?.count || 0;
  const averageScore = recentAnalyses.length > 0 
    ? (recentAnalyses.reduce((sum, analysis) => sum + analysis.score, 0) / recentAnalyses.length).toFixed(1)
    : '0.0';
  const favoriteSector = recentAnalyses.length > 0
    ? recentAnalyses.reduce((acc, analysis) => {
        acc[analysis.sector] = (acc[analysis.sector] || 0) + 1;
        return acc;
      }, {})
    : {};
  const topSector = Object.keys(favoriteSector).length > 0
    ? Object.keys(favoriteSector).reduce((a, b) => favoriteSector[a] > favoriteSector[b] ? a : b)
    : 'None';

  // Calculate sentiment aggregations from recent analyses
  const sentimentStats = recentAnalyses.reduce((acc, analysis) => {
    if (analysis.components?.sentiment?.raw) {
      const sentiment = analysis.components.sentiment.raw;
      acc.total++;
      if (sentiment.label === 'positive') acc.positive++;
      else if (sentiment.label === 'negative') acc.negative++;
      else acc.neutral++;
      
      if (sentiment.sentiment) {
        acc.scores.push(parseFloat(sentiment.sentiment));
      }
      if (sentiment.newsCount) {
        acc.totalNews += sentiment.newsCount;
      }
    }
    return acc;
  }, { total: 0, positive: 0, negative: 0, neutral: 0, scores: [], totalNews: 0 });

  const avgSentimentScore = sentimentStats.scores.length > 0
    ? (sentimentStats.scores.reduce((sum, score) => sum + score, 0) / sentimentStats.scores.length)
    : 0;

  const dominantSentiment = sentimentStats.total > 0
    ? (sentimentStats.positive > sentimentStats.negative && sentimentStats.positive > sentimentStats.neutral
       ? 'positive'
       : sentimentStats.negative > sentimentStats.neutral
       ? 'negative'
       : 'neutral')
    : 'neutral';

  // Sample data for the chart (replace with real data from API)
  const chartData = [
    { name: 'Jan', value: 4000, profit: 2400 },
    { name: 'Feb', value: 3000, profit: 1398 },
    { name: 'Mar', value: 2000, profit: 9800 },
    { name: 'Apr', value: 2780, profit: 3908 },
    { name: 'May', value: 1890, profit: 4800 },
    { name: 'Jun', value: 2390, profit: 3800 },
    { name: 'Jul', value: 3490, profit: 4300 },
  ];

  // const marketData = [ // Not used currently
  //   { name: 'AAPL', price: 178.50, change: 2.3 },
  //   { name: 'MSFT', price: 425.20, change: -1.2 },
  //   { name: 'GOOGL', price: 142.30, change: 0.8 },
  //   { name: 'AMZN', price: 180.90, change: 3.1 },
  //   { name: 'TSLA', price: 240.50, change: -2.5 },
  // ];

  const handleQuickSearch = () => {
    if (quickSearch.trim()) {
      navigate('/stocks', { 
        state: { 
          searchTicker: quickSearch.trim(),
          autoAnalyze: true  // Flag to trigger automatic analysis
        } 
      });
    }
  };

  const getScoreColor = (score) => {
    if (score >= 7) return 'success';
    if (score >= 4) return 'warning';
    return 'error';
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <SentimentSatisfied color="success" />;
      case 'negative':
        return <SentimentDissatisfied color="error" />;
      case 'neutral':
      default:
        return <SentimentNeutral color="warning" />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'success';
      case 'negative':
        return 'error';
      case 'neutral':
      default:
        return 'warning';
    }
  };

  const formatSentimentScore = (score) => {
    if (score === null || score === undefined || isNaN(score)) {
      return 'N/A';
    }
    const numScore = parseFloat(score);
    if (numScore > 0) {
      return `+${numScore.toFixed(3)}`;
    }
    return numScore.toFixed(3);
  };

  return (
    <Container component="main" role="main" aria-label="Dashboard main content" maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          {t('dashboard.welcome')}, {user?.username || 'Investor'}!
        </Typography>
        <Typography variant="body1" color="textSecondary">
          {t('dashboard.subtitle')}
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Credit Balance Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ backgroundColor: 'primary.main', color: 'white', height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AccountBalanceWallet sx={{ mr: 1 }} />
                <Typography variant="h6">
                  {t('dashboard.availableCredits')}
                </Typography>
              </Box>
              <Typography variant="h3" gutterBottom>
                {formatLargeNumber(userCredits)}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9, mb: 2 }}>
                {t('dashboard.creditEquation')}
              </Typography>
              <Button
                variant="outlined"
                size="small"
                onClick={() => navigate('/store')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    borderColor: 'white'
                  }
                }}
                startIcon={<ShoppingCart />}
              >
                {t('dashboard.buyMoreCredits')}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Stock Search */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Search sx={{ mr: 1 }} />
                {t('dashboard.quickSearch')}
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  fullWidth
                  placeholder={t('dashboard.searchPlaceholder')}
                  value={quickSearch}
                  onChange={(e) => setQuickSearch(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && handleQuickSearch()}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleQuickSearch}
                  disabled={!quickSearch.trim()}
                  sx={{ minWidth: 120 }}
                >
                  {t('dashboard.analyseButton')}
                </Button>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {t('dashboard.searchDescription')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Dashboard Stats */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.analysesThisMonth')}
              </Typography>
              <Typography variant="h5" component="div">
                {formatLargeNumber(totalAnalyses)}
              </Typography>
              <Typography variant="body2" color="success.main">
                {recentAnalyses.length} {t('dashboard.recentCount')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.averageScore')}
              </Typography>
              <Typography variant="h5" component="div">
                {formatScore(parseFloat(averageScore))}/10
              </Typography>
              <Typography variant="body2" color={parseFloat(averageScore) >= 6 ? "success.main" : parseFloat(averageScore) >= 4 ? "warning.main" : "error.main"}>
                {parseFloat(averageScore) >= 6 ? t('dashboard.aboveAverage') : parseFloat(averageScore) >= 4 ? t('dashboard.average') : t('dashboard.belowAverage')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.favoriteSector')}
              </Typography>
              <Typography variant="h5" component="div">
                {topSector}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {t('dashboard.mostAnalysed')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.marketSentiment')}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                {getSentimentIcon(dominantSentiment)}
                <Typography variant="h5" component="div" sx={{ ml: 1, textTransform: 'capitalize' }}>
                  {dominantSentiment}
                </Typography>
              </Box>
              <Typography variant="body2" color={`${getSentimentColor(dominantSentiment)}.main`}>
                {sentimentStats.total > 0 ? t('dashboard.fromAnalyses', { count: sentimentStats.total }) : t('dashboard.noDataYet')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Score Card */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.avgSentimentScore')}
              </Typography>
              <Typography variant="h5" component="div">
                {formatSentimentScore(avgSentimentScore)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {sentimentStats.totalNews > 0 ? t('dashboard.newsArticles', { count: sentimentStats.totalNews }) : t('dashboard.noNewsData')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* News Coverage Card */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <NewspaperOutlined color="action" sx={{ mr: 1 }} />
                <Typography color="textSecondary">
                  {t('dashboard.newsCoverage')}
                </Typography>
              </Box>
              <Typography variant="h5" component="div">
                {sentimentStats.totalNews}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {t('dashboard.articlesAnalysed')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Distribution Card */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                {t('dashboard.sentimentBreakdown')}
              </Typography>
              {sentimentStats.total > 0 ? (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <SentimentSatisfied color="success" fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">{t('dashboard.positive')}</Typography>
                    </Box>
                    <Typography variant="body2" color="success.main">
                      {Math.round((sentimentStats.positive / sentimentStats.total) * 100)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <SentimentNeutral color="warning" fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">{t('dashboard.neutral')}</Typography>
                    </Box>
                    <Typography variant="body2" color="warning.main">
                      {Math.round((sentimentStats.neutral / sentimentStats.total) * 100)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <SentimentDissatisfied color="error" fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">{t('dashboard.negative')}</Typography>
                    </Box>
                    <Typography variant="body2" color="error.main">
                      {Math.round((sentimentStats.negative / sentimentStats.total) * 100)}%
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  {t('dashboard.noSentimentData')}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Analysis History */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
                <History sx={{ mr: 1 }} />
                {t('dashboard.recentAnalyses')}
              </Typography>
              <Button 
                variant="outlined" 
                size="small"
                onClick={() => navigate('/reports')}
              >
                {t('dashboard.viewAll')}
              </Button>
            </Box>
            
            {analysisLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : recentAnalyses.length > 0 ? (
              <List>
                {recentAnalyses.map((analysis, index) => (
                  <React.Fragment key={analysis.id}>
                    <ListItem sx={{ px: 0 }}>
                      <ListItemIcon>
                        <Analytics color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body1" sx={{ fontWeight: 500 }}>
                              {analysis.symbol}
                            </Typography>
                            <Chip 
                              label={`${analysis.score}/10`} 
                              size="small" 
                              color={getScoreColor(analysis.score)}
                            />
                          </Box>
                        }
                        secondary={`${analysis.sector} • ${new Date(analysis.analysis_date).toLocaleDateString()}`}
                      />
                      <IconButton 
                        size="small"
                        onClick={() => navigate(`/analysis/${analysis.id}`)}
                        title={t('dashboard.viewAnalysisResults')}
                      >
                        <TrendingUp />
                      </IconButton>
                    </ListItem>
                    {index < recentAnalyses.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            ) : (
              <Alert severity="info">
                {t('dashboard.noAnalysisHistory')}
              </Alert>
            )}
          </Paper>
        </Grid>

        {/* Quick Actions & Tools */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              {t('dashboard.quickActions')}
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<Analytics />}
                onClick={() => navigate('/stocks')}
              >
                {t('dashboard.stockAnalysis')}
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                startIcon={<TrendingUp />}
                onClick={() => navigate('/compare')}
              >
                {t('dashboard.compareStocks')}
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                startIcon={<AccountBalanceWallet />}
                onClick={() => navigate('/sectors')}
              >
                {t('dashboard.sectorAnalysis')}
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                startIcon={<ShoppingCart />}
                onClick={() => navigate('/store')}
              >
                {t('dashboard.buyCredits')}
              </Button>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Typography variant="h6" gutterBottom>
              {t('dashboard.popularStocks')}
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'].map((symbol) => (
                <Button
                  key={symbol}
                  variant="text"
                  size="small"
                  onClick={() => navigate('/stocks', { state: { searchTicker: symbol, autoAnalyze: true } })}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  <Star sx={{ mr: 1, fontSize: 16 }} />
                  {symbol}
                </Button>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Market Overview Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              {t('dashboard.marketOverview')}
            </Typography>
            {analysisLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#1976d2"
                    fill="#1976d2"
                    fillOpacity={0.6}
                    name={t('dashboard.portfolioValue')}
                  />
                  <Area
                    type="monotone"
                    dataKey="profit"
                    stroke="#2e7d32"
                    fill="#2e7d32"
                    fillOpacity={0.6}
                    name={t('dashboard.profitLoss')}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;