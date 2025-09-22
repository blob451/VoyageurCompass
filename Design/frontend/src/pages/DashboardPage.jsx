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
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
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
  Psychology,
  Info,
  CheckCircle,
  Speed,
  Visibility
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { selectCurrentUser } from '../features/auth/authSlice';
import { useGetUserAnalysisHistoryQuery, useGetUserProfileQuery, useAnalyzeStockMutation } from '../features/api/apiSlice';

const DashboardPage = () => {
  const user = useSelector(selectCurrentUser);
  const navigate = useNavigate();
  const { t, i18n } = useTranslation('common');
  const { formatLargeNumber } = useLocaleFormat();
  const { formatScore } = useFinancialFormat();
  
  // API queries
  const { data: analysisHistoryData, isLoading: analysisLoading, refetch: refetchHistory } = useGetUserAnalysisHistoryQuery({ limit: 5 });
  const { data: userProfile, refetch: refetchUserProfile } = useGetUserProfileQuery();
  const [analyzeStock] = useAnalyzeStockMutation();

  // Get user credits from user profile API
  const userCredits = userProfile?.credits || 0;

  // Enhanced search and analysis state
  const [quickSearch, setQuickSearch] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(false);

  // Analysis state machine
  const [analysisMode, setAnalysisMode] = useState('none'); // 'none' | 'manual' | 'auto'
  const [analysisPhase, setAnalysisPhase] = useState('idle'); // 'idle' | 'confirming' | 'syncing' | 'analyzing' | 'completed' | 'failed'
  
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

  // Calculate best performing stock from recent analyses
  const bestStock = recentAnalyses.length > 0
    ? recentAnalyses.reduce((best, current) =>
        current.score > (best?.score || 0) ? current : best, null)
    : null;

  // Remove unused market data - was not being used

  // Enhanced search functionality
  const handleQuickSearch = () => {
    if (!quickSearch.trim()) {
      setError(t('stockSearch.enterTickerError'));
      return;
    }

    if (userCredits < 1) {
      setError(t('stockSearch.insufficientCreditsError'));
      return;
    }

    // Set manual analysis mode and show confirmation
    setAnalysisMode('manual');
    setAnalysisPhase('confirming');
    setError('');
    setConfirmDialog(true);
  };

  // Analysis workflow functions
  const performAnalysis = async (targetSymbol, mode = 'manual') => {
    const trimmedSymbol = targetSymbol.trim().toUpperCase();

    try {
      setAnalysisPhase('analyzing');
      console.log(`[ANALYSIS] Starting analysis for ${trimmedSymbol} (mode: ${mode})`);

      const result = await analyzeStock({
        symbol: trimmedSymbol,
        includeExplanation: true,
        explanationDetail: 'standard',
        language: i18n.language
      });

      if (result.error) {
        console.log('[ANALYSIS] Error occurred:', result.error);

        // Handle different types of errors
        let errorMessage;
        if (result.error.status === 401) {
          errorMessage = t('stockSearch.authenticationError', 'Authentication expired. Please try again.');
        } else if (result.error.status === 402) {
          errorMessage = t('stockSearch.insufficientCreditsError');
        } else if (result.error.status === 429) {
          errorMessage = t('stockSearch.rateLimitError', 'Too many requests. Please try again later.');
        } else if (result.error.status === 'FETCH_ERROR') {
          errorMessage = t('stockSearch.networkError', 'Network error. Please check your connection.');
        } else {
          errorMessage = result.error.data?.error || result.error.data?.detail || t('stockSearch.analysisFailedError');
        }

        setError(errorMessage);
        setAnalysisPhase('failed');
        return;
      }

      const analysisResult = result.data;
      console.log(`[ANALYSIS] Analysis completed successfully:`, analysisResult);

      // Format analysis data for display
      const formattedData = {
        symbol: analysisResult.symbol,
        name: analysisResult.name || `${analysisResult.symbol} Corporation`,
        sector: analysisResult.sector || 'Unknown',
        industry: analysisResult.industry || 'Unknown',
        score: analysisResult.composite_score,
        recommendation: getScoreLabel(analysisResult.composite_score),
        indicators: analysisResult.indicators || {},
        weightedScores: analysisResult.weighted_scores || {},
        explanation: analysisResult.explanation || null,
        timestamp: analysisResult.analysis_date,
        creditsUsed: 1
      };

      console.log(`[ANALYSIS] Analysis data formatted:`, formattedData);
      setAnalysisData(formattedData);
      setAnalysisPhase('completed');

      // Refetch user profile to update credit balance
      console.log(`[ANALYSIS] Refetching user profile for updated credit balance`);
      refetchUserProfile();

      // Refetch history to include the new analysis
      if (analysisHistoryData !== undefined) {
        console.log(`[ANALYSIS] Refetching analysis history`);
        refetchHistory();
      }

      console.log(`[ANALYSIS] Analysis complete.`);

    } catch (err) {
      console.error('[ANALYSIS] Unexpected error:', err);

      // Handle network errors or other unexpected issues
      let errorMessage;
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        errorMessage = t('stockSearch.networkError', 'Network error. Please check your connection and try again.');
      } else if (err.message?.includes('timeout')) {
        errorMessage = t('stockSearch.timeoutError', 'Request timed out. Please try again.');
      } else if (err.message?.includes('AbortError')) {
        errorMessage = t('stockSearch.requestCancelled', 'Request was cancelled. Please try again.');
      } else {
        errorMessage = t('stockSearch.unexpectedError', 'An unexpected error occurred. Please try again.');
      }

      setError(errorMessage);
      setAnalysisPhase('failed');
    }
  };

  const handleConfirmAnalysis = async () => {
    console.log(`[ANALYSIS] Confirmation dialog confirmed for ${quickSearch} (mode: ${analysisMode})`);
    setConfirmDialog(false);
    await performAnalysis(quickSearch, analysisMode);
  };

  const handleCancelAnalysis = () => {
    setConfirmDialog(false);
    setAnalysisPhase('idle');
    setAnalysisMode('none');
  };

  const getScoreColor = (score) => {
    if (score >= 7) return 'success';
    if (score >= 4) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score) => {
    if (score >= 8) return 'Strong Buy';
    if (score >= 6) return 'Buy';
    if (score >= 4) return 'Hold';
    return 'Sell';
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

      <Grid container spacing={4}>
        {/* Top Row - Key Metrics */}
        {/* Enhanced Credit & Account Summary */}
        <Grid item xs={12} md={3}>
          <Card sx={{
            background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
            color: 'white',
            height: '100%',
            borderRadius: 3,
            boxShadow: '0 8px 32px rgba(25, 118, 210, 0.3)'
          }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AccountBalanceWallet sx={{ mr: 1, fontSize: 28 }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {t('dashboard.availableCredits')}
                </Typography>
              </Box>
              <Typography variant="h2" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
                {formatLargeNumber(userCredits)}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                {t('dashboard.creditEquation')}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.8, mb: 3 }}>
                {totalAnalyses} {t('dashboard.totalAnalyses')}
              </Typography>
              <Button
                variant="outlined"
                fullWidth
                onClick={() => navigate('/store')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  fontWeight: 600,
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

        {/* Enhanced Analysis Search */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontWeight: 600 }}>
                <Search sx={{ mr: 1 }} />
                Stock Analysis
              </Typography>

              {/* Search Input */}
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  fullWidth
                  placeholder={t('dashboard.searchPlaceholder')}
                  value={quickSearch}
                  onChange={(e) => setQuickSearch(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && handleQuickSearch()}
                  error={!!error}
                  helperText={error}
                  disabled={analysisPhase === 'analyzing' || analysisPhase === 'syncing'}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 2
                    }
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleQuickSearch}
                  disabled={
                    analysisPhase === 'analyzing' ||
                    analysisPhase === 'syncing' ||
                    analysisPhase === 'confirming' ||
                    !quickSearch.trim()
                  }
                  sx={{
                    minWidth: 120,
                    borderRadius: 2,
                    boxShadow: '0 4px 12px rgba(25, 118, 210, 0.3)'
                  }}
                >
                  {analysisPhase === 'analyzing' || analysisPhase === 'syncing' ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Analyze'
                  )}
                </Button>
              </Box>

              {/* Progress indicator */}
              {(analysisPhase === 'analyzing' || analysisPhase === 'syncing') && (
                <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />
              )}

              {/* AI Explanation Info */}
              <Box sx={{ p: 2, backgroundColor: 'background.default', borderRadius: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Psychology color="primary" />
                  <Typography variant="subtitle2">
                    AI Explanation Included
                  </Typography>
                  <Chip
                    label="Phi3 & LLaMA 3.1"
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  AI explanation will be automatically generated with your analysis to help you understand the results in plain language
                </Typography>
              </Box>

              {/* Recent searches as chips */}
              {recentAnalyses.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ width: '100%', mb: 1 }}>
                    Recent searches:
                  </Typography>
                  {recentAnalyses.slice(0, 5).map((analysis) => (
                    <Chip
                      key={analysis.id}
                      label={analysis.symbol}
                      size="small"
                      onClick={() => {
                        setQuickSearch(analysis.symbol);
                        handleQuickSearch();
                      }}
                      sx={{
                        cursor: 'pointer',
                        '&:hover': { backgroundColor: 'primary.light', color: 'white' }
                      }}
                    />
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Snapshot */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%', borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                <TrendingUp sx={{ mr: 1 }} />
                Performance
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography color="textSecondary" variant="body2">
                  {t('dashboard.averageScore')}
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, color: parseFloat(averageScore) >= 6 ? "success.main" : parseFloat(averageScore) >= 4 ? "warning.main" : "error.main" }}>
                  {formatScore(parseFloat(averageScore))}/10
                </Typography>
              </Box>
              {bestStock && (
                <Box sx={{ mb: 2 }}>
                  <Typography color="textSecondary" variant="body2">
                    Best Performer
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {bestStock.symbol}
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    Score: {bestStock.score}/10
                  </Typography>
                </Box>
              )}
              <Box>
                <Typography color="textSecondary" variant="body2">
                  {t('dashboard.favoriteSector')}
                </Typography>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                  {topSector}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Middle Section - Recent Analyses */}

        {/* Enhanced Recent Analyses */}
        <Grid item xs={12} md={8}>
          <Card sx={{ borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.1)', height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', fontWeight: 600 }}>
                  <History sx={{ mr: 1 }} />
                  {t('dashboard.recentAnalyses')}
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => navigate('/reports')}
                  sx={{ borderRadius: 2 }}
                >
                  {t('dashboard.viewAll')}
                </Button>
              </Box>

              {analysisLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : recentAnalyses.length > 0 ? (
                <List sx={{ p: 0 }}>
                  {recentAnalyses.slice(0, 5).map((analysis) => (
                    <React.Fragment key={analysis.id}>
                      <ListItem sx={{
                        px: 2,
                        py: 2,
                        borderRadius: 2,
                        mb: 1,
                        backgroundColor: 'rgba(0,0,0,0.02)',
                        '&:hover': {
                          backgroundColor: 'rgba(25, 118, 210, 0.08)',
                          cursor: 'pointer'
                        }
                      }}
                      onClick={() => navigate(`/analysis/${analysis.id}`)}
                      >
                        <ListItemIcon>
                          <Analytics color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Box>
                                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                                  {analysis.symbol}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {analysis.name || `${analysis.symbol} Corporation`}
                                </Typography>
                              </Box>
                              <Box sx={{ textAlign: 'right' }}>
                                <Chip
                                  label={`${analysis.score}/10`}
                                  size="small"
                                  color={getScoreColor(analysis.score)}
                                  sx={{ fontWeight: 600 }}
                                />
                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                                  {analysis.sector}
                                </Typography>
                              </Box>
                            </Box>
                          }
                          secondary={new Date(analysis.analysis_date).toLocaleDateString()}
                        />
                        <IconButton
                          size="small"
                          sx={{ ml: 1 }}
                        >
                          <TrendingUp />
                        </IconButton>
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Alert severity="info" sx={{ borderRadius: 2 }}>
                  {t('dashboard.noAnalysisHistory')}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Market Sentiment Dashboard */}
        <Grid item xs={12} md={4}>
          <Card sx={{ borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.1)', height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                <SentimentSatisfied sx={{ mr: 1 }} />
                Market Sentiment
              </Typography>

              {sentimentStats.total > 0 ? (
                <Box>
                  {/* Sentiment Chart */}
                  <Box sx={{ height: 150, mb: 2 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Positive', value: sentimentStats.positive, color: '#2e7d32' },
                            { name: 'Neutral', value: sentimentStats.neutral, color: '#ed6c02' },
                            { name: 'Negative', value: sentimentStats.negative, color: '#d32f2f' }
                          ]}
                          cx="50%"
                          cy="50%"
                          innerRadius={30}
                          outerRadius={60}
                          dataKey="value"
                        >
                          <Cell fill="#2e7d32" />
                          <Cell fill="#ed6c02" />
                          <Cell fill="#d32f2f" />
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </Box>

                  {/* Sentiment Stats */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Dominant Sentiment
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      {getSentimentIcon(dominantSentiment)}
                      <Typography variant="h6" sx={{ ml: 1, textTransform: 'capitalize', fontWeight: 600 }}>
                        {dominantSentiment}
                      </Typography>
                    </Box>
                  </Box>

                  {/* News Coverage */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      News Coverage
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {sentimentStats.totalNews} articles
                    </Typography>
                  </Box>

                  {/* Average Score */}
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Average Sentiment Score
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: avgSentimentScore > 0 ? 'success.main' : avgSentimentScore < 0 ? 'error.main' : 'warning.main' }}>
                      {formatSentimentScore(avgSentimentScore)}
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <SentimentNeutral sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    No sentiment data available
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Run some analyses to see market sentiment insights
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Bottom Row - Smart Actions Grid */}
        <Grid item xs={12}>
          <Card sx={{ borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Quick Actions
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    size="large"
                    startIcon={<Analytics />}
                    onClick={() => {
                      setQuickSearch('');
                      setError('');
                      document.querySelector('input[placeholder*="search"]')?.focus();
                    }}
                    sx={{
                      py: 2,
                      borderRadius: 3,
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'primary.main',
                        color: 'white',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 25px rgba(25, 118, 210, 0.3)'
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    New Analysis
                  </Button>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    size="large"
                    startIcon={<TrendingUp />}
                    onClick={() => navigate('/compare')}
                    sx={{
                      py: 2,
                      borderRadius: 3,
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'success.main',
                        color: 'white',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 25px rgba(46, 125, 50, 0.3)'
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    Compare Stocks
                  </Button>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    size="large"
                    startIcon={<AccountBalanceWallet />}
                    onClick={() => navigate('/sectors')}
                    sx={{
                      py: 2,
                      borderRadius: 3,
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'warning.main',
                        color: 'white',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 25px rgba(237, 108, 2, 0.3)'
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    Sector Analysis
                  </Button>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    size="large"
                    startIcon={<History />}
                    onClick={() => navigate('/reports')}
                    sx={{
                      py: 2,
                      borderRadius: 3,
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'info.main',
                        color: 'white',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 25px rgba(2, 136, 209, 0.3)'
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    View Reports
                  </Button>
                </Grid>
              </Grid>

              {/* Popular/Recent Stocks */}
              {recentAnalyses.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Quick access to recent analyses:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                    {recentAnalyses.slice(0, 8).map((analysis) => (
                      <Chip
                        key={analysis.id}
                        label={analysis.symbol}
                        size="medium"
                        onClick={() => {
                          setQuickSearch(analysis.symbol);
                          setAnalysisMode('auto');
                          setAnalysisPhase('confirming');
                          setError('');
                          setConfirmDialog(true);
                        }}
                        sx={{
                          cursor: 'pointer',
                          fontWeight: 600,
                          '&:hover': {
                            backgroundColor: 'primary.main',
                            color: 'white',
                            transform: 'scale(1.05)'
                          },
                          transition: 'all 0.2s ease'
                        }}
                        icon={<Star sx={{ fontSize: 16 }} />}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Results Section */}
        {analysisData && analysisPhase === 'completed' && (
          <Grid item xs={12}>
            <Card sx={{ borderRadius: 3, boxShadow: '0 8px 32px rgba(25, 118, 210, 0.15)', border: '1px solid', borderColor: 'primary.light' }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
                    <Analytics sx={{ mr: 2, color: 'primary.main' }} />
                    Analysis Results: {analysisData.symbol}
                  </Typography>
                  <Chip
                    label={analysisData.recommendation}
                    color={getScoreColor(analysisData.score)}
                    size="large"
                    sx={{ fontWeight: 600, fontSize: '1rem', px: 2 }}
                  />
                </Box>

                <Grid container spacing={3}>
                  {/* Company Info */}
                  <Grid item xs={12} md={6}>
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        {analysisData.name}
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                        {analysisData.sector} • {analysisData.industry}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h3" sx={{ fontWeight: 700, mr: 2, color: getScoreColor(analysisData.score) + '.main' }}>
                          {analysisData.score}/10
                        </Typography>
                        <Box>
                          <Typography variant="body1" sx={{ fontWeight: 600 }}>
                            Technical Score
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Based on 14 indicators
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Grid>

                  {/* Quick Stats */}
                  <Grid item xs={12} md={6}>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.default', borderRadius: 2 }}>
                          <Typography variant="h6" color="primary.main" sx={{ fontWeight: 600 }}>
                            {analysisData.creditsUsed}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Credits Used
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.default', borderRadius: 2 }}>
                          <Typography variant="h6" color="success.main" sx={{ fontWeight: 600 }}>
                            14
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Indicators
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Grid>

                  {/* AI Explanation */}
                  {analysisData.explanation && (
                    <Grid item xs={12}>
                      <Box sx={{ p: 3, backgroundColor: 'primary.light', borderRadius: 2, border: '1px solid', borderColor: 'primary.main' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Psychology sx={{ mr: 1, color: 'primary.main' }} />
                          <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            AI Analysis Explanation
                          </Typography>
                          <Chip
                            label="Standard detail"
                            size="small"
                            sx={{ ml: 2 }}
                          />
                        </Box>
                        <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
                          {analysisData.explanation.content || 'Analysis explanation will appear here when available.'}
                        </Typography>
                        {analysisData.explanation.confidence_score && (
                          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                              Confidence:
                            </Typography>
                            <Chip
                              label={`${(analysisData.explanation.confidence_score * 100).toFixed(0)}%`}
                              size="small"
                              color={analysisData.explanation.confidence_score > 0.7 ? 'success' : 'warning'}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>
                  )}

                  {/* Action Buttons */}
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 2 }}>
                      <Button
                        variant="contained"
                        startIcon={<Visibility />}
                        onClick={() => navigate(`/analysis/${analysisData.id || 'latest'}`)}
                        sx={{ borderRadius: 2 }}
                      >
                        View Full Report
                      </Button>
                      <Button
                        variant="outlined"
                        startIcon={<Search />}
                        onClick={() => {
                          setAnalysisData(null);
                          setAnalysisPhase('idle');
                          setQuickSearch('');
                        }}
                        sx={{ borderRadius: 2 }}
                      >
                        New Analysis
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

      </Grid>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog} onClose={handleCancelAnalysis} maxWidth="sm" fullWidth>
        <DialogTitle>
          {analysisMode === 'auto' ? 'Auto Analysis Confirmation' : 'Confirm Stock Analysis'}
        </DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            {analysisMode === 'auto'
              ? `Auto-analyze ${quickSearch} (triggered from dashboard quick search)`
              : `Analyze ${quickSearch} for 1 credit?`
            }
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This will use 1 credit from your balance ({userCredits} remaining)
          </Alert>
          {analysisMode === 'auto' && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              Analysis will start automatically after confirmation
            </Alert>
          )}
          <Alert severity="info" sx={{ mt: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Psychology sx={{ mr: 1 }} />
              AI explanation will be automatically generated with your analysis
            </Box>
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelAnalysis}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirmAnalysis}
            variant="contained"
            disabled={analysisPhase === 'analyzing'}
          >
            {analysisPhase === 'analyzing' ? (
              <CircularProgress size={24} />
            ) : (
              'Confirm Analysis'
            )}
          </Button>
        </DialogActions>
      </Dialog>

    </Container>
  );
};

export default DashboardPage;