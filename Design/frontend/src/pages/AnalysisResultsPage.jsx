import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  Container,
  Paper,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  CircularProgress,
  Alert,
  Button,
  IconButton,
  Card,
  CardContent,
  Grid,
  Divider
} from '@mui/material';
import { 
  ArrowBack, 
  TrendingUp, 
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  NewspaperOutlined,
  Psychology
} from '@mui/icons-material';
import { useGetAnalysisByIdQuery, useGetExplanationStatusQuery } from '../features/api/apiSlice';
import TechnicalExplanation from '../components/TechnicalExplanation';

const AnalysisResultsPage = () => {
  const { analysisId } = useParams();
  const navigate = useNavigate();
  const { t } = useTranslation('common');
  
  const {
    data: analysisData,
    error,
    isLoading
  } = useGetAnalysisByIdQuery(analysisId);

  const {
    data: explanationStatus,
    error: statusError
  } = useGetExplanationStatusQuery();

  const getScoreColor = (score) => {
    if (score >= 7) return 'success';
    if (score >= 4) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score) => {
    if (score >= 8) return t('recommendations.buy');
    if (score >= 6) return t('recommendations.buy');
    if (score >= 4) return t('recommendations.hold');
    return t('recommendations.sell');
  };

  // Sentiment helper functions
  const getSentimentIcon = (label) => {
    switch (label?.toLowerCase()) {
      case 'positive':
        return <SentimentSatisfied color="success" />;
      case 'negative':
        return <SentimentDissatisfied color="error" />;
      case 'neutral':
      default:
        return <SentimentNeutral color="warning" />;
    }
  };

  const getSentimentColor = (label) => {
    switch (label?.toLowerCase()) {
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

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6">{t('analysis.loading')}</Typography>
          </Box>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 3 }}>
          <Button
            startIcon={<ArrowBack />}
            onClick={() => navigate('/dashboard')}
            sx={{ mb: 2 }}
          >
            {t('analysis.backToDashboard')}
          </Button>
        </Box>
        <Alert severity="error">
          {error.data?.error || t('analysis.failedToLoad')}
        </Alert>
      </Container>
    );
  }

  if (!analysisData) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 3 }}>
          <Button
            startIcon={<ArrowBack />}
            onClick={() => navigate('/dashboard')}
            sx={{ mb: 2 }}
          >
            {t('analysis.backToDashboard')}
          </Button>
        </Box>
        <Alert severity="warning">
          {t('analysis.analysisNotFound')}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header with back button */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/dashboard')}
        >
          {t('analysis.backToDashboard')}
        </Button>
        <Button
          variant="outlined"
          startIcon={<TrendingUp />}
          onClick={() => navigate('/stocks', { 
            state: { searchTicker: analysisData.symbol, autoAnalyze: true } 
          })}
        >
          {t('analysis.runNewAnalysis')}
        </Button>
      </Box>

      {/* Analysis Results */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              {analysisData.symbol} - {analysisData.name}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
              <Chip label={analysisData.sector} />
              <Chip label={analysisData.industry} variant="outlined" />
            </Box>
            <Typography variant="body2" color="text.secondary">
              {t('analysis.analysisDate')}: {new Date(analysisData.analysis_date).toLocaleString()}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h2" color={`${getScoreColor(analysisData.score)}.main`}>
              {analysisData.score}/10
            </Typography>
            <Chip 
              label={getScoreLabel(analysisData.score)} 
              color={getScoreColor(analysisData.score)}
              size="large"
            />
          </Box>
        </Box>

        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          {t('analysis.technicalIndicators')}
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t('analysis.indicator')}</TableCell>
                <TableCell>{t('dashboard.averageScore')}</TableCell>
                <TableCell>{t('analysis.description')}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {analysisData.indicators && Object.entries(analysisData.indicators).map(([key, indicator]) => {
                let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                
                // Handle NaN values
                if (isNaN(score) || !isFinite(score)) {
                  score = 0;
                }
                
                const displayScore = score * 10;
                const isValidScore = !isNaN(displayScore) && isFinite(displayScore);
                const isSentiment = key.toLowerCase() === 'sentiment';
                
                // Get translated indicator name and description
                let indicatorName = t(`indicators.${key}`, key.toUpperCase());
                let description = t(`indicators.descriptions.${key}`, indicator.description || indicator.desc || 'Technical indicator');
                let sentimentIcon = null;
                
                if (isSentiment) {
                  const sentimentLabel = indicator.raw?.label;
                  const newsCount = indicator.raw?.newsCount;
                  sentimentIcon = getSentimentIcon(sentimentLabel);
                  indicatorName = t('analysis.newsSentiment');
                  description = t('analysis.sentimentFromNews', { sentiment: sentimentLabel?.toUpperCase() || t('dashboard.neutral').toUpperCase(), count: newsCount || 0 });
                }
                
                return (
                  <TableRow key={key} sx={isSentiment ? { backgroundColor: 'rgba(25, 118, 210, 0.04)' } : {}}>
                    <TableCell sx={{ fontWeight: 500 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {sentimentIcon}
                        {indicatorName}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={Math.max(0, Math.min(100, score * 100))}
                          sx={{ width: 60, height: 8, borderRadius: 4 }}
                          color={isSentiment ? 
                            getSentimentColor(indicator.raw?.label) : 
                            (score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error')
                          }
                        />
                        <Typography variant="body2">
                          {isValidScore ? displayScore.toFixed(1) : 'N/A'}
                          {isSentiment && indicator.raw?.sentiment && (
                            <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                              ({formatSentimentScore(indicator.raw.sentiment)})
                            </Typography>
                          )}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {description}
                        {isSentiment && indicator.raw?.confidence && (
                          <Typography component="span" variant="caption" display="block" color="text.secondary">
                            {t('analysis.confidence')}: {(indicator.raw.confidence * 100).toFixed(1)}%
                          </Typography>
                        )}
                      </Typography>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>


        {/* Sentiment Analysis Section */}
        {analysisData.indicators?.sentiment && (
          <>
            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
              {t('analysis.newsSentimentAnalysis')}
            </Typography>
            <Grid container spacing={3} sx={{ mb: 3 }}>
              {/* Sentiment Overview Card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                      {getSentimentIcon(analysisData.indicators.sentiment.raw?.label)}
                      <Typography variant="h6">
                        {t('analysis.overallSentiment')}
                      </Typography>
                    </Box>
                    <Typography variant="h4" color={`${getSentimentColor(analysisData.indicators.sentiment.raw?.label)}.main`} gutterBottom>
                      {analysisData.indicators.sentiment.raw?.label?.toUpperCase() || 'NEUTRAL'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {t('analysis.sentimentScore')}: {formatSentimentScore(analysisData.indicators.sentiment.raw?.sentiment)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {t('analysis.confidence')}: {analysisData.indicators.sentiment.raw?.confidence ?
                        `${(analysisData.indicators.sentiment.raw.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
                      <NewspaperOutlined fontSize="small" color="action" />
                      <Typography variant="body2" color="text.secondary">
                        {t('analysis.basedOnNews', { count: analysisData.indicators.sentiment.raw?.newsCount || 0 })}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Sentiment Impact Card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {t('analysis.impactOnAnalysis')}
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {t('analysis.contributionToScore')}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={10} // 10% weight
                        sx={{ height: 8, borderRadius: 4, mb: 1 }}
                        color={getSentimentColor(analysisData.indicators.sentiment.raw?.label)}
                      />
                      <Typography variant="body2">
                        {t('analysis.weightInScore')}
                      </Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      {t('analysis.sentimentDescription')}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* News Sources Breakdown */}
              {analysisData.indicators.sentiment.raw?.sources && 
               Object.keys(analysisData.indicators.sentiment.raw.sources).length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {t('analysis.newsSourcesAnalysis')}
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>{t('analysis.source')}</TableCell>
                              <TableCell align="center">{t('analysis.articles')}</TableCell>
                              <TableCell align="center">{t('analysis.averageSentiment')}</TableCell>
                              <TableCell align="center">{t('analysis.impact')}</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(analysisData.indicators.sentiment.raw.sources).map(([source, data]) => {
                              const avgScore = data.avg_score || 0;
                              const sentimentLabel = avgScore > 0.1 ? 'positive' : avgScore < -0.1 ? 'negative' : 'neutral';
                              return (
                                <TableRow key={source}>
                                  <TableCell sx={{ fontWeight: 500 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <NewspaperOutlined fontSize="small" color="action" />
                                      {source}
                                    </Box>
                                  </TableCell>
                                  <TableCell align="center">
                                    <Chip 
                                      label={data.count || 0} 
                                      size="small" 
                                      variant="outlined"
                                    />
                                  </TableCell>
                                  <TableCell align="center">
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                                      {getSentimentIcon(sentimentLabel)}
                                      <Typography variant="body2">
                                        {formatSentimentScore(avgScore)}
                                      </Typography>
                                    </Box>
                                  </TableCell>
                                  <TableCell align="center">
                                    <LinearProgress
                                      variant="determinate"
                                      value={Math.max(0, Math.min(100, (Math.abs(avgScore) + 0.5) * 50))}
                                      sx={{ width: 50, height: 6, borderRadius: 3 }}
                                      color={getSentimentColor(sentimentLabel)}
                                    />
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          </>
        )}

        {/* Analysis Summary */}
        <Box sx={{ mt: 4, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
          <Typography variant="h6" gutterBottom>
            {t('analysis.analysisSummary')}
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>{t('analysis.compositeScore')}:</strong> {analysisData.score}/10 ({getScoreLabel(analysisData.score)})
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>{t('analysis.analysisHorizon')}:</strong> {analysisData.horizon || t('analysis.standard')}
          </Typography>
          {analysisData.composite_raw && (
            <Typography variant="body2" paragraph>
              <strong>{t('analysis.rawScore')}:</strong> {analysisData.composite_raw.toFixed(4)}
            </Typography>
          )}
          <Typography variant="body2" color="text.secondary">
            {t('analysis.analysisPerformed', { date: new Date(analysisData.analysis_date).toLocaleDateString() })}
          </Typography>
        </Box>
      </Paper>

      {/* AI-Powered Explanations Section */}
      <Box sx={{ mt: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology color="primary" />
            <Typography variant="h5">
              {t('analysis.aiExplanations')}
            </Typography>
          </Box>
          
          {/* Service Status Indicator */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {explanationStatus?.status?.llm_available ? (
              <Chip 
                label={t('analysis.multiModelActive')} 
                color="success" 
                size="small"
                variant="outlined"
              />
            ) : statusError ? (
              <Chip 
                label={t('analysis.serviceUnavailable')} 
                color="error" 
                size="small"
                variant="outlined"
              />
            ) : (
              <Chip 
                label={t('analysis.templateMode')} 
                color="warning" 
                size="small"
                variant="outlined"
              />
            )}
          </Box>
        </Box>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          {t('analysis.explanationDescription')}
          {!explanationStatus?.status?.llm_available && (
            <Typography component="span" color="warning.main" sx={{ ml: 1 }}>
              {t('analysis.llmUnavailable')}
            </Typography>
          )}
        </Typography>

        {/* Model Information Cards */}
        {explanationStatus?.status?.llm_available && (
          <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
            <Paper 
              sx={{ 
                p: 2, 
                flex: '1 1 300px', 
                minWidth: 300,
                background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
                border: '1px solid rgba(33, 150, 243, 0.2)'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Chip 
                  label="Phi3 (3.8B)" 
                  color="primary" 
                  size="small" 
                  variant="filled"
                />
                <Typography variant="subtitle2" color="primary">
                  {t('analysis.quickAnalysis')}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {t('analysis.phi3Description')}
              </Typography>
            </Paper>

            <Paper 
              sx={{ 
                p: 2, 
                flex: '1 1 300px', 
                minWidth: 300,
                background: 'linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%)',
                border: '1px solid rgba(255, 152, 0, 0.2)'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Chip 
                  label="LLaMA 3.1 (8B)" 
                  color="warning" 
                  size="small" 
                  variant="filled"
                />
                <Typography variant="subtitle2" color="warning.dark">
                  {t('analysis.detailedAnalysis')}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {t('analysis.llamaDescription')}
              </Typography>
            </Paper>
          </Box>
        )}

        {/* Technical Analysis Explanation */}
        <TechnicalExplanation
          analysisId={analysisId}
          analysisData={analysisData}
          defaultExpanded={true}
        />

      </Box>
    </Container>
  );
};

export default AnalysisResultsPage;