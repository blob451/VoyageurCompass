import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
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
import SentimentExplanation from '../components/SentimentExplanation';
import TechnicalExplanation from '../components/TechnicalExplanation';

const AnalysisResultsPage = () => {
  const { analysisId } = useParams();
  const navigate = useNavigate();
  
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
    if (score >= 8) return 'Strong Buy';
    if (score >= 6) return 'Buy';
    if (score >= 4) return 'Hold';
    return 'Sell';
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
            <Typography variant="h6">Loading analysis results...</Typography>
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
            Back to Dashboard
          </Button>
        </Box>
        <Alert severity="error">
          {error.data?.error || 'Failed to load analysis results'}
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
            Back to Dashboard
          </Button>
        </Box>
        <Alert severity="warning">
          Analysis not found
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
          Back to Dashboard
        </Button>
        <Button
          variant="outlined"
          startIcon={<TrendingUp />}
          onClick={() => navigate('/stocks', { 
            state: { searchTicker: analysisData.symbol, autoAnalyze: true } 
          })}
        >
          Run New Analysis
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
              Analysis Date: {new Date(analysisData.analysis_date).toLocaleString()}
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
          Technical Indicators
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Indicator</TableCell>
                <TableCell>Score</TableCell>
                <TableCell>Description</TableCell>
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
                
                // Special handling for sentiment indicator
                let indicatorName = key.toUpperCase();
                let description = indicator.description || indicator.desc || 'Technical indicator';
                let sentimentIcon = null;
                
                if (isSentiment) {
                  const sentimentLabel = indicator.raw?.label;
                  const newsCount = indicator.raw?.newsCount;
                  sentimentIcon = getSentimentIcon(sentimentLabel);
                  indicatorName = 'NEWS SENTIMENT';
                  description = `${sentimentLabel?.toUpperCase() || 'NEUTRAL'} sentiment from ${newsCount || 0} news articles`;
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
                            Confidence: {(indicator.raw.confidence * 100).toFixed(1)}%
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

        {/* Weighted Scores Section */}
        {analysisData.weighted_scores && (
          <>
            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
              Component Contribution Analysis
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Component</TableCell>
                    <TableCell align="center">Weighted Score</TableCell>
                    <TableCell align="center">Contribution</TableCell>
                    <TableCell align="center">Impact</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(analysisData.weighted_scores)
                    .filter(([, value]) => value !== null && value !== undefined)
                    .sort(([,a], [,b]) => parseFloat(b) - parseFloat(a))
                    .map(([key, value]) => {
                      const displayValue = parseFloat(value);
                      const percentage = (displayValue / (analysisData.composite_raw || 1) * 100);
                      const componentName = key.replace('w_', '').toUpperCase();
                      
                      // Enhanced score display (scale up for better readability)
                      const scaledScore = (displayValue * 100).toFixed(1);
                      
                      return (
                        <TableRow key={key}>
                          <TableCell sx={{ fontWeight: 500 }}>
                            {componentName}
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={Math.max(0, Math.min(100, percentage))}
                                sx={{ width: 50, height: 6, borderRadius: 3 }}
                                color={percentage >= 15 ? 'success' : percentage >= 8 ? 'warning' : 'error'}
                              />
                              <Typography variant="body2" sx={{ minWidth: 45 }}>
                                {scaledScore}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Chip 
                              label={`${percentage.toFixed(1)}%`}
                              size="small"
                              color={percentage >= 15 ? 'success' : percentage >= 8 ? 'warning' : 'default'}
                              variant={percentage >= 10 ? 'filled' : 'outlined'}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Typography variant="body2" color="text.secondary">
                              {percentage >= 15 ? 'High' : percentage >= 8 ? 'Medium' : 'Low'}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                </TableBody>
              </Table>
            </TableContainer>
            <Box sx={{ mt: 2, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary">
                <strong>Note:</strong> Weighted scores are scaled (×100) for readability. 
                Contribution percentages show each component's relative impact on the final analysis.
                Components are sorted by contribution from highest to lowest.
              </Typography>
            </Box>
          </>
        )}

        {/* Sentiment Analysis Section */}
        {analysisData.indicators?.sentiment && (
          <>
            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
              News Sentiment Analysis
            </Typography>
            <Grid container spacing={3} sx={{ mb: 3 }}>
              {/* Sentiment Overview Card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                      {getSentimentIcon(analysisData.indicators.sentiment.raw?.label)}
                      <Typography variant="h6">
                        Overall Sentiment
                      </Typography>
                    </Box>
                    <Typography variant="h4" color={`${getSentimentColor(analysisData.indicators.sentiment.raw?.label)}.main`} gutterBottom>
                      {analysisData.indicators.sentiment.raw?.label?.toUpperCase() || 'NEUTRAL'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Score: {formatSentimentScore(analysisData.indicators.sentiment.raw?.sentiment)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Confidence: {analysisData.indicators.sentiment.raw?.confidence ? 
                        `${(analysisData.indicators.sentiment.raw.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
                      <NewspaperOutlined fontSize="small" color="action" />
                      <Typography variant="body2" color="text.secondary">
                        Based on {analysisData.indicators.sentiment.raw?.newsCount || 0} news articles
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
                      Impact on Analysis
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Contribution to Final Score
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={10} // 10% weight
                        sx={{ height: 8, borderRadius: 4, mb: 1 }}
                        color={getSentimentColor(analysisData.indicators.sentiment.raw?.label)}
                      />
                      <Typography variant="body2">
                        10% Weight in Composite Score
                      </Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      Sentiment analysis uses FinBERT to evaluate financial news and incorporate 
                      market sentiment into the technical analysis framework.
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
                        News Sources Analysis
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Source</TableCell>
                              <TableCell align="center">Articles</TableCell>
                              <TableCell align="center">Average Sentiment</TableCell>
                              <TableCell align="center">Impact</TableCell>
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
            Analysis Summary
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>Composite Score:</strong> {analysisData.score}/10 ({getScoreLabel(analysisData.score)})
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>Analysis Horizon:</strong> {analysisData.horizon || 'Standard'}
          </Typography>
          {analysisData.composite_raw && (
            <Typography variant="body2" paragraph>
              <strong>Raw Score:</strong> {analysisData.composite_raw.toFixed(4)}
            </Typography>
          )}
          <Typography variant="body2" color="text.secondary">
            This analysis was performed on {new Date(analysisData.analysis_date).toLocaleDateString()} 
            using 12 technical indicators with weighted scoring methodology.
          </Typography>
        </Box>
      </Paper>

      {/* AI-Powered Explanations Section */}
      <Box sx={{ mt: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology color="primary" />
            <Typography variant="h5">
              AI-Powered Analysis Explanations
            </Typography>
          </Box>
          
          {/* Service Status Indicator */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {explanationStatus?.status?.llm_available ? (
              <Chip 
                label="LLaMA 3.1 70B Online" 
                color="success" 
                size="small"
                variant="outlined"
              />
            ) : statusError ? (
              <Chip 
                label="Service Unavailable" 
                color="error" 
                size="small"
                variant="outlined"
              />
            ) : (
              <Chip 
                label="Template Mode" 
                color="warning" 
                size="small"
                variant="outlined"
              />
            )}
          </Box>
        </Box>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Get natural language explanations powered by LLaMA 3.1 70B to better understand your analysis results.
          {!explanationStatus?.status?.llm_available && (
            <Typography component="span" color="warning.main" sx={{ ml: 1 }}>
              LLM service unavailable - using template explanations.
            </Typography>
          )}
        </Typography>

        {/* Technical Analysis Explanation */}
        <TechnicalExplanation
          analysisId={analysisId}
          analysisData={analysisData}
          defaultExpanded={true}
        />

        {/* Sentiment Analysis Explanation (only if sentiment data exists) */}
        {analysisData.indicators?.sentiment && (
          <SentimentExplanation
            analysisId={analysisId}
            sentimentData={analysisData.indicators.sentiment}
            defaultExpanded={false}
          />
        )}
      </Box>
    </Container>
  );
};

export default AnalysisResultsPage;