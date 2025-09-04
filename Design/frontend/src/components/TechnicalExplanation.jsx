import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  Speed,
  ShowChart,
  Timeline,
  Insights,
  BarChart,
  CandlestickChart
} from '@mui/icons-material';
import ExplanationCard from './ExplanationCard';
import { 
  useGenerateExplanationMutation, 
  useGetExplanationQuery 
} from '../features/api/apiSlice';
import { explanationLogger, performanceUtils } from '../utils/logger';

const TechnicalExplanation = ({ 
  analysisId, 
  analysisData,
  defaultExpanded = true 
}) => {
  const [currentDetailLevel, setCurrentDetailLevel] = React.useState('summary');
  const [generateExplanation, { isLoading: isGenerating }] = useGenerateExplanationMutation();
  const { 
    data: explanation, 
    error: explanationError, 
    isLoading: isLoadingExplanation,
    refetch: refetchExplanation 
  } = useGetExplanationQuery(
    { analysisId, detailLevel: currentDetailLevel }, 
    { skip: !analysisId }
  );

  if (!analysisData) {
    return null;
  }

  const {
    score,
    indicators = {},
    weighted_scores = {},
    composite_raw,
    symbol,
    name
  } = analysisData;

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

  const getIndicatorIcon = (indicatorKey) => {
    const key = indicatorKey.toLowerCase();
    if (key.includes('sma') || key.includes('moving')) return <Timeline />;
    if (key.includes('rsi')) return <Speed />;
    if (key.includes('macd')) return <ShowChart />;
    if (key.includes('bollinger')) return <BarChart />;
    if (key.includes('volume')) return <Assessment />;
    if (key.includes('candlestick')) return <CandlestickChart />;
    return <Insights />;
  };

  const getTopIndicators = (indicators, count = 5) => {
    return Object.entries(indicators)
      .filter(([key, indicator]) => key !== 'sentiment')
      .map(([key, indicator]) => ({
        key,
        name: key.toUpperCase(),
        score: indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0),
        description: indicator.description || indicator.desc || 'Technical indicator'
      }))
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .slice(0, count);
  };

  const getTopContributors = (weightedScores, count = 5) => {
    return Object.entries(weightedScores)
      .filter(([key, value]) => value !== null && value !== undefined)
      .map(([key, value]) => ({
        key,
        name: key.replace('w_', '').toUpperCase(),
        value: parseFloat(value),
        percentage: (parseFloat(value) / (composite_raw || 1) * 100)
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, count);
  };

  const handleGenerateExplanation = async (detailLevel) => {
    const startTime = performance.now();
    explanationLogger.workflow(analysisId, 'Technical explanation generation started', {
      symbol,
      detailLevel,
      score
    });
    
    try {
      const result = await generateExplanation({
        analysisId,
        detailLevel
      }).unwrap();
      
      const duration = performance.now() - startTime;
      explanationLogger.workflow(analysisId, 'Technical explanation generation completed', {
        symbol,
        detailLevel,
        duration: `${duration.toFixed(2)}ms`,
        success: true
      });
      
      // Refetch to get the newly generated explanation
      refetchExplanation();
    } catch (error) {
      const duration = performance.now() - startTime;
      explanationLogger.error('Failed to generate technical explanation', {
        analysisId,
        symbol,
        detailLevel,
        duration: `${duration.toFixed(2)}ms`,
        error: error.message || error
      });
      // Error will be handled by the ExplanationCard component
      // through the explanationError from the query
    }
  };

  const handleRefreshExplanation = async (detailLevel) => {
    // Update detail level first
    setCurrentDetailLevel(detailLevel);
    
    // Force refetch to see if explanation exists for this detail level
    const result = await refetchExplanation();
    
    // If no explanation exists for this detail level, generate it
    if (!result?.data?.explanation?.content) {
      explanationLogger.workflow(analysisId, 'No explanation found for detail level, generating new one', {
        symbol,
        detailLevel,
        score
      });
      await handleGenerateExplanation(detailLevel);
    }
  };

  const technicalExplanation = explanation?.explanation;
  const topIndicators = getTopIndicators(indicators);
  const topContributors = getTopContributors(weighted_scores);

  return (
    <ExplanationCard
      title="Technical Analysis Summary Explanation"
      analysisId={analysisId}
      explanation={technicalExplanation}
      isLoading={isGenerating || isLoadingExplanation}
      error={explanationError}
      onGenerate={handleGenerateExplanation}
      onRefresh={handleRefreshExplanation}
      defaultExpanded={defaultExpanded}
      variant={currentDetailLevel}
      confidence={technicalExplanation?.confidence}
      method={technicalExplanation?.method}
      timestamp={technicalExplanation?.explained_at}
    >
      {/* Analysis Overview */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color={`${getScoreColor(score)}.main`} gutterBottom>
              {score}/10
            </Typography>
            <Chip 
              label={getScoreLabel(score)} 
              color={getScoreColor(score)}
              size="small"
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Composite Score
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              {symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {name}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Stock Analysis
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              {Object.keys(indicators).length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Technical Indicators
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Analyzed
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Top Performing Indicators */}
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
        Top Performing Indicators
      </Typography>
      
      <TableContainer component={Paper} sx={{ mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Indicator</TableCell>
              <TableCell align="center">Score</TableCell>
              <TableCell>Signal Strength</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {topIndicators.map((indicator) => (
              <TableRow key={indicator.key}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getIndicatorIcon(indicator.key)}
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {indicator.name}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2">
                    {(indicator.score * 10).toFixed(1)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={Math.max(0, Math.min(100, indicator.score * 100))}
                      sx={{ width: 60, height: 6, borderRadius: 3 }}
                      color={indicator.score >= 0.7 ? 'success' : indicator.score >= 0.4 ? 'warning' : 'error'}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {indicator.score >= 0.7 ? 'Strong' : indicator.score >= 0.4 ? 'Moderate' : 'Weak'}
                    </Typography>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Component Contributions */}
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
        Analysis Component Contributions
      </Typography>
      
      <Grid container spacing={1} sx={{ mb: 3 }}>
        {topContributors.map((contributor) => (
          <Grid item xs={12} sm={6} md={4} key={contributor.key}>
            <Paper sx={{ p: 1.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }} gutterBottom>
                {contributor.name}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(0, Math.min(100, contributor.percentage))}
                  sx={{ flex: 1, height: 6, borderRadius: 3 }}
                  color={contributor.percentage >= 15 ? 'success' : contributor.percentage >= 8 ? 'warning' : 'error'}
                />
                <Typography variant="caption">
                  {contributor.percentage.toFixed(1)}%
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                Impact: {contributor.percentage >= 15 ? 'High' : contributor.percentage >= 8 ? 'Medium' : 'Low'}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Analysis Methodology */}
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
        Analysis Framework
      </Typography>
      
      <List dense>
        <ListItem>
          <ListItemIcon>
            <Assessment />
          </ListItemIcon>
          <ListItemText
            primary="Composite Scoring"
            secondary="12 technical indicators with weighted contributions to final score"
          />
        </ListItem>
        
        <ListItem>
          <ListItemIcon>
            <Timeline />
          </ListItemIcon>
          <ListItemText
            primary="Multi-Timeframe Analysis"
            secondary="Short-term, medium-term, and long-term technical signals"
          />
        </ListItem>
        
        <ListItem>
          <ListItemIcon>
            <Insights />
          </ListItemIcon>
          <ListItemText
            primary="Risk Assessment"
            secondary="Volatility, momentum, and market sentiment consideration"
          />
        </ListItem>
      </List>

      {/* Methodology Note */}
      <Box sx={{ mt: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
        <Typography variant="body2" color="text.secondary">
          <strong>Analysis Methodology:</strong> This technical analysis uses a comprehensive 
          framework incorporating 12 technical indicators including moving averages (SMA), 
          momentum oscillators (RSI), trend indicators (MACD), volatility measures (Bollinger Bands), 
          volume analysis, and candlestick patterns. Each indicator is weighted based on its 
          historical predictive accuracy and combined to produce a composite investment score.
        </Typography>
      </Box>
    </ExplanationCard>
  );
};

export default TechnicalExplanation;