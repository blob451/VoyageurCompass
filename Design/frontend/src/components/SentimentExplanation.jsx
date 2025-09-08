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
  LinearProgress
} from '@mui/material';
import {
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  NewspaperOutlined,
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Assessment
} from '@mui/icons-material';
import ExplanationCard from './ExplanationCard';
import { 
  useGenerateExplanationMutation, 
  useGetExplanationQuery 
} from '../features/api/apiSlice';
import { explanationLogger } from '../utils/logger';

const SentimentExplanation = ({ 
  analysisId, 
  sentimentData,
  defaultExpanded = false 
}) => {
  const [currentDetailLevel, setCurrentDetailLevel] = React.useState('standard');
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

  if (!sentimentData || !sentimentData.raw) {
    return null;
  }

  const { raw: sentiment } = sentimentData;
  const {
    label,
    sentiment: score,
    confidence,
    newsCount,
    sources
  } = sentiment;

  const getSentimentIcon = (sentimentLabel) => {
    switch (sentimentLabel?.toLowerCase()) {
      case 'positive':
        return <SentimentSatisfied color="success" />;
      case 'negative':
        return <SentimentDissatisfied color="error" />;
      case 'neutral':
      default:
        return <SentimentNeutral color="warning" />;
    }
  };

  const getSentimentColor = (sentimentLabel) => {
    switch (sentimentLabel?.toLowerCase()) {
      case 'positive':
        return 'success';
      case 'negative':
        return 'error';
      case 'neutral':
      default:
        return 'warning';
    }
  };

  const getTrendIcon = (sentimentLabel) => {
    switch (sentimentLabel?.toLowerCase()) {
      case 'positive':
        return <TrendingUp color="success" />;
      case 'negative':
        return <TrendingDown color="error" />;
      case 'neutral':
      default:
        return <TrendingFlat color="warning" />;
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

  const getImpactLevel = (score, confidence) => {
    const absScore = Math.abs(score || 0);
    const conf = confidence || 0;
    const impact = absScore * conf;
    
    if (impact >= 0.4) return { level: 'High', color: 'error' };
    if (impact >= 0.2) return { level: 'Medium', color: 'warning' };
    return { level: 'Low', color: 'success' };
  };

  const handleGenerateExplanation = async (detailLevel) => {
    const startTime = performance.now();
    explanationLogger.workflow(analysisId, 'Sentiment explanation generation started', {
      sentimentLabel: label,
      sentimentScore: score,
      detailLevel,
      newsCount
    });
    
    try {
      await generateExplanation({
        analysisId,
        detailLevel
      }).unwrap();
      
      const duration = performance.now() - startTime;
      explanationLogger.workflow(analysisId, 'Sentiment explanation generation completed', {
        sentimentLabel: label,
        detailLevel,
        duration: `${duration.toFixed(2)}ms`,
        success: true
      });
      
      // Refetch to get the newly generated explanation
      refetchExplanation();
    } catch (error) {
      const duration = performance.now() - startTime;
      explanationLogger.error('Failed to generate sentiment explanation', {
        analysisId,
        sentimentLabel: label,
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
        sentimentLabel: label,
        detailLevel,
        newsCount
      });
      await handleGenerateExplanation(detailLevel);
    }
  };

  const sentimentExplanation = explanation?.explanation;
  const impact = getImpactLevel(score, confidence);

  return (
    <ExplanationCard
      title="News Sentiment Analysis Explanation"
      analysisId={analysisId}
      explanation={sentimentExplanation}
      isLoading={isGenerating || isLoadingExplanation}
      error={explanationError}
      onGenerate={handleGenerateExplanation}
      onRefresh={handleRefreshExplanation}
      defaultExpanded={defaultExpanded}
      confidence={sentimentExplanation?.confidence}
      method={sentimentExplanation?.method}
      timestamp={sentimentExplanation?.explained_at}
    >
      {/* Sentiment Overview */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
              {getSentimentIcon(label)}
            </Box>
            <Typography variant="h6" color={`${getSentimentColor(label)}.main`}>
              {label?.toUpperCase() || 'NEUTRAL'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Overall Sentiment
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
              {getTrendIcon(label)}
            </Box>
            <Typography variant="h6">
              {formatSentimentScore(score)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sentiment Score
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Key Metrics */}
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
        Sentiment Analysis Metrics
      </Typography>
      
      <List dense>
        <ListItem>
          <ListItemIcon>
            <Assessment />
          </ListItemIcon>
          <ListItemText
            primary="Confidence Level"
            secondary={
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                <LinearProgress
                  variant="determinate"
                  value={(confidence || 0) * 100}
                  sx={{ width: 100, height: 6, borderRadius: 3 }}
                  color={confidence >= 0.8 ? 'success' : confidence >= 0.6 ? 'warning' : 'error'}
                />
                <span style={{ fontSize: '0.875rem' }}>
                  {confidence ? `${(confidence * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </span>
            }
          />
        </ListItem>
        
        <ListItem>
          <ListItemIcon>
            <NewspaperOutlined />
          </ListItemIcon>
          <ListItemText
            primary="News Articles Analyzed"
            secondary={`${newsCount || 0} articles from financial news sources`}
          />
        </ListItem>
        
        <ListItem>
          <ListItemIcon>
            <TrendingUp />
          </ListItemIcon>
          <ListItemText
            primary="Market Impact Potential"
            secondary={
              <Typography variant="body2" component="div" sx={{ mt: 0.5 }}>
                <Chip
                  label={impact.level}
                  size="small"
                  color={impact.color}
                  variant="outlined"
                />
              </Typography>
            }
          />
        </ListItem>
      </List>

      {/* Sources Breakdown */}
      {sources && Object.keys(sources).length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
            News Sources Analysis
          </Typography>
          
          <Grid container spacing={1}>
            {Object.entries(sources).map(([source, data]) => {
              const avgScore = data.avg_score || 0;
              const sourceLabel = avgScore > 0.1 ? 'positive' : avgScore < -0.1 ? 'negative' : 'neutral';
              
              return (
                <Grid item xs={12} sm={6} md={4} key={source}>
                  <Paper sx={{ p: 1.5, textAlign: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 1 }}>
                      {getSentimentIcon(sourceLabel)}
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {source}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" display="block">
                      {data.count || 0} articles
                    </Typography>
                    <Typography variant="caption" color={`${getSentimentColor(sourceLabel)}.main`}>
                      {formatSentimentScore(avgScore)}
                    </Typography>
                  </Paper>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      )}

      {/* FinBERT Model Info */}
      <Box sx={{ mt: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
        <Typography variant="body2" color="text.secondary">
          <strong>About FinBERT:</strong> This sentiment analysis uses FinBERT, a financial domain-specific 
          BERT model trained on financial text. It's specifically designed to understand financial 
          language and context, providing more accurate sentiment analysis for investment decisions 
          compared to general-purpose sentiment models.
        </Typography>
      </Box>
    </ExplanationCard>
  );
};

export default SentimentExplanation;