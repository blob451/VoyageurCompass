import React from 'react';
import { useTranslation } from 'react-i18next';
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
import { explanationLogger } from '../utils/logger';

const TechnicalExplanation = ({
  analysisId,
  analysisData,
  defaultExpanded = true
}) => {
  const { t, i18n } = useTranslation();
  const [currentDetailLevel, setCurrentDetailLevel] = React.useState('summary'); // 'summary' = Standard level
  const [generateExplanation, { isLoading: isGenerating }] = useGenerateExplanationMutation();
  const {
    data: explanation,
    error: explanationError,
    isLoading: isLoadingExplanation,
    refetch: refetchExplanation
  } = useGetExplanationQuery(
    { analysisId, detailLevel: currentDetailLevel, language: i18n.language },
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
    if (score >= 8) return t('recommendations.strongBuy');
    if (score >= 6) return t('recommendations.buy');
    if (score >= 4) return t('recommendations.hold');
    return t('recommendations.sell');
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
      .filter(([key]) => key !== 'sentiment')
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
      .filter(([, value]) => value !== null && value !== undefined)
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
      await generateExplanation({
        analysisId,
        detailLevel,
        forceRegenerate: true,  // Always force regeneration when user clicks Generate
        language: i18n.language
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
    
    // Only refetch to check if explanation exists for this detail level
    // Don't automatically generate - let the user click Generate if needed
    explanationLogger.workflow(analysisId, 'Detail level changed, checking for existing explanation', {
      symbol,
      detailLevel,
      score
    });
    
    // Just refetch - this will either return existing explanation or empty state
    refetchExplanation();
  };

  const technicalExplanation = explanation?.explanation;
  const topIndicators = getTopIndicators(indicators);
  const topContributors = getTopContributors(weighted_scores);

  return (
    <ExplanationCard
      title={t('explanations.title')}
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
      modelName={technicalExplanation?.model_used}
      timestamp={technicalExplanation?.explained_at}
    >



    </ExplanationCard>
  );
};

export default TechnicalExplanation;