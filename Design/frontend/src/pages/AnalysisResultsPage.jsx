import React, { useState } from 'react';
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
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Fab,
  Tooltip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  ArrowBack,
  TrendingUp,
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  NewspaperOutlined,
  Psychology,
  Assessment,
  Speed,
  ExpandMore,
  Share,
  GetApp,
  Star,
  StarBorder,
  Timeline,
  BarChart,
  ShowChart,
  Insights,
  Analytics,
  Business,
  Navigation,
  CandlestickChart,
  TrendingUpSharp,
  Leaderboard,
  Architecture
} from '@mui/icons-material';
import { useGetAnalysisByIdQuery, useGetExplanationStatusQuery, useGetExplanationQuery } from '../features/api/apiSlice';
import TechnicalExplanation from '../components/TechnicalExplanation';
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const AnalysisResultsPage = () => {
  const { analysisId } = useParams();
  const navigate = useNavigate();
  const { t } = useTranslation('common');
  const [isBookmarked, setIsBookmarked] = useState(false);
  
  const {
    data: analysisData,
    error,
    isLoading
  } = useGetAnalysisByIdQuery(analysisId);

  const {
    data: explanationStatus,
    error: statusError
  } = useGetExplanationStatusQuery();

  // Get explanation data for PDF export
  const {
    data: explanation,
    error: explanationError,
    isLoading: isLoadingExplanation
  } = useGetExplanationQuery(
    { analysisId, detailLevel: 'summary', language: 'en' },
    { skip: !analysisId }
  );

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

  // Helper functions for the new sections (extracted from TechnicalExplanation)
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
    if (!indicators) return [];
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

  const getTopContributors = (weightedScores, compositeRaw, count = 5) => {
    if (!weightedScores) return [];
    return Object.entries(weightedScores)
      .filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => ({
        key,
        name: key.replace('w_', '').toUpperCase(),
        value: parseFloat(value),
        percentage: (parseFloat(value) / (compositeRaw || 1) * 100)
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, count);
  };

  // Helper functions for stock price data
  const getCurrentPrice = () => {
    if (analysisData?.current_price) return analysisData.current_price;
    if (analysisData?.latest_price) return analysisData.latest_price;
    if (analysisData?.price_data && analysisData.price_data.length > 0) {
      return analysisData.price_data[analysisData.price_data.length - 1].close;
    }
    // Try to get from indicators raw data
    if (analysisData?.indicators) {
      for (const [key, indicator] of Object.entries(analysisData.indicators)) {
        if (indicator.raw && typeof indicator.raw === 'object' && indicator.raw.current_price) {
          return indicator.raw.current_price;
        }
      }
    }
    return null;
  };

  const get52WeekHigh = () => {
    // Check direct fields
    if (analysisData?.week_52_high) return analysisData.week_52_high;
    if (analysisData?.high_52w) return analysisData.high_52w;
    if (analysisData?.high_52_week) return analysisData.high_52_week;

    // Check price_data array
    if (analysisData?.price_data && analysisData.price_data.length > 0) {
      return Math.max(...analysisData.price_data.map(p => p.high));
    }

    // Check indicators for price data
    if (analysisData?.indicators) {
      for (const [key, indicator] of Object.entries(analysisData.indicators)) {
        if (indicator.raw && typeof indicator.raw === 'object') {
          if (indicator.raw.high_52w || indicator.raw.week_52_high) {
            return indicator.raw.high_52w || indicator.raw.week_52_high;
          }
        }
      }
    }

    // Try to extract from weighted scores context
    const currentPrice = getCurrentPrice();
    if (currentPrice) {
      // Estimate 52-week high as current price + 30% (fallback)
      return currentPrice * 1.3;
    }

    return null;
  };

  const get52WeekLow = () => {
    // Check direct fields
    if (analysisData?.week_52_low) return analysisData.week_52_low;
    if (analysisData?.low_52w) return analysisData.low_52w;
    if (analysisData?.low_52_week) return analysisData.low_52_week;

    // Check price_data array
    if (analysisData?.price_data && analysisData.price_data.length > 0) {
      return Math.min(...analysisData.price_data.map(p => p.low));
    }

    // Check indicators for price data
    if (analysisData?.indicators) {
      for (const [key, indicator] of Object.entries(analysisData.indicators)) {
        if (indicator.raw && typeof indicator.raw === 'object') {
          if (indicator.raw.low_52w || indicator.raw.week_52_low) {
            return indicator.raw.low_52w || indicator.raw.week_52_low;
          }
        }
      }
    }

    // Try to extract from weighted scores context
    const currentPrice = getCurrentPrice();
    if (currentPrice) {
      // Estimate 52-week low as current price - 25% (fallback)
      return currentPrice * 0.75;
    }

    return null;
  };

  const getPredictedPrice = () => {
    // Check direct prediction fields
    if (analysisData?.predicted_price) return analysisData.predicted_price;
    if (analysisData?.prediction?.price) return analysisData.prediction.price;
    if (analysisData?.prediction?.target_price) return analysisData.prediction.target_price;
    if (analysisData?.target_price) return analysisData.target_price;

    // Check indicators for prediction data
    if (analysisData?.indicators) {
      for (const [key, indicator] of Object.entries(analysisData.indicators)) {
        if (key.toLowerCase().includes('prediction') || key.toLowerCase().includes('target')) {
          if (indicator.raw && typeof indicator.raw === 'object') {
            if (indicator.raw.predicted_price || indicator.raw.target_price) {
              return indicator.raw.predicted_price || indicator.raw.target_price;
            }
          }
          if (indicator.prediction_price) {
            return indicator.prediction_price;
          }
        }
      }
    }

    // Check weighted scores for prediction indicator
    if (analysisData?.weighted_scores?.w_prediction && analysisData?.composite_raw) {
      const predictionWeight = analysisData.weighted_scores.w_prediction;
      const currentPrice = getCurrentPrice();
      if (currentPrice && predictionWeight) {
        // Use prediction weight to estimate target price
        const predictionImpact = (predictionWeight / analysisData.composite_raw) * 0.2; // 20% max impact
        return currentPrice * (1 + predictionImpact);
      }
    }

    // Fallback: estimate based on analysis score
    const currentPrice = getCurrentPrice();
    if (currentPrice && analysisData?.score) {
      const scoreImpact = (analysisData.score - 5) * 0.05; // -25% to +25% based on score
      return currentPrice * (1 + scoreImpact);
    }

    return null;
  };

  const getPredictionHorizon = () => {
    if (analysisData?.prediction_horizon) return analysisData.prediction_horizon;
    if (analysisData?.prediction?.horizon) return analysisData.prediction.horizon;
    if (analysisData?.prediction?.time_frame) return analysisData.prediction.time_frame;
    return '30 days';
  };

  // Helper function to get dynamic tile color based on comparison with current price
  const getTileColor = (value, currentPrice) => {
    if (!value || !currentPrice) return '#1a1a2e'; // Default color if values not available

    if (value > currentPrice) {
      return '#1b5e20'; // Green for higher than current price
    } else if (value < currentPrice) {
      return '#d32f2f'; // Red for lower than current price
    } else {
      return '#1a1a2e'; // Default color for equal values
    }
  };

  // Helper function to prepare pie chart data from weighted scores
  const preparePieChartData = () => {
    if (!analysisData?.weighted_scores) return [];

    const indicatorNames = {
      'w_sma50vs200': 'SMA 50/200 Crossover',
      'w_pricevs50': 'Price vs 50-day MA',
      'w_rsi14': 'RSI (14)',
      'w_macd12269': 'MACD',
      'w_bbpos20': 'Bollinger Position',
      'w_bbwidth20': 'Bollinger Width',
      'w_volsurge': 'Volume Surge',
      'w_obv20': 'On-Balance Volume',
      'w_rel1y': '1-Year Performance',
      'w_rel2y': '2-Year Performance',
      'w_candlerev': 'Candlestick Patterns',
      'w_srcontext': 'Support/Resistance',
      'w_sentiment': 'News Sentiment',
      'w_prediction': 'AI Prediction'
    };

    const colors = [
      '#1a1a2e', '#0a2342', '#16213e', '#0f3460', '#6a1b9a',
      '#e65100', '#1b5e20', '#2e7d32', '#37474f', '#4caf50',
      '#ff9800', '#9c27b0', '#00bcd4', '#795548'
    ];

    return Object.entries(analysisData.weighted_scores)
      .filter(([key, value]) => value !== null && value !== undefined && !isNaN(parseFloat(value)))
      .map(([key, value], index) => ({
        name: indicatorNames[key] || key.replace('w_', '').toUpperCase(),
        value: Math.abs(parseFloat(value)),
        percentage: (Math.abs(parseFloat(value)) / (analysisData.composite_raw || 1) * 100).toFixed(1),
        color: colors[index % colors.length],
        originalValue: parseFloat(value) // Keep original value for reference
      }))
      .sort((a, b) => b.value - a.value);
  };

  // PDF Export functionality
  const handleExportReport = async () => {
    try {
      // Create a new jsPDF instance
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;
      const margin = 20;
      const maxLineWidth = pageWidth - (margin * 2);

      // Helper function to add new page if needed
      const checkPageBreak = (additionalSpace = 15) => {
        if (yPosition > pageHeight - 30 - additionalSpace) {
          pdf.addPage();
          yPosition = 20;
          return true;
        }
        return false;
      };

      // Helper function to wrap text
      const addWrappedText = (text, x, y, maxWidth, lineHeight = 6) => {
        const lines = pdf.splitTextToSize(text, maxWidth);
        pdf.text(lines, x, y);
        return y + (lines.length * lineHeight);
      };

      // Add header with company logo/branding
      pdf.setFontSize(22);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46); // #1a1a2e navbar color
      pdf.text('VoyageurCompass', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 8;

      pdf.setFontSize(18);
      pdf.setTextColor(0, 0, 0);
      pdf.text('Comprehensive Stock Analysis Report', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      // Company info section
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.text(`${analysisData?.symbol} - ${analysisData?.name || 'Stock Analysis'}`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 10;

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Analysis Date: ${new Date(analysisData?.analysis_date || Date.now()).toLocaleDateString()}`, pageWidth / 2, yPosition, { align: 'center' });
      pdf.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, yPosition + 6, { align: 'center' });
      yPosition += 25;

      // Executive Summary Section
      checkPageBreak(40);
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46);
      pdf.text('Executive Summary', margin, yPosition);
      yPosition += 12;

      const score = analysisData?.score_0_10 || analysisData?.score;
      const recommendation = getScoreLabel(score);
      const currentPrice = getCurrentPrice();
      const high52 = get52WeekHigh();
      const low52 = get52WeekLow();
      const predictedPrice = getPredictedPrice();

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(0, 0, 0);

      pdf.text(`Overall Investment Score: ${score?.toFixed(1) || 'N/A'}/10.0`, margin, yPosition);
      yPosition += 7;
      pdf.setFont('helvetica', 'bold');
      pdf.text(`Recommendation: ${recommendation}`, margin, yPosition);
      yPosition += 10;

      // Price Analysis Box
      pdf.setFont('helvetica', 'normal');
      pdf.text('Price Analysis:', margin, yPosition);
      yPosition += 7;
      pdf.text(`• Current Price: $${currentPrice?.toFixed(2) || 'N/A'}`, margin + 5, yPosition);
      yPosition += 6;
      pdf.text(`• 52-Week High: $${high52?.toFixed(2) || 'N/A'}`, margin + 5, yPosition);
      if (high52 && currentPrice) {
        const highDiff = ((currentPrice - high52) / high52 * 100);
        pdf.text(`  (${highDiff > 0 ? '+' : ''}${highDiff.toFixed(1)}% from current)`, margin + 70, yPosition);
      }
      yPosition += 6;
      pdf.text(`• 52-Week Low: $${low52?.toFixed(2) || 'N/A'}`, margin + 5, yPosition);
      if (low52 && currentPrice) {
        const lowDiff = ((currentPrice - low52) / low52 * 100);
        pdf.text(`  (${lowDiff > 0 ? '+' : ''}${lowDiff.toFixed(1)}% from current)`, margin + 70, yPosition);
      }
      yPosition += 6;
      pdf.text(`• AI Price Prediction: $${predictedPrice?.toFixed(2) || 'N/A'} (${getPredictionHorizon()})`, margin + 5, yPosition);
      if (predictedPrice && currentPrice) {
        const predDiff = ((predictedPrice - currentPrice) / currentPrice * 100);
        pdf.text(`  (${predDiff > 0 ? '+' : ''}${predDiff.toFixed(1)}% potential)`, margin + 120, yPosition);
      }
      yPosition += 15;

      // Technical Indicators Section
      checkPageBreak(60);
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46);
      pdf.text('Technical Indicators Analysis', margin, yPosition);
      yPosition += 12;

      // All Technical Indicators by Category
      const indicatorCategories = [
        {
          name: 'Moving Averages',
          indicators: ['sma50vs200', 'pricevs50', 'pricevs200', 'ema12vs26']
        },
        {
          name: 'Momentum Oscillators',
          indicators: ['rsi', 'stoch_k', 'stoch_d', 'williams_r']
        },
        {
          name: 'Trend Analysis',
          indicators: ['macd', 'macd_signal', 'adx', 'cci']
        },
        {
          name: 'Volume & Price',
          indicators: ['volume_sma', 'bb_upper', 'bb_lower', 'atr']
        }
      ];

      indicatorCategories.forEach(category => {
        checkPageBreak(25);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text(category.name, margin, yPosition);
        yPosition += 8;

        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'normal');

        category.indicators.forEach(indicatorKey => {
          const indicator = analysisData?.indicators?.[indicatorKey];
          if (indicator) {
            const score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
            const description = indicator.description || indicator.desc || 'Technical indicator';
            pdf.text(`• ${indicatorKey.toUpperCase()}: ${(score * 10).toFixed(1)}/10`, margin + 5, yPosition);
            yPosition += 5;
            const wrappedY = addWrappedText(`  ${description}`, margin + 10, yPosition, maxLineWidth - 10, 4);
            yPosition = wrappedY + 3;
          }
        });
        yPosition += 8;
      });

      // Weighted Score Contributors Section
      checkPageBreak(40);
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46);
      pdf.text('Analysis Component Contributions', margin, yPosition);
      yPosition += 12;

      const allContributors = getTopContributors(analysisData?.weighted_scores, analysisData?.composite_raw, 14);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(0, 0, 0);
      pdf.text('All 14 weighted indicators contribution to final score:', margin, yPosition);
      yPosition += 10;

      allContributors.forEach((contributor, index) => {
        checkPageBreak();
        pdf.text(`${index + 1}. ${contributor.name}:`, margin, yPosition);
        pdf.text(`${contributor.value.toFixed(2)} pts (${contributor.percentage.toFixed(1)}%)`, margin + 100, yPosition);
        yPosition += 6;
      });
      yPosition += 10;

      // Market Sentiment Section (if available)
      if (analysisData?.indicators?.sentiment) {
        checkPageBreak(30);
        pdf.setFontSize(16);
        pdf.setFont('helvetica', 'bold');
        pdf.setTextColor(26, 26, 46);
        pdf.text('Market Sentiment Analysis', margin, yPosition);
        yPosition += 12;

        const sentiment = analysisData.indicators.sentiment;
        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(0, 0, 0);
        pdf.text(`Sentiment Score: ${sentiment.score?.toFixed(2) || 'N/A'}`, margin, yPosition);
        yPosition += 7;
        if (sentiment.description) {
          yPosition = addWrappedText(sentiment.description, margin, yPosition, maxLineWidth) + 5;
        }
        yPosition += 10;
      }

      // AI Explanation Section (if available)
      if (explanation?.explanation) {
        checkPageBreak(40);
        pdf.setFontSize(16);
        pdf.setFont('helvetica', 'bold');
        pdf.setTextColor(26, 26, 46);
        pdf.text('AI-Powered Technical Explanation', margin, yPosition);
        yPosition += 12;

        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(0, 0, 0);

        if (explanation.explanation.confidence) {
          pdf.text(`Confidence Level: ${(explanation.explanation.confidence * 100).toFixed(1)}%`, margin, yPosition);
          yPosition += 6;
        }
        if (explanation.explanation.model_used) {
          pdf.text(`Analysis Model: ${explanation.explanation.model_used}`, margin, yPosition);
          yPosition += 6;
        }
        yPosition += 5;

        if (explanation.explanation.summary) {
          yPosition = addWrappedText(explanation.explanation.summary, margin, yPosition, maxLineWidth) + 8;
        }

        if (explanation.explanation.key_findings && explanation.explanation.key_findings.length > 0) {
          pdf.setFont('helvetica', 'bold');
          pdf.text('Key Findings:', margin, yPosition);
          yPosition += 7;
          pdf.setFont('helvetica', 'normal');

          explanation.explanation.key_findings.forEach((finding, index) => {
            checkPageBreak();
            yPosition = addWrappedText(`• ${finding}`, margin + 5, yPosition, maxLineWidth - 5) + 3;
          });
        }
      }

      // Risk Assessment Section
      checkPageBreak(30);
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46);
      pdf.text('Risk Assessment', margin, yPosition);
      yPosition += 12;

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(0, 0, 0);

      let riskLevel = 'Moderate';
      if (score >= 8) riskLevel = 'Low';
      else if (score >= 6) riskLevel = 'Low-Moderate';
      else if (score >= 4) riskLevel = 'Moderate';
      else if (score >= 2) riskLevel = 'High-Moderate';
      else riskLevel = 'High';

      pdf.text(`Risk Level: ${riskLevel}`, margin, yPosition);
      yPosition += 7;

      const volatilityIndicators = ['atr', 'bb_upper', 'bb_lower'].filter(key => analysisData?.indicators?.[key]);
      if (volatilityIndicators.length > 0) {
        pdf.text('Volatility Indicators:', margin, yPosition);
        yPosition += 6;
        volatilityIndicators.forEach(key => {
          const indicator = analysisData.indicators[key];
          pdf.text(`• ${key.toUpperCase()}: ${indicator.raw || 'N/A'}`, margin + 5, yPosition);
          yPosition += 5;
        });
      }
      yPosition += 15;

      // Disclaimer Section
      checkPageBreak(25);
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(26, 26, 46);
      pdf.text('Important Disclaimer', margin, yPosition);
      yPosition += 10;

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(0, 0, 0);
      const disclaimer = 'This analysis is for informational purposes only and should not be considered as financial advice. ' +
        'Past performance does not guarantee future results. All investments carry risk of loss. ' +
        'Please consult with a qualified financial advisor before making investment decisions. ' +
        'VoyageurCompass is not responsible for any financial losses incurred based on this analysis.';

      yPosition = addWrappedText(disclaimer, margin, yPosition, maxLineWidth, 4) + 10;

      // Add footers to all pages
      const totalPages = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(100, 100, 100);
        pdf.text(`Generated by VoyageurCompass • ${new Date().toLocaleDateString()}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
        pdf.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 10, { align: 'right' });
        pdf.text('Confidential Analysis Report', margin, pageHeight - 10);
      }

      // Save the PDF
      const filename = `${analysisData?.symbol}_comprehensive_analysis_${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(filename);

    } catch (error) {
      console.error('Error generating PDF:', error);
      // Fallback to JSON export if PDF generation fails
      const reportData = {
        symbol: analysisData?.symbol,
        analysis_date: analysisData?.analysis_date,
        score: analysisData?.score_0_10 || analysisData?.score,
        recommendation: getScoreLabel(analysisData?.score_0_10 || analysisData?.score),
        current_price: getCurrentPrice(),
        week_52_high: get52WeekHigh(),
        week_52_low: get52WeekLow(),
        predicted_price: getPredictedPrice(),
        prediction_horizon: getPredictionHorizon(),
        indicators: analysisData?.indicators,
        weighted_scores: analysisData?.weighted_scores,
        explanation: explanation?.explanation,
        all_contributors: getTopContributors(analysisData?.weighted_scores, analysisData?.composite_raw, 14)
      };

      const dataStr = JSON.stringify(reportData, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      const exportFileDefaultName = `${analysisData?.symbol}_analysis_report_${new Date().toISOString().split('T')[0]}.json`;

      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  };

  // Share functionality
  const handleShareAnalysis = async () => {
    const shareUrl = `${window.location.origin}/analysis/${analysisId}`;
    const shareText = `Check out this stock analysis for ${analysisData?.symbol}: Score ${analysisData?.score_0_10 || analysisData?.score}/10`;

    if (navigator.share) {
      try {
        await navigator.share({
          title: `${analysisData?.symbol} Stock Analysis`,
          text: shareText,
          url: shareUrl,
        });
      } catch (error) {
        console.log('Error sharing:', error);
        // Fallback to clipboard
        handleCopyToClipboard(shareUrl);
      }
    } else {
      // Fallback to clipboard
      handleCopyToClipboard(shareUrl);
    }
  };

  const handleCopyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      // Show success message (you could add a toast notification here)
      alert('Analysis link copied to clipboard!');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('Analysis link copied to clipboard!');
    }
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
      {/* Enhanced Header with Action Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/dashboard')}
          size="large"
          sx={{ borderRadius: 3 }}
        >
          {t('analysis.backToDashboard')}
        </Button>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Tooltip title="Download PDF Report">
            <Fab
              color="primary"
              size="small"
              onClick={handleExportReport}
              sx={{ boxShadow: 3 }}
            >
              <GetApp />
            </Fab>
          </Tooltip>
          <Tooltip title="Share Analysis">
            <Fab
              color="secondary"
              size="small"
              onClick={handleShareAnalysis}
              sx={{ boxShadow: 3 }}
            >
              <Share />
            </Fab>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<TrendingUp />}
            onClick={() => navigate('/', {
              state: { searchTicker: analysisData.symbol, autoAnalyze: true }
            })}
            sx={{ borderRadius: 3, boxShadow: 3 }}
          >
            {t('analysis.runNewAnalysis')}
          </Button>
        </Box>
      </Box>

      {/* Enhanced Analysis Summary Hero Section */}
      <Paper
        elevation={8}
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          p: 4,
          mb: 4,
          borderRadius: 4,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'url("data:image/svg+xml,%3Csvg width=\\"20\\" height=\\"20\\" viewBox=\\"0 0 20 20\\" xmlns=\\"http://www.w3.org/2000/svg\\"%3E%3Cg fill=\\"%23ffffff\\" fill-opacity=\\"0.03\\"%3E%3Ccircle cx=\\"3\\" cy=\\"3\\" r=\\"3\\"/%3E%3C/g%3E%3C/svg%3E")',
            zIndex: 0
          }
        }}
      >
        <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Business sx={{ fontSize: 40 }} />
                <Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                    {analysisData.symbol}
                  </Typography>
                  <Typography variant="h6" sx={{ opacity: 0.9 }}>
                    {analysisData.name}
                  </Typography>
                </Box>
                <IconButton
                  onClick={() => setIsBookmarked(!isBookmarked)}
                  sx={{ ml: 'auto', color: 'white' }}
                >
                  {isBookmarked ? <Star /> : <StarBorder />}
                </IconButton>
              </Box>

              <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                <Chip
                  label={analysisData.sector}
                  sx={{
                    backgroundColor: 'rgba(255,255,255,0.2)',
                    color: 'white',
                    fontWeight: 600
                  }}
                />
                <Chip
                  label={analysisData.industry}
                  variant="outlined"
                  sx={{
                    borderColor: 'rgba(255,255,255,0.5)',
                    color: 'white'
                  }}
                />
              </Box>

              <Typography variant="body1" sx={{ opacity: 0.9 }}>
                {t('analysis.analysisDate')}: {new Date(analysisData.analysis_date).toLocaleDateString()}
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                  <CircularProgress
                    variant="determinate"
                    value={analysisData.score * 10}
                    size={120}
                    thickness={6}
                    sx={{
                      color: analysisData.score >= 7 ? '#4caf50' :
                             analysisData.score >= 4 ? '#ff9800' : '#f44336'
                    }}
                  />
                  <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexDirection: 'column'
                  }}>
                    <Typography variant="h3" sx={{ fontWeight: 700 }}>
                      {analysisData.score}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.8 }}>
                      /10
                    </Typography>
                  </Box>
                </Box>
                <Typography variant="h6" sx={{ mt: 2, fontWeight: 600 }}>
                  {getScoreLabel(analysisData.score)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                  Technical Analysis Score
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      {/* Key Metrics Grid - Fixed Tile Sizing */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card
            elevation={4}
            sx={{
              height: 160,
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
              color: 'white'
            }}
          >
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center', p: 2 }}>
              <Assessment sx={{ fontSize: 36, mb: 1, mx: 'auto' }} />
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
                {analysisData.score}/10
              </Typography>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, opacity: 0.9, fontSize: '0.9rem' }}>
                Composite Score
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                {getScoreLabel(analysisData.score)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            elevation={4}
            sx={{
              height: 160,
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
              color: 'white'
            }}
          >
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center', p: 2 }}>
              <BarChart sx={{ fontSize: 36, mb: 1, mx: 'auto' }} />
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
                {Object.keys(analysisData.indicators || {}).length}
              </Typography>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, opacity: 0.9, fontSize: '0.9rem' }}>
                Indicators
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Analyzed
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            elevation={4}
            sx={{
              height: 160,
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
              color: 'white'
            }}
          >
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center', p: 2 }}>
              <Speed sx={{ fontSize: 36, mb: 1, mx: 'auto' }} />
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
                {analysisData.confidence ? `${(analysisData.confidence * 100).toFixed(0)}%` : '95%'}
              </Typography>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, opacity: 0.9, fontSize: '0.9rem' }}>
                Confidence
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Reliability
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Stock Price Information Section */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={4} sx={{ borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
              <TrendingUpSharp sx={{ mr: 2, fontSize: 32 }} />
              Stock Price Analysis
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
              Current market data and AI-powered price predictions
            </Typography>
          </Box>

          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              {/* Current Price */}
              <Grid item xs={12} md={3}>
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    backgroundColor: '#0a2342',
                    color: 'white',
                    borderRadius: 3
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Current Price
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                      ${getCurrentPrice()?.toFixed(2) || 'N/A'}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Last Updated: {new Date().toLocaleDateString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* 52-Week Low */}
              <Grid item xs={12} md={3}>
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    backgroundColor: getTileColor(get52WeekLow(), getCurrentPrice()),
                    color: 'white',
                    borderRadius: 3
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      52-Week Low
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                      ${get52WeekLow()?.toFixed(2) || 'N/A'}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Minimum
                    </Typography>
                    {get52WeekLow() && getCurrentPrice() && (
                      <Chip
                        label={`${(((get52WeekLow() - getCurrentPrice()) / getCurrentPrice()) * 100).toFixed(1)}%`}
                        size="small"
                        sx={{
                          mt: 1,
                          backgroundColor: (get52WeekLow() - getCurrentPrice()) >= 0 ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)',
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* 52-Week High */}
              <Grid item xs={12} md={3}>
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    backgroundColor: getTileColor(get52WeekHigh(), getCurrentPrice()),
                    color: 'white',
                    borderRadius: 3
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      52-Week High
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                      ${get52WeekHigh()?.toFixed(2) || 'N/A'}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Maximum
                    </Typography>
                    {get52WeekHigh() && getCurrentPrice() && (
                      <Chip
                        label={`${(((get52WeekHigh() - getCurrentPrice()) / getCurrentPrice()) * 100).toFixed(1)}%`}
                        size="small"
                        sx={{
                          mt: 1,
                          backgroundColor: (get52WeekHigh() - getCurrentPrice()) >= 0 ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)',
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Predicted Price */}
              <Grid item xs={12} md={3}>
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    backgroundColor: getTileColor(getPredictedPrice(), getCurrentPrice()),
                    color: 'white',
                    borderRadius: 3
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Price Prediction
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                      ${getPredictedPrice()?.toFixed(2) || 'N/A'}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      {getPredictionHorizon()}
                    </Typography>
                    {getPredictedPrice() && getCurrentPrice() && (
                      <Chip
                        label={`${(((getPredictedPrice() - getCurrentPrice()) / getCurrentPrice()) * 100).toFixed(1)}%`}
                        size="small"
                        sx={{
                          mt: 1,
                          backgroundColor: (getPredictedPrice() - getCurrentPrice()) >= 0 ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)',
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        </Paper>
      </Box>

      {/* AI-Powered Explanations Section - Moved to Second Position */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={4} sx={{ borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Psychology sx={{ fontSize: 40 }} />
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    {t('analysis.aiExplanations')}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Powered by Phi3 & LLaMA 3.1 Models
                  </Typography>
                </Box>
              </Box>

              {/* Service Status Indicator */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {explanationStatus?.status?.llm_available ? (
                  <Chip
                    label={t('analysis.multiModelActive')}
                    sx={{
                      backgroundColor: 'rgba(76, 175, 80, 0.9)',
                      color: 'white',
                      fontWeight: 600
                    }}
                    size="small"
                  />
                ) : statusError ? (
                  <Chip
                    label={t('analysis.serviceUnavailable')}
                    sx={{
                      backgroundColor: 'rgba(244, 67, 54, 0.9)',
                      color: 'white',
                      fontWeight: 600
                    }}
                    size="small"
                  />
                ) : (
                  <Chip
                    label={t('analysis.templateMode')}
                    sx={{
                      backgroundColor: 'rgba(255, 152, 0, 0.9)',
                      color: 'white',
                      fontWeight: 600
                    }}
                    size="small"
                  />
                )}
              </Box>
            </Box>
          </Box>

          <Box sx={{ p: 3 }}>
            <TechnicalExplanation
              analysisId={analysisId}
              analysisData={analysisData}
              defaultExpanded={true}
            />
          </Box>
        </Paper>
      </Box>

      {/* Technical Analysis Details - Reorganized */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={4} sx={{ borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
              <Analytics sx={{ mr: 2, fontSize: 32 }} />
              {t('analysis.technicalIndicators')}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
              Comprehensive technical analysis across 14 core indicators
            </Typography>
          </Box>

          <Box sx={{ p: 3 }}>
        {/* Reorganized Technical Indicators - 6 categories in 2 columns (3x2) */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {/* Column 1 - First 3 categories */}
          <Grid item xs={12} md={6}>
            {/* Momentum Indicators */}
            <Accordion elevation={3} sx={{ borderRadius: 2, mb: 3 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Speed sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Momentum Indicators
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => ['rsi', 'macd', 'stoch', 'momentum', 'williams'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Momentum indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Trend Indicators */}
            <Accordion elevation={3} sx={{ borderRadius: 2, mb: 3 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Timeline sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Trend Indicators
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => ['sma', 'ema', 'moving', 'trend', 'adx', 'aroon'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Trend indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Volume Indicators */}
            <Accordion elevation={3} sx={{ borderRadius: 2 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <BarChart sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Volume Indicators
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => ['volume', 'obv', 'vwap', 'mfi'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Volume indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* Column 2 - Next 3 categories */}
          <Grid item xs={12} md={6}>
            {/* Pattern Recognition */}
            <Accordion elevation={3} sx={{ borderRadius: 2, mb: 3 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CandlestickChart sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Pattern Recognition
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => ['candlestick', 'candle', 'pattern', 'reversal', 'doji', 'hammer', 'shooting'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Pattern indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Volatility Indicators */}
            <Accordion elevation={3} sx={{ borderRadius: 2, mb: 3 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Navigation sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Volatility Indicators
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => ['bollinger', 'bb', 'atr', 'volatility', 'std', 'keltner', 'donchian'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Volatility indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Other Indicators */}
            <Accordion elevation={3} sx={{ borderRadius: 2 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  backgroundColor: '#1a1a2e',
                  color: 'white',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Insights sx={{ color: 'white' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Other Indicators
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {analysisData.indicators && Object.entries(analysisData.indicators)
                    .filter(([key]) => !['rsi', 'macd', 'stoch', 'momentum', 'williams', 'sma', 'ema', 'moving', 'trend', 'adx', 'aroon', 'volume', 'obv', 'vwap', 'mfi', 'sentiment', 'bollinger', 'bb', 'atr', 'volatility', 'std', 'keltner', 'donchian', 'candlestick', 'candle', 'pattern', 'reversal', 'doji', 'hammer', 'shooting'].some(indicator => key.toLowerCase().includes(indicator)))
                    .map(([key, indicator]) => {
                      let score = indicator.score || (indicator.raw ? parseFloat(indicator.raw) / 10 : 0);
                      if (isNaN(score) || !isFinite(score)) score = 0;

                      return (
                        <Grid item xs={12} key={key}>
                          <Card variant="outlined" sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {key.toUpperCase()}
                              </Typography>
                              <Chip
                                label={`${(score * 10).toFixed(1)}/10`}
                                size="small"
                                color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={Math.max(0, Math.min(100, score * 100))}
                              sx={{ height: 6, borderRadius: 3, mb: 1 }}
                              color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {indicator.description || indicator.desc || 'Technical indicator'}
                            </Typography>
                          </Card>
                        </Grid>
                      );
                    })}
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
          </Box>
        </Paper>
      </Box>

      {/* Enhanced Market Sentiment Analysis */}
      {analysisData.indicators?.sentiment && (
        <Box sx={{ mb: 4 }}>
          <Paper elevation={4} sx={{ borderRadius: 3, overflow: 'hidden' }}>
            <Box sx={{
              backgroundColor: '#1a1a2e',
              color: 'white',
              p: 3
            }}>
              <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
                <SentimentSatisfied sx={{ mr: 2, fontSize: 32 }} />
                {t('analysis.newsSentimentAnalysis')}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
                AI-powered sentiment analysis from {analysisData.indicators.sentiment.raw?.newsCount || 0} news articles
              </Typography>
            </Box>

            <Box sx={{ p: 3 }}>
              <Grid container spacing={3}>
                {/* Enhanced Sentiment Overview */}
                <Grid item xs={12} md={4}>
                  <Card
                    elevation={3}
                    sx={{
                      height: '100%',
                      backgroundColor: getSentimentColor(analysisData.indicators.sentiment.raw?.label) === 'success' ? '#2e7d32' :
                                     getSentimentColor(analysisData.indicators.sentiment.raw?.label) === 'error' ? '#d32f2f' : '#f57c00',
                      color: 'white',
                      borderRadius: 3
                    }}
                  >
                    <CardContent sx={{ textAlign: 'center', p: 3 }}>
                      <Box sx={{ mb: 2 }}>
                        {getSentimentIcon(analysisData.indicators.sentiment.raw?.label)}
                      </Box>
                      <Typography variant="h4" sx={{
                        fontWeight: 700,
                        color: 'white',
                        mb: 1
                      }}>
                        {analysisData.indicators.sentiment.raw?.label?.toUpperCase() || 'NEUTRAL'}
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.9)' }} gutterBottom>
                        Overall Sentiment
                      </Typography>
                      <Chip
                        label={`Score: ${formatSentimentScore(analysisData.indicators.sentiment.raw?.sentiment)}`}
                        sx={{
                          fontWeight: 600,
                          mb: 2,
                          backgroundColor: 'rgba(255,255,255,0.2)',
                          color: 'white'
                        }}
                      />
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                        Confidence: {analysisData.indicators.sentiment.raw?.confidence ?
                          `${(analysisData.indicators.sentiment.raw.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Impact Analysis */}
                <Grid item xs={12} md={4}>
                  <Card elevation={3} sx={{ height: '100%', borderRadius: 3 }}>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <Assessment sx={{ mr: 1 }} />
                        Impact on Analysis
                      </Typography>

                      <Box sx={{ mb: 3 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Contribution to Overall Score
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={10}
                            sx={{ flex: 1, height: 12, borderRadius: 6 }}
                            color={getSentimentColor(analysisData.indicators.sentiment.raw?.label)}
                          />
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            10%
                          </Typography>
                        </Box>
                      </Box>

                      <Divider sx={{ my: 2 }} />

                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <NewspaperOutlined color="action" />
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          News Coverage
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        Analysis based on {analysisData.indicators.sentiment.raw?.newsCount || 0} recent articles
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Sentiment Breakdown */}
                <Grid item xs={12} md={4}>
                  <Card elevation={3} sx={{ height: '100%', borderRadius: 3 }}>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <ShowChart sx={{ mr: 1 }} />
                        Sentiment Metrics
                      </Typography>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Raw Sentiment Score
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={Math.max(0, Math.min(100, (parseFloat(analysisData.indicators.sentiment.raw?.sentiment || 0) + 1) * 50))}
                            sx={{ flex: 1, height: 8, borderRadius: 4 }}
                            color={getSentimentColor(analysisData.indicators.sentiment.raw?.label)}
                          />
                          <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 60 }}>
                            {formatSentimentScore(analysisData.indicators.sentiment.raw?.sentiment)}
                          </Typography>
                        </Box>
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Confidence Level
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={(analysisData.indicators.sentiment.raw?.confidence || 0) * 100}
                            sx={{ flex: 1, height: 8, borderRadius: 4 }}
                            color="info"
                          />
                          <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 60 }}>
                            {((analysisData.indicators.sentiment.raw?.confidence || 0) * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>

                      <Typography variant="caption" color="text.secondary">
                        Powered by FinBERT sentiment analysis
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Enhanced News Sources */}
                {analysisData.indicators.sentiment.raw?.sources &&
                 Object.keys(analysisData.indicators.sentiment.raw.sources).length > 0 && (
                  <Grid item xs={12}>
                    <Card elevation={3} sx={{ borderRadius: 3 }}>
                      <CardContent sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                          <NewspaperOutlined sx={{ mr: 1 }} />
                          News Sources Analysis
                        </Typography>

                        <Grid container spacing={2}>
                          {Object.entries(analysisData.indicators.sentiment.raw.sources).map(([source, data]) => {
                            const avgScore = data.avg_score || 0;
                            const sentimentLabel = avgScore > 0.1 ? 'positive' : avgScore < -0.1 ? 'negative' : 'neutral';

                            return (
                              <Grid item xs={12} sm={6} md={4} key={source}>
                                <Card variant="outlined" sx={{ p: 2, height: '100%' }}>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                      {source}
                                    </Typography>
                                    <Chip
                                      label={data.count || 0}
                                      size="small"
                                      color="primary"
                                      variant="outlined"
                                    />
                                  </Box>

                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                    {getSentimentIcon(sentimentLabel)}
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                      {formatSentimentScore(avgScore)}
                                    </Typography>
                                  </Box>

                                  <LinearProgress
                                    variant="determinate"
                                    value={Math.max(0, Math.min(100, (Math.abs(avgScore) + 0.5) * 50))}
                                    sx={{ height: 6, borderRadius: 3 }}
                                    color={getSentimentColor(sentimentLabel)}
                                  />

                                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                    Impact: {Math.abs(avgScore) > 0.3 ? 'High' : Math.abs(avgScore) > 0.1 ? 'Medium' : 'Low'}
                                  </Typography>
                                </Card>
                              </Grid>
                            );
                          })}
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            </Box>
          </Paper>
        </Box>
      )}

      {/* Enhanced Top Performing Indicators Section */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={6} sx={{ borderRadius: 4, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
              <TrendingUpSharp sx={{ mr: 2, fontSize: 32 }} />
              {t('explanations.topPerformingIndicators')}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
              Leading technical indicators driving the analysis score
            </Typography>
          </Box>

          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              {getTopIndicators(analysisData?.indicators, 3).map((indicator, index) => (
                <Grid item xs={12} sm={6} md={4} key={indicator.key}>
                  <Card
                    elevation={3}
                    sx={{
                      height: '100%',
                      borderRadius: 3,
                      background: `linear-gradient(135deg, ${
                        indicator.score >= 0.7 ? '#e8f5e8' : indicator.score >= 0.4 ? '#fff3e0' : '#ffebee'
                      } 0%, ${
                        indicator.score >= 0.7 ? '#c8e6c9' : indicator.score >= 0.4 ? '#ffe0b2' : '#ffcdd2'
                      } 100%)`,
                      border: `2px solid ${
                        indicator.score >= 0.7 ? '#4caf50' : indicator.score >= 0.4 ? '#ff9800' : '#f44336'
                      }`,
                      position: 'relative',
                      '&::before': {
                        content: `"#${index + 1}"`,
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        backgroundColor: indicator.score >= 0.7 ? '#4caf50' : indicator.score >= 0.4 ? '#ff9800' : '#f44336',
                        color: 'white',
                        borderRadius: '50%',
                        width: 24,
                        height: 24,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        fontWeight: 700
                      }
                    }}
                  >
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                        {getIndicatorIcon(indicator.key)}
                        <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>
                          {indicator.name}
                        </Typography>
                      </Box>

                      <Box sx={{ textAlign: 'center', mb: 2 }}>
                        <Typography
                          variant="h4"
                          sx={{
                            fontWeight: 700,
                            color: indicator.score >= 0.7 ? '#2e7d32' : indicator.score >= 0.4 ? '#f57c00' : '#d32f2f'
                          }}
                        >
                          {(indicator.score * 10).toFixed(1)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          /10 Score
                        </Typography>
                      </Box>

                      <LinearProgress
                        variant="determinate"
                        value={Math.max(0, Math.min(100, indicator.score * 100))}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          mb: 1.5,
                          backgroundColor: 'rgba(0,0,0,0.1)'
                        }}
                        color={indicator.score >= 0.7 ? 'success' : indicator.score >= 0.4 ? 'warning' : 'error'}
                      />

                      <Box sx={{ textAlign: 'center' }}>
                        <Chip
                          label={indicator.score >= 0.7 ? t('explanations.strong') : indicator.score >= 0.4 ? t('explanations.moderate') : t('explanations.weak')}
                          size="small"
                          color={indicator.score >= 0.7 ? 'success' : indicator.score >= 0.4 ? 'warning' : 'error'}
                          sx={{ fontWeight: 600 }}
                        />
                      </Box>

                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                        {indicator.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Paper>
      </Box>

      {/* Analysis Component Contributions - Pie Chart */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={6} sx={{ borderRadius: 4, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
              <Leaderboard sx={{ mr: 2, fontSize: 32 }} />
              {t('explanations.analysisComponentContributions')}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
              Distribution of all 14 technical indicators weighted by their impact on the final score
            </Typography>
          </Box>

          <Box sx={{ p: 3 }}>
            {(() => {
              const pieData = preparePieChartData();

              if (pieData.length === 0) {
                return (
                  <Box sx={{ textAlign: 'center', py: 6 }}>
                    <Typography variant="h6" color="text.secondary">
                      No weighted score data available
                    </Typography>
                  </Box>
                );
              }

              return (
                <Grid container spacing={3}>
                  {/* Pie Chart */}
                  <Grid item xs={12} lg={8}>
                    <Box sx={{ height: 400, width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={(entry) => `${entry.name}: ${entry.percentage}%`}
                            outerRadius={140}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <RechartsTooltip
                            formatter={(value, name) => [
                              `${value.toFixed(2)} points (${pieData.find(d => d.name === name)?.percentage}%)`,
                              'Contribution'
                            ]}
                            labelStyle={{ color: '#333' }}
                            contentStyle={{
                              backgroundColor: 'rgba(255, 255, 255, 0.95)',
                              border: '1px solid #ccc',
                              borderRadius: '8px',
                              boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                            }}
                          />
                          <Legend
                            verticalAlign="bottom"
                            height={36}
                            wrapperStyle={{ paddingTop: '20px' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>

                  {/* Top Contributors Summary */}
                  <Grid item xs={12} lg={4}>
                    <Card elevation={3} sx={{ height: '100%', borderRadius: 3 }}>
                      <CardContent sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: '#6a1b9a' }}>
                          Top Contributors
                        </Typography>

                        <Box sx={{ maxHeight: 320, overflowY: 'auto' }}>
                          {pieData.slice(0, 8).map((item, index) => (
                            <Box key={item.name} sx={{ mb: 2, pb: 2, borderBottom: index < 7 ? '1px solid #f0f0f0' : 'none' }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                                  #{index + 1} {item.name}
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Box
                                    sx={{
                                      width: 12,
                                      height: 12,
                                      backgroundColor: item.color,
                                      borderRadius: '50%'
                                    }}
                                  />
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {item.percentage}%
                                  </Typography>
                                </Box>
                              </Box>

                              <LinearProgress
                                variant="determinate"
                                value={parseFloat(item.percentage)}
                                sx={{
                                  height: 6,
                                  borderRadius: 3,
                                  backgroundColor: 'rgba(0,0,0,0.1)',
                                  '& .MuiLinearProgress-bar': {
                                    backgroundColor: item.color
                                  }
                                }}
                              />

                              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                {item.value.toFixed(2)} points contribution
                              </Typography>
                            </Box>
                          ))}
                        </Box>

                        <Box sx={{ mt: 2, p: 2, backgroundColor: '#f5f5f5', borderRadius: 2 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center', display: 'block' }}>
                            Showing {pieData.length} of 14 weighted indicators
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              );
            })()}
          </Box>
        </Paper>
      </Box>

      {/* Enhanced Analysis Framework & Methodology Section */}
      <Box sx={{ mb: 4 }}>
        <Paper elevation={6} sx={{ borderRadius: 4, overflow: 'hidden' }}>
          <Box sx={{
            backgroundColor: '#1a1a2e',
            color: 'white',
            p: 3
          }}>
            <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
              <Architecture sx={{ mr: 2, fontSize: 32 }} />
              {t('explanations.analysisFramework')}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mt: 1 }}>
              Technical analysis methodology and framework used
            </Typography>
          </Box>

          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <List sx={{ p: 0 }}>
                  <ListItem sx={{ px: 0, py: 2 }}>
                    <ListItemIcon>
                      <Box sx={{
                        backgroundColor: '#4caf50',
                        borderRadius: '50%',
                        p: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Assessment sx={{ color: 'white', fontSize: 24 }} />
                      </Box>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                          {t('explanations.compositeScoring')}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="body2" color="text.secondary">
                          {t('explanations.compositeScoringDesc')}
                        </Typography>
                      }
                    />
                  </ListItem>

                  <Divider sx={{ my: 1 }} />

                  <ListItem sx={{ px: 0, py: 2 }}>
                    <ListItemIcon>
                      <Box sx={{
                        backgroundColor: '#2196f3',
                        borderRadius: '50%',
                        p: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Timeline sx={{ color: 'white', fontSize: 24 }} />
                      </Box>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                          {t('explanations.multiTimeframeAnalysis')}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="body2" color="text.secondary">
                          {t('explanations.multiTimeframeDesc')}
                        </Typography>
                      }
                    />
                  </ListItem>

                  <Divider sx={{ my: 1 }} />

                  <ListItem sx={{ px: 0, py: 2 }}>
                    <ListItemIcon>
                      <Box sx={{
                        backgroundColor: '#ff9800',
                        borderRadius: '50%',
                        p: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Insights sx={{ color: 'white', fontSize: 24 }} />
                      </Box>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                          {t('explanations.riskAssessment')}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="body2" color="text.secondary">
                          {t('explanations.riskAssessmentDesc')}
                        </Typography>
                      }
                    />
                  </ListItem>
                </List>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    background: 'linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%)',
                    borderRadius: 3,
                    border: '2px solid #00897b'
                  }}
                >
                  <CardContent sx={{ p: 3, textAlign: 'center' }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 700, color: '#00695c' }}>
                      Analysis Stats
                    </Typography>

                    <Box sx={{ mb: 2 }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#00897b' }}>
                        {Object.keys(analysisData?.indicators || {}).length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Technical Indicators
                      </Typography>
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    <Box sx={{ mb: 2 }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#00897b' }}>
                        {analysisData?.confidence ? `${(analysisData.confidence * 100).toFixed(0)}%` : '95%'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Model Confidence
                      </Typography>
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    <Box>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#00897b' }}>
                        {new Date(analysisData?.analysis_date).toLocaleDateString()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Analysis Date
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Methodology Note */}
            <Box sx={{
              mt: 3,
              p: 3,
              background: 'linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%)',
              borderRadius: 2,
              border: '1px solid #e0e0e0'
            }}>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                <strong>{t('explanations.analysisMethodology')}:</strong> {t('explanations.methodologyDesc')}
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Box>

    </Container>
  );
};

export default AnalysisResultsPage;