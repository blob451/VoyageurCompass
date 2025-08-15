import React, { useState, useEffect } from 'react';
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
  IconButton
} from '@mui/material';
import { ArrowBack, TrendingUp } from '@mui/icons-material';
import { useGetAnalysisByIdQuery } from '../features/api/apiSlice';

const AnalysisResultsPage = () => {
  const { analysisId } = useParams();
  const navigate = useNavigate();
  
  const {
    data: analysisData,
    error,
    isLoading
  } = useGetAnalysisByIdQuery(analysisId);

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
                
                return (
                  <TableRow key={key}>
                    <TableCell sx={{ fontWeight: 500 }}>
                      {key.toUpperCase()}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={Math.max(0, Math.min(100, score * 100))}
                          sx={{ width: 60, height: 8, borderRadius: 4 }}
                          color={score >= 0.7 ? 'success' : score >= 0.4 ? 'warning' : 'error'}
                        />
                        <Typography variant="body2">
                          {isValidScore ? displayScore.toFixed(1) : 'N/A'}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {indicator.description || indicator.desc || 'Technical indicator'}
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
                    .filter(([key, value]) => value !== null && value !== undefined)
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
    </Container>
  );
};

export default AnalysisResultsPage;