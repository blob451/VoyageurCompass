import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  Search,
  TrendingUp,
  TrendingDown,
  Analytics,
  History,
  AccountBalance,
  Speed,
  Info,
  Refresh,
  Visibility,
  Psychology
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { useAnalyzeStockMutation, useGetUserAnalysisHistoryQuery, useGetUserLatestAnalysisQuery } from '../features/api/apiSlice';
import { analysisLogger, performanceUtils } from '../utils/logger';

const StockSearchPage = () => {
  const { user } = useSelector((state) => state.auth);
  const location = useLocation();
  const navigate = useNavigate();
  const [searchTicker, setSearchTicker] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(false);
  const [userCredits] = useState(25); // Mock credit balance
  
  // AI Explanation controls - automatically generate Summary explanations
  const includeExplanation = true;
  const explanationDetail = 'summary';
  
  // Proper state machine for analysis flow
  const [analysisMode, setAnalysisMode] = useState('none'); // 'none' | 'manual' | 'auto'
  const [analysisPhase, setAnalysisPhase] = useState('idle'); // 'idle' | 'confirming' | 'syncing' | 'analyzing' | 'completed' | 'failed'
  
  // RTK Query hooks
  const [analyzeStock, { isLoading: analyzing }] = useAnalyzeStockMutation();
  const { data: analysisHistoryData, refetch: refetchHistory } = useGetUserAnalysisHistoryQuery({ limit: 10 });

  // Mock recent searches for demonstration
  const recentSearches = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'];
  
  // Mock popular stocks
  const popularStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology' },
    { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Consumer Cyclical' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer Cyclical' },
    { symbol: 'META', name: 'Meta Platforms Inc.', sector: 'Technology' },
  ];

  // Get analysis history from API
  const analysisHistory = analysisHistoryData?.analyses || [];
  
  // Check if we're coming from dashboard with a ticker to show
  const dashboardTicker = location.state?.searchTicker;
  const autoAnalyze = location.state?.autoAnalyze;
  
  // Query for latest analysis if coming from dashboard (but not auto-analyzing)
  const { data: latestAnalysisData, isLoading: loadingLatestAnalysis } = useGetUserLatestAnalysisQuery(
    dashboardTicker, 
    { skip: !dashboardTicker || autoAnalyze }
  );

  // Function to perform analysis (handles both auto and manual modes)
  const performAnalysis = useCallback(async (symbol, mode) => {
    const targetSymbol = symbol || searchTicker;
    
    analysisLogger.stage(targetSymbol, 'Analysis Started', { mode, phase: 'analyzing' });
    
    if (!targetSymbol?.trim()) {
      console.log(`[ANALYSIS] Error: No ticker symbol provided`);
      setError('Please enter a stock ticker symbol');
      setAnalysisPhase('failed');
      return;
    }
    
    if (userCredits < 1) {
      console.log(`[ANALYSIS] Error: Insufficient credits (${userCredits})`);
      setError('Insufficient credits. Please purchase more credits to continue.');
      setAnalysisPhase('failed');
      return;
    }

    analysisLogger.stage(targetSymbol, 'API Request Starting', { userCredits });
    setError('');
    setAnalysisPhase('analyzing');

    const analysisStartTime = performance.now();
    try {
      analysisLogger.stage(targetSymbol, 'API Request Initiated');
      const result = await analyzeStock({ 
        symbol: targetSymbol.toUpperCase(),
        includeExplanation,
        explanationDetail
      });
      
      if (result.error) {
        console.log(`[ANALYSIS] API error received:`, result.error);
        const errorMessage = result.error.data?.error || 'Analysis failed. Please try again.';
        console.log(`[ANALYSIS] Error message: ${errorMessage}`);
        setError(errorMessage);
        setAnalysisPhase('failed');
        
        // Check if it's a "No price data" error that might need auto-sync
        if (errorMessage.includes('No price data available')) {
          console.log(`[ANALYSIS] Switching to syncing phase`);
          setAnalysisPhase('syncing');
          // The backend should handle auto-sync, but we show appropriate UI
          setTimeout(() => {
            console.log(`[ANALYSIS] Updating sync status message`);
            setError('Stock data is being synchronized. This may take a few moments...');
          }, 1000);
          
          // Set a longer timeout for sync completion
          setTimeout(() => {
            if (analysisPhase === 'syncing') {
              console.log(`[ANALYSIS] Sync timeout reached`);
              setError('Data synchronization is taking longer than expected. Please try again.');
              setAnalysisPhase('failed');
            }
          }, 30000); // 30 seconds timeout
        }
        return;
      }

      const analysisResult = result.data;
      const analysisDuration = performance.now() - analysisStartTime;
      analysisLogger.performance('API Response Received', analysisDuration, {
        symbol: analysisResult.symbol,
        score: analysisResult.composite_score,
        analysisId: analysisResult.analytics_result_id
      });
      
      analysisLogger.stage(analysisResult.symbol, 'Processing Analysis Result', {
        score: analysisResult.composite_score,
        creditsUsed: 1
      });
      
      // Format the analysis data for display
      const formattedData = {
        symbol: analysisResult.symbol,
        name: analysisResult.name || `${analysisResult.symbol} Corporation`,
        sector: analysisResult.sector || 'Unknown',
        industry: analysisResult.industry || 'Unknown',
        score: analysisResult.composite_score,
        indicators: analysisResult.indicators,
        weighted_scores: analysisResult.weighted_scores,
        timestamp: analysisResult.analysis_date,
        creditsUsed: 1
      };

      console.log(`[ANALYSIS] Analysis data formatted:`, formattedData);
      setAnalysisData(formattedData);
      setAnalysisPhase('completed');
      
      // Refetch history to include the new analysis (only if the query was started)
      if (analysisHistoryData !== undefined) {
        console.log(`[ANALYSIS] Refetching analysis history`);
        refetchHistory();
      }

      console.log(`[ANALYSIS] Analysis complete.`);

    } catch (err) {
      console.log(`[ANALYSIS] Analysis failed with exception:`, err);
      setError('Analysis failed. Please try again.');
      setAnalysisPhase('failed');
      console.error('Analysis error:', err);
    }
  }, [searchTicker, userCredits, analyzeStock, refetchHistory, analysisHistoryData]);

  // Handle navigation from dashboard
  useEffect(() => {
    if (dashboardTicker) {
      setSearchTicker(dashboardTicker);
      setAnalysisPhase('idle'); // Reset phase
      
      if (autoAnalyze) {
        // Set auto-analysis mode and show confirmation dialog
        console.log(`[ANALYSIS] Auto-analysis mode activated for ${dashboardTicker}`);
        setAnalysisMode('auto');
        setError('');
        console.log(`[ANALYSIS] Showing confirmation dialog for auto-analysis`);
        setConfirmDialog(true); // Show confirmation for auto-analysis too
      } else if (latestAnalysisData) {
        // Show existing analysis data
        setAnalysisMode('none');
        setAnalysisPhase('completed');
        
        const formattedData = {
          symbol: latestAnalysisData.symbol,
          name: latestAnalysisData.name || `${latestAnalysisData.symbol} Corporation`,
          sector: latestAnalysisData.sector || 'Unknown',
          industry: latestAnalysisData.industry || 'Unknown',
          score: latestAnalysisData.score,
          indicators: latestAnalysisData.indicators,
          weighted_scores: latestAnalysisData.weighted_scores,
          timestamp: latestAnalysisData.analysis_date,
          creditsUsed: 0 // This is a saved analysis, not a new one
        };

        setAnalysisData(formattedData);
        setError('');
      } else {
        // Fresh page load, manual mode
        setAnalysisMode('manual');
        setAnalysisPhase('idle');
      }
    } else {
      // Direct navigation to page
      setAnalysisMode('manual');
      setAnalysisPhase('idle');
    }
  }, [dashboardTicker, autoAnalyze, latestAnalysisData]);

  const handleSearch = (ticker = searchTicker) => {
    if (!ticker.trim()) {
      setError('Please enter a stock ticker symbol');
      return;
    }
    
    if (userCredits < 1) {
      setError('Insufficient credits. Please purchase more credits to continue.');
      return;
    }

    // Set manual analysis mode and show confirmation
    setAnalysisMode('manual');
    setAnalysisPhase('confirming');
    setError('');
    setConfirmDialog(true);
  };

  const handleConfirmAnalysis = async () => {
    console.log(`[ANALYSIS] Confirmation dialog confirmed for ${searchTicker} (mode: ${analysisMode})`);
    setConfirmDialog(false);
    await performAnalysis(searchTicker, analysisMode);
  };

  const handleCancelAnalysis = () => {
    setConfirmDialog(false);
    setAnalysisPhase('idle');
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

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Stock Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Get comprehensive technical analysis for any stock with 12 professional indicators
        </Typography>
      </Box>

      {/* Credit Balance */}
      <Card sx={{ mb: 4, backgroundColor: 'primary.main', color: 'white' }}>
        <CardContent sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">Available Credits</Typography>
            <Typography variant="h4">{userCredits}</Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              1 Credit = 1 Analysis
            </Typography>
            <Button 
              variant="outlined" 
              size="small" 
              sx={{ 
                mt: 1, 
                borderColor: 'white', 
                color: 'white',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' }
              }}
            >
              Buy More Credits
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={4}>
        {/* Search Section */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Create New Report
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                fullWidth
                label="Stock Ticker Symbol"
                placeholder="e.g., AAPL, MSFT, GOOGL"
                value={searchTicker}
                onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                error={!!error}
                helperText={error}
                disabled={analysisPhase === 'analyzing' || analysisPhase === 'syncing'}
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                }}
              />
              <Button
                variant="contained"
                size="large"
                onClick={() => handleSearch()}
                disabled={
                  analysisPhase === 'analyzing' || 
                  analysisPhase === 'syncing' || 
                  analysisPhase === 'confirming' || 
                  !searchTicker.trim()
                }
                sx={{ minWidth: 120 }}
              >
                {analysisPhase === 'analyzing' || analysisPhase === 'syncing' ? (
                  <CircularProgress size={24} />
                ) : (
                  'Analyze'
                )}
              </Button>
            </Box>

            {/* AI Explanation Controls */}
            <Box sx={{ mt: 2, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Psychology color="primary" />
                <Typography variant="subtitle2">
                  AI-Powered Explanations
                </Typography>
                <Chip 
                  label="LLaMA 3.1 70B" 
                  size="small" 
                  color="primary" 
                  variant="outlined" 
                />
              </Box>
              
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                🤖 Summary AI explanations will be automatically generated to help you understand your analysis results
              </Typography>
            </Box>
            
            {/* Recent Searches */}
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Recent searches:
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {recentSearches.map((symbol) => (
                  <Chip
                    key={symbol}
                    label={symbol}
                    onClick={() => {
                      setSearchTicker(symbol);
                      handleSearch(symbol);
                    }}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Box>
            </Box>
          </Paper>

          {/* Analysis Results */}
          {(analysisPhase === 'analyzing' || analysisPhase === 'syncing') && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ textAlign: 'center' }}>
                <CircularProgress size={60} sx={{ mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {analysisPhase === 'syncing' ? (
                    `Fetching data for ${searchTicker}...`
                  ) : analysisMode === 'auto' ? (
                    `Auto-analyzing ${searchTicker}...`
                  ) : (
                    `Analyzing ${searchTicker}...`
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {analysisPhase === 'syncing' ? (
                    'Getting latest stock data from Yahoo Finance (this may take 10-30 seconds)'
                  ) : analysisMode === 'auto' ? (
                    'Analysis triggered from dashboard • Running 12 technical indicators • 1 credit will be used'
                  ) : (
                    'Running 12 technical indicators'
                  )}
                </Typography>
                <LinearProgress sx={{ mt: 2 }} />
              </Box>
            </Paper>
          )}

          {analysisData && analysisPhase === 'completed' && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                  <Typography variant="h5" gutterBottom>
                    {analysisData.symbol} - {analysisData.name}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip label={analysisData.sector} />
                    <Chip label={analysisData.industry} variant="outlined" />
                  </Box>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="h3" color={`${getScoreColor(analysisData.score)}.main`}>
                    {analysisData.score}/10
                  </Typography>
                  <Chip 
                    label={getScoreLabel(analysisData.score)} 
                    color={getScoreColor(analysisData.score)}
                  />
                </Box>
              </Box>

              <Typography variant="h6" gutterBottom>
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
            </Paper>
          )}

          {/* Browse Existing Reports Section */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Browse Existing Reports
              </Typography>
              <Button
                variant="outlined"
                onClick={() => navigate('/reports')}
                sx={{ minWidth: 120 }}
              >
                View All Reports
              </Button>
            </Box>
            
            {analysisHistory.length > 0 ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Stock</TableCell>
                      <TableCell align="center">Score</TableCell>
                      <TableCell>Date</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analysisHistory.slice(0, 5).map((analysis) => (
                      <TableRow key={analysis.id} hover>
                        <TableCell>
                          <Box>
                            <Typography variant="subtitle2">
                              {analysis.symbol}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {analysis.name || 'N/A'}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={`${analysis.score}/10`}
                            size="small"
                            color={analysis.score >= 7 ? 'success' : analysis.score >= 4 ? 'warning' : 'error'}
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {new Date(analysis.analysis_date).toLocaleDateString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Tooltip title="View Analysis Results">
                            <IconButton
                              size="small"
                              onClick={() => navigate(`/analysis/${analysis.id}`)}
                              color="primary"
                            >
                              <Visibility fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  No analysis reports yet
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Create your first report using the form above
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Sidebar */}
        <Grid item xs={12} md={4}>
          {/* Popular Stocks */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Popular Stocks
            </Typography>
            <List dense>
              {popularStocks.map((stock, index) => (
                <React.Fragment key={stock.symbol}>
                  <ListItem
                    component="button"
                    onClick={() => {
                      setSearchTicker(stock.symbol);
                      handleSearch(stock.symbol);
                    }}
                    sx={{ cursor: 'pointer' }}
                  >
                    <ListItemText
                      primary={`${stock.symbol} - ${stock.name}`}
                      secondary={stock.sector}
                    />
                    <Analytics />
                  </ListItem>
                  {index < popularStocks.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </Paper>

          {/* Analysis History */}
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Recent Analysis</Typography>
              <IconButton size="small">
                <Refresh />
              </IconButton>
            </Box>
            {analysisHistory.length > 0 ? (
              <List dense>
                {analysisHistory.map((analysis, index) => (
                  <React.Fragment key={analysis.id}>
                    <ListItem sx={{ px: 0 }}>
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
                    </ListItem>
                    {index < analysisHistory.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                No analysis history yet
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog} onClose={() => setConfirmDialog(false)}>
        <DialogTitle>
          {analysisMode === 'auto' ? 'Auto-Analysis from Dashboard' : 'Confirm Analysis'}
        </DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            {analysisMode === 'auto' 
              ? `Auto-analyze ${searchTicker} (triggered from dashboard quick search)`
              : `Analyze ${searchTicker} for 1 credit?`
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
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelAnalysis}>Cancel</Button>
          <Button onClick={handleConfirmAnalysis} variant="contained">
            {analysisMode === 'auto' ? 'Start Auto-Analysis' : 'Confirm Analysis'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default StockSearchPage;