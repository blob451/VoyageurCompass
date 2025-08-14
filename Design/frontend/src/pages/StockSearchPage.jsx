import React, { useState, useEffect } from 'react';
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
  Tooltip
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
  Refresh
} from '@mui/icons-material';
// import { useSelector } from 'react-redux'; // Not used currently

const StockSearchPage = () => {
  // const { user } = useSelector((state) => state.auth); // Not used currently
  const [searchTicker, setSearchTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [userCredits] = useState(25); // Mock credit balance

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

  useEffect(() => {
    // Load analysis history (mock data)
    setAnalysisHistory([
      {
        id: 1,
        symbol: 'AAPL',
        score: 7,
        date: '2025-01-14',
        sector: 'Technology'
      },
      {
        id: 2,
        symbol: 'MSFT',
        score: 8,
        date: '2025-01-13',
        sector: 'Technology'
      }
    ]);
  }, []);

  const handleSearch = (ticker = searchTicker) => {
    if (!ticker.trim()) {
      setError('Please enter a stock ticker symbol');
      return;
    }
    
    if (userCredits < 1) {
      setError('Insufficient credits. Please purchase more credits to continue.');
      return;
    }

    setError('');
    setConfirmDialog(true);
  };

  const handleConfirmAnalysis = async () => {
    setConfirmDialog(false);
    setLoading(true);
    setError('');

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Mock analysis data - replace with actual API call
      const mockAnalysisData = {
        symbol: searchTicker.toUpperCase(),
        name: `${searchTicker.toUpperCase()} Corporation`,
        sector: 'Technology',
        industry: 'Software',
        score: Math.floor(Math.random() * 11), // 0-10
        indicators: {
          sma50vs200: { score: 1.0, description: 'SMA50 above SMA200 - Strong trend' },
          pricevs50: { score: 0.813, description: 'Price 6.3% above 50-day average' }, // cSpell:ignore pricevs50
          rsi14: { score: 0.599, description: 'RSI at 59.8 - Neutral momentum' },
          macd12269: { score: 0.740, description: 'MACD bullish crossover' },
          bbpos20: { score: 0.039, description: 'Near upper Bollinger Band' }, // cSpell:ignore bbpos20 Bollinger
          bbwidth20: { score: 0.548, description: 'Average volatility' }, // cSpell:ignore bbwidth20
          volsurge: { score: 0.600, description: 'Volume below average' }, // cSpell:ignore volsurge
          obv20: { score: 0.594, description: 'OBV showing accumulation' },
          rel1y: { score: 0.500, description: 'Neutral 1-year performance' },
          rel2y: { score: 0.466, description: 'Underperforming 2-year' },
          candlerev: { score: 0.500, description: 'Neutral candlestick pattern' }, // cSpell:ignore candlerev
          srcontext: { score: 0.500, description: 'Between support and resistance' } // cSpell:ignore srcontext
        },
        timestamp: new Date().toISOString(),
        creditsUsed: 1
      };

      setAnalysisData(mockAnalysisData);
      
      // Add to history
      setAnalysisHistory(prev => [
        {
          id: Date.now(),
          symbol: mockAnalysisData.symbol,
          score: mockAnalysisData.score,
          date: new Date().toISOString().split('T')[0],
          sector: mockAnalysisData.sector
        },
        ...prev.slice(0, 9) // Keep only last 10
      ]);

    } catch (err) {
      setError('Analysis failed. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
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
              Search Stock
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
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />
                }}
              />
              <Button
                variant="contained"
                size="large"
                onClick={() => handleSearch()}
                disabled={loading || !searchTicker.trim()}
                sx={{ minWidth: 120 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Analyze'}
              </Button>
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
          {loading && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ textAlign: 'center' }}>
                <CircularProgress size={60} sx={{ mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Analyzing {searchTicker}...
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Running 12 technical indicators
                </Typography>
                <LinearProgress sx={{ mt: 2 }} />
              </Box>
            </Paper>
          )}

          {analysisData && !loading && (
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
                    {Object.entries(analysisData.indicators).map(([key, indicator]) => (
                      <TableRow key={key}>
                        <TableCell sx={{ fontWeight: 500 }}>
                          {key.toUpperCase()}
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={indicator.score * 100}
                              sx={{ width: 60, height: 8, borderRadius: 4 }}
                              color={indicator.score >= 0.7 ? 'success' : indicator.score >= 0.4 ? 'warning' : 'error'}
                            />
                            <Typography variant="body2">
                              {(indicator.score * 10).toFixed(1)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {indicator.description}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}
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
                    button
                    onClick={() => {
                      setSearchTicker(stock.symbol);
                      handleSearch(stock.symbol);
                    }}
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
                        secondary={`${analysis.sector} • ${analysis.date}`}
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
        <DialogTitle>Confirm Analysis</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Analyze <strong>{searchTicker}</strong> for 1 credit?
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This will use 1 credit from your balance ({userCredits} remaining)
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog(false)}>Cancel</Button>
          <Button onClick={handleConfirmAnalysis} variant="contained">
            Confirm Analysis
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default StockSearchPage;