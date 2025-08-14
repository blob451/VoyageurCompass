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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Autocomplete,
  LinearProgress,
  Divider,
  Box as MuiBox
} from '@mui/material';
import {
  Add,
  Remove,
  Compare,
  TrendingUp,
  TrendingDown,
  Analytics,
  FileDownload,
  Refresh,
  // Clear // Not used currently
} from '@mui/icons-material';
// import { useSelector } from 'react-redux'; // Not used currently

const ComparisonPage = () => {
  // const { user } = useSelector((state) => state.auth); // Not used currently
  const [selectedStocks, setSelectedStocks] = useState(['AAPL', 'MSFT']);
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [error, setError] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(false);
  const [userCredits] = useState(25); // Mock credit balance

  // Mock stock suggestions
  const stockSuggestions = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
    'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'BAC', 'XOM',
    'CRM', 'NFLX', 'ADBE', 'PYPL', 'INTC', 'VZ', 'KO', 'NKE', 'MRK'
  ];

  const comparisonMetrics = [
    'Overall Score',
    'SMA Trend',
    'Price vs 50MA',
    'RSI (14)',
    'MACD',
    'Bollinger Position', // cSpell:ignore Bollinger
    'Volume Surge',
    'OBV Trend',
    'Relative 1Y',
    'Relative 2Y'
  ];

  useEffect(() => {
    // Auto-run comparison if we have 2+ stocks
    if (selectedStocks.length >= 2) {
      handleCompare();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  const handleAddStock = (newStock) => {
    if (newStock && !selectedStocks.includes(newStock) && selectedStocks.length < 5) {
      setSelectedStocks([...selectedStocks, newStock]);
    }
  };

  const handleRemoveStock = (stockToRemove) => {
    if (selectedStocks.length > 1) {
      setSelectedStocks(selectedStocks.filter(stock => stock !== stockToRemove));
    }
  };

  const handleCompare = () => {
    if (selectedStocks.length < 2) {
      setError('Please select at least 2 stocks to compare');
      return;
    }
    
    const creditsNeeded = selectedStocks.length;
    if (userCredits < creditsNeeded) {
      setError(`Insufficient credits. Need ${creditsNeeded} credits for this comparison.`);
      return;
    }

    setError('');
    setConfirmDialog(true);
  };

  const handleConfirmComparison = async () => {
    setConfirmDialog(false);
    setLoading(true);
    setError('');

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Mock comparison data
      const mockData = {
        stocks: selectedStocks.map(symbol => ({
          symbol,
          name: `${symbol} Corporation`,
          sector: symbol === 'AAPL' || symbol === 'MSFT' || symbol === 'GOOGL' ? 'Technology' : 'Various',
          price: Math.random() * 200 + 50,
          change: (Math.random() - 0.5) * 10,
          changePercent: (Math.random() - 0.5) * 5,
          metrics: {
            overallScore: Math.floor(Math.random() * 11),
            smaTrend: Math.random(),
            priceVs50MA: Math.random(),
            rsi: Math.random() * 100,
            macd: (Math.random() - 0.5) * 2,
            bollingerPosition: Math.random(),
            volumeSurge: Math.random(),
            obvTrend: Math.random(),
            relative1Y: (Math.random() - 0.5) * 50,
            relative2Y: (Math.random() - 0.5) * 100
          }
        })),
        timestamp: new Date().toISOString(),
        creditsUsed: selectedStocks.length
      };

      setComparisonData(mockData);

    } catch (err) {
      setError('Comparison failed. Please try again.');
      console.error('Comparison error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 7) return 'success';
    if (score >= 4) return 'warning';
    return 'error';
  };

  const getChangeColor = (change) => {
    return change >= 0 ? 'success' : 'error';
  };

  const formatMetricValue = (metric, value) => {
    switch (metric) {
      case 'Overall Score':
        return `${value}/10`;
      case 'RSI (14)':
        return value.toFixed(1);
      case 'Price vs 50MA':
      case 'SMA Trend':
      case 'Bollinger Position': // cSpell:ignore Bollinger
      case 'Volume Surge':
      case 'OBV Trend':
        return (value * 10).toFixed(1);
      case 'Relative 1Y':
      case 'Relative 2Y':
        return `${value.toFixed(1)}%`;
      case 'MACD':
        return value.toFixed(2);
      default:
        return value.toString();
    }
  };

  const exportComparison = () => {
    if (!comparisonData) return;
    
    // Mock export functionality
    const csvContent = [
      ['Stock', ...comparisonMetrics].join(','),
      ...comparisonData.stocks.map(stock => [
        stock.symbol,
        stock.metrics.overallScore,
        (stock.metrics.smaTrend * 10).toFixed(1),
        (stock.metrics.priceVs50MA * 10).toFixed(1),
        stock.metrics.rsi.toFixed(1),
        stock.metrics.macd.toFixed(2),
        (stock.metrics.bollingerPosition * 10).toFixed(1), // cSpell:ignore bollingerPosition
        (stock.metrics.volumeSurge * 10).toFixed(1),
        (stock.metrics.obvTrend * 10).toFixed(1),
        stock.metrics.relative1Y.toFixed(1),
        stock.metrics.relative2Y.toFixed(1)
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stock-comparison-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Stock Comparison
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Compare multiple stocks side-by-side with comprehensive technical analysis
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
              Cost: {selectedStocks.length} Credits ({selectedStocks.length} stocks)
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

      {/* Stock Selection */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Select Stocks to Compare
        </Typography>
        
        <Grid container spacing={2} alignItems="center" sx={{ mb: 3 }}>
          <Grid item xs={12} md={8}>
            <Autocomplete
              options={stockSuggestions.filter(stock => !selectedStocks.includes(stock))}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  label="Add Stock Symbol" 
                  placeholder="Type to search..."
                />
              )}
              onChange={(event, value) => {
                if (value) {
                  handleAddStock(value);
                }
              }}
              disabled={selectedStocks.length >= 5}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              variant="contained"
              size="large"
              onClick={handleCompare}
              disabled={loading || selectedStocks.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : <Compare />}
              fullWidth
            >
              {loading ? 'Comparing...' : 'Compare Stocks'}
            </Button>
          </Grid>
        </Grid>

        {/* Selected Stocks */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
          {selectedStocks.map((stock) => (
            <Chip
              key={stock}
              label={stock}
              onDelete={() => handleRemoveStock(stock)}
              deleteIcon={<Remove />}
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <Typography variant="body2" color="text.secondary">
          Select 2-5 stocks for comparison. Each stock costs 1 credit to analyze.
        </Typography>
      </Paper>

      {/* Loading State */}
      {loading && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Comparing {selectedStocks.length} stocks...
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Running technical analysis for each stock
            </Typography>
            <LinearProgress sx={{ mt: 2 }} />
          </Box>
        </Paper>
      )}

      {/* Comparison Results */}
      {comparisonData && !loading && (
        <>
          {/* Stock Overview Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {comparisonData.stocks.map((stock) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={stock.symbol}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {stock.symbol}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {stock.name}
                    </Typography>
                    <Chip label={stock.sector} size="small" sx={{ mb: 2 }} />
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h5">
                        ${stock.price.toFixed(2)}
                      </Typography>
                      <Box sx={{ textAlign: 'right' }}>
                        <Typography 
                          variant="body2" 
                          color={`${getChangeColor(stock.change)}.main`}
                          sx={{ display: 'flex', alignItems: 'center' }}
                        >
                          {stock.change >= 0 ? <TrendingUp fontSize="small" /> : <TrendingDown fontSize="small" />}
                          {stock.changePercent.toFixed(2)}%
                        </Typography>
                      </Box>
                    </Box>
                    
                    <Box sx={{ textAlign: 'center', mt: 2 }}>
                      <Typography variant="h4" color={`${getScoreColor(stock.metrics.overallScore)}.main`}>
                        {stock.metrics.overallScore}/10
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Technical Score
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Detailed Comparison Table */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">
                Detailed Comparison
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  startIcon={<FileDownload />}
                  onClick={exportComparison}
                  variant="outlined"
                  size="small"
                >
                  Export CSV
                </Button>
                <Button
                  startIcon={<Refresh />}
                  onClick={handleCompare}
                  variant="outlined"
                  size="small"
                >
                  Refresh
                </Button>
              </Box>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>Metric</TableCell>
                    {comparisonData.stocks.map((stock) => (
                      <TableCell key={stock.symbol} align="center" sx={{ fontWeight: 600 }}>
                        {stock.symbol}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {comparisonMetrics.map((metric, index) => (
                    <TableRow key={metric} sx={{ backgroundColor: index % 2 === 0 ? 'grey.50' : 'white' }}>
                      <TableCell sx={{ fontWeight: 500 }}>
                        {metric}
                      </TableCell>
                      {comparisonData.stocks.map((stock) => {
                        let value;
                        switch (metric) {
                          case 'Overall Score':
                            value = stock.metrics.overallScore;
                            break;
                          case 'SMA Trend':
                            value = stock.metrics.smaTrend;
                            break;
                          case 'Price vs 50MA':
                            value = stock.metrics.priceVs50MA;
                            break;
                          case 'RSI (14)':
                            value = stock.metrics.rsi;
                            break;
                          case 'MACD':
                            value = stock.metrics.macd;
                            break;
                          case 'Bollinger Position': // cSpell:ignore Bollinger
                            value = stock.metrics.bollingerPosition; // cSpell:ignore bollingerPosition
                            break;
                          case 'Volume Surge':
                            value = stock.metrics.volumeSurge;
                            break;
                          case 'OBV Trend':
                            value = stock.metrics.obvTrend;
                            break;
                          case 'Relative 1Y':
                            value = stock.metrics.relative1Y;
                            break;
                          case 'Relative 2Y':
                            value = stock.metrics.relative2Y;
                            break;
                          default:
                            value = 0;
                        }

                        return (
                          <TableCell key={stock.symbol} align="center">
                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {formatMetricValue(metric, value)}
                              </Typography>
                              {metric !== 'Overall Score' && metric !== 'RSI (14)' && metric !== 'MACD' && 
                               metric !== 'Relative 1Y' && metric !== 'Relative 2Y' && (
                                <LinearProgress
                                  variant="determinate"
                                  value={metric === 'Overall Score' ? (value / 10) * 100 : value * 100}
                                  sx={{ width: 40, height: 4, borderRadius: 2, mt: 0.5 }}
                                  color={
                                    metric === 'Overall Score' 
                                      ? getScoreColor(value)
                                      : value >= 0.7 ? 'success' : value >= 0.4 ? 'warning' : 'error'
                                  }
                                />
                              )}
                            </Box>
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Analysis Summary */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Comparison Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                  Best Performers
                </Typography>
                {[...comparisonData.stocks]
                  .sort((a, b) => b.metrics.overallScore - a.metrics.overallScore)
                  .slice(0, 2)
                  .map((stock, index) => (
                    <Box key={stock.symbol} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Chip 
                        label={index + 1} 
                        size="small" 
                        color={index === 0 ? 'success' : 'warning'}
                        sx={{ mr: 1, minWidth: 32 }}
                      />
                      <Typography variant="body2">
                        <strong>{stock.symbol}</strong> - Score: {stock.metrics.overallScore}/10
                      </Typography>
                    </Box>
                  ))}
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                  Analysis Details
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Analyzed {comparisonData.stocks.length} stocks
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Used {comparisonData.creditsUsed} credits
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Analysis completed at {new Date(comparisonData.timestamp).toLocaleString()}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </>
      )}

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog} onClose={() => setConfirmDialog(false)}>
        <DialogTitle>Confirm Stock Comparison</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Compare <strong>{selectedStocks.join(', ')}</strong> for {selectedStocks.length} credits?
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This will use {selectedStocks.length} credits from your balance ({userCredits} remaining)
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog(false)}>Cancel</Button>
          <Button onClick={handleConfirmComparison} variant="contained">
            Confirm Comparison
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ComparisonPage;