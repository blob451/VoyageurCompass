import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  CompareArrows,
  Analytics,
  Business,
  Computer,
  LocalHospital,
  AccountBalance,
  ShoppingCart,
  Build,
  Home,
  FlightTakeoff,
  Phone,
  ElectricBolt,
  Sentiment,
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral
} from '@mui/icons-material';
import { useGetUserAnalysisHistoryQuery } from '../features/api/apiSlice';

const SectorPage = () => {
  const [selectedSectors, setSelectedSectors] = useState(['technology', 'healthcare']);
  const [timeframe, setTimeframe] = useState('1M');
  const [loading, setLoading] = useState(false);
  const [sectorData, setSectorData] = useState(null);
  const [confirmDialog, setConfirmDialog] = useState(false);
  const [userCredits] = useState(25); // Mock credit balance
  
  // Fetch user analysis history to get real sentiment data
  const { data: analysisHistoryData } = useGetUserAnalysisHistoryQuery({ 
    limit: 100, 
    fields: 'symbol,sector,components' 
  });

  const sectorIcons = {
    technology: <Computer />,
    healthcare: <LocalHospital />,
    financials: <AccountBalance />, // cSpell:ignore financials
    consumer_discretionary: <ShoppingCart />,
    industrials: <Build />,
    real_estate: <Home />,
    utilities: <FlightTakeoff />,
    telecommunications: <Phone />,
    energy: <ElectricBolt />,
    materials: <Business />
  };

  const availableSectors = [
    { id: 'technology', name: 'Technology', description: 'Software, hardware, semiconductors' },
    { id: 'healthcare', name: 'Healthcare', description: 'Pharmaceuticals, medical devices, biotech' },
    { id: 'financials', name: 'Financials', description: 'Banks, insurance, investment services' }, // cSpell:ignore financials Financials
    { id: 'consumer_discretionary', name: 'Consumer Discretionary', description: 'Retail, automotive, entertainment' },
    { id: 'industrials', name: 'Industrials', description: 'Manufacturing, aerospace, transportation' },
    { id: 'real_estate', name: 'Real Estate', description: 'REITs, property development' },
    { id: 'utilities', name: 'Utilities', description: 'Electric, gas, water utilities' },
    { id: 'telecommunications', name: 'Telecommunications', description: 'Telecom services, media' },
    { id: 'energy', name: 'Energy', description: 'Oil, gas, renewable energy' },
    { id: 'materials', name: 'Materials', description: 'Mining, chemicals, metals' }
  ];

  const timeframes = ['1D', '1W', '1M', '3M', '6M', '1Y'];

  useEffect(() => {
    // Auto-load sector data on component mount
    handleAnalyzeSectors();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  const handleSectorChange = (event) => {
    const value = event.target.value;
    setSelectedSectors(typeof value === 'string' ? value.split(',') : value);
  };

  const handleAnalyzeSectors = () => {
    if (selectedSectors.length < 2) {
      return;
    }
    
    const creditsNeeded = Math.ceil(selectedSectors.length / 2); // 2 sectors per credit
    if (userCredits < creditsNeeded) {
      return;
    }

    setConfirmDialog(true);
  };

  const handleConfirmAnalysis = async () => {
    setConfirmDialog(false);
    setLoading(true);

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Mock sector data
      const mockData = {
        sectors: selectedSectors.map(sectorId => {
          const sector = availableSectors.find(s => s.id === sectorId);
          const performance = (Math.random() - 0.5) * 20; // -10% to +10%
          const trend = performance > 2 ? 'up' : performance < -2 ? 'down' : 'flat';
          
          return {
            id: sectorId,
            name: sector.name,
            description: sector.description,
            performance: performance,
            trend: trend,
            marketCap: Math.random() * 5000 + 1000, // Billions
            avgVolume: Math.random() * 100 + 50, // Millions
            topStocks: [
              { symbol: 'STOCK1', performance: (Math.random() - 0.5) * 15 },
              { symbol: 'STOCK2', performance: (Math.random() - 0.5) * 15 },
              { symbol: 'STOCK3', performance: (Math.random() - 0.5) * 15 }
            ],
            metrics: {
              momentum: Math.random() * 10,
              volatility: Math.random() * 10,
              volume: Math.random() * 10,
              sentiment: calculateRealSentimentForSector(sectorId)
            }
          };
        }),
        timeframe: timeframe,
        timestamp: new Date().toISOString(),
        creditsUsed: Math.ceil(selectedSectors.length / 2)
      };

      setSectorData(mockData);

    } catch (err) {
      console.error('Sector analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp color="success" />;
      case 'down': return <TrendingDown color="error" />;
      default: return <TrendingFlat color="warning" />;
    }
  };

  const getTrendColor = (performance) => {
    if (performance > 2) return 'success';
    if (performance < -2) return 'error';
    return 'warning';
  };

  const getMetricColor = (value) => {
    if (value >= 7) return 'success';
    if (value >= 4) return 'warning';
    return 'error';
  };

  const getSentimentIcon = (sentiment) => {
    if (sentiment >= 6) return <SentimentSatisfied color="success" />;
    if (sentiment >= 4) return <SentimentNeutral color="warning" />;
    return <SentimentDissatisfied color="error" />;
  };

  const calculateRealSentimentForSector = (sectorId) => {
    if (!analysisHistoryData?.analyses) {
      return Math.random() * 10; // Fallback to random if no data
    }

    const sectorName = availableSectors.find(s => s.id === sectorId)?.name;
    if (!sectorName) {
      return Math.random() * 10;
    }

    // Filter analyses for this sector and extract sentiment scores
    const sectorAnalyses = analysisHistoryData.analyses.filter(
      analysis => analysis.sector?.toLowerCase() === sectorName.toLowerCase()
    );

    const sentimentScores = sectorAnalyses
      .map(analysis => {
        const sentimentData = analysis.components?.sentiment?.raw;
        if (sentimentData?.sentiment) {
          // Convert sentiment score (-1 to 1) to 0-10 scale
          const normalizedScore = (parseFloat(sentimentData.sentiment) + 1) * 5;
          return Math.max(0, Math.min(10, normalizedScore));
        }
        return null;
      })
      .filter(score => score !== null);

    if (sentimentScores.length === 0) {
      return Math.random() * 10; // Fallback if no sentiment data
    }

    // Return average sentiment score for the sector
    return sentimentScores.reduce((sum, score) => sum + score, 0) / sentimentScores.length;
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Sector Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Compare sector performance and identify market trends across industries
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
              Cost: {Math.ceil(selectedSectors.length / 2)} Credits ({selectedSectors.length} sectors)
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7, fontSize: '0.8rem' }}>
              2 sectors = 1 credit
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Select Sectors</InputLabel>
              <Select
                multiple
                value={selectedSectors}
                onChange={handleSectorChange}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => {
                      const sector = availableSectors.find(s => s.id === value);
                      return (
                        <Chip 
                          key={value} 
                          label={sector?.name || value} 
                          size="small"
                          icon={sectorIcons[value]}
                        />
                      );
                    })}
                  </Box>
                )}
              >
                {availableSectors.map((sector) => (
                  <MenuItem key={sector.id} value={sector.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {sectorIcons[sector.id]}
                      <Box>
                        <Typography variant="body1">{sector.name}</Typography>
                        <Typography variant="body2" color="text.secondary" fontSize="0.8rem">
                          {sector.description}
                        </Typography>
                      </Box>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                label="Timeframe"
              >
                {timeframes.map((tf) => (
                  <MenuItem key={tf} value={tf}>{tf}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <Button
              variant="contained"
              size="large"
              onClick={handleAnalyzeSectors}
              disabled={loading || selectedSectors.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : <Analytics />}
              fullWidth
            >
              {loading ? 'Analyzing...' : 'Analyze Sectors'}
            </Button>
          </Grid>
        </Grid>

        {selectedSectors.length < 2 && (
          <Alert severity="info" sx={{ mt: 2 }}>
            Please select at least 2 sectors for comparison
          </Alert>
        )}
      </Paper>

      {/* Loading State */}
      {loading && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Analyzing {selectedSectors.length} sectors...
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Processing market data and performance metrics
            </Typography>
            <LinearProgress sx={{ mt: 2 }} />
          </Box>
        </Paper>
      )}

      {/* Sector Performance Overview */}
      {sectorData && !loading && (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {sectorData.sectors.map((sector) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={sector.id}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      {sectorIcons[sector.id]}
                      <Typography variant="h6" sx={{ ml: 1 }}>
                        {sector.name}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h4" color={`${getTrendColor(sector.performance)}.main`}>
                        {sector.performance > 0 ? '+' : ''}{sector.performance.toFixed(2)}%
                      </Typography>
                      {getTrendIcon(sector.trend)}
                    </Box>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2, height: 40 }}>
                      {sector.description}
                    </Typography>

                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Market Cap: ${sector.marketCap.toFixed(0)}B
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Avg Volume: {sector.avgVolume.toFixed(0)}M
                      </Typography>
                    </Box>

                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                      Top Performers:
                    </Typography>
                    {sector.topStocks.map((stock, index) => (
                      <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">{stock.symbol}</Typography>
                        <Typography 
                          variant="body2" 
                          color={stock.performance >= 0 ? 'success.main' : 'error.main'}
                        >
                          {stock.performance > 0 ? '+' : ''}{stock.performance.toFixed(1)}%
                        </Typography>
                      </Box>
                    ))}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Detailed Metrics Table */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Sector Metrics Comparison
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>Sector</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Performance</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Momentum</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Volatility</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Volume</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Sentiment</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Market Cap</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sectorData.sectors.map((sector) => (
                    <TableRow key={sector.id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {sectorIcons[sector.id]}
                          <Typography sx={{ ml: 1, fontWeight: 500 }}>
                            {sector.name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Typography 
                            variant="body2" 
                            color={`${getTrendColor(sector.performance)}.main`}
                            sx={{ fontWeight: 500 }}
                          >
                            {sector.performance > 0 ? '+' : ''}{sector.performance.toFixed(2)}%
                          </Typography>
                          <Box sx={{ ml: 1 }}>
                            {getTrendIcon(sector.trend)}
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Typography variant="body2">{sector.metrics.momentum.toFixed(1)}/10</Typography>
                          <LinearProgress
                            variant="determinate"
                            value={sector.metrics.momentum * 10}
                            sx={{ width: 40, height: 4, borderRadius: 2 }}
                            color={getMetricColor(sector.metrics.momentum)}
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Typography variant="body2">{sector.metrics.volatility.toFixed(1)}/10</Typography>
                          <LinearProgress
                            variant="determinate"
                            value={sector.metrics.volatility * 10}
                            sx={{ width: 40, height: 4, borderRadius: 2 }}
                            color={getMetricColor(sector.metrics.volatility)}
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Typography variant="body2">{sector.metrics.volume.toFixed(1)}/10</Typography>
                          <LinearProgress
                            variant="determinate"
                            value={sector.metrics.volume * 10}
                            sx={{ width: 40, height: 4, borderRadius: 2 }}
                            color={getMetricColor(sector.metrics.volume)}
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                            {getSentimentIcon(sector.metrics.sentiment)}
                            <Typography variant="body2">{sector.metrics.sentiment.toFixed(1)}/10</Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={sector.metrics.sentiment * 10}
                            sx={{ width: 40, height: 4, borderRadius: 2 }}
                            color={getMetricColor(sector.metrics.sentiment)}
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Typography variant="body2">
                          ${sector.marketCap.toFixed(0)}B
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Analysis Summary */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Sector Analysis Summary
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                  Best Performing Sectors
                </Typography>
                {[...sectorData.sectors]
                  .sort((a, b) => b.performance - a.performance)
                  .slice(0, 3)
                  .map((sector, index) => (
                    <Box key={sector.id} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Chip 
                        label={index + 1} 
                        size="small" 
                        color={index === 0 ? 'success' : index === 1 ? 'warning' : 'info'}
                        sx={{ mr: 1, minWidth: 32 }}
                      />
                      <Box sx={{ display: 'flex', alignItems: 'center', mr: 1 }}>
                        {sectorIcons[sector.id]}
                      </Box>
                      <Typography variant="body2">
                        <strong>{sector.name}</strong> - {sector.performance > 0 ? '+' : ''}{sector.performance.toFixed(2)}%
                      </Typography>
                    </Box>
                  ))}
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                  Analysis Details
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Analyzed {sectorData.sectors.length} sectors
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Timeframe: {sectorData.timeframe}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Used {sectorData.creditsUsed} credits
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Analysis completed at {new Date(sectorData.timestamp).toLocaleString()}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </>
      )}

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog} onClose={() => setConfirmDialog(false)}>
        <DialogTitle>Confirm Sector Analysis</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Analyze <strong>{selectedSectors.length} sectors</strong> for {Math.ceil(selectedSectors.length / 2)} credits?
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This will use {Math.ceil(selectedSectors.length / 2)} credits from your balance ({userCredits} remaining)
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

export default SectorPage;