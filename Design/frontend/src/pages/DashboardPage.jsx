import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useSelector } from 'react-redux';
import { selectCurrentUser } from '../features/auth/authSlice';
import { useGetPortfoliosQuery, useGetStocksQuery } from '../features/api/apiSlice';

const DashboardPage = () => {
  const user = useSelector(selectCurrentUser);
  const { data: portfolios, isLoading: portfoliosLoading } = useGetPortfoliosQuery();
  const { data: stocks, isLoading: stocksLoading } = useGetStocksQuery({ page_size: 5 });

  // Sample data for the chart (replace with real data from API)
  const chartData = [
    { name: 'Jan', value: 4000, profit: 2400 },
    { name: 'Feb', value: 3000, profit: 1398 },
    { name: 'Mar', value: 2000, profit: 9800 },
    { name: 'Apr', value: 2780, profit: 3908 },
    { name: 'May', value: 1890, profit: 4800 },
    { name: 'Jun', value: 2390, profit: 3800 },
    { name: 'Jul', value: 3490, profit: 4300 },
  ];

  const marketData = [
    { name: 'AAPL', price: 178.50, change: 2.3 },
    { name: 'MSFT', price: 425.20, change: -1.2 },
    { name: 'GOOGL', price: 142.30, change: 0.8 },
    { name: 'AMZN', price: 180.90, change: 3.1 },
    { name: 'TSLA', price: 240.50, change: -2.5 },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome back, {user?.username || 'Investor'}!
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Here's your portfolio overview and market insights
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Portfolio Summary Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Value
              </Typography>
              <Typography variant="h5" component="div">
                $124,563
              </Typography>
              <Typography variant="body2" color="success.main">
                +12.3% this month
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Today's Gain
              </Typography>
              <Typography variant="h5" component="div">
                $2,458
              </Typography>
              <Typography variant="body2" color="success.main">
                +1.98%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Portfolios
              </Typography>
              <Typography variant="h5" component="div">
                {portfoliosLoading ? <CircularProgress size={20} /> : portfolios?.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Active portfolios
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Holdings
              </Typography>
              <Typography variant="h5" component="div">
                23
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Across all portfolios
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Performance Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Performance
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="profit"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Market Overview */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Market Movers
            </Typography>
            <Box sx={{ mt: 2 }}>
              {marketData.map((stock) => (
                <Box
                  key={stock.name}
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    py: 1,
                    borderBottom: '1px solid #e0e0e0',
                  }}
                >
                  <Typography variant="body2" fontWeight="bold">
                    {stock.name}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Typography variant="body2">
                      ${stock.price}
                    </Typography>
                    <Typography
                      variant="body2"
                      color={stock.change > 0 ? 'success.main' : 'error.main'}
                    >
                      {stock.change > 0 ? '+' : ''}{stock.change}%
                    </Typography>
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Recent Stocks */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recently Tracked Stocks
            </Typography>
            {stocksLoading ? (
              <CircularProgress />
            ) : stocks?.results?.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={stocks.results.map(s => ({
                  name: s.symbol,
                  price: s.latest_price?.close || 0,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="price" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">
                No stocks tracked yet. Start by searching and adding stocks to your watchlist.
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;