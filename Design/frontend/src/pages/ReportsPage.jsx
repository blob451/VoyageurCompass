import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  IconButton,
  CircularProgress,
  Alert,
  InputAdornment,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  Search,
  Visibility,
  Add,
  TrendingUp,
  FilterList,
  Sort
} from '@mui/icons-material';
import { useGetUserAnalysisHistoryQuery } from '../features/api/apiSlice';

const ReportsPage = () => {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchFilter, setSearchFilter] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');

  // Fetch user's analysis history with server-side pagination
  const { 
    data: analysisData, 
    error, 
    isLoading,
    isFetching 
  } = useGetUserAnalysisHistoryQuery({ 
    limit: rowsPerPage,
    offset: page * rowsPerPage,
    fields: 'id,symbol,name,score,analysis_date,sector,industry' // Exclude heavy fields
  }, {
    // Skip the query if we don't have basic params
    skip: false,
    // Reduce polling frequency to improve performance
    pollingInterval: 0, // Disable automatic polling
    // Cache for better performance
    refetchOnMountOrArgChange: 30,
  });

  const reports = Array.isArray(analysisData?.analyses) ? analysisData.analyses : [];
  const pagination = analysisData?.pagination || {};

  // Use server-side paginated data directly (filtering/sorting will be moved to server-side in future)
  const paginatedReports = reports;

  const handleChangePage = (event, newPage) => {
    try {
      setPage(newPage);
    } catch (error) {
      console.warn('Error changing page:', error);
      setPage(0);
    }
  };

  const handleChangeRowsPerPage = (event) => {
    try {
      setRowsPerPage(parseInt(event.target.value, 10));
      setPage(0);
    } catch (error) {
      console.warn('Error changing rows per page:', error);
      setRowsPerPage(10);
      setPage(0);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 7) return 'success';
    if (score >= 4) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score) => {
    const numScore = typeof score === 'number' ? score : 0;
    if (numScore >= 8) return 'Strong Buy';
    if (numScore >= 6) return 'Buy';
    if (numScore >= 4) return 'Hold';
    return 'Sell';
  };

  // Safe navigation handlers
  const handleViewAnalysis = (reportId) => {
    try {
      if (reportId) {
        navigate(`/analysis/${reportId}`);
      } else {
        console.warn('No report ID provided for navigation');
      }
    } catch (error) {
      console.error('Error navigating to analysis:', error);
    }
  };

  const handleRunNewAnalysis = (symbol) => {
    try {
      if (symbol) {
        navigate('/stocks', { 
          state: { searchTicker: symbol, autoAnalyze: true } 
        });
      } else {
        navigate('/stocks');
      }
    } catch (error) {
      console.error('Error navigating to stocks page:', error);
      navigate('/stocks');
    }
  };

  // Show loading only on initial load, not on pagination changes
  if (isLoading && !analysisData) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6">Loading your reports...</Typography>
          </Box>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Alert severity="error">
          Failed to load reports: {error.data?.error || 'Please try again later'}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Analysis Reports
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Browse and manage all your stock analysis reports
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => navigate('/stocks')}
          sx={{ minWidth: 180 }}
        >
          Create New Report
        </Button>
      </Box>

      <Paper sx={{ p: 3 }}>
        {/* Search and Filter Controls - Temporarily disabled for server-side pagination */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <TextField
            placeholder="Search by symbol, company, sector... (Coming soon with server-side search)"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            disabled
            sx={{ minWidth: 300, flex: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
          />
          
          <FormControl sx={{ minWidth: 150 }} disabled>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              label="Sort By"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="date">Date</MenuItem>
              <MenuItem value="symbol">Symbol</MenuItem>
              <MenuItem value="score">Score</MenuItem>
              <MenuItem value="sector">Sector</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }} disabled>
            <InputLabel>Order</InputLabel>
            <Select
              value={sortOrder}
              label="Order"
              onChange={(e) => setSortOrder(e.target.value)}
            >
              <MenuItem value="desc">Descending</MenuItem>
              <MenuItem value="asc">Ascending</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Statistics */}
        <Box sx={{ mb: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            <strong>{pagination.total_count || 0}</strong> reports total
            {reports.length > 0 && ` • Showing ${pagination.count || 0} on this page`}
          </Typography>
          {isFetching && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption" color="text.secondary">
                Loading...
              </Typography>
            </Box>
          )}
        </Box>

        {/* Reports Table */}
        {paginatedReports.length > 0 ? (
          <>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Stock</TableCell>
                    <TableCell>Company</TableCell>
                    <TableCell>Sector</TableCell>
                    <TableCell align="center">Score</TableCell>
                    <TableCell align="center">Recommendation</TableCell>
                    <TableCell>Analysis Date</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedReports.map((report) => {
                    if (!report || !report.id) return null;
                    
                    // Safe data extraction with fallbacks
                    const symbol = report.symbol || 'N/A';
                    const name = report.name || 'N/A';
                    const sector = report.sector || 'Unknown';
                    const score = typeof report.score === 'number' ? report.score : 0;
                    
                    // Safe date handling
                    let formattedDate = 'N/A';
                    let formattedTime = '';
                    
                    try {
                      if (report.analysis_date) {
                        const date = new Date(report.analysis_date);
                        if (!isNaN(date.getTime())) {
                          formattedDate = date.toLocaleDateString();
                          formattedTime = date.toLocaleTimeString();
                        }
                      }
                    } catch (error) {
                      console.warn('Error parsing date for report:', report.id, error);
                    }
                    
                    return (
                      <TableRow key={report.id} hover>
                        <TableCell>
                          <Typography variant="h6" component="span">
                            {symbol}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {name}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={sector} 
                            size="small" 
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Typography variant="h6" color={`${getScoreColor(score)}.main`}>
                            {score}/10
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={getScoreLabel(score)}
                            color={getScoreColor(score)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {formattedDate}
                          </Typography>
                          {formattedTime && (
                            <Typography variant="caption" color="text.secondary">
                              {formattedTime}
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell align="center">
                          <IconButton
                            onClick={() => handleViewAnalysis(report.id)}
                            color="primary"
                            title="View Analysis Results"
                          >
                            <Visibility />
                          </IconButton>
                          <IconButton
                            onClick={() => handleRunNewAnalysis(symbol)}
                            color="secondary"
                            title="Run New Analysis"
                          >
                            <TrendingUp />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    );
                  }).filter(Boolean)}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              component="div"
              count={pagination.total_count || 0}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[5, 10, 25, 50]}
            />
          </>
        ) : (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" gutterBottom>
              {searchFilter ? 'No reports match your search' : 'No reports found'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {searchFilter 
                ? 'Try adjusting your search terms or filters'
                : 'Get started by creating your first analysis report'
              }
            </Typography>
            {!searchFilter && (
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => navigate('/stocks')}
              >
                Create Your First Report
              </Button>
            )}
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ReportsPage;