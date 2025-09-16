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
import { useTranslation } from 'react-i18next';
import { useGetUserAnalysisHistoryQuery } from '../features/api/apiSlice';

const ReportsPage = () => {
  const { t } = useTranslation();
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
    if (numScore >= 8) return t('recommendations.strongBuy');
    if (numScore >= 6) return t('recommendations.buy');
    if (numScore >= 4) return t('recommendations.hold');
    return t('recommendations.sell');
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
            <Typography variant="h6">{t('reports.loadingReports')}</Typography>
          </Box>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Alert severity="error">
          {t('reports.failedToLoad')}: {error.data?.error || t('reports.tryAgainLater')}
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
            {t('reports.title')}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {t('reports.subtitle')}
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => navigate('/stocks')}
          sx={{ minWidth: 180 }}
        >
          {t('reports.createNewReport')}
        </Button>
      </Box>

      <Paper sx={{ p: 3 }}>
        {/* Search and Filter Controls - Temporarily disabled for server-side pagination */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <TextField
            placeholder={t('reports.searchPlaceholder')}
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
            <InputLabel>{t('reports.sortBy')}</InputLabel>
            <Select
              value={sortBy}
              label={t('reports.sortBy')}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="date">{t('reports.sortOptions.date')}</MenuItem>
              <MenuItem value="symbol">{t('reports.sortOptions.symbol')}</MenuItem>
              <MenuItem value="score">{t('reports.sortOptions.score')}</MenuItem>
              <MenuItem value="sector">{t('reports.sortOptions.sector')}</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }} disabled>
            <InputLabel>{t('reports.order')}</InputLabel>
            <Select
              value={sortOrder}
              label={t('reports.order')}
              onChange={(e) => setSortOrder(e.target.value)}
            >
              <MenuItem value="desc">{t('reports.orderOptions.descending')}</MenuItem>
              <MenuItem value="asc">{t('reports.orderOptions.ascending')}</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Statistics */}
        <Box sx={{ mb: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {t('reports.reportsTotal', { count: pagination.total_count || 0 })}
            {reports.length > 0 && t('reports.showingOnPage', { count: pagination.count || 0 })}
          </Typography>
          {isFetching && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption" color="text.secondary">
                {t('reports.loading')}
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
                    <TableCell>{t('reports.table.stock')}</TableCell>
                    <TableCell>{t('reports.table.company')}</TableCell>
                    <TableCell>{t('reports.table.sector')}</TableCell>
                    <TableCell align="center">{t('reports.table.score')}</TableCell>
                    <TableCell align="center">{t('reports.table.recommendation')}</TableCell>
                    <TableCell>{t('reports.table.analysisDate')}</TableCell>
                    <TableCell align="center">{t('reports.table.actions')}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedReports.map((report) => {
                    if (!report || !report.id) return null;
                    
                    // Safe data extraction with fallbacks
                    const symbol = report.symbol || t('common.notAvailable');
                    const name = report.name || t('common.notAvailable');
                    const sector = report.sector || t('common.unknown');
                    const score = typeof report.score === 'number' ? report.score : 0;
                    
                    // Safe date handling
                    let formattedDate = t('common.notAvailable');
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
                            title={t('reports.actions.viewResults')}
                          >
                            <Visibility />
                          </IconButton>
                          <IconButton
                            onClick={() => handleRunNewAnalysis(symbol)}
                            color="secondary"
                            title={t('reports.actions.runNewAnalysis')}
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
              {searchFilter ? t('reports.noMatchingReports') : t('reports.noReportsFound')}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {searchFilter
                ? t('reports.tryAdjustingSearch')
                : t('reports.getStartedMessage')
              }
            </Typography>
            {!searchFilter && (
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => navigate('/stocks')}
              >
                {t('reports.createFirstReport')}
              </Button>
            )}
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ReportsPage;