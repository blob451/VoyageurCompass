import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Button,
  TextField,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab
} from '@mui/material';
import {
  ExpandMore,
  Help,
  Analytics,
  School,
  QuestionAnswer,
  Email,
  Phone,
  CheckCircle,
  TrendingUp,
  Compare,
  AccountBalance,
  Star,
  Info,
  Speed,
  Security,
  CreditCard
} from '@mui/icons-material';

const HelpPage = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [expandedFaq, setExpandedFaq] = useState(false);
  const [contactForm, setContactForm] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleFaqChange = (panel) => (event, isExpanded) => {
    setExpandedFaq(isExpanded ? panel : false);
  };

  const handleContactChange = (field) => (event) => {
    setContactForm(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  const handleContactSubmit = (event) => {
    event.preventDefault();
    // Mock form submission
    console.log('Contact form submitted:', contactForm);
    alert('Thank you for your message! We\'ll get back to you soon.');
    setContactForm({ name: '', email: '', subject: '', message: '' });
  };

  const features = [
    {
      icon: <Analytics color="primary" />,
      title: 'Technical Analysis',
      description: '12 comprehensive indicators providing detailed stock analysis',
      details: [
        'SMA50 vs SMA200 trend analysis',
        'Price vs 50-day moving average',
        'RSI (14-period) momentum indicator',
        'MACD signal and histogram',
        'Bollinger Bands position and width',
        'Volume surge detection',
        'On-Balance Volume trend',
        'Relative strength vs sector/market',
        'Candlestick pattern recognition',
        'Support/resistance context'
      ]
    },
    {
      icon: <Compare color="primary" />,
      title: 'Stock Comparison',
      description: 'Side-by-side analysis of multiple stocks',
      details: [
        'Compare up to 5 stocks simultaneously',
        'Visual metric comparison tables',
        'Export results to CSV',
        'Performance ranking',
        'Sector-specific comparisons'
      ]
    },
    {
      icon: <AccountBalance color="primary" />,
      title: 'Sector Analysis',
      description: 'Market sector performance and trends',
      details: [
        'Industry performance tracking',
        'Sector momentum analysis',
        'Market cap and volume metrics',
        'Top performing stocks by sector',
        'Trend identification'
      ]
    },
    {
      icon: <CreditCard color="primary" />,
      title: 'Credit System',
      description: 'Pay-per-analysis affordable pricing',
      details: [
        '1 Credit = $1 = 1 Stock Analysis',
        'No monthly subscriptions',
        'Credits never expire',
        'Student-friendly pricing',
        'Bulk purchase discounts'
      ]
    }
  ];

  const technicalIndicators = [
    { name: 'SMA50 vs SMA200', description: 'Trend direction based on moving average crossover', range: '0-10' },
    { name: 'Price vs 50MA', description: 'Current price relative to 50-day moving average', range: '0-10' },
    { name: 'RSI (14)', description: 'Relative Strength Index momentum oscillator', range: '0-100' },
    { name: 'MACD', description: 'Moving Average Convergence Divergence', range: 'Signal/Histogram' },
    { name: 'Bollinger Position', description: 'Position within Bollinger Bands', range: '0-10' },
    { name: 'Volume Surge', description: 'Trading volume compared to average', range: '0-10' },
    { name: 'OBV Trend', description: 'On-Balance Volume trend analysis', range: '0-10' },
    { name: 'Relative 1Y/2Y', description: 'Performance vs sector benchmark', range: 'Percentage' }
  ];

  const faqs = [
    {
      id: 'what-is-voyageur',
      question: 'What is VoyageurCompass?',
      answer: 'VoyageurCompass is a comprehensive financial analytics platform that provides professional-grade stock analysis using 12 technical indicators. Built for students and professionals, it offers affordable pay-per-analysis pricing without monthly subscriptions.'
    },
    {
      id: 'how-credits-work',
      question: 'How does the credit system work?',
      answer: 'Our credit system is simple: 1 Credit = $1 = 1 Stock Analysis. You only pay for what you use. Credits never expire, and you can purchase them in various packages with bulk discounts. No monthly fees or subscriptions required.'
    },
    {
      id: 'technical-indicators',
      question: 'What technical indicators do you provide?',
      answer: 'We provide 12 comprehensive technical indicators including SMA trends, RSI, MACD, Bollinger Bands, volume analysis, OBV, relative performance, and candlestick patterns. Each analysis receives a score from 0-10 based on these indicators.'
    },
    {
      id: 'data-sources',
      question: 'Where do you get your data?',
      answer: 'We use Yahoo Finance API for real-time and historical stock data. Our data is updated regularly and stored in a secure PostgreSQL database with Redis caching for fast access.'
    },
    {
      id: 'comparison-tool',
      question: 'How does stock comparison work?',
      answer: 'You can compare 2-5 stocks side-by-side with detailed metrics. Each stock in the comparison costs 1 credit. Results include visual charts, performance rankings, and exportable CSV reports.'
    },
    {
      id: 'sector-analysis',
      question: 'What is sector analysis?',
      answer: 'Sector analysis compares different market sectors (Technology, Healthcare, Finance, etc.) showing performance trends, momentum, volatility, and top performers. You can analyze 2+ sectors for credits (2 sectors = 1 credit).'
    },
    {
      id: 'school-project',
      question: 'Is this a real trading platform?',
      answer: 'VoyageurCompass is an educational project built for a university course. While it uses real market data and professional analysis techniques, it\'s designed for learning and demonstration purposes.'
    },
    {
      id: 'supported-stocks',
      question: 'Which stocks can I analyze?',
      answer: 'You can analyze any publicly traded stock with a valid ticker symbol. Our system supports major exchanges including NYSE, NASDAQ, and others through Yahoo Finance integration.'
    },
    {
      id: 'analysis-time',
      question: 'How long does analysis take?',
      answer: 'Stock analysis typically takes 2-5 seconds. The system fetches real-time data, runs 12 technical indicators, and calculates a comprehensive score. Results are displayed immediately.'
    },
    {
      id: 'mobile-access',
      question: 'Can I use this on mobile devices?',
      answer: 'Yes! VoyageurCompass is built with responsive design using Material-UI, making it fully functional on desktop, tablet, and mobile devices.'
    }
  ];

  const renderUserGuide = () => (
    <Grid container spacing={3}>
      {features.map((feature, index) => (
        <Grid item xs={12} md={6} key={index}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                {feature.icon}
                <Typography variant="h6" sx={{ ml: 1 }}>
                  {feature.title}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {feature.description}
              </Typography>
              <List dense>
                {feature.details.map((detail, idx) => (
                  <ListItem key={idx} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <CheckCircle color="success" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={detail}
                      primaryTypographyProps={{ fontSize: '0.9rem' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderTechnicalGuide = () => (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          Our technical analysis engine uses 12 professional indicators to generate a comprehensive score (0-10) for each stock.
          Higher scores indicate stronger technical performance.
        </Typography>
      </Alert>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Indicator</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Range/Scale</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {technicalIndicators.map((indicator, index) => (
              <TableRow key={index}>
                <TableCell sx={{ fontWeight: 500 }}>
                  {indicator.name}
                </TableCell>
                <TableCell>
                  {indicator.description}
                </TableCell>
                <TableCell>
                  <Chip label={indicator.range} size="small" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Score Interpretation
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'success.main', borderRadius: 2 }}>
              <Typography variant="h5" color="success.main">8-10</Typography>
              <Typography variant="body2">Strong Buy</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'success.light', borderRadius: 2 }}>
              <Typography variant="h5" color="success.light">6-7</Typography>
              <Typography variant="body2">Buy</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'warning.main', borderRadius: 2 }}>
              <Typography variant="h5" color="warning.main">4-5</Typography>
              <Typography variant="body2">Hold</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'error.main', borderRadius: 2 }}>
              <Typography variant="h5" color="error.main">0-3</Typography>
              <Typography variant="body2">Sell</Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );

  const renderFAQ = () => (
    <Box>
      {faqs.map((faq) => (
        <Accordion
          key={faq.id}
          expanded={expandedFaq === faq.id}
          onChange={handleFaqChange(faq.id)}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6" sx={{ fontWeight: 500 }}>
              {faq.question}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body1" color="text.secondary">
              {faq.answer}
            </Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );

  const renderContact = () => (
    <Grid container spacing={4}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Contact Information
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                <Email color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Email Support"
                secondary="support@voyageurcompass.edu"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <School color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Academic Project"
                secondary="University of Louisville - CM 3070"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <QuestionAnswer color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Response Time"
                secondary="Usually within 24 hours"
              />
            </ListItem>
          </List>

          <Alert severity="info" sx={{ mt: 2 }}>
            This is an educational project. For real financial advice, please consult a qualified financial advisor.
          </Alert>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Send us a Message
          </Typography>
          
          <Box component="form" onSubmit={handleContactSubmit}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Name"
                  value={contactForm.name}
                  onChange={handleContactChange('name')}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Email"
                  type="email"
                  value={contactForm.email}
                  onChange={handleContactChange('email')}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Subject"
                  value={contactForm.subject}
                  onChange={handleContactChange('subject')}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Message"
                  multiline
                  rows={4}
                  value={contactForm.message}
                  onChange={handleContactChange('message')}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  startIcon={<Email />}
                >
                  Send Message
                </Button>
              </Grid>
            </Grid>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Help & Documentation
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Everything you need to know about using VoyageurCompass
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Analytics sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6">12 Indicators</Typography>
              <Typography variant="body2" color="text.secondary">
                Professional analysis
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h6">2-5 Seconds</Typography>
              <Typography variant="body2" color="text.secondary">
                Analysis time
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CreditCard sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
              <Typography variant="h6">$1 = 1 Analysis</Typography>
              <Typography variant="body2" color="text.secondary">
                Affordable pricing
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Security sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h6">Real-time Data</Typography>
              <Typography variant="body2" color="text.secondary">
                Yahoo Finance API
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="User Guide" icon={<Help />} />
          <Tab label="Technical Indicators" icon={<Analytics />} />
          <Tab label="FAQ" icon={<QuestionAnswer />} />
          <Tab label="Contact" icon={<Email />} />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box>
        {activeTab === 0 && renderUserGuide()}
        {activeTab === 1 && renderTechnicalGuide()}
        {activeTab === 2 && renderFAQ()}
        {activeTab === 3 && renderContact()}
      </Box>

      {/* Footer Note */}
      <Paper sx={{ p: 3, mt: 4, backgroundColor: 'grey.50' }}>
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
          <strong>Educational Disclaimer:</strong> VoyageurCompass is a university project created for 
          educational purposes. This platform demonstrates technical analysis concepts and should not 
          be used for actual trading decisions. Always consult with qualified financial advisors 
          for investment guidance.
        </Typography>
      </Paper>
    </Container>
  );
};

export default HelpPage;