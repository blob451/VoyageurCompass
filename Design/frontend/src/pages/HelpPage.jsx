import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
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
  const { t } = useTranslation();
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
    alert(t('help.contactForm.success'));
    setContactForm({ name: '', email: '', subject: '', message: '' });
  };

  const features = [
    {
      icon: <Analytics color="primary" />,
      title: t('help.features.technicalAnalysis.title'),
      description: t('help.features.technicalAnalysis.description'),
      details: t('help.features.technicalAnalysis.details', { returnObjects: true })
    },
    {
      icon: <Compare color="primary" />,
      title: t('help.features.stockComparison.title'),
      description: t('help.features.stockComparison.description'),
      details: t('help.features.stockComparison.details', { returnObjects: true })
    },
    {
      icon: <AccountBalance color="primary" />,
      title: t('help.features.sectorAnalysis.title'),
      description: t('help.features.sectorAnalysis.description'),
      details: t('help.features.sectorAnalysis.details', { returnObjects: true })
    },
    {
      icon: <CreditCard color="primary" />,
      title: t('help.features.creditSystem.title'),
      description: t('help.features.creditSystem.description'),
      details: t('help.features.creditSystem.details', { returnObjects: true })
    }
  ];

  const technicalIndicators = t('help.technicalGuide.indicators', { returnObjects: true });

  const faqs = t('help.faq.items', { returnObjects: true }).map((faq, index) => ({
    id: `faq-${index}`,
    question: faq.question,
    answer: faq.answer
  }));

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
          {t('help.technicalGuide.description')}
        </Typography>
      </Alert>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>{t('help.technicalGuide.tableHeaders.indicator')}</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>{t('help.technicalGuide.tableHeaders.description')}</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>{t('help.technicalGuide.tableHeaders.range')}</TableCell>
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
{t('help.technicalGuide.scoreInterpretation.title')}
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'success.main', borderRadius: 2 }}>
              <Typography variant="h5" color="success.main">8-10</Typography>
              <Typography variant="body2">{t('recommendations.strongBuy')}</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'success.light', borderRadius: 2 }}>
              <Typography variant="h5" color="success.light">6-7</Typography>
              <Typography variant="body2">{t('recommendations.buy')}</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'warning.main', borderRadius: 2 }}>
              <Typography variant="h5" color="warning.main">4-5</Typography>
              <Typography variant="body2">{t('recommendations.hold')}</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'error.main', borderRadius: 2 }}>
              <Typography variant="h5" color="error.main">0-3</Typography>
              <Typography variant="body2">{t('recommendations.sell')}</Typography>
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
{t('help.contactForm.contactInfo')}
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
          {t('help.title')}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          {t('help.subtitle')}
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Analytics sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6">{t('help.stats.indicators.title')}</Typography>
              <Typography variant="body2" color="text.secondary">
                {t('help.stats.indicators.subtitle')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h6">{t('help.stats.analysisTime.title')}</Typography>
              <Typography variant="body2" color="text.secondary">
                {t('help.stats.analysisTime.subtitle')}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CreditCard sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
              <Typography variant="h6">{t('help.stats.pricing.title')}</Typography>
              <Typography variant="body2" color="text.secondary">
                {t('help.stats.pricing.subtitle')}
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
          <Tab label={t('help.tabs.userGuide')} icon={<Help />} />
          <Tab label={t('help.tabs.technical')} icon={<Analytics />} />
          <Tab label={t('help.tabs.faq')} icon={<QuestionAnswer />} />
          <Tab label={t('help.tabs.contact')} icon={<Email />} />
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
          <strong>{t('help.disclaimer.title')}</strong> {t('help.disclaimer.content')}
        </Typography>
      </Paper>
    </Container>
  );
};

export default HelpPage;