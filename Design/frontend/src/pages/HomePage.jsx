import React from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip
} from '@mui/material';
import {
  TrendingUp,
  Analytics,
  Compare,
  AccountBalance,
  CheckCircle,
  Star,
  Speed,
  Security
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';

const HomePage = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useSelector((state) => state.auth);
  const { t } = useTranslation();

  const features = [
    {
      icon: <Analytics sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t('home.features.technicalAnalysis.title'),
      description: t('home.features.technicalAnalysis.description'),
      highlight: t('home.features.technicalAnalysis.highlight')
    },
    {
      icon: <Compare sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t('home.features.stockComparison.title'),
      description: t('home.features.stockComparison.description'),
      highlight: t('home.features.stockComparison.highlight')
    },
    {
      icon: <TrendingUp sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t('home.features.sectorAnalysis.title'),
      description: t('home.features.sectorAnalysis.description'),
      highlight: t('home.features.sectorAnalysis.highlight')
    },
    {
      icon: <AccountBalance sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t('home.features.creditSystem.title'),
      description: t('home.features.creditSystem.description'),
      highlight: t('home.features.creditSystem.highlight')
    }
  ];

  const benefits = t('home.benefits.items', { returnObjects: true });

  const handleGetStarted = () => {
    if (isAuthenticated) {
      navigate('/dashboard');
    } else {
      navigate('/register');
    }
  };

  const handleLogin = () => {
    navigate('/login');
  };

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: 'background.default' }}>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
          color: 'white',
          pt: 8,
          pb: 12,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center" minHeight="60vh">
            <Grid item xs={12} md={6}>
              <Typography
                variant="h2"
                component="h1"
                gutterBottom
                sx={{
                  fontWeight: 700,
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  mb: 3
                }}
              >
                {t('home.hero.title')}
              </Typography>
              <Typography
                variant="h5"
                component="h2"
                gutterBottom
                sx={{
                  fontWeight: 400,
                  opacity: 0.9,
                  mb: 4,
                  lineHeight: 1.4
                }}
              >
                {t('home.hero.subtitle')}
              </Typography>
              <Typography
                variant="h6"
                component="p"
                sx={{
                  opacity: 0.8,
                  mb: 4,
                  fontSize: '1.1rem',
                  lineHeight: 1.6
                }}
              >
                {t('home.hero.description')}
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleGetStarted}
                  sx={{
                    backgroundColor: 'white',
                    color: 'primary.main',
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem',
                    '&:hover': {
                      backgroundColor: 'grey.100',
                    }
                  }}
                >
                  {isAuthenticated ? t('home.hero.goToDashboard') : t('home.hero.getStarted')}
                </Button>
                {!isAuthenticated && (
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={handleLogin}
                    sx={{
                      borderColor: 'white',
                      color: 'white',
                      px: 4,
                      py: 1.5,
                      fontSize: '1.1rem',
                      '&:hover': {
                        borderColor: 'white',
                        backgroundColor: 'rgba(255,255,255,0.1)',
                      }
                    }}
                  >
                    {t('home.hero.login')}
                  </Button>
                )}
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Paper
                  elevation={8}
                  sx={{
                    p: 3,
                    backgroundColor: 'rgba(255,255,255,0.95)',
                    color: 'text.primary',
                    borderRadius: 3,
                    backdropFilter: 'blur(10px)',
                  }}
                >
                  <Typography variant="h6" gutterBottom color="primary.main">
                    {t('home.sampleAnalysis.title')}
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {t('home.sampleAnalysis.technicalScore')}: <strong>7/10</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {t('home.sampleAnalysis.company')}
                  </Typography>
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-around' }}>
                    <Box textAlign="center">
                      <Typography variant="body2" color="success.main">{t('home.sampleAnalysis.smaTrend')}</Typography>
                      <Typography variant="h6">{t('home.sampleAnalysis.strong')}</Typography>
                    </Box>
                    <Box textAlign="center">
                      <Typography variant="body2" color="info.main">{t('home.sampleAnalysis.rsi')}</Typography>
                      <Typography variant="h6">59.8</Typography>
                    </Box>
                    <Box textAlign="center">
                      <Typography variant="body2" color="warning.main">{t('home.sampleAnalysis.macd')}</Typography>
                      <Typography variant="h6">{t('home.sampleAnalysis.bullish')}</Typography>
                    </Box>
                  </Box>
                </Paper>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Typography
          variant="h3"
          component="h2"
          textAlign="center"
          gutterBottom
          sx={{ mb: 6, fontWeight: 600 }}
        >
          {t('home.features.title')}
        </Typography>
        
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card 
                sx={{ 
                  height: '100%',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
                  }
                }}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Box sx={{ mb: 2 }}>
                    {feature.icon}
                    <Chip 
                      label={feature.highlight} 
                      size="small" 
                      color="primary" 
                      sx={{ ml: 1, mb: 2 }}
                    />
                  </Box>
                  <Typography variant="h6" component="h3" gutterBottom>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Benefits Section */}
      <Box sx={{ backgroundColor: 'grey.50', py: 8 }}>
        <Container maxWidth="lg">
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography
                variant="h4"
                component="h2"
                gutterBottom
                sx={{ fontWeight: 600, mb: 3 }}
              >
                {t('home.benefits.title')}
              </Typography>
              <List>
                {benefits.map((benefit, index) => (
                  <ListItem key={index} sx={{ px: 0 }}>
                    <ListItemIcon>
                      <CheckCircle color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={benefit}
                      primaryTypographyProps={{ fontSize: '1.1rem' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>
            <Grid item xs={12} md={6}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card sx={{ textAlign: 'center', p: 2 }}>
                    <Speed sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                    <Typography variant="h6">{t('home.benefits.fastAnalysis.title')}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {t('home.benefits.fastAnalysis.description')}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card sx={{ textAlign: 'center', p: 2 }}>
                    <Security sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                    <Typography variant="h6">{t('home.benefits.securePlatform.title')}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {t('home.benefits.securePlatform.description')}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={12}>
                  <Card sx={{ textAlign: 'center', p: 2, backgroundColor: 'primary.main', color: 'white' }}>
                    <Star sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h6">{t('home.benefits.creditSystemCard.title')}</Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      {t('home.benefits.creditSystemCard.description')}
                    </Typography>
                  </Card>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Call to Action */}
      <Box sx={{ py: 8, textAlign: 'center' }}>
        <Container maxWidth="md">
          <Typography
            variant="h4"
            component="h2"
            gutterBottom
            sx={{ fontWeight: 600, mb: 3 }}
          >
            {t('home.callToAction.title')}
          </Typography>
          <Typography
            variant="h6"
            color="text.secondary"
            gutterBottom
            sx={{ mb: 4 }}
          >
            {t('home.callToAction.subtitle')}
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="large"
              onClick={handleGetStarted}
              sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
            >
              {isAuthenticated ? t('home.hero.goToDashboard') : t('home.hero.createFreeAccount')}
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={() => navigate('/help')}
              sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
            >
              {t('home.hero.learnMore')}
            </Button>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default HomePage;