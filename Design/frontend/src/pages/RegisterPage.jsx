import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Link,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Chip,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import {
  AccountCircle,
  Email,
  Lock,
  Star,
  CardGiftcard,
  Security
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
// import { useDispatch, useSelector } from 'react-redux'; // Not used currently

const RegisterPage = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  // const dispatch = useDispatch(); // Not used currently
  // const { loading, error } = useSelector((state) => state.auth); // Not used currently
  
  // Mock loading and error states
  const loading = false;
  const error = null;

  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    agreeTerms: false
  });
  const [validationErrors, setValidationErrors] = useState({});

  const handleInputChange = (e) => {
    const { name, value, checked, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
    // Clear validation error when user starts typing
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const errors = {};

    if (!formData.username.trim()) {
      errors.username = t('register.validation.usernameRequired');
    } else if (formData.username.length < 3) {
      errors.username = t('register.validation.usernameMinLength');
    }

    if (!formData.email.trim()) {
      errors.email = t('register.validation.emailRequired');
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = t('register.validation.emailInvalid');
    }

    if (!formData.password) {
      errors.password = t('register.validation.passwordRequired');
    } else if (formData.password.length < 8) {
      errors.password = t('register.validation.passwordMinLength');
    }

    if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = t('register.validation.passwordsNoMatch');
    }

    if (!formData.firstName.trim()) {
      errors.firstName = t('register.validation.firstNameRequired');
    }

    if (!formData.lastName.trim()) {
      errors.lastName = t('register.validation.lastNameRequired');
    }

    if (!formData.agreeTerms) {
      errors.agreeTerms = t('register.validation.agreeTermsRequired');
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      
      // Simulate registration success
      setTimeout(() => {
        navigate('/login', {
          state: {
            message: t('register.registrationSuccess')
          }
        });
      }, 1000);
    } catch {
      setValidationErrors({ general: 'Registration failed. Please try again.' });
    }
  };

  const benefits = [
    {
      icon: <CardGiftcard sx={{ color: 'success.main' }} />,
      title: t('register.benefits.welcomeBonus'),
      description: t('register.benefits.welcomeBonusDesc')
    },
    {
      icon: <Star sx={{ color: 'warning.main' }} />,
      title: t('register.benefits.professionalAnalysis'),
      description: t('register.benefits.professionalAnalysisDesc')
    },
    {
      icon: <Security sx={{ color: 'info.main' }} />,
      title: t('register.benefits.securePlatform'),
      description: t('register.benefits.securePlatformDesc')
    }
  ];

  return (
    <Box sx={{ 
      minHeight: '100vh', 
      backgroundColor: 'background.default',
      py: 4
    }}>
      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="stretch">
          {/* Registration Form */}
          <Grid item xs={12} md={6}>
            <Paper
              elevation={3}
              sx={{
                p: 4,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center'
              }}
            >
              <Box sx={{ textAlign: 'center', mb: 3 }}>
                <Typography
                  variant="h4"
                  component="h1"
                  gutterBottom
                  sx={{ fontWeight: 600, color: 'primary.main' }}
                >
                  {t('register.title')}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {t('register.subtitle')}
                </Typography>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {error}
                </Alert>
              )}

              <Box component="form" onSubmit={handleSubmit}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label={t('register.firstName')}
                      name="firstName"
                      value={formData.firstName}
                      onChange={handleInputChange}
                      error={!!validationErrors.firstName}
                      helperText={validationErrors.firstName}
                      InputProps={{
                        startAdornment: <AccountCircle sx={{ mr: 1, color: 'action.active' }} />
                      }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label={t('register.lastName')}
                      name="lastName"
                      value={formData.lastName}
                      onChange={handleInputChange}
                      error={!!validationErrors.lastName}
                      helperText={validationErrors.lastName}
                    />
                  </Grid>
                </Grid>

                <TextField
                  fullWidth
                  margin="normal"
                  label={t('register.username')}
                  name="username"
                  value={formData.username}
                  onChange={handleInputChange}
                  error={!!validationErrors.username}
                  helperText={validationErrors.username}
                  InputProps={{
                    startAdornment: <AccountCircle sx={{ mr: 1, color: 'action.active' }} />
                  }}
                />

                <TextField
                  fullWidth
                  margin="normal"
                  label={t('register.email')}
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  error={!!validationErrors.email}
                  helperText={validationErrors.email}
                  InputProps={{
                    startAdornment: <Email sx={{ mr: 1, color: 'action.active' }} />
                  }}
                />

                <TextField
                  fullWidth
                  margin="normal"
                  label={t('register.password')}
                  name="password"
                  type="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  error={!!validationErrors.password}
                  helperText={validationErrors.password}
                  InputProps={{
                    startAdornment: <Lock sx={{ mr: 1, color: 'action.active' }} />
                  }}
                />

                <TextField
                  fullWidth
                  margin="normal"
                  label={t('register.confirmPassword')}
                  name="confirmPassword"
                  type="password"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  error={!!validationErrors.confirmPassword}
                  helperText={validationErrors.confirmPassword}
                  InputProps={{
                    startAdornment: <Lock sx={{ mr: 1, color: 'action.active' }} />
                  }}
                />

                <FormControlLabel
                  control={
                    <Checkbox
                      name="agreeTerms"
                      checked={formData.agreeTerms}
                      onChange={handleInputChange}
                      color="primary"
                    />
                  }
                  label={
                    <Typography variant="body2">
                      {t('register.agreeTerms', {
                        termsLink: <Link href="#" color="primary">{t('register.termsOfService')}</Link>,
                        privacyLink: <Link href="#" color="primary">{t('register.privacyPolicy')}</Link>
                      })}
                    </Typography>
                  }
                  sx={{ mt: 2, mb: 1 }}
                />
                {validationErrors.agreeTerms && (
                  <Typography variant="body2" color="error" sx={{ ml: 4 }}>
                    {validationErrors.agreeTerms}
                  </Typography>
                )}

                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  disabled={loading}
                  sx={{ mt: 3, mb: 2, py: 1.5 }}
                >
                  {loading ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    t('register.createAccount')
                  )}
                </Button>

                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    {t('register.alreadyHaveAccount')}{' '}
                    <Link
                      component="button"
                      type="button"
                      onClick={() => navigate('/login')}
                      color="primary"
                      sx={{ textDecoration: 'none' }}
                    >
                      {t('register.signIn')}
                    </Link>
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Grid>

          {/* Benefits Section */}
          <Grid item xs={12} md={6}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography
                variant="h5"
                component="h2"
                gutterBottom
                sx={{ fontWeight: 600, mb: 3, textAlign: 'center' }}
              >
                {t('register.whyJoin')}
              </Typography>

              <Grid container spacing={2} sx={{ mb: 3 }}>
                {benefits.map((benefit, index) => (
                  <Grid item xs={12} key={index}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent sx={{ display: 'flex', alignItems: 'center', p: 2 }}>
                        <Box sx={{ mr: 2 }}>
                          {benefit.icon}
                        </Box>
                        <Box>
                          <Typography variant="h6" component="h3">
                            {benefit.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {benefit.description}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>

              <Card sx={{ backgroundColor: 'primary.main', color: 'white', mt: 'auto' }}>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    {t('register.studentPricing')}
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mb: 2 }}>
                    <Chip
                      label={t('register.pricing.creditLabel')}
                      sx={{ backgroundColor: 'white', color: 'primary.main' }}
                    />
                    <Chip
                      label={t('register.pricing.analysisLabel')}
                      sx={{ backgroundColor: 'white', color: 'primary.main' }}
                    />
                  </Box>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    {t('register.pricingDescription')}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default RegisterPage;