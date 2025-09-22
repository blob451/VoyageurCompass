import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import {
  ShoppingCart,
  CreditCard,
  AccountBalanceWallet,
  Star,
  LocalOffer,
  CheckCircle,
  Payment,
  Security,
  History,
  CardGiftcard
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import { useGetUserProfileQuery } from '../features/api/apiSlice';
// import { useSelector } from 'react-redux'; // Not used currently

const StorePage = () => {
  const { t } = useTranslation();
  // const { user } = useSelector((state) => state.auth); // Not used currently
  const { data: userProfile } = useGetUserProfileQuery();
  const userCredits = userProfile?.credits || 0;
  const [selectedPackage, setSelectedPackage] = useState(null);
  const [purchaseDialog, setPurchaseDialog] = useState(false);
  const [paymentMethod, setPaymentMethod] = useState('credit_card');
  const [processing, setProcessing] = useState(false);
  const [purchaseComplete, setPurchaseComplete] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  // Mock purchase history
  const purchaseHistory = [
    { id: 1, package: 'Professional Pack', credits: 60, amount: 50, date: '2025-01-12', status: 'completed' },
    { id: 2, package: 'Value Pack', credits: 25, amount: 20, date: '2025-01-10', status: 'completed' },
    { id: 3, package: 'Basic Pack', credits: 5, amount: 5, date: '2025-01-08', status: 'completed' }
  ];

  const creditPackages = [
    {
      id: 'single',
      name: 'Single Credit',
      credits: 1,
      price: 1,
      originalPrice: 1,
      discount: 0,
      popular: false,
      description: 'Perfect for trying out the platform'
    },
    {
      id: 'basic',
      name: 'Basic Pack',
      credits: 5,
      price: 5,
      originalPrice: 5,
      discount: 0,
      popular: false,
      description: 'Great for occasional use'
    },
    {
      id: 'value',
      name: 'Value Pack',
      credits: 25,
      price: 20,
      originalPrice: 25,
      discount: 20,
      popular: true,
      description: 'Most popular choice for regular users'
    },
    {
      id: 'professional',
      name: 'Professional Pack',
      credits: 60,
      price: 50,
      originalPrice: 60,
      discount: 17,
      popular: false,
      description: 'Ideal for professional traders'
    },
    {
      id: 'enterprise',
      name: 'Enterprise Pack',
      credits: 100,
      price: 80,
      originalPrice: 100,
      discount: 20,
      popular: false,
      description: 'Perfect for heavy users and institutions'
    }
  ];

  const paymentMethods = [
    { id: 'credit_card', name: t('store.checkout.paymentMethods.creditCard'), icon: <CreditCard /> },
    { id: 'paypal', name: t('store.checkout.paymentMethods.paypal'), icon: <AccountBalanceWallet /> },
    { id: 'bank_transfer', name: t('store.checkout.paymentMethods.bankTransfer'), icon: <Payment /> }
  ];

  const steps = [
    t('store.checkout.steps.selectPackage'),
    t('store.checkout.steps.paymentMethod'),
    t('store.checkout.steps.confirmation')
  ];

  const handleSelectPackage = (pkg) => {
    setSelectedPackage(pkg);
    setPurchaseDialog(true);
    setActiveStep(0);
    setPurchaseComplete(false);
  };

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      handlePurchase();
    } else {
      setActiveStep(activeStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const handlePurchase = async () => {
    setProcessing(true);
    
    try {
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock successful purchase
      setPurchaseComplete(true);
      setProcessing(false);
      
      // Close dialog after showing success
      setTimeout(() => {
        setPurchaseDialog(false);
        setSelectedPackage(null);
        setActiveStep(0);
        setPurchaseComplete(false);
      }, 3000);
      
    } catch (error) {
      setProcessing(false);
      console.error('Purchase failed:', error);
    }
  };

  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              {t('store.checkout.selectedPackage')}
            </Typography>
            {selectedPackage && (
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">{selectedPackage.name}</Typography>
                    {selectedPackage.popular && (
                      <Chip label={t('store.mostPopular')} color="primary" icon={<Star />} />
                    )}
                  </Box>
                  <Typography variant="h4" color="primary.main" gutterBottom>
                    ${selectedPackage.price}
                    {selectedPackage.discount > 0 && (
                      <Typography 
                        component="span" 
                        variant="h6" 
                        color="text.secondary"
                        sx={{ textDecoration: 'line-through', ml: 1 }}
                      >
                        ${selectedPackage.originalPrice}
                      </Typography>
                    )}
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {selectedPackage.credits} {t('store.credits')} • {selectedPackage.description}
                  </Typography>
                  <List dense>
                    {selectedPackage.features.map((feature, index) => (
                      <ListItem key={index} sx={{ px: 0 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircle color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            )}
          </Box>
        );
      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              {t('store.checkout.selectPaymentMethod')}
            </Typography>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>{t('store.checkout.selectPaymentMethod')}</InputLabel>
              <Select
                value={paymentMethod}
                onChange={(e) => setPaymentMethod(e.target.value)}
                label={t('store.checkout.selectPaymentMethod')}
              >
                {paymentMethods.map((method) => (
                  <MenuItem key={method.id} value={method.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {method.icon}
                      {method.name}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            {paymentMethod === 'credit_card' && (
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label={t('store.checkout.cardNumber')}
                    placeholder="1234 5678 9012 3456"
                    InputProps={{
                      startAdornment: <CreditCard sx={{ mr: 1, color: 'action.active' }} />
                    }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label={t('store.checkout.expiryDate')}
                    placeholder="MM/YY"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label={t('store.checkout.cvv')}
                    placeholder="123"
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label={t('store.checkout.cardholderName')}
                    placeholder="John Doe"
                  />
                </Grid>
              </Grid>
            )}
            
            <Alert severity="info" sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Security />
                {t('store.checkout.securityNotice')}
              </Box>
            </Alert>
          </Box>
        );
      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              {t('store.checkout.orderConfirmation')}
            </Typography>
            <Card>
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={8}>
                    <Typography variant="body1">{selectedPackage?.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {selectedPackage?.credits} {t('store.credits')}
                    </Typography>
                  </Grid>
                  <Grid item xs={4} sx={{ textAlign: 'right' }}>
                    <Typography variant="h6">${selectedPackage?.price}</Typography>
                  </Grid>
                </Grid>
                <Divider sx={{ my: 2 }} />
                <Grid container spacing={2}>
                  <Grid item xs={8}>
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>{t('store.checkout.total')}</Typography>
                  </Grid>
                  <Grid item xs={4} sx={{ textAlign: 'right' }}>
                    <Typography variant="h6" color="primary.main">
                      ${selectedPackage?.price}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
            
            <Alert severity="success" sx={{ mt: 2 }}>
              {t('store.checkout.creditsAddedNotice')}
            </Alert>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          {t('store.title')}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          {t('store.subtitle')}
        </Typography>
      </Box>

      {/* Current Balance */}
      <Card sx={{ mb: 4, backgroundColor: 'primary.main', color: 'white' }}>
        <CardContent sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">{t('store.currentBalance')}</Typography>
            <Typography variant="h3">{userCredits} {t('store.credits')}</Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              {t('store.creditEquation')}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <AccountBalanceWallet sx={{ fontSize: 60, opacity: 0.7 }} />
          </Box>
        </CardContent>
      </Card>

      {/* Credit Packages */}
      <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        {t('store.choosePackage')}
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {creditPackages.map((pkg) => (
          <Grid item xs={12} sm={6} md={4} key={pkg.id}>
            <Card 
              sx={{ 
                height: '100%',
                position: 'relative',
                border: pkg.popular ? '2px solid' : '1px solid',
                borderColor: pkg.popular ? 'primary.main' : 'divider',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
                }
              }}
            >
              {pkg.popular && (
                <Chip
                  label={t('store.mostPopular')}
                  color="primary"
                  icon={<Star />}
                  sx={{
                    position: 'absolute',
                    top: 8,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 1
                  }}
                />
              )}
              
              <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  {pkg.name}
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h3" color="primary.main" sx={{ fontWeight: 700 }}>
                    ${pkg.price}
                    {pkg.discount > 0 && (
                      <Typography 
                        component="span" 
                        variant="h5" 
                        color="text.secondary"
                        sx={{ textDecoration: 'line-through', ml: 1 }}
                      >
                        ${pkg.originalPrice}
                      </Typography>
                    )}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {pkg.credits} {t('store.credits')}
                  </Typography>
                  {pkg.discount > 0 && (
                    <Chip
                      label={t('store.save', { percent: pkg.discount })}
                      color="success"
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                  {pkg.description}
                </Typography>

                <Button
                  variant={pkg.popular ? "contained" : "outlined"}
                  size="large"
                  fullWidth
                  onClick={() => handleSelectPackage(pkg)}
                  startIcon={<ShoppingCart />}
                  sx={{ mt: 2 }}
                >
                  {t('store.purchaseNow')}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Purchase History */}
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <History sx={{ mr: 1 }} />
          <Typography variant="h6">{t('store.purchaseHistory')}</Typography>
        </Box>
        
        {purchaseHistory.length > 0 ? (
          <List>
            {purchaseHistory.map((purchase, index) => (
              <React.Fragment key={purchase.id}>
                <ListItem sx={{ px: 0 }}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body1" sx={{ fontWeight: 500 }}>
                          {purchase.package}
                        </Typography>
                        <Typography variant="h6" color="primary.main">
                          ${purchase.amount}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          {purchase.credits} {t('store.credits')} • {purchase.date}
                        </Typography>
                        <Chip
                          label={purchase.status}
                          size="small"
                          color="success"
                        />
                      </Box>
                    }
                    primaryTypographyProps={{ component: 'div' }}
                    secondaryTypographyProps={{ component: 'div' }}
                  />
                </ListItem>
                {index < purchaseHistory.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body1" color="text.secondary">
              {t('store.noPurchaseHistory')}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {t('store.purchaseFirstPackage')}
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Purchase Dialog */}
      <Dialog 
        open={purchaseDialog} 
        onClose={() => setPurchaseDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {purchaseComplete ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CheckCircle color="success" />
              {t('store.purchaseComplete')}
            </Box>
          ) : (
            t('store.completePurchase')
          )}
        </DialogTitle>
        
        <DialogContent>
          {purchaseComplete ? (
            <Box sx={{ textAlign: 'center', py: 3 }}>
              <CheckCircle color="success" sx={{ fontSize: 60, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                {t('store.thankYou')}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                {t('store.creditsAdded', { credits: selectedPackage?.credits })}
              </Typography>
            </Box>
          ) : (
            <>
              <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
                {steps.map((label) => (
                  <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>
              
              {getStepContent(activeStep)}
            </>
          )}
        </DialogContent>
        
        {!purchaseComplete && (
          <DialogActions>
            <Button onClick={() => setPurchaseDialog(false)}>
              {t('store.cancel')}
            </Button>
            {activeStep > 0 && (
              <Button onClick={handleBack}>
                {t('store.back')}
              </Button>
            )}
            <Button 
              onClick={handleNext}
              variant="contained"
              disabled={processing}
            >
              {processing ? (
                <CircularProgress size={24} />
              ) : activeStep === steps.length - 1 ? (
                t('store.completePurchaseButton')
              ) : (
                t('store.next')
              )}
            </Button>
          </DialogActions>
        )}
      </Dialog>
    </Container>
  );
};

export default StorePage;