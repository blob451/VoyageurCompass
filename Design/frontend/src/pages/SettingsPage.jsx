import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
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
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Tabs,
  Tab
} from '@mui/material';
import {
  Person,
  Security,
  Notifications,
  Palette,
  Email,
  Lock,
  Delete,
  Save,
  Edit,
  Visibility,
  VisibilityOff,
  Warning,
  Check
} from '@mui/icons-material';
import { useSelector } from 'react-redux';

const SettingsPage = () => {
  const { user } = useSelector((state) => state.auth);
  const { t, i18n } = useTranslation('common');
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [saveDialog, setSaveDialog] = useState(false);
  
  // Profile settings
  const [profileData, setProfileData] = useState({
    firstName: user?.first_name || 'John',
    lastName: user?.last_name || 'Doe',
    email: user?.email || 'john.doe@example.com',
    username: user?.username || 'johndoe',
    bio: 'Financial analyst and investor',
    location: 'New York, NY',
    company: 'Investment Corp'
  });

  // Security settings
  const [securityData, setSecurityData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
    twoFactorEnabled: false,
    loginNotifications: true
  });

  // Notification settings
  const [notificationData, setNotificationData] = useState({
    emailNotifications: true,
    analysisComplete: true,
    weeklyReport: false,
    marketAlerts: true,
    creditLowAlert: true,
    promotionalEmails: false
  });

  // Preference settings
  const [preferenceData, setPreferenceData] = useState({
    theme: 'light',
    language: 'en',
    timezone: 'America/New_York',
    defaultAnalysisView: 'detailed',
    autoRefresh: true,
    soundEffects: false
  });

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleProfileChange = (field) => (event) => {
    setProfileData(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  const handleSecurityChange = (field) => (event) => {
    setSecurityData(prev => ({
      ...prev,
      [field]: event.target.type === 'checkbox' ? event.target.checked : event.target.value
    }));
  };

  const handleNotificationChange = (field) => (event) => {
    setNotificationData(prev => ({
      ...prev,
      [field]: event.target.checked
    }));
  };

  const handlePreferenceChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setPreferenceData(prev => ({ ...prev, [field]: value }));

    // Handle language change immediately
    if (field === 'language') {
      i18n.changeLanguage(value);
    }
  };

  const handleSaveSettings = async () => {
    setLoading(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setSaveDialog(true);
      
      // Close success dialog after 2 seconds
      setTimeout(() => {
        setSaveDialog(false);
      }, 2000);
      
    } catch {
      // Error handling could be added here if needed
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    setLoading(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // In real app, would redirect to login or show confirmation
      
    } catch {
      // Error handling could be added here if needed
    } finally {
      setLoading(false);
      setDeleteDialog(false);
    }
  };

  const validatePasswordChange = () => {
    if (!securityData.currentPassword) return false;
    if (securityData.newPassword.length < 8) return false;
    if (securityData.newPassword !== securityData.confirmPassword) return false;
    return true;
  };

  const renderProfileTab = () => (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Person />
        {t('settings.profile.personalInfo')}
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label={t('settings.profile.firstName')}
            value={profileData.firstName}
            onChange={handleProfileChange('firstName')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label={t('settings.profile.lastName')}
            value={profileData.lastName}
            onChange={handleProfileChange('lastName')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label={t('settings.profile.username')}
            value={profileData.username}
            onChange={handleProfileChange('username')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label={t('settings.profile.email')}
            type="email"
            value={profileData.email}
            onChange={handleProfileChange('email')}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Bio"
            multiline
            rows={3}
            value={profileData.bio}
            onChange={handleProfileChange('bio')}
            placeholder="Tell us about yourself..."
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Location"
            value={profileData.location}
            onChange={handleProfileChange('location')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Company"
            value={profileData.company}
            onChange={handleProfileChange('company')}
          />
        </Grid>
      </Grid>
    </Paper>
  );

  const renderSecurityTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Lock />
            {t('settings.security.changePassword')}
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                type={showPassword ? 'text' : 'password'}
                label={t('settings.security.currentPassword')}
                value={securityData.currentPassword}
                onChange={handleSecurityChange('currentPassword')}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="password"
                label={t('settings.security.newPassword')}
                value={securityData.newPassword}
                onChange={handleSecurityChange('newPassword')}
                helperText="Minimum 8 characters"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="password"
                label={t('settings.security.confirmPassword')}
                value={securityData.confirmPassword}
                onChange={handleSecurityChange('confirmPassword')}
                error={securityData.newPassword !== securityData.confirmPassword && securityData.confirmPassword !== ''}
                helperText={securityData.newPassword !== securityData.confirmPassword && securityData.confirmPassword !== '' ? 'Passwords do not match' : ''}
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="outlined"
                disabled={!validatePasswordChange()}
                onClick={() => {
                  // Handle password change
                }}
              >
                Update Password
              </Button>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
      
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Security />
            {t('settings.security.title')}
          </Typography>
          
          <List>
            <ListItem>
              <ListItemText
                primary={t('settings.security.twoFactor')}
                secondary={t('settings.security.twoFactorDesc')}
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={securityData.twoFactorEnabled}
                  onChange={handleSecurityChange('twoFactorEnabled')}
                />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary={t('settings.security.loginNotifications')}
                secondary={t('settings.security.loginNotificationsDesc')}
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={securityData.loginNotifications}
                  onChange={handleSecurityChange('loginNotifications')}
                />
              </ListItemSecondaryAction>
            </ListItem>
          </List>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderNotificationsTab = () => (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Notifications />
        {t('settings.notifications.title')}
      </Typography>
      
      <List>
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.email')}
            secondary={t('settings.notifications.emailDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.emailNotifications}
              onChange={handleNotificationChange('emailNotifications')}
            />
          </ListItemSecondaryAction>
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.analysisComplete')}
            secondary={t('settings.notifications.analysisCompleteDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.analysisComplete}
              onChange={handleNotificationChange('analysisComplete')}
              disabled={!notificationData.emailNotifications}
            />
          </ListItemSecondaryAction>
        </ListItem>
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.weeklyReport')}
            secondary={t('settings.notifications.weeklyReportDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.weeklyReport}
              onChange={handleNotificationChange('weeklyReport')}
              disabled={!notificationData.emailNotifications}
            />
          </ListItemSecondaryAction>
        </ListItem>
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.marketAlerts')}
            secondary={t('settings.notifications.marketAlertsDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.marketAlerts}
              onChange={handleNotificationChange('marketAlerts')}
              disabled={!notificationData.emailNotifications}
            />
          </ListItemSecondaryAction>
        </ListItem>
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.creditLowAlert')}
            secondary={t('settings.notifications.creditLowAlertDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.creditLowAlert}
              onChange={handleNotificationChange('creditLowAlert')}
              disabled={!notificationData.emailNotifications}
            />
          </ListItemSecondaryAction>
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText
            primary={t('settings.notifications.promotionalEmails')}
            secondary={t('settings.notifications.promotionalEmailsDesc')}
          />
          <ListItemSecondaryAction>
            <Switch
              checked={notificationData.promotionalEmails}
              onChange={handleNotificationChange('promotionalEmails')}
              disabled={!notificationData.emailNotifications}
            />
          </ListItemSecondaryAction>
        </ListItem>
      </List>
    </Paper>
  );

  const renderPreferencesTab = () => (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Palette />
        {t('settings.preferences.title')}
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>{t('settings.preferences.theme')}</InputLabel>
            <Select
              value={preferenceData.theme}
              onChange={handlePreferenceChange('theme')}
              label={t('settings.preferences.theme')}
            >
              <MenuItem value="light">{t('settings.preferences.themeLight')}</MenuItem>
              <MenuItem value="dark">{t('settings.preferences.themeDark')}</MenuItem>
              <MenuItem value="auto">{t('settings.preferences.themeAuto')}</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>{t('settings.preferences.language')}</InputLabel>
            <Select
              value={preferenceData.language}
              onChange={handlePreferenceChange('language')}
              label={t('settings.preferences.language')}
            >
              <MenuItem value="en">{t('languages.en')}</MenuItem>
              <MenuItem value="fr">{t('languages.fr')}</MenuItem>
              <MenuItem value="es">{t('languages.es')}</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>{t('settings.preferences.timezone')}</InputLabel>
            <Select
              value={preferenceData.timezone}
              onChange={handlePreferenceChange('timezone')}
              label={t('settings.preferences.timezone')}
            >
              <MenuItem value="America/New_York">{t('settings.preferences.timezoneEastern')}</MenuItem>
              <MenuItem value="America/Chicago">{t('settings.preferences.timezoneCentral')}</MenuItem>
              <MenuItem value="America/Denver">{t('settings.preferences.timezoneMountain')}</MenuItem>
              <MenuItem value="America/Los_Angeles">{t('settings.preferences.timezonePacific')}</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>{t('settings.preferences.defaultAnalysisView')}</InputLabel>
            <Select
              value={preferenceData.defaultAnalysisView}
              onChange={handlePreferenceChange('defaultAnalysisView')}
              label={t('settings.preferences.defaultAnalysisView')}
            >
              <MenuItem value="summary">{t('settings.preferences.viewSummary')}</MenuItem>
              <MenuItem value="detailed">{t('settings.preferences.viewDetailed')}</MenuItem>
              <MenuItem value="charts">{t('settings.preferences.viewCharts')}</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12}>
          <List>
            <ListItem>
              <ListItemText
                primary={t('settings.preferences.autoRefresh')}
                secondary={t('settings.preferences.autoRefreshDesc')}
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={preferenceData.autoRefresh}
                  onChange={handlePreferenceChange('autoRefresh')}
                />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary={t('settings.preferences.soundEffects')}
                secondary={t('settings.preferences.soundEffectsDesc')}
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={preferenceData.soundEffects}
                  onChange={handlePreferenceChange('soundEffects')}
                />
              </ListItemSecondaryAction>
            </ListItem>
          </List>
        </Grid>
      </Grid>
    </Paper>
  );

  const renderDangerZone = () => (
    <Paper sx={{ p: 3, border: '1px solid', borderColor: 'error.main' }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'error.main' }}>
        <Warning />
        Danger Zone
      </Typography>
      
      <Alert severity="warning" sx={{ mb: 2 }}>
        These actions cannot be undone. Please proceed with caution.
      </Alert>
      
      <Button
        variant="outlined"
        color="error"
        startIcon={<Delete />}
        onClick={() => setDeleteDialog(true)}
      >
        Delete Account
      </Button>
    </Paper>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          {t('settings.title')}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          {t('settings.subtitle')}
        </Typography>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label={t('settings.tabs.profile')} />
          <Tab label={t('settings.tabs.security')} />
          <Tab label={t('settings.tabs.notifications')} />
          <Tab label={t('settings.tabs.appearance')} />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box sx={{ mb: 3 }}>
        {activeTab === 0 && renderProfileTab()}
        {activeTab === 1 && renderSecurityTab()}
        {activeTab === 2 && renderNotificationsTab()}
        {activeTab === 3 && renderPreferencesTab()}
      </Box>

      {/* Save Button */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button
          variant="contained"
          size="large"
          startIcon={loading ? <CircularProgress size={20} /> : <Save />}
          onClick={handleSaveSettings}
          disabled={loading}
        >
          {loading ? t('settings.profile.updating') : t('settings.profile.saveChanges')}
        </Button>
      </Box>

      {/* Danger Zone */}
      {renderDangerZone()}

      {/* Delete Account Dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle sx={{ color: 'error.main' }}>
          Delete Account
        </DialogTitle>
        <DialogContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            This action cannot be undone. All your data will be permanently deleted.
          </Alert>
          <Typography>
            Are you sure you want to delete your account? This will:
          </Typography>
          <List dense>
            <ListItem>• Permanently delete all your analysis history</ListItem>
            <ListItem>• Remove all your credits and purchase history</ListItem>
            <ListItem>• Cancel any active subscriptions</ListItem>
            <ListItem>• Delete your profile and preferences</ListItem>
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteAccount}
            color="error"
            variant="contained"
            disabled={loading}
          >
            {loading ? <CircularProgress size={20} /> : 'Delete Account'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Save Success Dialog */}
      <Dialog open={saveDialog} onClose={() => setSaveDialog(false)}>
        <DialogContent sx={{ textAlign: 'center', py: 4 }}>
          <Check color="success" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Settings Saved Successfully!
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Your changes have been applied.
          </Typography>
        </DialogContent>
      </Dialog>
    </Container>
  );
};

export default SettingsPage;