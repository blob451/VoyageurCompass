import React, { useState } from 'react';
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
    setPreferenceData(prev => ({
      ...prev,
      [field]: event.target.type === 'checkbox' ? event.target.checked : event.target.value
    }));
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
      
    } catch (error) {
      console.error('Failed to save settings:', error);
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
      console.log('Account deletion requested');
      
    } catch (error) {
      console.error('Failed to delete account:', error);
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
        Profile Information
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="First Name"
            value={profileData.firstName}
            onChange={handleProfileChange('firstName')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Last Name"
            value={profileData.lastName}
            onChange={handleProfileChange('lastName')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Username"
            value={profileData.username}
            onChange={handleProfileChange('username')}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Email"
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
            Change Password
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                type={showPassword ? 'text' : 'password'}
                label="Current Password"
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
                label="New Password"
                value={securityData.newPassword}
                onChange={handleSecurityChange('newPassword')}
                helperText="Minimum 8 characters"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="password"
                label="Confirm New Password"
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
                  console.log('Password change requested');
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
            Security Settings
          </Typography>
          
          <List>
            <ListItem>
              <ListItemText
                primary="Two-Factor Authentication"
                secondary="Add an extra layer of security to your account"
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
                primary="Login Notifications"
                secondary="Get notified when someone signs into your account"
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
        Notification Preferences
      </Typography>
      
      <List>
        <ListItem>
          <ListItemText
            primary="Email Notifications"
            secondary="Receive notifications via email"
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
            primary="Analysis Complete"
            secondary="Notify when stock analysis is finished"
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
            primary="Weekly Report"
            secondary="Get a summary of your activity"
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
            primary="Market Alerts"
            secondary="Important market news and updates"
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
            primary="Low Credit Alert"
            secondary="Notify when your credits are running low"
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
            primary="Promotional Emails"
            secondary="Special offers and product updates"
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
        Application Preferences
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Theme</InputLabel>
            <Select
              value={preferenceData.theme}
              onChange={handlePreferenceChange('theme')}
              label="Theme"
            >
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
              <MenuItem value="auto">Auto</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Language</InputLabel>
            <Select
              value={preferenceData.language}
              onChange={handlePreferenceChange('language')}
              label="Language"
            >
              <MenuItem value="en">English</MenuItem>
              <MenuItem value="es">Spanish</MenuItem>
              <MenuItem value="fr">French</MenuItem>
              <MenuItem value="de">German</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Timezone</InputLabel>
            <Select
              value={preferenceData.timezone}
              onChange={handlePreferenceChange('timezone')}
              label="Timezone"
            >
              <MenuItem value="America/New_York">Eastern Time</MenuItem>
              <MenuItem value="America/Chicago">Central Time</MenuItem>
              <MenuItem value="America/Denver">Mountain Time</MenuItem>
              <MenuItem value="America/Los_Angeles">Pacific Time</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Default Analysis View</InputLabel>
            <Select
              value={preferenceData.defaultAnalysisView}
              onChange={handlePreferenceChange('defaultAnalysisView')}
              label="Default Analysis View"
            >
              <MenuItem value="summary">Summary</MenuItem>
              <MenuItem value="detailed">Detailed</MenuItem>
              <MenuItem value="charts">Charts</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12}>
          <List>
            <ListItem>
              <ListItemText
                primary="Auto Refresh Data"
                secondary="Automatically refresh market data"
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
                primary="Sound Effects"
                secondary="Play sounds for notifications and alerts"
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
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your account preferences and security settings
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
          <Tab label="Profile" />
          <Tab label="Security" />
          <Tab label="Notifications" />
          <Tab label="Preferences" />
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
          {loading ? 'Saving...' : 'Save Changes'}
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