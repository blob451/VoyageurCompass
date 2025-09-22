import React, { useState } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box, 
  IconButton,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout, selectCurrentUser, selectCurrentToken } from '../../features/auth/authSlice';
import { useLogoutMutation, useGetUserProfileQuery } from '../../features/api/apiSlice';
import { 
  TrendingUp, 
  ExpandMore,
  Analytics,
  Compare,
  AccountBalance,
  ShoppingCart,
  Settings,
  Help,
  Assessment,
  DarkMode,
  LightMode
} from '@mui/icons-material';
import { useThemeMode } from '../../theme/ThemeModeContext.jsx';
import { useTranslation } from 'react-i18next';
import LanguageSwitcher from '../LanguageSwitcher';

const Navbar = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const user = useSelector(selectCurrentUser);
  const token = useSelector(selectCurrentToken);
  const [logoutApi] = useLogoutMutation();
  const [toolsMenuAnchor, setToolsMenuAnchor] = useState(null);
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const { mode, toggleMode } = useThemeMode();
  const { t } = useTranslation();

  // Get user profile data including credits - only when authenticated
  const { data: userProfile } = useGetUserProfileQuery(undefined, {
    skip: !user || !token
  });
  const userCredits = userProfile?.credits || 0;

  const handleLogout = async () => {
    try {
      setUserMenuAnchor(null);
      
      // Refresh token retrieval for secure API logout
      const refreshToken = localStorage.getItem('refreshToken');
      
      // Call logout API
      if (refreshToken) {
        try {
          await logoutApi(refreshToken).unwrap();
        } catch (error) {
          console.warn('Logout API call failed:', error);
          // Client-side logout continuation despite API failure
        }
      }
      
      // Dispatch logout action to Redux store
      dispatch(logout({ reason: 'manual' }));
      
      // Redirect to logout confirmation page
      navigate('/logout', { 
        replace: true, 
        state: { reason: 'manual' } 
      });
      
    } catch (error) {
      console.error('Logout error:', error);
      // Fallback logout enforcement regardless of API failures
      dispatch(logout({ reason: 'manual' }));
      navigate('/logout', { 
        replace: true, 
        state: { reason: 'error' } 
      });
    }
  };

  const handleToolsMenuOpen = (event) => {
    setToolsMenuAnchor(event.currentTarget);
  };

  const handleToolsMenuClose = () => {
    setToolsMenuAnchor(null);
  };

  const handleUserMenuOpen = (event) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleNavigation = (path) => {
    navigate(path);
    handleToolsMenuClose();
    handleUserMenuClose();
  };

  return (
    <AppBar position="static" component="nav" role="navigation" sx={{ bgcolor: '#1a1a2e' }}>
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="logo"
          onClick={() => navigate('/')}
          sx={{ mr: 2 }}
        >
          <TrendingUp />
        </IconButton>
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ flexGrow: 1, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          VoyageurCompass
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {user && token ? (
            <>
              <Button color="inherit" onClick={() => navigate('/dashboard')}>
                {t('navbar.dashboard')}
              </Button>
              
              <Button
                color="inherit"
                onClick={handleToolsMenuOpen}
                endIcon={<ExpandMore />}
              >
                {t('navbar.tools')}
              </Button>
              
              <Menu
                anchorEl={toolsMenuAnchor}
                open={Boolean(toolsMenuAnchor)}
                onClose={handleToolsMenuClose}
              >
                <MenuItem onClick={() => handleNavigation('/reports')}>
                  <Assessment sx={{ mr: 1 }} />
                  {t('navbar.analysisReports')}
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/compare')}>
                  <Compare sx={{ mr: 1 }} />
                  {t('navbar.compareStocks')}
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/sectors')}>
                  <AccountBalance sx={{ mr: 1 }} />
                  {t('navbar.sectorAnalysis')}
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/store')}>
                  <ShoppingCart sx={{ mr: 1 }} />
                  {t('navbar.creditStore')}
                </MenuItem>
              </Menu>

              <Button color="inherit" onClick={() => navigate('/help')}>
                <Help sx={{ mr: 0.5 }} />
                {t('navbar.help')}
              </Button>

              <LanguageSwitcher sx={{ mr: 1 }} />

              <IconButton
                color="inherit"
                aria-label={t(`navbar.switchTo${mode === 'light' ? 'Dark' : 'Light'}Mode`)}
                onClick={toggleMode}
                title={t(`navbar.switchTo${mode === 'light' ? 'Dark' : 'Light'}Mode`)}
              >
                {mode === 'light' ? <DarkMode /> : <LightMode />}
              </IconButton>

              <Chip
                label={`${userCredits} ${t('navbar.credits')}`}
                size="small"
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white',
                  mx: 1
                }}
              />
              
              <Button 
                color="inherit" 
                onClick={handleUserMenuOpen}
                endIcon={<ExpandMore />}
              >
                {user.username}
              </Button>
              
              <Menu
                anchorEl={userMenuAnchor}
                open={Boolean(userMenuAnchor)}
                onClose={handleUserMenuClose}
              >
                <MenuItem onClick={() => handleNavigation('/settings')}>
                  <Settings sx={{ mr: 1 }} />
                  {t('navbar.settings')}
                </MenuItem>
                <MenuItem onClick={handleLogout}>
                  {t('navbar.signOut')}
                </MenuItem>
              </Menu>
            </>
          ) : (
            <>
              <Button color="inherit" onClick={() => navigate('/help')}>
                {t('navbar.help')}
              </Button>

              <LanguageSwitcher sx={{ mr: 1 }} />

              <IconButton
                color="inherit"
                aria-label={t(`navbar.switchTo${mode === 'light' ? 'Dark' : 'Light'}Mode`)}
                onClick={toggleMode}
                title={t(`navbar.switchTo${mode === 'light' ? 'Dark' : 'Light'}Mode`)}
              >
                {mode === 'light' ? <DarkMode /> : <LightMode />}
              </IconButton>
              <Button color="inherit" onClick={() => navigate('/login')}>
                {t('auth.login')}
              </Button>
              <Button
                color="inherit"
                onClick={() => navigate('/register')}
                variant="outlined"
                sx={{
                  borderColor: 'white',
                  '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' }
                }}
              >
                {t('auth.register')}
              </Button>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
