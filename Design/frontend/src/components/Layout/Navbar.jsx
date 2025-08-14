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
import { logout, selectCurrentUser } from '../../features/auth/authSlice';
import { 
  TrendingUp, 
  ExpandMore,
  Analytics,
  Compare,
  AccountBalance,
  ShoppingCart,
  Settings,
  Help
} from '@mui/icons-material';

const Navbar = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const user = useSelector(selectCurrentUser);
  const [toolsMenuAnchor, setToolsMenuAnchor] = useState(null);
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);

  // Mock user credits
  const userCredits = 25;

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
    setUserMenuAnchor(null);
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
    <AppBar position="static" sx={{ bgcolor: '#1a1a2e' }}>
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
          {user ? (
            <>
              <Button color="inherit" onClick={() => navigate('/dashboard')}>
                Dashboard
              </Button>
              
              <Button 
                color="inherit" 
                onClick={handleToolsMenuOpen}
                endIcon={<ExpandMore />}
              >
                Tools
              </Button>
              
              <Menu
                anchorEl={toolsMenuAnchor}
                open={Boolean(toolsMenuAnchor)}
                onClose={handleToolsMenuClose}
              >
                <MenuItem onClick={() => handleNavigation('/stocks')}>
                  <Analytics sx={{ mr: 1 }} />
                  Stock Analysis
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/compare')}>
                  <Compare sx={{ mr: 1 }} />
                  Compare Stocks
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/sectors')}>
                  <AccountBalance sx={{ mr: 1 }} />
                  Sector Analysis
                </MenuItem>
                <MenuItem onClick={() => handleNavigation('/store')}>
                  <ShoppingCart sx={{ mr: 1 }} />
                  Credit Store
                </MenuItem>
              </Menu>

              <Button color="inherit" onClick={() => navigate('/help')}>
                <Help sx={{ mr: 0.5 }} />
                Help
              </Button>

              <Chip 
                label={`${userCredits} Credits`}
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
                  Settings
                </MenuItem>
                <MenuItem onClick={handleLogout}>
                  Logout
                </MenuItem>
              </Menu>
            </>
          ) : (
            <>
              <Button color="inherit" onClick={() => navigate('/help')}>
                Help
              </Button>
              <Button color="inherit" onClick={() => navigate('/login')}>
                Login
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
                Register
              </Button>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;