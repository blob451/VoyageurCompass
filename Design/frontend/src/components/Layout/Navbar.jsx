import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, IconButton } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout, selectCurrentUser } from '../../features/auth/authSlice';
import { TrendingUp } from '@mui/icons-material';

const Navbar = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const user = useSelector(selectCurrentUser);

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
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
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Voyageur Compass
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          {user ? (
            <>
              <Button color="inherit" onClick={() => navigate('/dashboard')}>
                Dashboard
              </Button>
              <Button color="inherit" onClick={() => navigate('/portfolios')}>
                Portfolios
              </Button>
              <Button color="inherit" onClick={() => navigate('/analysis')}>
                Analysis
              </Button>
              <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                Welcome, {user.username}
              </Typography>
              <Button color="inherit" onClick={handleLogout}>
                Logout
              </Button>
            </>
          ) : (
            <>
              <Button color="inherit" onClick={() => navigate('/login')}>
                Login
              </Button>
              <Button color="inherit" onClick={() => navigate('/register')}>
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