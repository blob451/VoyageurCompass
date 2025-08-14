import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  LinearProgress,
  Alert,
} from '@mui/material';
import { Warning, AccessTime } from '@mui/icons-material';

const SessionWarningDialog = ({ 
  open, 
  onExtendSession, 
  onLogout,
  warningDuration = 3 * 60 * 1000 // 3 minutes in milliseconds
}) => {
  const [timeRemaining, setTimeRemaining] = useState(warningDuration);

  useEffect(() => {
    if (!open) {
      setTimeRemaining(warningDuration);
      return;
    }

    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1000) {
          // Auto-logout when time expires
          onLogout();
          return 0;
        }
        return prev - 1000;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [open, onLogout, warningDuration]);

  const formatTime = (milliseconds) => {
    const minutes = Math.floor(milliseconds / 60000);
    const seconds = Math.floor((milliseconds % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const progressValue = ((warningDuration - timeRemaining) / warningDuration) * 100;

  return (
    <Dialog
      open={open}
      disableEscapeKeyDown
      disableBackdropClick
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          background: 'linear-gradient(145deg, #fff3e0 0%, #ffffff 100%)',
          border: '2px solid #ff9800'
        }
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
          <Warning sx={{ color: 'warning.main', fontSize: 32 }} />
          <Typography variant="h5" component="span" color="warning.main">
            Session Expiring Soon
          </Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ textAlign: 'center', pb: 2 }}>
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body1">
            Your session will expire in <strong>{formatTime(timeRemaining)}</strong> due to inactivity.
          </Typography>
        </Alert>

        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 2 }}>
            <AccessTime sx={{ color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">
              Time remaining: {formatTime(timeRemaining)}
            </Typography>
          </Box>
          
          <LinearProgress
            variant="determinate"
            value={progressValue}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: 'grey.200',
              '& .MuiLinearProgress-bar': {
                backgroundColor: timeRemaining < 60000 ? 'error.main' : 'warning.main',
                borderRadius: 4,
              }
            }}
          />
        </Box>

        <Typography variant="body2" color="text.secondary" paragraph>
          For your security, we automatically log you out after 15 minutes of inactivity.
          Click "Stay Logged In" to continue your session, or "Logout" to end it now.
        </Typography>

        <Box sx={{ 
          p: 2, 
          bgcolor: 'grey.50', 
          borderRadius: 1,
          border: '1px solid',
          borderColor: 'grey.200'
        }}>
          <Typography variant="caption" color="text.secondary">
            💡 <strong>Tip:</strong> Any activity (clicking, typing, scrolling) will automatically extend your session.
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ justifyContent: 'center', gap: 2, pb: 3 }}>
        <Button
          onClick={onLogout}
          variant="outlined"
          color="inherit"
          sx={{ minWidth: 120 }}
        >
          Logout Now
        </Button>
        <Button
          onClick={onExtendSession}
          variant="contained"
          color="warning"
          sx={{ minWidth: 120 }}
          autoFocus
        >
          Stay Logged In
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SessionWarningDialog;