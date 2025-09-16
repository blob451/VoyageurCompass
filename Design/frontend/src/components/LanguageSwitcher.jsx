import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  Menu,
  MenuItem,
  Typography,
  Box,
  Tooltip
} from '@mui/material';
import { Language, ExpandMore } from '@mui/icons-material';

const LanguageSwitcher = ({ sx = {} }) => {
  const { t, i18n } = useTranslation('common');
  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLanguageChange = (languageCode) => {
    i18n.changeLanguage(languageCode);
    handleClose();
  };

  const languages = {
    en: { flag: '🇬🇧', name: 'English', shortName: 'EN' },
    fr: { flag: '🇫🇷', name: 'Français', shortName: 'FR' },
    es: { flag: '🇪🇸', name: 'Español', shortName: 'ES' }
  };

  const currentLanguage = languages[i18n.language] || languages.en;

  return (
    <Box sx={sx}>
      <Tooltip title={t('settings.language')} arrow>
        <Button
          color="inherit"
          onClick={handleClick}
          endIcon={<ExpandMore />}
          size="small"
          sx={{
            minWidth: 'auto',
            color: 'inherit',
            textTransform: 'none',
            borderRadius: 1,
            px: 1,
            '&:hover': {
              backgroundColor: 'action.hover',
            },
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span style={{ fontSize: '20px' }}>{currentLanguage.flag}</span>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {currentLanguage.shortName}
            </Typography>
          </Box>
        </Button>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        sx={{
          '& .MuiPaper-root': {
            minWidth: 160,
            mt: 0.5,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
          },
        }}
      >
        {Object.entries(languages).map(([code, lang]) => (
          <MenuItem
            key={code}
            onClick={() => handleLanguageChange(code)}
            selected={i18n.language === code}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              py: 1,
              '&:hover': {
                backgroundColor: 'rgba(25, 118, 210, 0.08)',
              },
              '&.Mui-selected': {
                backgroundColor: 'rgba(25, 118, 210, 0.12)',
                '&:hover': {
                  backgroundColor: 'rgba(25, 118, 210, 0.16)',
                },
              },
            }}
          >
            <span style={{ fontSize: '18px' }}>{lang.flag}</span>
            <Typography variant="body2">{lang.name}</Typography>
          </MenuItem>
        ))}
      </Menu>
    </Box>
  );
};

export default LanguageSwitcher;