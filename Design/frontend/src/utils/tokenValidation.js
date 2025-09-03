/**
 * JWT token validation utilities for client-side authentication management.
 * Provides token decoding, expiration checking, and localStorage cleanup functions.
 */

/**
 * Decode JWT token payload without signature verification (client-side inspection only).
 */
export const decodeJWTPayload = (token) => {
  try {
    if (!token) return null;
    
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    
    const payload = parts[1];
    const decodedPayload = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decodedPayload);
  } catch (error) {
    console.warn('Failed to decode JWT payload:', error);
    return null;
  }
};

/**
 * Verify JWT token expiration status.
 */
export const isTokenExpired = (token) => {
  try {
    const payload = decodeJWTPayload(token);
    if (!payload || !payload.exp) return true;
    
    const currentTime = Date.now() / 1000; // Convert to seconds
    return payload.exp < currentTime;
  } catch (error) {
    console.warn('Error checking token expiration:', error);
    return true; // Assume expired if validation fails
  }
};

/**
 * Extract token expiration timestamp.
 */
export const getTokenExpiration = (token) => {
  try {
    const payload = decodeJWTPayload(token);
    if (!payload || !payload.exp) return null;
    
    return new Date(payload.exp * 1000);
  } catch (error) {
    console.warn('Error getting token expiration:', error);
    return null;
  }
};

/**
 * Calculate remaining time until token expiration (milliseconds).
 */
export const getTimeUntilExpiration = (token) => {
  try {
    const expiration = getTokenExpiration(token);
    if (!expiration) return 0;
    
    const timeRemaining = expiration.getTime() - Date.now();
    return Math.max(0, timeRemaining);
  } catch (error) {
    console.warn('Error calculating time until expiration:', error);
    return 0;
  }
};

/**
 * Check if token will expire within specified minutes
 */
export const willExpireSoon = (token, minutes = 5) => {
  try {
    const timeRemaining = getTimeUntilExpiration(token);
    const thresholdMs = minutes * 60 * 1000;
    return timeRemaining <= thresholdMs;
  } catch (error) {
    console.warn('Error checking if token expires soon:', error);
    return true; // Assume expiring soon if we can't validate
  }
};

/**
 * Extract user ID from JWT token
 */
export const getUserIdFromToken = (token) => {
  try {
    const payload = decodeJWTPayload(token);
    return payload?.user_id || null;
  } catch (error) {
    console.warn('Error extracting user ID from token:', error);
    return null;
  }
};

/**
 * Validate token format and basic structure
 */
export const isValidTokenFormat = (token) => {
  if (!token || typeof token !== 'string') return false;
  
  const parts = token.split('.');
  if (parts.length !== 3) return false;
  
  try {
    // Try to decode each part to ensure it's valid base64
    atob(parts[0].replace(/-/g, '+').replace(/_/g, '/'));
    atob(parts[1].replace(/-/g, '+').replace(/_/g, '/'));
    return true;
  } catch (error) {
    return false;
  }
};

/**
 * Remove invalid or expired tokens from localStorage.
 */
export const cleanupInvalidTokens = () => {
  const accessToken = localStorage.getItem('token');
  const refreshToken = localStorage.getItem('refreshToken');
  
  let tokensRemoved = 0;
  
  // Check access token
  if (accessToken && (isTokenExpired(accessToken) || !isValidTokenFormat(accessToken))) {
    localStorage.removeItem('token');
    tokensRemoved++;
    console.log('Removed invalid/expired access token');
  }
  
  // Check refresh token
  if (refreshToken && (isTokenExpired(refreshToken) || !isValidTokenFormat(refreshToken))) {
    localStorage.removeItem('refreshToken');
    tokensRemoved++;
    console.log('Removed invalid/expired refresh token');
  }
  
  return tokensRemoved;
};

/**
 * Get valid tokens from localStorage (returns null for invalid tokens)
 */
export const getValidTokensFromStorage = () => {
  const accessToken = localStorage.getItem('token');
  const refreshToken = localStorage.getItem('refreshToken');
  
  const validAccessToken = (
    accessToken && 
    isValidTokenFormat(accessToken) && 
    !isTokenExpired(accessToken)
  ) ? accessToken : null;
  
  const validRefreshToken = (
    refreshToken && 
    isValidTokenFormat(refreshToken) && 
    !isTokenExpired(refreshToken)
  ) ? refreshToken : null;
  
  return {
    accessToken: validAccessToken,
    refreshToken: validRefreshToken,
    hasValidTokens: !!(validAccessToken && validRefreshToken),
    hasExpiredTokens: !!(
      (accessToken && !validAccessToken) || 
      (refreshToken && !validRefreshToken)
    )
  };
};

/**
 * Format time remaining for display
 */
export const formatTimeRemaining = (milliseconds) => {
  if (milliseconds <= 0) return 'Expired';
  
  const minutes = Math.floor(milliseconds / 60000);
  const seconds = Math.floor((milliseconds % 60000) / 1000);
  
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  } else {
    return `${seconds}s`;
  }
};

/**
 * Comprehensive token validation summary
 */
export const getTokenValidationSummary = (accessToken, refreshToken) => {
  const validation = {
    accessToken: {
      exists: !!accessToken,
      valid: false,
      expired: true,
      timeRemaining: 0,
      payload: null
    },
    refreshToken: {
      exists: !!refreshToken,
      valid: false,
      expired: true,
      timeRemaining: 0,
      payload: null
    },
    overall: {
      isAuthenticated: false,
      needsRefresh: false,
      shouldLogout: false
    }
  };
  
  // Validate access token
  if (accessToken) {
    validation.accessToken.valid = isValidTokenFormat(accessToken);
    validation.accessToken.expired = isTokenExpired(accessToken);
    validation.accessToken.timeRemaining = getTimeUntilExpiration(accessToken);
    validation.accessToken.payload = decodeJWTPayload(accessToken);
  }
  
  // Validate refresh token
  if (refreshToken) {
    validation.refreshToken.valid = isValidTokenFormat(refreshToken);
    validation.refreshToken.expired = isTokenExpired(refreshToken);
    validation.refreshToken.timeRemaining = getTimeUntilExpiration(refreshToken);
    validation.refreshToken.payload = decodeJWTPayload(refreshToken);
  }
  
  // Determine overall status
  const hasValidAccess = validation.accessToken.valid && !validation.accessToken.expired;
  const hasValidRefresh = validation.refreshToken.valid && !validation.refreshToken.expired;
  
  validation.overall.isAuthenticated = hasValidAccess && hasValidRefresh;
  validation.overall.needsRefresh = !hasValidAccess && hasValidRefresh;
  validation.overall.shouldLogout = !hasValidRefresh;
  
  return validation;
};