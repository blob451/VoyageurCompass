/**
 * Production-safe logging utility
 * Only logs important events without sensitive data
 */

const isDevelopment = import.meta.env.MODE === 'development';

export const authLogger = {
  info: (event, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[AUTH] ${event}`, sanitizedDetails);
    }
  },
  
  error: (event, error = {}) => {
    const sanitizedError = sanitizeForLogging(error);
    console.error(`[AUTH ERROR] ${event}`, sanitizedError);
  },
  
  warn: (event, details = {}) => {
    const sanitizedDetails = sanitizeForLogging(details);
    console.warn(`[AUTH WARNING] ${event}`, sanitizedDetails);
  }
};

/**
 * Remove sensitive data from logging output
 */
function sanitizeForLogging(obj) {
  if (!obj || typeof obj !== 'object') {
    return obj;
  }
  
  const sanitized = { ...obj };
  
  // Remove or mask sensitive fields
  const sensitiveFields = ['token', 'access', 'refresh', 'password', 'secret'];
  
  for (const field of sensitiveFields) {
    if (sanitized[field]) {
      sanitized[field] = '[REDACTED]';
    }
  }
  
  // If it's a user object, only keep safe fields
  if (sanitized.user) {
    const { username, email, id } = sanitized.user;
    sanitized.user = { username, email, id };
  }
  
  return sanitized;
}

export default authLogger;