/**
 * Production-safe logging utility
 * Comprehensive logging for analysis, explanations, and system operations
 */

const isDevelopment = import.meta.env.MODE === 'development';

// Performance tracking utilities
const performanceTracker = {
  timers: new Map(),
  
  start: (operation) => {
    const startTime = performance.now();
    performanceTracker.timers.set(operation, startTime);
    return startTime;
  },
  
  end: (operation) => {
    const startTime = performanceTracker.timers.get(operation);
    if (startTime) {
      const duration = performance.now() - startTime;
      performanceTracker.timers.delete(operation);
      return duration;
    }
    return null;
  }
};

// Enhanced logging categories
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

export const analysisLogger = {
  info: (event, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[ANALYSIS] ${event}`, sanitizedDetails);
    }
  },
  
  error: (event, error = {}) => {
    const sanitizedError = sanitizeForLogging(error);
    console.error(`[ANALYSIS ERROR] ${event}`, sanitizedError);
  },
  
  warn: (event, details = {}) => {
    const sanitizedDetails = sanitizeForLogging(details);
    console.warn(`[ANALYSIS WARNING] ${event}`, sanitizedDetails);
  },
  
  stage: (symbol, stage, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[ANALYSIS] ${symbol} → ${stage}`, sanitizedDetails);
    }
  },
  
  performance: (operation, duration, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[ANALYSIS PERF] ${operation}: ${duration.toFixed(2)}ms`, sanitizedDetails);
    }
  }
};

export const explanationLogger = {
  info: (event, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[EXPLANATION] ${event}`, sanitizedDetails);
    }
  },
  
  error: (event, error = {}) => {
    const sanitizedError = sanitizeForLogging(error);
    console.error(`[EXPLANATION ERROR] ${event}`, sanitizedError);
  },
  
  warn: (event, details = {}) => {
    const sanitizedDetails = sanitizeForLogging(details);
    console.warn(`[EXPLANATION WARNING] ${event}`, sanitizedDetails);
  },
  
  workflow: (analysisId, stage, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[EXPLANATION] Analysis ${analysisId} → ${stage}`, sanitizedDetails);
    }
  },
  
  generation: (analysisId, detailLevel, startTime) => {
    if (isDevelopment) {
      const operation = `explanation_${analysisId}_${detailLevel}`;
      if (startTime) {
        performanceTracker.timers.set(operation, startTime);
        console.log(`[EXPLANATION] Starting ${detailLevel} generation for analysis ${analysisId}`);
      } else {
        const duration = performanceTracker.end(operation);
        if (duration) {
          console.log(`[EXPLANATION] Generated ${detailLevel} for analysis ${analysisId} in ${duration.toFixed(2)}ms`);
        }
      }
    }
  }
};

export const llmLogger = {
  info: (event, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[LLM] ${event}`, sanitizedDetails);
    }
  },
  
  error: (event, error = {}) => {
    const sanitizedError = sanitizeForLogging(error);
    console.error(`[LLM ERROR] ${event}`, sanitizedError);
  },
  
  warn: (event, details = {}) => {
    const sanitizedDetails = sanitizeForLogging(details);
    console.warn(`[LLM WARNING] ${event}`, sanitizedDetails);
  },
  
  operation: (operation, stage, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      console.log(`[LLM] ${operation} → ${stage}`, sanitizedDetails);
    }
  },
  
  request: (analysisId, detailLevel, method = 'generate') => {
    if (isDevelopment) {
      const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
      console.groupCollapsed(`[LLM REQUEST] ${method.toUpperCase()} - Analysis ${analysisId} (${detailLevel}) - ${timestamp}`);
      console.log('Request initiated for LLM explanation generation');
      console.log('Detail Level:', detailLevel);
      console.log('Analysis ID:', analysisId);
      console.log('Method:', method);
    }
  },
  
  response: (analysisId, success, details = {}) => {
    if (isDevelopment) {
      const sanitizedDetails = sanitizeForLogging(details);
      if (success) {
        console.log(`[LLM SUCCESS] Analysis ${analysisId}:`, sanitizedDetails);
      } else {
        console.error(`[LLM FAILURE] Analysis ${analysisId}:`, sanitizedDetails);
      }
      console.groupEnd();
    }
  },
  
  debounce: (action, reason) => {
    if (isDevelopment) {
      console.log(`[LLM DEBOUNCE] ${action} blocked - ${reason}`);
    }
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
  const sensitiveFields = ['token', 'access', 'refresh', 'password', 'secret', 'authorization'];
  
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
  
  // Handle error objects with stack traces
  if (sanitized.error && sanitized.error instanceof Error) {
    sanitized.error = {
      message: sanitized.error.message,
      status: sanitized.error.status,
      type: sanitized.error.constructor.name
    };
  }
  
  // Truncate very long content for readability
  if (sanitized.content && typeof sanitized.content === 'string' && sanitized.content.length > 200) {
    sanitized.content = sanitized.content.substring(0, 200) + '...[truncated]';
  }
  
  return sanitized;
}

// Utility exports for performance tracking
export const performanceUtils = {
  startTimer: performanceTracker.start,
  endTimer: performanceTracker.end,
  
  // Wrapper for timing operations
  timeOperation: async (operation, operationName) => {
    performanceTracker.start(operationName);
    try {
      const result = await operation();
      const duration = performanceTracker.end(operationName);
      return { result, duration };
    } catch (error) {
      performanceTracker.end(operationName);
      throw error;
    }
  }
};

// Combined logger export for convenience
export const logger = {
  auth: authLogger,
  analysis: analysisLogger,
  explanation: explanationLogger,
  llm: llmLogger
};

export default logger;