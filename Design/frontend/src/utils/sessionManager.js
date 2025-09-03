/**
 * Session management utilities for handling timeouts and logout scenarios.
 * Provides automated session tracking, user activity monitoring, and timeout handling.
 */

class SessionManager {
  constructor() {
    this.sessionDuration = 15 * 60 * 1000; // 15 minutes in milliseconds
    this.warningTime = 3 * 60 * 1000; // 3 minutes before expiry
    this.checkInterval = 30 * 1000; // Check every 30 seconds
    
    this.sessionTimer = null;
    this.warningTimer = null;
    this.intervalTimer = null;
    this.lastActivity = Date.now();
    this.isWarningShown = false;
    this.onWarning = null;
    this.onTimeout = null;
    this.onActivity = null;
  }

  /**
   * Initialise session monitoring with callback functions.
   */
  init(callbacks = {}) {
    this.onWarning = callbacks.onWarning || (() => {});
    this.onTimeout = callbacks.onTimeout || (() => {});
    this.onActivity = callbacks.onActivity || (() => {});
    
    this.resetSession();
    this.bindActivityEvents();
    this.startMonitoring();
    
    // Session manager initialised
  }

  /**
   * Reset session timers and update activity tracking.
   */
  resetSession() {
    this.lastActivity = Date.now();
    this.isWarningShown = false;
    
    // Clear existing timers
    this.clearTimers();
    
    // Set warning timer (fires 3 minutes before expiry)
    this.warningTimer = setTimeout(() => {
      this.showWarning();
    }, this.sessionDuration - this.warningTime);
    
    // Set session expiry timer
    this.sessionTimer = setTimeout(() => {
      this.expireSession();
    }, this.sessionDuration);
    
    // Notify of activity
    this.onActivity();
  }

  /**
   * Display session expiration warning to user.
   */
  showWarning() {
    if (!this.isWarningShown) {
      this.isWarningShown = true;
      console.log('Session warning triggered');
      this.onWarning();
    }
  }

  /**
   * Expire session and trigger logout
   */
  expireSession() {
    console.log('Session expired');
    this.cleanup();
    this.onTimeout();
  }

  /**
   * Extend session duration when user confirms continued activity.
   */
  extendSession() {
    console.log('Session extended');
    this.resetSession();
  }

  /**
   * Handle manual user logout.
   */
  logout() {
    console.log('Manual logout');
    this.cleanup();
  }

  /**
   * Bind activity event listeners
   */
  bindActivityEvents() {
    const events = [
      'mousedown',
      'mousemove', 
      'keypress',
      'scroll',
      'touchstart',
      'click'
    ];
    
    const throttledActivity = this.throttle(() => {
      this.handleActivity();
    }, 1000); // Throttle to once per second
    
    events.forEach(event => {
      document.addEventListener(event, throttledActivity, true);
    });
    
    this.activityHandler = throttledActivity;
    this.boundEvents = events;
  }

  /**
   * Handle user activity
   */
  handleActivity() {
    const now = Date.now();
    const timeSinceLastActivity = now - this.lastActivity;
    
    // Only reset if significant time has passed or warning is showing
    if (timeSinceLastActivity > 30000 || this.isWarningShown) { // 30 seconds
      this.resetSession();
    }
  }

  /**
   * Start periodic monitoring
   */
  startMonitoring() {
    this.intervalTimer = setInterval(() => {
      const now = Date.now();
      const timeSinceActivity = now - this.lastActivity;
      
      // Check if we should show warning
      if (timeSinceActivity >= (this.sessionDuration - this.warningTime) && !this.isWarningShown) {
        this.showWarning();
      }
      
      // Check if session should expire
      if (timeSinceActivity >= this.sessionDuration) {
        this.expireSession();
      }
    }, this.checkInterval);
  }

  /**
   * Clear all timers
   */
  clearTimers() {
    if (this.sessionTimer) {
      clearTimeout(this.sessionTimer);
      this.sessionTimer = null;
    }
    
    if (this.warningTimer) {
      clearTimeout(this.warningTimer);
      this.warningTimer = null;
    }
    
    if (this.intervalTimer) {
      clearInterval(this.intervalTimer);
      this.intervalTimer = null;
    }
  }

  /**
   * Remove event listeners
   */
  removeEventListeners() {
    if (this.activityHandler && this.boundEvents) {
      this.boundEvents.forEach(event => {
        document.removeEventListener(event, this.activityHandler, true);
      });
    }
  }

  /**
   * Clean up session manager
   */
  cleanup() {
    this.clearTimers();
    this.removeEventListeners();
  }

  /**
   * Get time remaining in session
   */
  getTimeRemaining() {
    const now = Date.now();
    const timeSinceActivity = now - this.lastActivity;
    const remaining = this.sessionDuration - timeSinceActivity;
    return Math.max(0, remaining);
  }

  /**
   * Get time until warning
   */
  getTimeUntilWarning() {
    const now = Date.now();
    const timeSinceActivity = now - this.lastActivity;
    const warningTime = (this.sessionDuration - this.warningTime) - timeSinceActivity;
    return Math.max(0, warningTime);
  }

  /**
   * Check if session is active
   */
  isSessionActive() {
    return this.getTimeRemaining() > 0;
  }

  /**
   * Throttle function to limit execution frequency
   */
  throttle(func, limit) {
    let inThrottle;
    return function() {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /**
   * Format time for display
   */
  formatTime(milliseconds) {
    const minutes = Math.floor(milliseconds / 60000);
    const seconds = Math.floor((milliseconds % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }
}

// Create singleton instance
const sessionManager = new SessionManager();

export default sessionManager;

/**
 * Hook for handling page visibility changes
 */
export const handlePageVisibility = (onVisible, onHidden) => {
  const handleVisibilityChange = () => {
    if (document.hidden) {
      onHidden?.();
    } else {
      onVisible?.();
    }
  };

  document.addEventListener('visibilitychange', handleVisibilityChange);
  
  return () => {
    document.removeEventListener('visibilitychange', handleVisibilityChange);
  };
};

/**
 * Hook for handling page unload
 */
export const handlePageUnload = (callback) => {
  const handleBeforeUnload = (event) => {
    callback?.(event);
  };

  const handleUnload = (event) => {
    callback?.(event);
  };

  window.addEventListener('beforeunload', handleBeforeUnload);
  window.addEventListener('unload', handleUnload);
  
  return () => {
    window.removeEventListener('beforeunload', handleBeforeUnload);
    window.removeEventListener('unload', handleUnload);
  };
};