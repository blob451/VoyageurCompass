/**
 * Test utilities for RTK Query and API testing
 */

/**
 * Creates a proper Response object for RTK Query tests
 * @param {any} data - The data to return
 * @param {Object} options - Response options
 * @returns {Response} A proper Response object with clone method
 */
export const createMockResponse = (data, options = {}) => {
  const {
    status = 200,
    statusText = 'OK',
    headers = { 'Content-Type': 'application/json' }
  } = options;

  const responseInit = {
    status,
    statusText,
    headers: new Headers(headers)
  };

  const response = new Response(JSON.stringify(data), responseInit);
  
  // Ensure response has all required methods
  Object.defineProperty(response, 'ok', {
    value: status >= 200 && status < 300,
    writable: false
  });

  return response;
};

/**
 * Creates a proper fetch mock for RTK Query tests
 * @param {any} data - Default data to return
 * @param {Object} options - Default response options
 * @returns {Function} A mock fetch function
 */
export const createMockFetch = (data = {}, options = {}) => {
  return vi.fn().mockResolvedValue(createMockResponse(data, options));
};

/**
 * Creates multiple response mocks for different API endpoints
 * @param {Object} responseMap - Map of URL patterns to responses
 * @returns {Function} A mock fetch function that returns different responses based on URL
 */
export const createMockFetchWithRoutes = (responseMap = {}) => {
  return vi.fn().mockImplementation((url) => {
    // Find matching route
    for (const [pattern, responseData] of Object.entries(responseMap)) {
      if (url.toString().includes(pattern)) {
        return Promise.resolve(createMockResponse(responseData.data || responseData, responseData.options || {}));
      }
    }
    
    // Default response if no route matches
    return Promise.resolve(createMockResponse({ message: 'Not found' }, { status: 404 }));
  });
};

/**
 * Setup global fetch mock for RTK Query compatibility
 * @param {Object} responseMap - Optional response mapping
 */
export const setupFetchMock = (responseMap = {}) => {
  if (Object.keys(responseMap).length > 0) {
    global.fetch = createMockFetchWithRoutes(responseMap);
  } else {
    global.fetch = createMockFetch();
  }
};