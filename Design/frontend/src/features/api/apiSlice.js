import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { setCredentials, logout } from '../auth/authSlice';

// Token refresh mutex to prevent simultaneous refresh attempts
let isRefreshing = false;
let refreshPromise = null;

// Get API URL from environment or use default
const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const baseQuery = fetchBaseQuery({
  baseUrl: apiUrl,
  // Remove credentials: 'include' since JWT doesn't need cookies and it causes CORS issues
  timeout: 180000, // 180 second timeout for analysis operations that may take longer (includes auto-sync, FinBERT, and LLM)
  prepareHeaders: (headers, { getState }) => {
    const token = getState().auth.token;
    if (token) {
      headers.set('authorization', `Bearer ${token}`);
    }
    return headers;
  },
  fetchFn: async (...args) => {
    try {
      return await fetch(...args);
    } catch (error) {
      console.warn('Fetch operation failed:', error.message);
      // Return a minimal response object that won't cause clone errors
      return new Response(JSON.stringify({ data: null }), {
        status: 500,
        statusText: 'Fetch Error',
        headers: { 'Content-Type': 'application/json' }
      });
    }
  },
  responseHandler: async (response) => {
    // Comprehensive response validation for RTK Query
    if (!response || typeof response !== 'object') {
      return Promise.resolve({ data: null });
    }
    
    // Verify response has required methods before cloning
    if (typeof response.clone !== 'function') {
      console.warn('Response object missing clone method, returning null data');
      return Promise.resolve({ data: null });
    }
    
    // Clone the response safely to prevent clone errors
    try {
      const clonedResponse = response.clone();
      
      // Ensure headers exist before accessing
      const contentType = clonedResponse.headers ? 
        clonedResponse.headers.get('content-type') : null;
      
      if (contentType && contentType.includes('application/json')) {
        return await clonedResponse.json();
      }
      return await clonedResponse.text();
    } catch (error) {
      console.warn('Response cloning failed:', error.message);
      // Return null data instead of original response to prevent further errors
      return Promise.resolve({ data: null });
    }
  },
});

const baseQueryWithReauth = async (args, api, extraOptions) => {
  let result;
  
  // Safely execute baseQuery with comprehensive error handling
  try {
    result = await baseQuery(args, api, extraOptions);
  } catch (error) {
    console.warn('BaseQuery execution failed:', error.message);
    return {
      error: {
        status: 'FETCH_ERROR',
        error: 'BaseQuery execution failed'
      }
    };
  }

  // Validate result to prevent clone errors in tests
  if (!result || (typeof result !== 'object')) {
    return {
      error: {
        status: 'FETCH_ERROR',
        error: 'Invalid response format'
      }
    };
  }

  if (result?.error?.status === 401) {
    console.log('Token expired, attempting refresh...');

    // Use mutex to prevent simultaneous refresh attempts
    if (isRefreshing) {
      console.log('Token refresh already in progress, waiting...');
      // Wait for the existing refresh to complete
      try {
        await refreshPromise;
        // Retry the original query with potentially new token
        console.log('Using refreshed token from ongoing refresh, retrying original request...');
        result = await baseQuery(args, api, extraOptions);
        return result;
      } catch (error) {
        console.log('Waited refresh failed, not retrying');
        return result;
      }
    }

    // Start refresh process
    isRefreshing = true;

    const state = api.getState();
    const refreshToken = state.auth.refreshToken;

    if (refreshToken) {
      try {
        refreshPromise = baseQuery({
          url: '/auth/refresh/',
          method: 'POST',
          body: { refresh: refreshToken },
        }, api, extraOptions);

        const refreshResult = await refreshPromise;

        if (refreshResult?.data?.access) {
          const user = api.getState().auth.user;
          // Store the new token with correct field names
          api.dispatch(setCredentials({
            access: refreshResult.data.access,
            refresh: refreshToken, // Keep existing refresh token
            user
          }));
          // Retry the original query with new access token
          console.log('Token refreshed successfully, retrying original request...');
          result = await baseQuery(args, api, extraOptions);
        } else {
          console.log('Token refresh failed - invalid response, logging out...');
          api.dispatch(logout({ reason: 'token_refresh_failed' }));
        }
      } catch (refreshError) {
        console.log('Token refresh error:', refreshError);
        api.dispatch(logout({ reason: 'token_refresh_error' }));
      } finally {
        isRefreshing = false;
        refreshPromise = null;
      }
    } else {
      console.log('No refresh token available, logging out...');
      api.dispatch(logout({ reason: 'no_refresh_token' }));
      isRefreshing = false;
      refreshPromise = null;
    }
  }

  return result;
};

export const apiSlice = createApi({
  baseQuery: baseQueryWithReauth,
  tagTypes: ['User', 'Stock', 'Portfolio', 'Analysis'],
  endpoints: (builder) => ({
    login: builder.mutation({
      query: (credentials) => ({
        url: '/auth/login/',
        method: 'POST',
        body: credentials,
      }),
      transformResponse: (response) => {
        // Login successful - return response
        return response;
      },
      transformErrorResponse: (response) => {
        // Return a consistent error format
        return {
          status: response.status,
          data: response.data || { detail: 'Authentication failed' },
        };
      },
    }),
    register: builder.mutation({
      query: (userData) => ({
        url: '/auth/register/',
        method: 'POST',
        body: userData,
      }),
    }),
    logout: builder.mutation({
      query: (refreshToken) => ({
        url: '/auth/logout/',
        method: 'POST',
        body: { refresh_token: refreshToken },
      }),
      transformResponse: (response) => {
        // Logout successful - return response
        return response;
      },
      transformErrorResponse: (response) => {
        return {
          status: response.status,
          data: response.data || { detail: 'Logout failed' },
        };
      },
    }),
    validateToken: builder.query({
      query: () => '/auth/validate/',
      transformResponse: (response) => {
        // Token validation successful - return response
        return response;
      },
      transformErrorResponse: (response) => {
        return {
          status: response.status,
          data: response.data || { detail: 'Token validation failed' },
        };
      },
    }),
    getStocks: builder.query({
      query: (params = {}) => ({
        url: '/stocks/',
        params,
      }),
      providesTags: ['Stock'],
    }),
    getStock: builder.query({
      query: (symbol) => `/stocks/${symbol}/`,
      providesTags: (result, error, symbol) => [{ type: 'Stock', id: symbol }],
    }),
    analyzeStock: builder.mutation({
      query: ({ symbol, months = 6, includeExplanation = false, explanationDetail = 'standard', language = 'en' }) => ({
        url: `/analytics/analyze/${symbol}/`,
        method: 'GET',
        params: {
          months,
          include_explanation: includeExplanation,
          explanation_detail: explanationDetail,
          language: language
        },
      }),
      invalidatesTags: ['Analysis'],
    }),
    getUserProfile: builder.query({
      query: () => '/user/profile/',
      providesTags: ['User'],
    }),
    getPortfolios: builder.query({
      query: () => '/portfolios/',
      providesTags: ['Portfolio'],
    }),
    createPortfolio: builder.mutation({
      query: (portfolioData) => ({
        url: '/portfolios/',
        method: 'POST',
        body: portfolioData,
      }),
      invalidatesTags: ['Portfolio'],
    }),
    getUserAnalysisHistory: builder.query({
      query: (params = {}) => ({
        url: '/analytics/user/history/',
        params,
      }),
      providesTags: ['Analysis'],
      // Cache results for 5 minutes to avoid unnecessary re-fetches
      keepUnusedDataFor: 300,
      // Don't refetch on focus/reconnect for better perceived performance
      refetchOnMountOrArgChange: 60,
      refetchOnFocus: false,
      refetchOnReconnect: false,
    }),
    getUserLatestAnalysis: builder.query({
      query: (symbol) => `/analytics/user/${symbol}/latest/`,
      providesTags: (result, error, symbol) => [{ type: 'Analysis', id: symbol }],
    }),
    getAnalysisById: builder.query({
      query: (analysisId) => `/analytics/analysis/${analysisId}/`,
      providesTags: (result, error, analysisId) => [{ type: 'Analysis', id: analysisId }],
    }),
    // LLaMA 3.1 70B Explanation Endpoints
    generateExplanation: builder.mutation({
      query: ({ analysisId, detailLevel = 'standard', forceRegenerate = false, language = 'en' }) => ({
        url: `/analytics/explain/${analysisId}/`,
        method: 'POST',
        params: {
          detail_level: detailLevel,
          force_regenerate: forceRegenerate,
          language: language
        },
      }),
      invalidatesTags: (result, error, { analysisId }) => [{ type: 'Analysis', id: analysisId }],
    }),
    getExplanation: builder.query({
      query: ({ analysisId, detailLevel, language = 'en' }) => ({
        url: `/analytics/explanation/${analysisId}/`,
        params: {
          ...(detailLevel && { detail_level: detailLevel }),
          language: language
        },
      }),
      providesTags: (result, error, { analysisId, detailLevel }) => [
        { type: 'Analysis', id: `explanation-${analysisId}` },
        { type: 'Analysis', id: `explanation-${analysisId}-${detailLevel || 'default'}` }
      ],
    }),
    getExplanationStatus: builder.query({
      query: () => `/analytics/explanation-status/`,
      keepUnusedDataFor: 60, // Cache status for 1 minute
    }),
  }),
});

export const {
  useLoginMutation,
  useLogoutMutation,
  useRegisterMutation,
  useValidateTokenQuery,
  useGetStocksQuery,
  useGetStockQuery,
  useAnalyzeStockMutation,
  useGetUserProfileQuery,
  useGetPortfoliosQuery,
  useCreatePortfolioMutation,
  useGetUserAnalysisHistoryQuery,
  useGetUserLatestAnalysisQuery,
  useGetAnalysisByIdQuery,
  // LLaMA 3.1 70B Explanation Hooks
  useGenerateExplanationMutation,
  useGetExplanationQuery,
  useGetExplanationStatusQuery,
} = apiSlice;