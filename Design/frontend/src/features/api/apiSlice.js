import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { setCredentials, logout } from '../auth/authSlice';

// Get API URL from environment or use default
const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const baseQuery = fetchBaseQuery({
  baseUrl: apiUrl,
  // Remove credentials: 'include' since JWT doesn't need cookies and it causes CORS issues
  timeout: 90000, // 90 second timeout for analysis operations that may take longer
  prepareHeaders: (headers, { getState }) => {
    const token = getState().auth.token;
    if (token) {
      headers.set('authorization', `Bearer ${token}`);
    }
    return headers;
  },
});

const baseQueryWithReauth = async (args, api, extraOptions) => {
  let result = await baseQuery(args, api, extraOptions);

  if (result?.error?.status === 401) {
    console.log('Token expired, attempting refresh...');
    // Try to get a new token
    const refreshResult = await baseQuery('/auth/refresh/', api, extraOptions);
    
    if (refreshResult?.data) {
      const user = api.getState().auth.user;
      // Store the new token
      api.dispatch(setCredentials({ ...refreshResult.data, user }));
      // Retry the original query with new access token
      console.log('Token refreshed, retrying original request...');
      result = await baseQuery(args, api, extraOptions);
    } else {
      console.log('Token refresh failed, logging out...');
      api.dispatch(logout());
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
      transformResponse: (response, meta, arg) => {
        // Login successful - return response
        return response;
      },
      transformErrorResponse: (response, meta, arg) => {
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
      transformResponse: (response, meta, arg) => {
        // Logout successful - return response
        return response;
      },
      transformErrorResponse: (response, meta, arg) => {
        return {
          status: response.status,
          data: response.data || { detail: 'Logout failed' },
        };
      },
    }),
    validateToken: builder.query({
      query: () => '/auth/validate/',
      transformResponse: (response, meta, arg) => {
        // Token validation successful - return response
        return response;
      },
      transformErrorResponse: (response, meta, arg) => {
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
      query: ({ symbol, months = 6, includeExplanation = false, explanationDetail = 'standard' }) => ({
        url: `/analytics/analyze/${symbol}/`,
        method: 'GET',
        params: { 
          months,
          include_explanation: includeExplanation,
          explanation_detail: explanationDetail
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
      query: ({ analysisId, detailLevel = 'standard' }) => ({
        url: `/analytics/explain/${analysisId}/`,
        method: 'POST',
        params: { detail_level: detailLevel },
      }),
      invalidatesTags: (result, error, { analysisId }) => [{ type: 'Analysis', id: analysisId }],
    }),
    getExplanation: builder.query({
      query: (analysisId) => `/analytics/explanation/${analysisId}/`,
      providesTags: (result, error, analysisId) => [{ type: 'Analysis', id: `explanation-${analysisId}` }],
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