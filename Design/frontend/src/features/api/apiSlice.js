import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { setCredentials, logout } from '../auth/authSlice';

const baseQuery = fetchBaseQuery({
  baseUrl: 'http://localhost:8000/api/v1',
  credentials: 'include',
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
    console.log('Sending refresh token');
    // Try to get a new token
    const refreshResult = await baseQuery('/auth/refresh/', api, extraOptions);
    
    if (refreshResult?.data) {
      const user = api.getState().auth.user;
      // Store the new token
      api.dispatch(setCredentials({ ...refreshResult.data, user }));
      // Retry the original query with new access token
      result = await baseQuery(args, api, extraOptions);
    } else {
      api.dispatch(logout());
    }
  }

  return result;
};

export const apiSlice = createApi({
  baseQuery: baseQueryWithReauth,
  tagTypes: ['User', 'Stock', 'Portfolio'],
  endpoints: (builder) => ({
    login: builder.mutation({
      query: (credentials) => ({
        url: '/auth/login/',
        method: 'POST',
        body: credentials,
      }),
    }),
    register: builder.mutation({
      query: (userData) => ({
        url: '/auth/register/',
        method: 'POST',
        body: userData,
      }),
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
    analyzeStock: builder.query({
      query: ({ symbol, months = 6 }) => ({
        url: `/analyze/${symbol}/`,
        params: { months },
      }),
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
  }),
});

export const {
  useLoginMutation,
  useRegisterMutation,
  useGetStocksQuery,
  useGetStockQuery,
  useAnalyzeStockQuery,
  useGetUserProfileQuery,
  useGetPortfoliosQuery,
  useCreatePortfolioMutation,
} = apiSlice;