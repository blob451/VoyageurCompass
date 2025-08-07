import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { selectCurrentToken } from '../features/auth/authSlice';

const ProtectedRoute = () => {
  const token = useSelector(selectCurrentToken);
  
  return token ? <Outlet /> : <Navigate to="/login" replace />;
};

export default ProtectedRoute;