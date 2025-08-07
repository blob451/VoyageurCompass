"""
API Views for Core app.
Handles authentication, user management, and system utilities.
"""

from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth.models import User
from drf_spectacular.utils import extend_schema

from Core.serializers import (
    UserSerializer, UserRegistrationSerializer,
    ChangePasswordSerializer, UserProfileSerializer
)
from Core.services.auth import auth_service


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom JWT token view with additional user data."""
    
    @extend_schema(
        summary="Login",
        description="Authenticate user and receive JWT tokens",
        responses={
            200: {
                'description': 'Login successful',
                'example': {
                    'access': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                    'refresh': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                    'user': {
                        'id': 1,
                        'username': 'john_doe',
                        'email': 'john@example.com'
                    }
                }
            }
        }
    )
    def post(self, request, *args, **kwargs):
        """Override to add user data to response."""
        response = super().post(request, *args, **kwargs)
        
        if response.status_code == 200:
            username = request.data.get('username')
            user = User.objects.get(username=username)
            user_data = UserSerializer(user).data
            response.data['user'] = user_data
        
        return response


class RegisterView(generics.CreateAPIView):
    """User registration view."""
    
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Register new user",
        description="Create a new user account",
        responses={
            201: {
                'description': 'User created successfully',
                'example': {
                    'user': {
                        'id': 1,
                        'username': 'john_doe',
                        'email': 'john@example.com'
                    },
                    'tokens': {
                        'access': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                        'refresh': 'eyJ0eXAiOiJKV1QiLCJhbGc...'
                    }
                }
            }
        }
    )
    def create(self, request, *args, **kwargs):
        """Override to return tokens after registration."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate tokens for the new user
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': UserSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(APIView):
    """User profile management."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Get user profile",
        description="Get current user's profile information",
        responses={200: UserProfileSerializer}
    )
    def get(self, request):
        """Get current user profile."""
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)
    
    @extend_schema(
        summary="Update user profile",
        description="Update current user's profile information",
        request=UserProfileSerializer,
        responses={200: UserProfileSerializer}
    )
    def put(self, request):
        """Update user profile."""
        serializer = UserProfileSerializer(
            request.user, 
            data=request.data, 
            partial=True
        )
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class ChangePasswordView(APIView):
    """Change password view."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Change password",
        description="Change the current user's password",
        request=ChangePasswordSerializer,
        responses={
            200: {'description': 'Password changed successfully'},
            400: {'description': 'Invalid password data'}
        }
    )
    def post(self, request):
        """Change user password."""
        serializer = ChangePasswordSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            
            # Generate new tokens after password change
            refresh = RefreshToken.for_user(user)
            
            return Response({
                'message': 'Password changed successfully',
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                }
            })
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


@extend_schema(
    summary="Health check",
    description="Check if the API is running",
    responses={
        200: {
            'description': 'API is healthy',
            'example': {
                'status': 'healthy',
                'version': '1.0.0'
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'VoyageurCompass API'
    })


@extend_schema(
    summary="Get API statistics",
    description="Get usage statistics for the current user",
    responses={
        200: {
            'description': 'User statistics',
            'example': {
                'portfolios': 3,
                'total_holdings': 15,
                'analyses_today': 5
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_stats(request):
    """Get user statistics."""
    user = request.user
    
    stats = {
        'username': user.username,
        'portfolios': user.portfolios.count(),
        'total_holdings': sum(
            p.holdings.count() for p in user.portfolios.all()
        ),
        'member_since': user.date_joined.isoformat(),
        'last_login': user.last_login.isoformat() if user.last_login else None,
    }
    
    return Response(stats)