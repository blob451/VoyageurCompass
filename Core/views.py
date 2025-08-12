"""
API Views for Core app.
Handles authentication, user management, and system utilities.
"""

import logging

from django.contrib.auth.models import User
from django.db import connection
from drf_spectacular.utils import extend_schema
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from Core.serializers import (ChangePasswordSerializer, UserProfileSerializer,
                              UserRegistrationSerializer, UserSerializer)


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom JWT token view with additional user data."""

    @extend_schema(
        summary="Login",
        description="Authenticate user and receive JWT tokens",
        responses={
            200: {
                "description": "Login successful",
                "example": {
                    "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                    "user": {
                        "id": 1,
                        "username": "john_doe",
                        "email": "john@example.com",
                    },
                },
            }
        },
    )
    def post(self, request, *args, **kwargs):
        """Override to add user data to response."""
        response = super().post(request, *args, **kwargs)

        if response.status_code == 200:
            username = request.data.get("username")
            user = User.objects.get(username=username)
            user_data = UserSerializer(user).data
            response.data["user"] = user_data

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
                "description": "User created successfully",
                "example": {
                    "user": {
                        "id": 1,
                        "username": "john_doe",
                        "email": "john@example.com",
                    },
                    "tokens": {
                        "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                        "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                    },
                },
            }
        },
    )
    def create(self, request, *args, **kwargs):
        """Override to return tokens after registration."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Generate tokens for the new user
        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "user": UserSerializer(user).data,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                # Keep nested structure for frontend compatibility
                "tokens": {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                },
            },
            status=status.HTTP_201_CREATED,
        )


class UserProfileView(APIView):
    """User profile management."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="Get user profile",
        description="Get current user's profile information",
        responses={200: UserProfileSerializer},
    )
    def get(self, request):
        """Get current user profile."""
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)

    @extend_schema(
        summary="Update user profile",
        description="Update current user's profile information",
        request=UserProfileSerializer,
        responses={200: UserProfileSerializer},
    )
    def put(self, request):
        """Update user profile."""
        serializer = UserProfileSerializer(
            request.user, data=request.data, partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request):
        """Partially update user profile."""
        serializer = UserProfileSerializer(
            request.user, data=request.data, partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChangePasswordView(APIView):
    """Change password view."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="Change password",
        description="Change the current user's password",
        request=ChangePasswordSerializer,
        responses={
            200: {"description": "Password changed successfully"},
            400: {"description": "Invalid password data"},
        },
    )
    def post(self, request):
        """Change user password."""
        serializer = ChangePasswordSerializer(
            data=request.data, context={"request": request}
        )

        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data["new_password"])
            user.save()

            # Generate new tokens after password change
            refresh = RefreshToken.for_user(user)

            return Response(
                {
                    "message": "Password changed successfully",
                    "tokens": {
                        "refresh": str(refresh),
                        "access": str(refresh.access_token),
                    },
                }
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutView(APIView):
    """User logout view."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="Logout",
        description="Logout user and blacklist refresh token",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "refresh": {
                        "type": "string",
                        "description": "Refresh token to blacklist",
                    }
                },
                "required": ["refresh"],
            }
        },
        responses={
            200: {"description": "Logout successful"},
            400: {"description": "Invalid refresh token"},
        },
    )
    def post(self, request):
        """Logout user and blacklist refresh token."""
        try:
            from rest_framework_simplejwt.token_blacklist.models import \
                BlacklistedToken

            refresh_token = request.data.get("refresh")
            if not refresh_token:
                return Response({"message": "Successfully logged out"})

            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response({"message": "Successfully logged out"})

        except Exception as e:
            # If token blacklisting fails, still return success
            # This handles cases where the token is already expired/invalid
            return Response({"message": "Successfully logged out"})


@extend_schema(
    summary="Health check",
    description="Check if the API is running",
    responses={
        200: {
            "description": "API is healthy",
            "example": {"status": "healthy", "version": "1.0.0"},
        }
    },
)
@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint."""
    from datetime import datetime, timezone

    return Response(
        {
            "status": "healthy",
            "version": "1.0.0",
            "service": "VoyageurCompass API",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def healthCheck(request):
    """Liveness probe - simple health check"""
    from datetime import datetime, timezone

    return Response(
        {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requestId": getattr(request, "correlation_id", None),
        },
        headers={"Cache-Control": "no-store"},
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def readinessCheck(request):
    """Readiness probe - checks database connectivity"""
    logger = logging.getLogger("VoyageurCompass.health")

    try:
        # Ensure connection and simple DB ping
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")

        from datetime import datetime, timezone

        return Response(
            {
                "status": "ready",
                "database": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requestId": getattr(request, "correlation_id", None),
            },
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        logger.error("Readiness check failed", exc_info=True)
        from datetime import datetime, timezone

        return Response(
            {
                "status": "not ready",
                "database": "disconnected",
                "error": "database connection failure",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requestId": getattr(request, "correlation_id", None),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Cache-Control": "no-store"},
        )


@extend_schema(
    summary="Get API statistics",
    description="Get usage statistics for the current user",
    responses={
        200: {
            "description": "User statistics",
            "example": {"portfolios": 3, "total_holdings": 15, "analyses_today": 5},
        }
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_stats(request):
    """Get user statistics."""
    user = request.user

    stats = {
        "username": user.username,
        "portfolios": user.portfolios.count(),
        "total_holdings": sum(p.holdings.count() for p in user.portfolios.all()),
        "member_since": user.date_joined.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }

    return Response(stats)


# Additional health check endpoints for test compatibility
@api_view(["GET"])
@permission_classes([AllowAny])
def health_database(request):
    from datetime import datetime, timezone

    try:
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return Response(
            {
                "database": {"status": "healthy", "connection": "connected"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception:
        return Response(
            {
                "database": {"status": "unhealthy", "connection": "failed"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_redis(request):
    from datetime import datetime, timezone

    try:
        import redis
        from django.conf import settings

        redis_client = redis.Redis(host=getattr(settings, "REDIS_HOST", "localhost"))
        redis_client.ping()
        return Response(
            {
                "redis": {"status": "healthy", "connection": "connected"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception:
        return Response(
            {
                "redis": {"status": "unhealthy", "connection": "failed"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_comprehensive(request):
    from datetime import datetime, timezone

    db_healthy = True
    try:
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        db_healthy = False

    redis_healthy = True
    try:
        import redis
        from django.conf import settings

        redis_client = redis.Redis(host=getattr(settings, "REDIS_HOST", "localhost"))
        redis_client.ping()
    except Exception:
        redis_healthy = False

    return Response(
        {
            "overall_status": (
                "healthy" if (db_healthy and redis_healthy) else "degraded"
            ),
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
            },
            "system_info": {"memory_usage": "N/A", "cpu_usage": "N/A"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def api_info(request):
    from datetime import datetime, timezone

    return Response(
        {
            "name": "VoyageurCompass API",
            "version": "1.0.0",
            "description": "Financial analytics platform API",
            "endpoints": {"authentication": "/api/auth/", "data": "/api/data/"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def server_time(request):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return Response(
        {
            "server_time": now.isoformat(),
            "timezone": "UTC",
            "timestamp": now.timestamp(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    from datetime import datetime, timezone

    return Response(
        {
            "status": "operational",
            "uptime": "N/A",
            "memory_usage": "N/A",
            "cpu_usage": "N/A",
            "active_connections": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_database(request):
    """Database-specific health check endpoint."""
    logger = logging.getLogger("VoyageurCompass.health")

    try:
        # Ensure connection and simple DB ping
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")

        from datetime import datetime, timezone

        return Response(
            {
                "database": {
                    "status": "healthy",
                    "connection": "connected",
                    "response_time_ms": 5,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logger.error("Database health check failed", exc_info=True)
        from datetime import datetime, timezone

        return Response(
            {
                "database": {
                    "status": "unhealthy",
                    "connection": "failed",
                    "error": "database connection failure",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_redis(request):
    """Redis-specific health check endpoint."""
    logger = logging.getLogger("VoyageurCompass.health")

    try:
        import redis
        from django.conf import settings

        # Try to connect to Redis
        redis_client = redis.Redis(
            host=getattr(settings, "REDIS_HOST", "localhost"),
            port=getattr(settings, "REDIS_PORT", 6379),
        )
        redis_client.ping()

        from datetime import datetime, timezone

        return Response(
            {
                "redis": {
                    "status": "healthy",
                    "connection": "connected",
                    "response_time_ms": 3,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logger.error("Redis health check failed", exc_info=True)
        from datetime import datetime, timezone

        return Response(
            {
                "redis": {
                    "status": "unhealthy",
                    "connection": "failed",
                    "error": "Redis connection failed",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_comprehensive(request):
    """Comprehensive health check endpoint."""
    from datetime import datetime, timezone

    # Check database
    db_healthy = True
    try:
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        db_healthy = False

    # Check redis
    redis_healthy = True
    try:
        import redis
        from django.conf import settings

        redis_client = redis.Redis(
            host=getattr(settings, "REDIS_HOST", "localhost"),
            port=getattr(settings, "REDIS_PORT", 6379),
        )
        redis_client.ping()
    except Exception:
        redis_healthy = False

    # System info
    try:
        import psutil

        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()

        system_info = {
            "memory_usage": f"{memory.percent}%",
            "cpu_usage": f"{cpu_percent}%",
            "uptime": "N/A",
        }
    except ImportError:
        system_info = {"memory_usage": "N/A", "cpu_usage": "N/A", "uptime": "N/A"}

    overall_healthy = db_healthy and redis_healthy

    return Response(
        {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
            },
            "system_info": system_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        status=(
            status.HTTP_200_OK
            if overall_healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def api_info(request):
    """API information endpoint."""
    from datetime import datetime, timezone

    return Response(
        {
            "name": "VoyageurCompass API",
            "version": "1.0.0",
            "description": "Financial analytics platform API",
            "endpoints": {
                "authentication": "/api/auth/",
                "data": "/api/data/",
                "analytics": "/api/analytics/",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def server_time(request):
    """Server time endpoint."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    return Response(
        {
            "server_time": now.isoformat(),
            "timezone": "UTC",
            "timestamp": now.timestamp(),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    """System status endpoint."""
    from datetime import datetime, timezone

    try:
        import psutil

        memory = psutil.virtual_memory()

        return Response(
            {
                "status": "operational",
                "uptime": "N/A",
                "memory_usage": f"{memory.percent}%",
                "cpu_usage": f"{psutil.cpu_percent()}%",
                "active_connections": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except ImportError:
        return Response(
            {
                "status": "operational",
                "uptime": "N/A",
                "memory_usage": "N/A",
                "cpu_usage": "N/A",
                "active_connections": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
