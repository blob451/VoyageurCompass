"""
Custom CORS middleware for VoyageurCompass.
"""

from django.utils.deprecation import MiddlewareMixin


class CustomCorsMiddleware(MiddlewareMixin):
    """
    Custom CORS middleware to handle cross-origin requests.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Add CORS headers
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Requested-With, Accept"
        )
        response["Access-Control-Max-Age"] = "3600"

        return response

    def process_request(self, request):
        """Handle preflight requests."""
        if request.method == "OPTIONS":
            response = self.get_response(request)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            response["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Requested-With, Accept"
            )
            response["Access-Control-Max-Age"] = "3600"
            return response
        return None
