#!/bin/bash
# VoyageurCompass Docker Shutdown Script

echo "🛑 Stopping VoyageurCompass Docker Services..."
echo ""

# Change to the infrastructure directory
cd "$(dirname "$0")"

# Stop all services
echo "📴 Shutting down services..."
docker-compose down

echo ""
echo "🧹 Cleaning up..."

# Show remaining containers (if any)
CONTAINERS=$(docker-compose ps -q)
if [ -n "$CONTAINERS" ]; then
    echo "⚠️  Some containers are still running:"
    docker-compose ps
else
    echo "✅ All services stopped successfully"
fi

echo ""
echo "💾 Docker volumes preserved:"
docker volume ls | grep voyageur || echo "  No VoyageurCompass volumes found"

echo ""
echo "🔧 To remove all data volumes: docker-compose down -v"
echo "🧽 To clean up Docker system: docker system prune"
echo ""
echo "✅ VoyageurCompass services stopped!"