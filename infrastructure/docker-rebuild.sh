#!/bin/bash
# VoyageurCompass Docker Rebuild Script

echo "🔄 Rebuilding VoyageurCompass Docker Services..."
echo ""

# Change to the infrastructure directory
cd "$(dirname "$0")"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "🛑 Stopping existing services..."
docker-compose down

echo ""
echo "🧹 Cleaning up old images..."
docker-compose build --no-cache --force-rm

echo ""
echo "🏗️  Rebuilding all services..."
docker-compose up -d --build --force-recreate

echo ""
echo "⏳ Waiting for services to stabilize..."
sleep 15

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Check for any failed services
FAILED_SERVICES=$(docker-compose ps --filter "status=exited" -q)
if [ -n "$FAILED_SERVICES" ]; then
    echo ""
    echo "❌ Some services failed to start:"
    docker-compose ps --filter "status=exited"
    echo ""
    echo "📜 Check logs with: docker-compose logs [service_name]"
else
    echo ""
    echo "✅ All services rebuilt and running successfully!"
    echo ""
    echo "🌐 Service URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  Django Admin: http://localhost:8000/admin/"
    echo "  Ollama: http://localhost:11434"
fi

echo ""
echo "📜 To view logs: docker-compose logs -f [service_name]"
echo "🛑 To stop services: ./docker-stop.sh"