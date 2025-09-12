#!/bin/bash
# VoyageurCompass Docker Startup Script

echo "🚀 Starting VoyageurCompass Docker Services..."
echo ""

# Change to the infrastructure directory
cd "$(dirname "$0")"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "⚠️  .env file not found in parent directory. Using default configuration."
    echo "   Consider copying .env.example to .env and updating values."
fi

echo "🔧 Building and starting services..."
docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to become healthy..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Service URLs:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  Django Admin: http://localhost:8000/admin/"
echo "  Ollama: http://localhost:11434"
echo ""
echo "📜 To view logs: docker-compose logs -f [service_name]"
echo "🛑 To stop services: ./docker-stop.sh"
echo ""
echo "✅ VoyageurCompass is starting up!"