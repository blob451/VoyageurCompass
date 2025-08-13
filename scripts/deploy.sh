#!/bin/bash

# VoyageurCompass Blue-Green Deployment Script
# Usage: ./deploy.sh [blue|green] [version]

set -e

# Configuration
DEPLOYMENT_COLOR=${1:-green}
VERSION=${2:-latest}
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ ! -f ".env.production" ]; then
        log_error ".env.production file not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create backup
create_backup() {
    log_info "Creating database backup..."
    
    BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
    
    docker-compose -f docker-compose.blue.yml exec -T db pg_dump \
        -U ${DB_USER} ${DB_NAME} > backups/${BACKUP_FILE}
    
    if [ $? -eq 0 ]; then
        log_success "Backup created: ${BACKUP_FILE}"
        echo ${BACKUP_FILE} > .last_backup
    else
        log_error "Failed to create backup"
        exit 1
    fi
}

# Deploy new version
deploy() {
    local color=$1
    local version=$2
    
    log_info "Deploying ${version} to ${color} environment..."
    
    # Set environment variable for version
    export ${color^^}_VERSION=${version}
    
    # Pull new images
    log_info "Pulling images..."
    docker-compose -f docker-compose.${color}.yml pull
    
    # Start new containers
    log_info "Starting ${color} containers..."
    docker-compose -f docker-compose.${color}.yml up -d
    
    # Run migrations
    log_info "Running database migrations..."
    docker-compose -f docker-compose.${color}.yml exec -T web_${color} \
        python manage.py migrate --no-input
    
    # Collect static files
    log_info "Collecting static files..."
    docker-compose -f docker-compose.${color}.yml exec -T web_${color} \
        python manage.py collectstatic --no-input
    
    log_success "Deployment to ${color} completed"
}

# Health check
health_check() {
    local color=$1
    local port=$2
    
    log_info "Performing health checks on ${color} environment..."
    
    for i in $(seq 1 ${HEALTH_CHECK_RETRIES}); do
        if curl -f http://localhost:${port}/api/health/ &> /dev/null; then
            log_success "Health check passed (attempt ${i}/${HEALTH_CHECK_RETRIES})"
            return 0
        else
            log_warning "Health check failed (attempt ${i}/${HEALTH_CHECK_RETRIES})"
            sleep ${HEALTH_CHECK_INTERVAL}
        fi
    done
    
    log_error "Health checks failed after ${HEALTH_CHECK_RETRIES} attempts"
    return 1
}

# Switch traffic
switch_traffic() {
    local target_color=$1
    
    log_info "Switching traffic to ${target_color}..."
    
    # Update nginx/traefik configuration
    if [ "${target_color}" == "green" ]; then
        sed -i 's/upstream backend { server web_blue/upstream backend { server web_green/' nginx/nginx.conf
        sed -i 's/upstream frontend { server frontend_blue/upstream frontend { server frontend_green/' nginx/nginx.conf
    else
        sed -i 's/upstream backend { server web_green/upstream backend { server web_blue/' nginx/nginx.conf
        sed -i 's/upstream frontend { server frontend_green/upstream frontend { server frontend_blue/' nginx/nginx.conf
    fi
    
    # Reload nginx
    docker exec voyageur_prod_nginx nginx -s reload
    
    log_success "Traffic switched to ${target_color}"
}

# Rollback
rollback() {
    log_warning "Initiating rollback..."
    
    # Get last backup file
    if [ -f ".last_backup" ]; then
        BACKUP_FILE=$(cat .last_backup)
        
        log_info "Restoring database from ${BACKUP_FILE}..."
        docker-compose -f docker-compose.blue.yml exec -T db \
            psql -U ${DB_USER} ${DB_NAME} < backups/${BACKUP_FILE}
    fi
    
    # Switch traffic back
    if [ "${DEPLOYMENT_COLOR}" == "green" ]; then
        switch_traffic "blue"
        docker-compose -f docker-compose.green.yml down
    else
        switch_traffic "green"
        docker-compose -f docker-compose.blue.yml down
    fi
    
    log_success "Rollback completed"
}

# Cleanup old containers
cleanup() {
    local old_color=$1
    
    log_info "Cleaning up ${old_color} environment..."
    
    docker-compose -f docker-compose.${old_color}.yml down
    
    # Remove old images (keep last 3 versions)
    docker images voyageurcompass/backend --format "{{.Tag}}" | \
        sort -r | tail -n +4 | \
        xargs -I {} docker rmi voyageurcompass/backend:{} || true
    
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    log_info "Starting deployment process..."
    log_info "Target: ${DEPLOYMENT_COLOR} environment"
    log_info "Version: ${VERSION}"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Deploy to target environment
    deploy ${DEPLOYMENT_COLOR} ${VERSION}
    
    # Determine ports based on color
    if [ "${DEPLOYMENT_COLOR}" == "green" ]; then
        PORT=8001
        OLD_COLOR="blue"
    else
        PORT=8000
        OLD_COLOR="green"
    fi
    
    # Health check
    if health_check ${DEPLOYMENT_COLOR} ${PORT}; then
        # Switch traffic
        switch_traffic ${DEPLOYMENT_COLOR}
        
        # Final health check after switch
        sleep 10
        if health_check ${DEPLOYMENT_COLOR} ${PORT}; then
            log_success "Deployment successful!"
            
            # Cleanup old environment
            read -p "Do you want to stop the ${OLD_COLOR} environment? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cleanup ${OLD_COLOR}
            fi
        else
            log_error "Post-switch health check failed"
            rollback
            exit 1
        fi
    else
        log_error "Pre-switch health check failed"
        rollback
        exit 1
    fi
    
    log_success "Deployment process completed successfully!"
}

# Handle errors
trap 'log_error "Deployment failed!"; rollback; exit 1' ERR

# Run main function
main