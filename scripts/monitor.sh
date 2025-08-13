#!/bin/bash

# VoyageurCompass Deployment Monitoring Script
# Monitors the health and performance of the deployment

set -e

# Configuration
CHECK_INTERVAL=30
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
ALERT_THRESHOLD_RESPONSE_TIME=2000  # milliseconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
check_container_health() {
    local container=$1
    
    if docker ps --filter "name=${container}" --filter "health=healthy" | grep -q ${container}; then
        echo -e "${GREEN}✓${NC} ${container}: Healthy"
        return 0
    else
        echo -e "${RED}✗${NC} ${container}: Unhealthy"
        return 1
    fi
}

check_resource_usage() {
    local container=$1
    
    # Get CPU and memory usage
    local stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}" ${container} | tail -n 1)
    local cpu=$(echo $stats | awk '{print $2}' | sed 's/%//')
    local mem=$(echo $stats | awk '{print $3}' | sed 's/%//')
    
    # Check CPU
    if (( $(echo "$cpu > $ALERT_THRESHOLD_CPU" | bc -l) )); then
        echo -e "${RED}⚠${NC} ${container}: High CPU usage (${cpu}%)"
    else
        echo -e "${GREEN}✓${NC} ${container}: CPU usage normal (${cpu}%)"
    fi
    
    # Check Memory
    if (( $(echo "$mem > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
        echo -e "${RED}⚠${NC} ${container}: High memory usage (${mem}%)"
    else
        echo -e "${GREEN}✓${NC} ${container}: Memory usage normal (${mem}%)"
    fi
}

check_response_time() {
    local url=$1
    local name=$2
    
    # Measure response time
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' ${url})
    local response_ms=$(echo "$response_time * 1000" | bc)
    
    if (( $(echo "$response_ms > $ALERT_THRESHOLD_RESPONSE_TIME" | bc -l) )); then
        echo -e "${YELLOW}⚠${NC} ${name}: Slow response (${response_ms}ms)"
    else
        echo -e "${GREEN}✓${NC} ${name}: Response time normal (${response_ms}ms)"
    fi
}

check_database() {
    # Check database connections
    local connections=$(docker exec voyageur_prod_db psql -U postgres -d voyageur_compass -t -c \
        "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
    
    echo -e "${GREEN}✓${NC} Database: ${connections} active connections"
    
    # Check database size
    local db_size=$(docker exec voyageur_prod_db psql -U postgres -d voyageur_compass -t -c \
        "SELECT pg_size_pretty(pg_database_size('voyageur_compass'));")
    
    echo -e "${GREEN}✓${NC} Database size: ${db_size}"
}

check_redis() {
    # Check Redis memory usage
    local redis_info=$(docker exec voyageur_prod_redis redis-cli INFO memory | grep used_memory_human | cut -d: -f2)
    
    echo -e "${GREEN}✓${NC} Redis memory usage: ${redis_info}"
    
    # Check Redis connected clients
    local redis_clients=$(docker exec voyageur_prod_redis redis-cli INFO clients | grep connected_clients | cut -d: -f2)
    
    echo -e "${GREEN}✓${NC} Redis connected clients: ${redis_clients}"
}

check_logs() {
    local container=$1
    local error_count=$(docker logs --since 5m ${container} 2>&1 | grep -c ERROR || true)
    local warning_count=$(docker logs --since 5m ${container} 2>&1 | grep -c WARNING || true)
    
    if [ ${error_count} -gt 0 ]; then
        echo -e "${RED}✗${NC} ${container}: ${error_count} errors in last 5 minutes"
    fi
    
    if [ ${warning_count} -gt 5 ]; then
        echo -e "${YELLOW}⚠${NC} ${container}: ${warning_count} warnings in last 5 minutes"
    fi
}

check_disk_usage() {
    local usage=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ ${usage} -gt ${ALERT_THRESHOLD_DISK} ]; then
        echo -e "${RED}⚠${NC} Disk usage critical: ${usage}%"
    else
        echo -e "${GREEN}✓${NC} Disk usage: ${usage}%"
    fi
}

send_alert() {
    local message=$1
    
    # Send to Slack (if configured)
    if [ ! -z "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Alert: ${message}\"}" \
            ${SLACK_WEBHOOK}
    fi
    
    # Log to file
    echo "[$(date)] ALERT: ${message}" >> /var/log/voyageur_monitoring.log
}

# Main monitoring loop
main() {
    echo "========================================="
    echo "VoyageurCompass Deployment Monitor"
    echo "Time: $(date)"
    echo "========================================="
    
    # Determine active deployment
    if docker ps | grep -q "voyageur_prod_web_blue"; then
        ACTIVE_COLOR="blue"
        ACTIVE_PORT="8000"
    else
        ACTIVE_COLOR="green"
        ACTIVE_PORT="8001"
    fi
    
    echo -e "\nActive deployment: ${ACTIVE_COLOR}"
    echo -e "\n--- Container Health ---"
    
    # Check container health
    check_container_health "voyageur_prod_web_${ACTIVE_COLOR}"
    check_container_health "voyageur_prod_frontend_${ACTIVE_COLOR}"
    check_container_health "voyageur_prod_celery_${ACTIVE_COLOR}"
    check_container_health "voyageur_prod_db"
    check_container_health "voyageur_prod_redis"
    
    echo -e "\n--- Resource Usage ---"
    check_resource_usage "voyageur_prod_web_${ACTIVE_COLOR}"
    check_resource_usage "voyageur_prod_celery_${ACTIVE_COLOR}"
    check_resource_usage "voyageur_prod_db"
    check_resource_usage "voyageur_prod_redis"
    
    echo -e "\n--- Response Times ---"
    check_response_time "http://localhost:${ACTIVE_PORT}/api/health/" "API Health"
    check_response_time "http://localhost:3000/" "Frontend"
    
    echo -e "\n--- Database & Cache ---"
    check_database
    check_redis
    
    echo -e "\n--- System ---"
    check_disk_usage
    
    echo -e "\n--- Logs ---"
    check_logs "voyageur_prod_web_${ACTIVE_COLOR}"
    check_logs "voyageur_prod_celery_${ACTIVE_COLOR}"
    
    echo -e "\n========================================="
}

# Continuous monitoring
if [ "$1" == "--continuous" ]; then
    while true; do
        clear
        main
        sleep ${CHECK_INTERVAL}
    done
else
    main
fi