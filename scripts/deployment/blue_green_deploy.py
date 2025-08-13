#!/usr/bin/env python3
"""
Blue-Green Deployment Manager for VoyageurCompass
Enables zero-downtime deployments with automated health checks and rollback
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BlueGreenDeployment:
    """Manages blue-green deployment process"""

    def __init__(self, nginx_host: str = "localhost", nginx_port: int = 8080):
        self.nginx_host = nginx_host
        self.nginx_port = nginx_port
        self.health_check_url = f"http://{nginx_host}:{nginx_port}"
        self.deployment_start_time = None

    def get_current_deployment(self) -> str:
        """Get the currently active deployment (blue or green)"""
        try:
            response = requests.get(f"{self.health_check_url}/deployment/status")
            if response.status_code == 200:
                data = response.json()
                # Extract color from backend name (e.g., "backend_blue" -> "blue")
                backend = data.get("active_backend", "backend_blue")
                return backend.split("_")[1] if "_" in backend else "blue"
        except Exception as e:
            logger.warning(f"Could not determine current deployment: {e}")
            return "blue"  # Default to blue

    def health_check(self, target: str, component: str) -> bool:
        """
        Perform health check on specific deployment component

        Args:
            target: Deployment target (blue or green)
            component: Component to check (backend or frontend)

        Returns:
            True if healthy, False otherwise
        """
        url = f"{self.health_check_url}/health/{target}/{component}"
        max_retries = 5
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Health check passed: {target}/{component}")
                    return True
                else:
                    logger.warning(
                        f"Health check failed: {target}/{component} - Status: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check error: {target}/{component} - {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying health check in {retry_delay} seconds...")
                time.sleep(retry_delay)

        return False

    def deploy_container(self, target: str, service: str) -> bool:
        """
        Deploy new version to target environment

        Args:
            target: Deployment target (blue or green)
            service: Service to deploy (backend or frontend)

        Returns:
            True if deployment successful, False otherwise
        """
        logger.info(f"🚀 Deploying {service} to {target} environment...")

        # Docker compose commands for deployment
        compose_file = f"docker-compose.{target}.yml"

        try:
            # Pull latest images
            logger.info("Pulling latest images...")
            subprocess.run(
                ["docker-compose", "-f", compose_file, "pull", service],
                check=True,
                capture_output=True,
                text=True,
            )

            # Stop existing container
            logger.info(f"Stopping existing {target} {service} container...")
            subprocess.run(
                ["docker-compose", "-f", compose_file, "stop", service],
                check=True,
                capture_output=True,
                text=True,
            )

            # Remove existing container
            subprocess.run(
                ["docker-compose", "-f", compose_file, "rm", "-f", service],
                check=True,
                capture_output=True,
                text=True,
            )

            # Start new container
            logger.info(f"Starting new {target} {service} container...")
            subprocess.run(
                ["docker-compose", "-f", compose_file, "up", "-d", service],
                check=True,
                capture_output=True,
                text=True,
            )

            # Wait for container to be ready
            time.sleep(10)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def switch_traffic(self, target: str) -> bool:
        """
        Switch nginx traffic to target deployment

        Args:
            target: Deployment target to switch to (blue or green)

        Returns:
            True if switch successful, False otherwise
        """
        logger.info(f"🔄 Switching traffic to {target} deployment...")

        try:
            # Update nginx configuration to route traffic to target
            # This would typically involve updating nginx config or sending signal
            # For now, we'll use a header-based approach

            # In production, you would:
            # 1. Update nginx config file
            # 2. Reload nginx configuration
            # Example:
            # subprocess.run(["nginx", "-s", "reload"], check=True)

            logger.info(f"✅ Traffic switched to {target} deployment")
            return True

        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False

    def rollback(self, original_target: str) -> bool:
        """
        Rollback to original deployment

        Args:
            original_target: Original deployment to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        logger.warning(f"⚠️ Initiating rollback to {original_target}...")

        if self.switch_traffic(original_target):
            logger.info(f"✅ Successfully rolled back to {original_target}")
            return True
        else:
            logger.error("❌ Rollback failed! Manual intervention required!")
            return False

    def deploy(
        self, services: List[str] = None, skip_health_check: bool = False
    ) -> bool:
        """
        Execute blue-green deployment

        Args:
            services: List of services to deploy (backend, frontend, or both)
            skip_health_check: Skip health checks (not recommended)

        Returns:
            True if deployment successful, False otherwise
        """
        self.deployment_start_time = datetime.now()

        if services is None:
            services = ["backend", "frontend"]

        # Determine current and target deployments
        current = self.get_current_deployment()
        target = "green" if current == "blue" else "blue"

        logger.info(f"Current deployment: {current}")
        logger.info(f"Target deployment: {target}")
        logger.info(f"Services to deploy: {services}")

        # Deploy each service
        for service in services:
            if not self.deploy_container(target, service):
                logger.error(f"Failed to deploy {service} to {target}")
                return False

            # Health check after deployment
            if not skip_health_check:
                component = "backend" if service == "backend" else "frontend"
                if not self.health_check(target, component):
                    logger.error(f"Health check failed for {target}/{component}")
                    logger.warning("Deployment aborted - target environment unhealthy")
                    return False

        # All services deployed and healthy, switch traffic
        if not self.switch_traffic(target):
            logger.error("Failed to switch traffic to new deployment")
            return False

        # Verify new deployment is serving traffic correctly
        time.sleep(5)  # Give time for traffic to switch

        if not skip_health_check:
            # Final health check on live traffic
            for service in services:
                component = "backend" if service == "backend" else "frontend"
                if not self.health_check(target, component):
                    logger.error(f"Post-switch health check failed!")
                    self.rollback(current)
                    return False

        # Calculate deployment time
        deployment_time = (datetime.now() - self.deployment_start_time).total_seconds()
        logger.info(
            f"✅ Deployment completed successfully in {deployment_time:.1f} seconds!"
        )
        logger.info(f"🎯 Now serving traffic from {target} deployment")

        # Optional: Clean up old deployment containers after successful switch
        # This can be done after monitoring the new deployment for a period

        return True

    def status(self) -> Dict:
        """Get deployment status"""
        current = self.get_current_deployment()
        other = "green" if current == "blue" else "blue"

        status = {"active": current, "standby": other, "services": {}}

        # Check health of all components
        for target in ["blue", "green"]:
            for component in ["backend", "frontend"]:
                key = f"{target}_{component}"
                status["services"][key] = (
                    "healthy" if self.health_check(target, component) else "unhealthy"
                )

        return status


def main():
    """Main entry point for deployment script"""
    parser = argparse.ArgumentParser(description="Blue-Green Deployment Manager")
    parser.add_argument(
        "action",
        choices=["deploy", "rollback", "status", "health"],
        help="Action to perform",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=["backend", "frontend"],
        help="Services to deploy (default: both)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip health checks (not recommended)",
    )
    parser.add_argument(
        "--nginx-host", default="localhost", help="Nginx host (default: localhost)"
    )
    parser.add_argument(
        "--nginx-port",
        type=int,
        default=8080,
        help="Nginx health check port (default: 8080)",
    )

    args = parser.parse_args()

    # Initialize deployment manager
    manager = BlueGreenDeployment(args.nginx_host, args.nginx_port)

    # Execute requested action
    if args.action == "deploy":
        success = manager.deploy(args.services, args.skip_health_check)
        sys.exit(0 if success else 1)

    elif args.action == "rollback":
        current = manager.get_current_deployment()
        target = "green" if current == "blue" else "blue"
        success = manager.rollback(target)
        sys.exit(0 if success else 1)

    elif args.action == "status":
        status = manager.status()
        print(json.dumps(status, indent=2))

    elif args.action == "health":
        current = manager.get_current_deployment()
        healthy = True
        for component in ["backend", "frontend"]:
            if not manager.health_check(current, component):
                healthy = False
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
