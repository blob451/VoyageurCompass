# Use official Node.js runtime as base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Set environment variable to prevent interactive prompts
ENV CI=true

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy application files
COPY . .

# Create a non-root user to run the application (use node user that already exists)
# The node:alpine image already has a 'node' user with UID 1000
RUN chown -R node:node /app

# Switch to non-root user
USER node

# Expose port 3000 (configured in vite.config.js)
EXPOSE 3000

# Start Vite development server with host flag to allow external connections
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]