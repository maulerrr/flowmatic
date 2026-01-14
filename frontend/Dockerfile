# ====================
# Build Stage
# ====================
FROM oven/bun:latest AS builder

# Set the working directory
WORKDIR /app

# Build arguments for passing env variables at build time
ARG VITE_API_URL
ARG NGINX_SERVER_NAME

# Copy only package files first for better caching
COPY package*.json bun.lock ./

# Install dependencies using bun
RUN bun install

# Copy the rest of the project files
COPY . .

# Inject environment variables for Vite build
ENV VITE_API_URL=$VITE_API_URL
ENV NGINX_SERVER_NAME=$NGINX_SERVER_NAME

# Build the production files
RUN bun run build

# ====================
# Final Stage (Nginx)
# ====================
FROM nginx:alpine AS final

# Remove default config and copy template
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.template.conf /etc/nginx/nginx.template.conf

# Copy static files
COPY --from=builder /app/dist /usr/share/nginx/html

# Add custom entrypoint script
COPY scripts/envsubst_nginx.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port and set entrypoint
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]