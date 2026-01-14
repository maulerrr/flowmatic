#!/bin/sh
# Substitute environment variables in the Nginx template
envsubst '$NGINX_SERVER_NAME' < /etc/nginx/nginx.template.conf > /etc/nginx/nginx.conf

# Execute the original command (nginx)
exec "$@"