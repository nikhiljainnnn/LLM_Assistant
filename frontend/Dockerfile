# ── Frontend Dockerfile ──────────────────────────────────────────
# Multi-stage: Node builds the bundle, Nginx serves it.

# Stage 1 — build
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --frozen-lockfile

COPY . .

# Inject env at build time (can override with --build-arg)
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_API_KEY=dev-api-key
ARG VITE_APP_TITLE="AI Assistant"

ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_API_KEY=$VITE_API_KEY
ENV VITE_APP_TITLE=$VITE_APP_TITLE

RUN npm run build


# Stage 2 — serve with Nginx
FROM nginx:1.27-alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html

# Custom Nginx config: SPA routing + gzip
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -q -O /dev/null http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
