# ============================================================
# Web image: builds the React app and serves it with Nginx, which also
# reverse-proxies /caption and /health to the API container.
#
# Build context is the REPO ROOT (it needs ../nginx.conf):
#   docker build -f frontend/Dockerfile .
# ============================================================

# ---------- Stage 1: build the static site ---------------------------
FROM node:22-alpine AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build          # tsc (strict) + vite build -> /app/dist


# ---------- Stage 2: Nginx -------------------------------------------
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
