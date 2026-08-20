FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html tsconfig.json tsconfig.app.json vite.config.mjs ./
COPY src ./src

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_AUTH_REQUIRED=false
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_AUTH_REQUIRED=${VITE_AUTH_REQUIRED}
RUN npm run build

FROM node:22-alpine AS runtime

ENV NODE_ENV=production
WORKDIR /app
COPY --from=build --chown=node:node /app/package.json /app/package-lock.json ./
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/vite.config.mjs ./vite.config.mjs
COPY --from=build --chown=node:node /app/dist ./dist

USER node
EXPOSE 5173
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1:5173/ || exit 1

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
