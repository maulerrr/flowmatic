FROM oven/bun:1.2-alpine

WORKDIR /app

COPY package.json tsconfig.json ./
COPY src ./src

ENV PORT=8092
ENV HOST=0.0.0.0

EXPOSE 8092

HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -qO- http://127.0.0.1:8092/health || exit 1

CMD ["bun", "src/index.ts"]
