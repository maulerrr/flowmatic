#!/bin/bash

# ───────────────────────────────────────────────────────────
#  NestJS + Prisma Dev Runner
# ───────────────────────────────────────────────────────────

echo ""
echo "================================================"
echo "      🛠  NestJS + Prisma – Development Startup"
echo "================================================"
echo ""

# Function to handle errors
handle_error() {
  echo ""
  echo "================================================"
  echo "  ❌ ERROR: Step failed with exit code $?."
  echo "================================================"
  exit 1
}

# 1) Install dependencies
echo "[1/4] 📦 Installing dependencies..."
bun install || handle_error

# 2) Generate Prisma client
echo ""
echo "[2/4] 🔄 Generating Prisma client..."
bun run prisma:generate || handle_error

# 3) Run migrations
echo ""
echo "[3/4] 🚧 Running database migrations..."
bun run prisma:migrate || handle_error
bun run prisma:seed || handle_error

# 4) Start dev server
echo ""
echo "[4/4] 🚀 Launching NestJS in dev mode..."
bun run dev || handle_error

echo ""
echo "================================================"
echo "   ✅ All steps completed successfully!"
echo "================================================"

echo "Press Enter to exit..."
read -r # Waits for user to press Enter

exit 0