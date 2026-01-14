@echo off
REM ───────────────────────────────────────────────────────────
REM  NestJS + Prisma Dev Runner
REM ───────────────────────────────────────────────────────────
title NestJS + Prisma Dev Runner
color 0A

echo.
echo ================================================
echo      🛠  NestJS + Prisma – Development Startup
echo ================================================
echo.

REM 1) Install dependencies
echo [1/4] 📦 Installing dependencies...
bun install
if errorlevel 1 goto :error

REM 2) Generate Prisma client
echo.
echo [2/4] 🔄 Generating Prisma client...
bun run prisma:generate
if errorlevel 1 goto :error

REM 3) Run migrations
echo.
echo [3/4] 🚧 Running database migrations...
bun run prisma:migrate
bun run prisma:seed
if errorlevel 1 goto :error

REM 4) Start dev server
echo.
echo [4/4] 🚀 Launching NestJS in dev mode...
bun run dev
if errorlevel 1 goto :error

echo.
echo ================================================
echo   ✅ All steps completed successfully!
echo ================================================
pause
exit /b 0

:error
echo.
echo ================================================
echo   ❌ ERROR: Step failed with exit code %errorlevel%.
echo ================================================
pause
exit /b %errorlevel%
