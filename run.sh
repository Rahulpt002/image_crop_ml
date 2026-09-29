#!/bin/bash
# AI Crop Trainer — Development launcher
# Starts both backend (FastAPI) and frontend (Vite) servers

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}  AI Crop Trainer — Development Server${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Setup backend
echo -e "${BLUE}[1/4]${NC} Setting up Python virtual environment..."
if [ ! -d "backend/.venv" ]; then
    python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate

echo -e "${BLUE}[2/4]${NC} Installing Python dependencies..."
pip install -q -r backend/requirements.txt

# Setup frontend
echo -e "${BLUE}[3/4]${NC} Installing frontend dependencies..."
cd frontend
npm install --silent 2>/dev/null
cd ..

echo -e "${BLUE}[4/4]${NC} Starting servers..."
echo ""
echo -e "${GREEN}  Backend:${NC}  http://localhost:8001"
echo -e "${GREEN}  Frontend:${NC} http://localhost:5173"
echo -e "${GREEN}  API Docs:${NC} http://localhost:8001/docs"
echo ""
echo -e "Press ${RED}Ctrl+C${NC} to stop both servers."
echo ""

# Start backend
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# Start frontend
cd "$SCRIPT_DIR/frontend"
npm run dev -- --host &
FRONTEND_PID=$!

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${RED}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

wait
