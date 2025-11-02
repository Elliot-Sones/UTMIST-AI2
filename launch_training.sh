#!/bin/bash

# Launch Training Script
#
# This script automates the setup and launch of overnight RL training.
# It will:
# 1. Run pre-flight checks
# 2. (Optional) Run baseline evaluation
# 3. Start training in a persistent tmux session
# 4. Show monitoring commands
#
# Usage:
#   ./launch_training.sh                    # Launch with all checks
#   ./launch_training.sh --skip-checks      # Skip pre-flight (not recommended)
#   ./launch_training.sh --baseline         # Evaluate baseline first

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SESSION_NAME="rl_training"
TRAINING_SCRIPT="user/train_with_strategy_encoder.py"
LOG_FILE="training_output_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
SKIP_CHECKS=false
RUN_BASELINE=false

for arg in "$@"; do
    case $arg in
        --skip-checks)
            SKIP_CHECKS=true
            ;;
        --baseline)
            RUN_BASELINE=true
            ;;
        --help)
            echo "Usage: ./launch_training.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-checks    Skip pre-flight checks (not recommended)"
            echo "  --baseline       Run baseline evaluation before training"
            echo "  --help           Show this help message"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}"
echo "================================================================================"
echo "                    OVERNIGHT TRAINING LAUNCH SCRIPT"
echo "================================================================================"
echo -e "${NC}"

# Step 1: Pre-flight checks
if [ "$SKIP_CHECKS" = false ]; then
    echo -e "${BLUE}[1/4] Running pre-flight checks...${NC}"
    echo ""

    if python user/pre_flight_check.py; then
        echo ""
        echo -e "${GREEN}✓ All pre-flight checks passed!${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ Pre-flight checks failed!${NC}"
        echo "Please fix the issues before starting training."
        echo ""
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Skipping pre-flight checks (--skip-checks flag)${NC}"
    echo ""
fi

# Step 2: Baseline evaluation (optional)
if [ "$RUN_BASELINE" = true ]; then
    echo -e "${BLUE}[2/4] Running baseline evaluation...${NC}"
    echo ""

    # Check if model exists
    if [ -f "checkpoints/simplified_training/latest_model.zip" ]; then
        python user/evaluate_baseline.py --model checkpoints/simplified_training/latest_model.zip --episodes 10
        echo ""
        echo -e "${GREEN}✓ Baseline evaluation complete!${NC}"
        echo ""

        read -p "Press Enter to continue with training..."
    else
        echo -e "${YELLOW}⚠ No existing model found for baseline evaluation${NC}"
        echo "Skipping baseline evaluation"
        echo ""
    fi
else
    echo -e "${BLUE}[2/4] Skipping baseline evaluation${NC}"
    echo "(Run with --baseline flag to evaluate first)"
    echo ""
fi

# Step 3: Check if tmux is available
echo -e "${BLUE}[3/4] Checking for tmux...${NC}"

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ tmux is not installed!${NC}"
    echo ""
    echo "tmux is required for persistent training sessions."
    echo "Install it with:"
    echo "  macOS: brew install tmux"
    echo "  Linux: sudo apt-get install tmux"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ tmux is available${NC}"
echo ""

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Training session '$SESSION_NAME' already exists!${NC}"
    echo ""
    echo "Options:"
    echo "  1) Attach to existing session (resume)"
    echo "  2) Kill existing session and start new training"
    echo "  3) Cancel"
    echo ""
    read -p "Choose option (1/2/3): " choice

    case $choice in
        1)
            echo "Attaching to existing session..."
            exec tmux attach-session -t "$SESSION_NAME"
            ;;
        2)
            echo "Killing existing session..."
            tmux kill-session -t "$SESSION_NAME"
            ;;
        3)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid option. Cancelled."
            exit 1
            ;;
    esac
fi

# Step 4: Launch training
echo -e "${BLUE}[4/4] Launching training in tmux session...${NC}"
echo ""

# Create tmux session and start training
tmux new-session -d -s "$SESSION_NAME" "python $TRAINING_SCRIPT 2>&1 | tee $LOG_FILE"

sleep 2  # Give it time to start

# Check if session is still running (didn't immediately crash)
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${GREEN}✓ Training launched successfully!${NC}"
    echo ""
    echo "================================================================================"
    echo "                         TRAINING IS NOW RUNNING"
    echo "================================================================================"
    echo ""
    echo "Session: $SESSION_NAME"
    echo "Log file: $LOG_FILE"
    echo "Checkpoint dir: /tmp/strategy_encoder_training/"
    echo ""
    echo "MONITORING COMMANDS:"
    echo "-------------------"
    echo -e "${YELLOW}Attach to session:${NC}    tmux attach -t $SESSION_NAME"
    echo -e "${YELLOW}Detach from session:${NC}  Press Ctrl+B, then D"
    echo -e "${YELLOW}View live output:${NC}     tail -f $LOG_FILE"
    echo -e "${YELLOW}Check progress:${NC}       ls -lht /tmp/strategy_encoder_training/"
    echo ""
    echo "TENSORBOARD (if installed):"
    echo "---------------------------"
    echo "tensorboard --logdir /tmp/strategy_encoder_training/tb_logs"
    echo "Then open: http://localhost:6006"
    echo ""
    echo "================================================================================"
    echo ""
    echo "What happens next:"
    echo "  - Training will run for approximately 8-18 hours (5M steps)"
    echo "  - Checkpoints saved every 1M steps"
    echo "  - Metrics exported to CSV for analysis"
    echo "  - Session persists even if you disconnect"
    echo ""
    echo "To check on training later:"
    echo "  1. SSH back into this machine"
    echo "  2. Run: tmux attach -t $SESSION_NAME"
    echo ""
    echo -e "${GREEN}Good luck! The agent will learn while you sleep.${NC}"
    echo ""
    echo "Press Enter to see the training output (you'll be attached to the tmux session)"
    echo "To detach later: Press Ctrl+B, then D"
    read -p ""

    # Attach to session
    exec tmux attach-session -t "$SESSION_NAME"
else
    echo -e "${RED}✗ Training session failed to start!${NC}"
    echo "Check the error messages above."
    exit 1
fi
