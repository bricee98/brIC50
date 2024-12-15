#!/bin/bash

# Name of the PM2 process
APP_NAME="DoseResponseServer"

# Path to the Python script
SCRIPT_PATH="run.py"

# Path to the virtual environment
VENV_PATH="./venv"

# Activate the virtual environment and start with PM2
source $VENV_PATH/bin/activate && pm2 start $SCRIPT_PATH --name $APP_NAME --interpreter python3 