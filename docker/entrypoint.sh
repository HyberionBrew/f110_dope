#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/app/ws
cd /app/ws
pip3 install -e ftg_agents/
pip3 install -e f1tenth_orl_dataset/

# Run whatever command was passed to the Docker container
# (If you always run a specific command, place it here. Otherwise, the "$@" will execute any command passed to the Docker run command)
exec "/bin/bash"
