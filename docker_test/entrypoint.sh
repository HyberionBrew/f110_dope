#!/bin/bash
# Initialize Conda for bash shell
source /opt/conda/etc/profile.d/conda.sh

# Activate the Conda environment named "tf"
conda activate tf

export PYTHONPATH=$PYTHONPATH:/app/ws
cd /app/ws

pip install -e ftg_agents/
pip install -e f1tenth_orl_dataset/
pip install -e f1tenth_gym/
# Run whatever command was passed to the Docker container
# (If you always run a specific command, place it here. Otherwise, the "$@" will execute any command passed to the Docker run command)
exec "/bin/bash"
