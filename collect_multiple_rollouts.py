#!/usr/bin/env python3

import subprocess
import concurrent.futures
import os
import argparse
import glob

# Parse the new command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--agent_config_folder', type=str, required=True, help="Folder containing agent config files")
parser.add_argument('--reward_config', type=str, required=True, help="Reward config file")
parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
parser.add_argument('--dataset_folder', type=str, default="datasets", help="Dataset folder name")
parser.add_argument('--timesteps_per_agent', type=int, default=50_000, help="Number of timesteps per agent")
args = parser.parse_args()

# Create the logs directory if it doesn't exist
if not os.path.exists('logs_parallel'):
    os.makedirs('logs_parallel')

if not os.path.exists(args.dataset_folder):
    os.makedirs(args.dataset_folder)

# Get a list of all agent config files in the specified folder
agent_configs = glob.glob(os.path.join(args.agent_config_folder, '*.json'))
print(agent_configs)
# Define a function to run a single job
def run_job(agent_config):
    command = (
        f"python collect_rollouts.py --timesteps={args.timesteps_per_agent}"
        f" --agent_config={agent_config} --reward_config={args.reward_config}"
        f" --dataset={args.dataset_folder} --norender --record"
    )
    
    # Modify log file path to include agent config name
    agent_config_name = os.path.basename(agent_config).split('.')[0]
    log_file_path = f"logs_parallel/output_{agent_config_name}.log"
    
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

# Using concurrent.futures to run multiple processes in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    # Launch all the jobs
    futures = [executor.submit(run_job, agent_config) for agent_config in agent_configs]
    
    # Wait for all jobs to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Job raised an exception: {e}")