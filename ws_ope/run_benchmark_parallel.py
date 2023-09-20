#!/usr/bin/env python3

import subprocess
import concurrent.futures
import os

# Configuration
algo_values = ["dr","mb"]
target_policy_values = ['raceline', 'min_lida'] #, 'steering', 'td_progr', 'vel_chan', 'velocity','min_acti']
#policy_class = "trifinger_rl_example.example.TorchPushPolicy"
#std_values = ["0.05", "0.2"]
max_concurrent_processes = 3  # Set the number of concurrent processes here

# Create the logs directory if it doesn't exist
if not os.path.exists('logs_benchmark_parallel'):
    os.makedirs('logs_benchmark_parallel')

# Define a function to run a single job
def run_job(target_policy):
    num_updates = 200000
    
    # extra_flags = "--target_policy_noisy" if std != "0.05" else ""
    
    command = (
        f"python policy_eval/train_eval_f110_rl.py"
        f" --algo=dr --no_behavior_cloning --target_policy={target_policy}"
        f" --num_updates={num_updates} --alternate_reward --discount=0.85"
    )
    
    # Log file path
    log_file_path = f"logs_benchmark_parallel/output_algo_dr_{target_policy}.log"
    
    # Execute the command and redirect stdout and stderr to a file
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

# Using concurrent.futures to run multiple processes in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
    # Launch all the jobs
    futures = [
        executor.submit(run_job, target_policy)
        for target_policy in target_policy_values
    ]
    
    # Wait for all jobs to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Job raised an exception: {e}")