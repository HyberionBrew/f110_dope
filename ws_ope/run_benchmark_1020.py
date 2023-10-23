#!/usr/bin/env python3

import subprocess
import concurrent.futures
import os

# Configuration
algo_values = ["dr"] #,"dr"]#, "mb"]
#assert(len(algo_values) == 1)
target_policy_values = ['progress_weight', 'raceline_delta_weigh', 'min_action_weight']#,'min_lidar_ray_weight' 'steering_change_weig','velocity_change_weig', 'velocity_weight'
datasets = ['trajectories_td_prog.zarr','trajectories_raceline.zarr', 'trajectories_min_act.zarr'] #'trajectories_raceline.zarr', 'trajectories_min_act.zarr']
#,

discount_values = [0.95,0.99]  # New list for discount values
num_runs_per_job = 3  # Number of times each job should be run
max_concurrent_processes = 3

# Create the logs directory if it doesn't exist
if not os.path.exists('logs_benchmark_parallel'):
    os.makedirs('logs_benchmark_parallel')

# Define a function to run a single job
def run_job(target_policy, discount, run_num, algo,path):
    num_updates = 300_000
    command = (
        f"python policy_eval/train_eval_f110_rl.py"
        f" --algo={algo} --no_behavior_cloning --target_policy={target_policy}"
        f" --num_updates={num_updates} --alternate_reward --discount={discount} --seed={run_num} --path={path}" #
        f" --eval_interval=50_000"
    )
    
    # Modify log file path to include discount value and run number
    log_file_path = f"logs_benchmark_parallel/output_algo_{algo}_{target_policy}_discount_{discount}_run_{run_num}.log"
    
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

# Using concurrent.futures to run multiple processes in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
    # Launch all the jobs
    futures = [
        executor.submit(run_job, target_policy, discount, run_num, algo,path)
        for target_policy in target_policy_values
        for discount in discount_values
        for algo in algo_values
        for path in datasets
        for run_num in range(1, num_runs_per_job + 1)  # Starting run_num from 1 for clarity in logs
    ]
    
    # Wait for all jobs to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Job raised an exception: {e}")
