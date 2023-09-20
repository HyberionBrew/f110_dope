import subprocess

# List of target policies
target_policies = [
    "progress_50k", "progress_100k", "progress_150k", "progress_200k", "progress_300k"
    #'raceline', 'min_lida', 'steering', 'td_progr', 'vel_chan', 'velocity','min_acti',
]

# Common options for the command
common_options = [
    'python', 'policy_eval/train_eval_f110_rl.py',
    '--no_behavior_cloning', '--algo=dr', '--num_updates=200_000', #,'--alternate_reward',
]

# Iterate over target policies and run the command
for policy in target_policies:
    command = common_options + ['--target_policy=' + policy]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command for policy '{policy}': {e}")