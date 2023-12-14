# Install

Make sure you are on the branch 'release_new'.

After cloning init submodules:
`git submodule update --init --recursive`

First create a conda env:
`conda env create -f environment.yml`

Checkout the conda env:
`conda activate release`

Install the following:

`pip install -e f1tenth_gym`

`pip install -e f1tenth_orl_dataset`

`pip install -e f110_sim_env`

`pip install -e stochastic_ftg_agents`



To test if it works, just run the run_ope.ipynb

# Run
If you dont want to create your own dataset skip to step 2).

1) You will need a dataset, either create this by following the create_datatset notebook or get one from me.

2) In explore_dataset notebook you can look at the data, some examples are there - however, to use the plotting utilities library you currently need tf (not in the conda env) - ups, need to change that.

3) If you want to change a reward, the original reward is never overwritten, instead you set the flag --alternate_reward to true. Computing the alternate reward can like this (in the f1tenth_orl_dataset/f110_orl_dataset folder):
`python relabel_reward.py --reward_config=reward_sparse.json --path=/home/fabian/msc/f110_dope/ws_release/datasets_1412.zarr`

4) In run_ope you can definitely run two OPE methods (FQE and DR). The others are maybe not running and not tested in this new release enviroment. I changed the reward config so MB is most likely not working right now.

# Some Info

Agents can be added in stochastic_ftg_agents (the name is miss-leading, all agents should be there soon). Agents should output a value between -1 and 1 for two actions (steering delta, velocity delta).

If you want to change how to -1 and 1 is interpreted it has to be changed in f110_sim_env, but there should be an argument for it.

Many of the OPE methods are not tested right now, no clue if they run currently (if they don't its just small things, that I haven't yet adapted)
