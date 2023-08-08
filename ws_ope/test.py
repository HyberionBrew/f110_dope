import f110_gym
import f110_orl_dataset
import gym
from f110_gym.envs.base_classes import Integrator

# Assuming f110_orl_dataset has already updated F110Env with get_dataset
env = gym.make('f110_with_dataset-v0',
               flatten_obs=True,
               flatten_acts=True,
               laser_obs=False,
               flatten_trajectories=True,
               subsample_laser=10,
               max_trajectories=5, **dict(name='f110_with_dataset-v0',map="/mnt/hdd2/fabian/ws_f1tenth/f1tenth_gym/examples/example_map", map_ext=".png", num_agents=1, timestep=0.01, integrator=Integrator.RK4))
#x = f110_orl_dataset.F1tenthDatasetEnv("dw", dict())
#x.get_dataset()
# This should now be possible
tr = env.get_dataset(zarr_path="/mnt/hdd2/fabian/f1tenth/f1tenth_orl_datasets/collect_dataset/trajectories.zarr")
#print(tr["observations"][0:10])
#env.get_action_space()
#print(env.action_space)
#print(env.observation_space)
print(env.observation_space.shape)
print(env.action_space.shape)
print(".........")
print(env.observation_spec().shape[0])
print(env.action_spec().shape[0])