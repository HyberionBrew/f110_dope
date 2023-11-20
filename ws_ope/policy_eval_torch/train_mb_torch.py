# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Run off-policy evaluation training loop."""

from absl import app
from absl import flags
from absl import logging
from stable_baselines3 import PPO # this needs to be imported before tensorflow
from policy_eval_torch.model_based_2 import ModelBased2

import gc
import json
import os
import pickle


import f110_gym
import f110_orl_dataset
import gymnasium as gym

print	("hi")
from gymnasium.wrappers import time_limit
import numpy as np

# from tf_agents.environments import gym_wrapper
# from tf_agents.environments import suite_mujoco
import tqdm


from tensorboardX import SummaryWriter
# from ftg_agents.agents import *

from f110_orl_dataset.normalize_dataset import Normalize
from f110_orl_dataset.dataset_agents import F110Actor,F110Stupid

# import torch dataloader
from torch.utils.data import DataLoader

import os
import sys
import torch

from policy_eval_torch.dataset import F110Dataset


EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'f110_gym', 'Name of the environment.')
flags.DEFINE_string('target_policy', 'progress_weight', 'Name of target agent')
flags.DEFINE_float('speed', 1.0, 'Mean speed of the car, for the agent') 
#flags.DEFINE_string('d4rl_policy_filename', None,
#                    'Path to saved pickle of D4RL policy.')
#flags.DEFINE_string('trifinger_policy_class', "trifinger_rl_example.example.TorchPushPolicy",
#                    'Policy class name for Trifinger.')
flags.DEFINE_bool('load_mb_model', False, 'Whether to load a model-based model.')
flags.DEFINE_integer('seed', 1, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-5, 'Critic learning rate.')
flags.DEFINE_float('weight_decay', 1e-7, 'Weight decay.') # 1e-5
flags.DEFINE_float('behavior_policy_std', None,
                   'Noise scale of behavior policy.')
flags.DEFINE_float('target_policy_std', 0.0, 'Noise scale of target policy.')
flags.DEFINE_bool('target_policy_noisy', False, 'inject noise into the actions of the target policy')
# flags.DEFINE_integer('num_trajectories', 1000, 'Number of trajectories.') # this is not actually used
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')

flags.DEFINE_integer('num_updates', 1_000_000, 'Number of updates.')
flags.DEFINE_integer('eval_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_string('save_dir', 'logdir_torch',
                    'Directory to save results to.')
flags.DEFINE_string(
    'data_dir',
    '/tmp/policy_eval/trajectory_datasets/',
    'Directory with data for evaluation.')
flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
flags.DEFINE_boolean('bootstrap', True,
                     'Whether to generated bootstrap weights.')
flags.DEFINE_float('noise_scale', 0.0, 'Noise scaling for data augmentation.') # 0.25
flags.DEFINE_string('model_path', None, 'Path to saved model.')
flags.DEFINE_bool('no_behavior_cloning', False, 'Whether to use behavior cloning')
flags.DEFINE_bool('alternate_reward', False, 'Whether to use alternate reward')
flags.DEFINE_string('path', "trajectories.zarr", "The reward dataset to use")
flags.DEFINE_bool('use_torch', False, 'Whether to use torch (which is the new model)')

def make_hparam_string(json_parameters=None, **hparam_str_dict):
  if json_parameters:
    for key, value in json.loads(json_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])

def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def main(_):
  
  np.random.seed(FLAGS.seed)
  # assert not FLAGS.d4rl and FlAGS.trifinger
  
  import datetime
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d-%H-%M-%S")

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name, algo='mb',
      target_policy=FLAGS.target_policy, 
      std=FLAGS.target_policy_std, time=time, target_policy_noisy=FLAGS.target_policy_noisy, noise_scale=FLAGS.noise_scale)

  writer = SummaryWriter(log_dir= os.path.join(FLAGS.save_dir, f"f110_rl_{FLAGS.discount}_mb_{FLAGS.path}_1411", hparam_str))
  subsample_laser = 20
  
  F110Env = gym.make('f110_with_dataset-v0',
  # only terminals are available as of right now 
      **dict(name='f110_with_dataset-v0',
          config = dict(map="Infsaal", num_agents=1,
          params=dict(vmin=0.5, vmax=2.0)),
            render_mode="human")
  )
  env = F110Env

  behavior_dataset = F110Dataset(
      env,
      normalize_states=FLAGS.normalize_states,
      normalize_rewards=FLAGS.normalize_rewards,
      path = f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
      exclude_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight'],#['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
      alternate_reward=FLAGS.alternate_reward,
      include_timesteps_in_obs = True,)
  eval_datasets = []
  
  eval_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight']
  print("means and stds")
  print(behavior_dataset.reward_mean, behavior_dataset.reward_std,
        behavior_dataset.state_mean,
      behavior_dataset.state_std,)
  
  for i, agent in enumerate(eval_agents):
    evaluation_dataset = F110Dataset(
      env,
      normalize_states=FLAGS.normalize_states,
      normalize_rewards=FLAGS.normalize_rewards,
      #noise_scale=FLAGS.noise_scale,
      #bootstrap=FLAGS.bootstrap,
      #debug=False,
      path = f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
      only_agents = [agent], #['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
      #scans_as_states=False,
      alternate_reward=FLAGS.alternate_reward,
      include_timesteps_in_obs = True,
      reward_mean = behavior_dataset.reward_mean,
      reward_std = behavior_dataset.reward_std,
      state_mean = behavior_dataset.state_mean,
      state_std = behavior_dataset.state_std,
      )
    eval_datasets.append(evaluation_dataset)
      

  print("Finished loading F110 Dataset")
  
  dataloader = DataLoader(behavior_dataset, batch_size=FLAGS.sample_batch_size, shuffle=True)
  inf_dataloader = get_infinite_iterator(dataloader)

  data_iter = iter(inf_dataloader)
  min_state = behavior_dataset.states.min(axis=0)[0]
  max_state = behavior_dataset.states.max(axis=0)[0]
  print(min_state)  
  print(max_state)
  print("marker 1")
  model = ModelBased2(behavior_dataset.states.shape[1],
                    env.action_spec().shape[1], [256,256,256,256], 
                    dt=1/20, 
                    min_state=min_state, 
                    max_state=max_state, 
                    logger=writer, 
                    dataset=behavior_dataset,
                    fn_normalize=behavior_dataset.normalize_states,
                    fn_unnormalize=behavior_dataset.unnormalize_states,
                    learning_rate=FLAGS.lr,
                    weight_decay=FLAGS.weight_decay,
                    target_reward="trajectories_raceline.zarr")
  
  if FLAGS.load_mb_model:
    model.load("/home/fabian/msc/f110_dope/ws_ope/logdir/mb/mb_model_110000", "new_model")


  #print(min_state)
  #print(max_state)

  actor = F110Actor(FLAGS.target_policy, deterministic=True) #F110Stupid()
  model_input_normalizer = Normalize()

  def get_target_actions(states, scans= None, batch_size=5000):
    num_batches = int(np.ceil(len(states) / batch_size))
    actions_list = []
    # batching, s.t. we dont run OOM
    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, len(states))
      batch_states = states[start_idx:end_idx].clone()

      # unnormalize from the dope dataset normalization
      batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states) # this needs batches
      del batch_states
      batch_states_unnorm = batch_states_unnorm.cpu().numpy()

      # get scans
      if scans is not None:
        laser_scan = scans[start_idx:end_idx]
      else:
        laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
        #print("Scan 1")
        #print(laser_scan)
        laser_scan = model_input_normalizer.normalize_laser_scan(laser_scan)
        #print("Scan 2")
        #print(laser_scan)
      # back to dict
      model_input_dict = model_input_normalizer.unflatten_batch(batch_states_unnorm)
      # normalize back to model input
      model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
     
      # now also append the laser scan
      model_input_dict['lidar_occupancy'] = laser_scan
      #print("model input dict")
      #print(model_input_dict)
      #print(model_input_dict)
      batch_actions = actor(
        model_input_dict,
        std=FLAGS.target_policy_std)[1]
      #print(batch_actions)
      
      actions_list.append(batch_actions)
    # tf.concat(actions_list, axis=0)
    # with torch
    # convert to torch tensor
    actions_list = [torch.from_numpy(action) for action in actions_list]
    actions = torch.concat(actions_list, axis=0)
    # print(actions)
    return actions


  #@tf.function
  def update_step():
    import time

    (states, scans, actions, next_states, next_scans, rewards, masks, weights,
     log_prob, timesteps) = next(data_iter)
    #print(behavior_dataset.states[0])
    #print(behavior_dataset.actions[0])
    #x = behavior_dataset.states[0].unsqueeze(0)
    #print(x)
    #scans = behavior_dataset.scans[0].unsqueeze(0)
    #inferred = get_target_actions(x, scans=scans)
    #print("--------")
    #print(inferred)
    #print(behavior_dataset.actions[0])


    #exit()
    #if not(FLAGS.load_mb_model):
    model.update(states, actions, next_states, rewards, masks,
                  weights)

  gc.collect()

  for i in tqdm.tqdm(range(FLAGS.num_updates), desc='Running Training',  mininterval=5.0):
    #indices = np.where(behavior_dataset.mask_inital == 1)[0][:5]
    #print(indices)
    # print(tf.tensor(indices))
    #selected_states = tf.gather(behavior_dataset.states, indices)
    #print(selected_states)
    #print(behavior_dataset.initial_states[:5])
    if not FLAGS.load_mb_model:
      update_step()

    if i % FLAGS.eval_interval == 0:
      horizon = 500
      print("Starting evaluation")
      if False:

        for j, evaluation_dataset in enumerate(eval_datasets):
          eval_ds = model.evaluate_fast(evaluation_dataset,
                                  behavior_dataset.unnormalize_rewards,
                                  horizon=50, num_samples=512, 
                                  get_target_action = None,
                                  tag = eval_agents[j])
      
      print("*returns*")
      pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
                             behavior_dataset.initial_weights,
                             get_target_actions, horizon=50,
                             discount=FLAGS.discount,)

      print("pred returns")
      print(pred_returns)
      print("std")
      print(std)
      

      model.evaluate_fast(eval_datasets[0], 
                          behavior_dataset.unnormalize_rewards,
                          horizon=50, num_samples=512, 
                          get_target_action = get_target_actions,
                          tag="with_target_actions")
      
      model.plot_rollouts_fast(eval_datasets[0],
                               behavior_dataset.unnormalize_rewards,
                          horizon=50, num_samples=15, 
                          get_target_action = None,
                          path= f"plts/{i}.png")
      
      model.plot_rollouts_fast(eval_datasets[0],
                               behavior_dataset.unnormalize_rewards,
                          horizon=50, num_samples=15, 
                          get_target_action = get_target_actions,
                          path= f"plts/{i}_target_actions.png")

      #model.save(f"/app/ws/logdir/mb/mb_model_{i}", "new_model")
      # print saved model
      # print(f"saved model as /app/ws/logdir/mb/mb_model_{i}")

app.run(main)
if __name__ == '__main__':
  print("Running main")
  app.run(main)
