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
from policy_eval_torch.q_fitter import QFitter
from policy_eval_torch.doubly_robust import DR_estimator

import gc
import json
import os
import pickle


import f110_gym
import f110_orl_dataset
import gymnasium as gym


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
from policy_eval_torch.q_model import FQEMB

EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'f110_gym', 'Name of the environment.')
flags.DEFINE_string('algo', 'fqe', 'Name of the algorithm.')
flags.DEFINE_string('target_policy', 'progress_weight', 'Name of target agent')
flags.DEFINE_float('speed', 1.0, 'Mean speed of the car, for the agent') 
#flags.DEFINE_string('d4rl_policy_filename', None,
#                    'Path to saved pickle of D4RL policy.')
#flags.DEFINE_string('trifinger_policy_class', "trifinger_rl_example.example.TorchPushPolicy",
#                    'Policy class name for Trifinger.')
flags.DEFINE_bool('load_mb_model', False, 'Whether to load a model-based model.')
flags.DEFINE_integer('seed', 1, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-5, 'Critic learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
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
flags.DEFINE_integer('clip_trajectory_max', 0, 'Max trajectory length')
flags.DEFINE_bool('dr', False, 'Whether to use doubly robust')

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

def create_save_dir(experiment_directory):
    save_directory = os.path.join(FLAGS.save_dir, experiment_directory)
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the algo directory
    if not FLAGS.dr:
      save_directory = os.path.join(save_directory, f"{FLAGS.algo}")
    else:
      save_directory = os.path.join(save_directory, f"{FLAGS.algo}_dr")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # path
    save_directory = os.path.join(save_directory, f"{FLAGS.path}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    #now the max_timesteps directory
    save_directory = os.path.join(save_directory, f"{FLAGS.clip_trajectory_max}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the target policy directory
    save_directory = os.path.join(save_directory, f"{FLAGS.target_policy}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    return save_directory

def main(_):
  
  save_path = create_save_dir("test")
  

  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)
  # assert not FLAGS.d4rl and FlAGS.trifinger
  
  import datetime
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d-%H-%M-%S")

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name, algo='mb',
      target_policy=FLAGS.target_policy, 
      std=FLAGS.target_policy_std, time=time, target_policy_noisy=FLAGS.target_policy_noisy, noise_scale=FLAGS.noise_scale)

  writer = SummaryWriter(log_dir= os.path.join(save_path, hparam_str))
  subsample_laser = 20
  
  F110Env = gym.make('f110_with_dataset-v0',
  # only terminals are available as of right now 
      **dict(name='f110_with_dataset-v0',
          config = dict(map="Infsaal", num_agents=1,
          params=dict(vmin=0.5, vmax=2.0)),
            render_mode="human")
  )
  env = F110Env
  clip_trajectory_length = None
  if FLAGS.clip_trajectory_max > 0:
    clip_trajectory_length = (0,FLAGS.clip_trajectory_max)
  
  behavior_dataset = F110Dataset(
      env,
      normalize_states=FLAGS.normalize_states,
      normalize_rewards=False,#FLAGS.normalize_rewards,
      path = f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
      exclude_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight'],#['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
      alternate_reward=FLAGS.alternate_reward,
      include_timesteps_in_obs = True,
      only_terminals=True,
      clip_trajectory_length= clip_trajectory_length,
      )
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
      only_terminals=True,
      clip_trajectory_length= clip_trajectory_length,
      )
    eval_datasets.append(evaluation_dataset)
      

  print("Finished loading F110 Dataset")
  
  dataloader = DataLoader(behavior_dataset, batch_size=FLAGS.sample_batch_size, shuffle=True)
  inf_dataloader = get_infinite_iterator(dataloader)

  data_iter = iter(inf_dataloader)
  min_state = behavior_dataset.states.min(axis=0)[0]
  max_state = behavior_dataset.states.max(axis=0)[0]
  min_reward = behavior_dataset.rewards.min()
  max_reward = behavior_dataset.rewards.max()
  print(min_reward)
  print(max_reward)
  actor = F110Actor(FLAGS.target_policy, deterministic=False) #F110Stupid()
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

  def get_target_logprobs(states,actions,scans=None, batch_size=5000):
    num_batches = int(np.ceil(len(states) / batch_size))
    log_probs_list = []
    for i in range(num_batches):
      # print(i)
      # Calculate start and end indices for the current batch
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, len(states))
      # Extract the current batch of states
      batch_states = states[start_idx:end_idx]
      batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states)
      
      # Extract the current batch of actions
      batch_actions = actions[start_idx:end_idx]

      # get scans
      if scans is not None:
        laser_scan = scans[start_idx:end_idx]
      else:
        laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = model_input_normalizer.normalize_laser_scan(laser_scan)

      # back to dict
      model_input_dict = model_input_normalizer.unflatten_batch(batch_states_unnorm)
      # normalize back to model input
      model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
      # now also append the laser scan
      model_input_dict['lidar_occupancy'] = laser_scan

      # Compute log_probs for the current batch
      batch_log_probs = actor(
          model_input_dict,
          actions=batch_actions,
          std=FLAGS.target_policy_std)[2]
      
      # Sum along the last axis if the rank is greater than 1
      # print("len logprobs", print(batch_log_probs.shape))
      
      # Collect the batch_log_probs
      log_probs_list.append(batch_log_probs)
    # Concatenate the collected log_probs from all batches
    log_probs = [torch.from_numpy(log_prob) for log_prob in log_probs_list]
    log_probs = torch.concat(log_probs, axis=0)
    return log_probs

  if FLAGS.algo == "mb":
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
  if FLAGS.algo == "fqe":
    model = QFitter(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                    env.action_spec().shape[1], FLAGS.lr, FLAGS.weight_decay,
                    FLAGS.tau, 
                    use_time=True, 
                    timestep_constant = behavior_dataset.timestep_constant,
                    writer=writer)
    
    if True:
      fqe_load = f"/home/fabian/msc/f110_dope/ws_ope/logdir_torch/2011/fqe/{FLAGS.path}/{FLAGS.clip_trajectory_max}/{FLAGS.target_policy}"
      print("Loading from ", fqe_load)
      model.load(fqe_load,
                    i=0)
    #print(behavior_dataset.timestep_constant)
    #print(behavior_dataset.timesteps[0:20])
    #exit()
  if FLAGS.algo == "fqe_mb":
    model_mb = ModelBased2(behavior_dataset.states.shape[1],
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
                  target_reward=FLAGS.path)
    model_mb.load("/home/fabian/msc/f110_dope/ws_ope/logdir/mb/mb_model_110000", "new_model")

    model_fqe = QFitter(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                env.action_spec().shape[1], FLAGS.lr, FLAGS.weight_decay,
                FLAGS.tau, 
                use_time=True, 
                timestep_constant = behavior_dataset.timestep_constant,
                writer=writer)
    fqe_load = f"/home/fabian/msc/f110_dope/ws_ope/logdir_torch/2011/fqe/{FLAGS.path}/{FLAGS.clip_trajectory_max}/{FLAGS.target_policy}"
    print("Loading from ", fqe_load)
    model_fqe.load(fqe_load,
                   i=0)

    model = FQEMB(model_fqe, model_mb, FLAGS.discount, 
                  behavior_dataset.timestep_constant,
                  max_timestep=behavior_dataset.timesteps.max(),
                  rollouts=1,
                  mb_steps=5,
                  single_step_fqe = True,
                  min_reward=min_reward, max_reward=max_reward,
                  writer=writer,
                  target_actions = get_target_actions,
                  )
  if FLAGS.dr:
    print("enabled dr")
    dr_model = DR_estimator(model, behavior_dataset, FLAGS.discount)

  #else:
  #  raise ValueError(f"Unknown algo {FLAGS.algo}")

  if FLAGS.load_mb_model and FLAGS.algo =="mb":
    model.load("/home/fabian/msc/f110_dope/ws_ope/logdir/mb/mb_model_110000", "new_model")


  #print(min_state)
  #print(max_state)

  


  #@tf.function
  def update_step():
    import time

    (states, scans, actions, next_states, next_scans, rewards, masks, weights,
     log_prob, timesteps) = next(data_iter)
    if FLAGS.dr:
      # already trained
      pass

    elif FLAGS.algo == "mb":
      model.update(states, actions, next_states, rewards, masks,
                    weights)
    elif FLAGS.algo == "fqe":
      
      #weights = torch.ones
      #create debug batch, first states, actions...
      #next_actions = get_target_actions(next_states, scans=next_scans)
      """
      states = behavior_dataset.states[0].unsqueeze(0)
      actions = behavior_dataset.actions[0].unsqueeze(0)
      next_states = behavior_dataset.states_next[0].unsqueeze(0)
      # next_actions = behavior_dataset.actions_next[0].unsqueeze(0)
      next_scans = behavior_dataset.scans_next[0].unsqueeze(0)
      rewards = behavior_dataset.rewards[0].unsqueeze(0)
      masks = behavior_dataset.masks[0].unsqueeze(0)
      weights = weights[0].unsqueeze(0)
      next_actions = get_target_actions(next_states, scans=next_scans)
      
      print("min_reward",min_reward)
      print("max_reward",max_reward)
      print("states",states)
      print("actions", actions)
      print("next_states", next_states)
      print("next_actions", next_actions)
      print("rewards", rewards)
      print("masks", masks)
      print("weights", weights)
      print("timesteps", behavior_dataset.timesteps[0])
      print("mean_reward", behavior_dataset.reward_mean)
      print("std_reward", behavior_dataset.reward_std)
      print("rewards:", behavior_dataset.rewards[0:5])
      print("unnorm rewards:", behavior_dataset.unnormalize_rewards(behavior_dataset.rewards[0:5]))
      print("masks", behavior_dataset.masks[48:51])
      print("timesteps", behavior_dataset.timesteps[49:51])
      print("states", behavior_dataset.states[49:51])
      print(next_scans)
      exit()
      """
      
      next_actions = get_target_actions(next_states, scans=next_scans)
      #print("raw")
      #print(rewards.mean())
      max_r = (max_reward*max(FLAGS.clip_trajectory_max,50))
      #print(max_r)
      #print(min_reward)
      model.update(states, actions, next_states, 
                   next_actions, rewards, masks,
                    weights, FLAGS.discount, 
                    min_reward = min_reward, 
                    max_reward= max_r, 
                    timesteps=timesteps)
      #exit()
    elif FLAGS.algo == "fqe_mb":
      # in this case we are using already trained models!
      # so there is no updating involved
      pass 
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
      if FLAGS.dr:
        print("Starting DR")
        pred_return, pred_std = dr_model.estimate_returns(get_target_actions, get_target_logprobs)
        pred_returns = behavior_dataset.unnormalize_rewards(pred_return)
        std = behavior_dataset.unnormalize_rewards(pred_std)
        pred_returns *= (1-FLAGS.discount)
        std *= (1-FLAGS.discount)
        print("pred returns", pred_returns)
        print("std", std)
        print("############# DR finished ##############")
      if FLAGS.algo == "mb":
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
      if FLAGS.algo == "fqe":
        pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
                                behavior_dataset.initial_weights,
                            get_target_actions)
        #print("Raw predicted returns", pred_returns)
        pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
        print(std)
        std = behavior_dataset.unnormalize_rewards(std)
        pred_returns *= (1-FLAGS.discount)
        std *= (1-FLAGS.discount)
        print("normed,", pred_returns)
        # print(std)
        """
        for j, evaluation_dataset in enumerate(eval_datasets):
          eval_ds = model.estimate_returns(evaluation_dataset.initial_states,
                                  evaluation_dataset.initial_weights,
                                  get_action=get_target_actions,)
          print(f"eval ds {eval_agents[j]}: {eval_ds}")
        """
        model.save(save_path, i=0)
        print("saved as", save_path, 0)
      if FLAGS.algo == "fqe_mb":
        print("Running fqe mb")
        pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
                                behavior_dataset.initial_weights,
                            get_target_actions)
        # print raw, normalized and then discount
        print("Raw predicted returns", pred_returns)
        pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
        std = behavior_dataset.unnormalize_rewards(std)
        pred_returns *= (1-FLAGS.discount)
        std *= (1-FLAGS.discount)
        #print("normed,", pred_returns)
        #exit()


      writer.add_scalar(f"eval/mean_{FLAGS.algo}", pred_returns, global_step=i)
      writer.add_scalar(f"eval/std_{FLAGS.algo}", std, global_step=i)
      print(pred_returns)
      print(std)
    #model.save(f"/app/ws/logdir/mb/mb_model_{i}", "new_model")
    # print saved model
    # print(f"saved model as /app/ws/logdir/mb/mb_model_{i}")

app.run(main)
if __name__ == '__main__':
  print("Running main")
  app.run(main)
