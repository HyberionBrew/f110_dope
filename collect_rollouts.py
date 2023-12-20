import gymnasium
from argparse import Namespace
import yaml
import numpy as np
import torch
import pickle as pkl
#from absl import flags, app

from f110_sim_env.base_env import make_base_env
import argparse
from gymnasium.wrappers import TimeLimit
#from stable_baselines3 import PPO
from f110_agents.agents_numpy import StochasticContinousFTGAgent
from f110_agents.agents_numpy import DoubleAgentWrapper
from f110_agents.agents_numpy import SwitchingAgentWrapper
from f110_orl_dataset.config_new import Config as RewardConfig
from f110_agents.agent import Agent

parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--timesteps', type=int, default=1_000, help='Number of timesteps to run for')
parser.add_argument('--episode_length', type=int, default=500, help='Number of timesteps to run for')
parser.add_argument('--record', action='store_true', default=False, help='Whether to record the run')
parser.add_argument('--norender', action='store_false', default=True, dest='render', help='Whether to render the run')
parser.add_argument('--track', type=str, default='Infsaal', help='Track to train on')
parser.add_argument('--fixed_speed', type=float, default=None, help='Fixing the speed to the provided value')
# model path
parser.add_argument('--dataset', type=str, default="datasets", help="dataset name")
parser.add_argument('--std', type=float, default=0.3, help="std of the noise")
parser.add_argument('--agent_config', type=str, default="agent_config.json", help="agent config file")
parser.add_argument('--reward_config', type=str, default="reward_config.json", help="reward config file")

args = parser.parse_args()


import matplotlib.pyplot as plt
import pickle as pkl

def main(args):
    reward_config = RewardConfig(args.reward_config)
    agents = Agent()
    model = agents.load(args.agent_config)
    print("Reward config", reward_config)
    model_name = str(model)
    print("Agent name", model_name)
    eval_env = make_base_env(map= args.track,
                fixed_speed=args.fixed_speed,
                random_start =True,
                train_random_start = False,
                reward_config = reward_config,
                eval=False, # somewhat deprecated, TODO!
                use_org_reward=True,
                min_vel=0.0, # this is the minimum random START velocity
                max_vel=0.0,) # this is the maximum random START velocity
    
    eval_env = TimeLimit(eval_env, max_episode_steps=args.episode_length)

    #model1 = StochasticContinousFTGAgent(gap_blocker = args.gap_blocker, speed_multiplier=0.5, std=args.std) #PPO.load(args.model_path)
    #model2 = StochasticContinousFTGAgent(gap_blocker = args.gap_blocker, speed_multiplier=5.0, std=args.std) #PPO.load(args.model_path)
    #model = DoubleAgentWrapper(model2, model1, 100)
    #model = SwitchingAgentWrapper(model1, model2)

    
    episode = 0
    timesteps = 0
    with open(f"{args.dataset}/{model_name}", 'wb') as f:
        pass

    while timesteps < args.timesteps:

        obs, _ = eval_env.reset() # unfortunately reset does not provide infos which we would need
        done = False
        truncated = False
        episode += 1
        #print("Episode:", episode)
        rewards = []
        steerings = []
        vels = []
        progress_sin = []
        log_probs=[]
        progress_cos = []
        theta_sin = []
        theta_cos = []

        action = np.array([0.0,0.0]) # just a zero zero action to get the juicy infos

        episode_timestep = 0
        
        while not done and not truncated:
            # print(obs)
            timesteps += 1
            episode_timestep += 1
            obs, reward, done, truncated, info = eval_env.step(action)
            # add dimension to lidar and previous action
            info_obs = info["observations"]
            #for key in info_obs.keys():
            #    info_obs[key] = np.expand_dims(info_obs[key], axis=0)
            info_obs["lidar_occupancy"] = np.expand_dims(info_obs["lidar_occupancy"], axis=0)
            print("prev, action", info["observations"]["previous_action"])
            print(info_obs)
            print("++++")
            print(obs)
            print("---")
            _, action, log_prob = model(info_obs, current_timestep =np.array([episode_timestep]))
            #print("action", action)
            #action = np.expand_dims(action, axis=0)
            #if timesteps == 30:
            #    exit()
            log_prob = float(log_prob)
            #print(action)
            new_infos = dict()
            new_infos["lidar_timestamp"] = 0.0
            new_infos["pose_timestamp"] = 0.0
            print(action)
            if args.record:
                with open(f"{args.dataset}/{model_name}", 'ab') as f:
                    pkl.dump((action, info["observations"], 
                              float(reward), done, 
                              truncated, log_prob, 
                              timesteps, model_name, 
                              info["collision"], info["action_raw"], new_infos), f)
            log_probs.append(log_prob)
            progress_sin.append(info["observations"]["progress_sin"])
            progress_cos.append(info["observations"]["progress_cos"])
            steerings.append(info["action_raw"][0][0])
            vels.append(info["observations"]["linear_vels_x"])
            rewards.append(reward)
            theta_sin.append(info["observations"]["theta_sin"])
            theta_cos.append(info["observations"]["theta_cos"])

            if args.render:
                eval_env.render()
                if done or truncated:
                    discounts = np.array([0.99**i for i in range(len(rewards))])
                    discounted = np.array(rewards) * discounts
                    print("return:" , np.sum(discounted))
                    theta = np.arctan2(theta_sin, theta_cos)

                    plt.plot(vels)
                    plt.title("velocities")
                    plt.show()
             
                    plt.plot(rewards)
                    plt.title("rewards")
                    plt.show()

                    plt.plot(theta)
                    plt.title("theta")
                    plt.show()

                    plt.plot(discounted)
                    plt.xlabel("timesteps")
                    plt.ylabel("discounted reward")
                    plt.title("discounted reward vs timesteps")
                    plt.show()
                    plt.plot(log_probs)
                    plt.title("log_prob")
                    plt.show()
                    plt.plot(progress_sin)
                    plt.plot(progress_cos)
                    plt.title("progress")
                    plt.show()
                    plt.plot(steerings)
                    plt.title("steerings")
                    plt.show()
                    plt.plot(vels)
                    plt.title("velocities")
                    plt.show()
                    plt.plot(rewards)
                    plt.title("rewards")
                    plt.show()
if __name__ == "__main__":
    main(args)