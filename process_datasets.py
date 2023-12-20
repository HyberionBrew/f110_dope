
import zarr
import os
import argparse
import pickle as pkl
import numpy as np
# for now we do not! recompute the reward
parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--input_folder', type=str, default='datasets', help='Logging directory')
parser.add_argument('--file', type=str, default=None, help='Specific File to process')
parser.add_argument('--noappend', action='store_false', 
                    default=True,dest='append', help='Whether to append to existing file')
args = parser.parse_args()


obs_keys = ['poses_x', 'poses_y', 'linear_vels_x', 
            'linear_vels_y', 'ang_vels_z', 'theta_sin', 
            'theta_cos','progress_sin','progress_cos','lidar_occupancy', 
            'previous_action']

def main(args):
    all_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join( args.input_folder, f))]

    if args.file is not None:
        all_files = [args.file]
    print(f"Available files: {all_files}")
    #exit()
    #root = zarr.open('trajectories.zarr', mode='w')

    chunks_size = 10000 
    # Create an extendible array named "actions"
    if args.append:
        root = zarr.open(f"{args.input_folder}.zarr", mode='a')
    else:
        root = zarr.open(f"{args.input_folder}.zarr", mode='w')

    if not(args.append):
        # write new
        
        actions_array = root.zeros("actions", shape=(0, 2), chunks=(chunks_size, 2), dtype='float32',
                                    overwrite=False, maxshape=(None, 2))
        raw_actions_array = root.zeros("raw_actions", shape=(0, 2), chunks=(chunks_size, 2), dtype='float32',
                            overwrite=False, maxshape=(None, 2))
        
        rewards_array = root.zeros("rewards", shape=(0, 1), chunks=(chunks_size, 1), dtype='float32',
                                    overwrite=False, maxshape=(None, 1))
        done_array = root.zeros("done", shape=(0, 1), chunks=(chunks_size, 1), dtype='bool',
                                    overwrite=False, maxshape=(None, 1))
        truncated_array = root.zeros("truncated", shape=(0, 1), chunks=(chunks_size, 1), dtype='bool',
                                    overwrite=False, maxshape=(None, 1))
        timesteps_array = root.zeros("timesteps", shape=(0, 1), chunks=(chunks_size, 1), dtype='int32',
                                    overwrite=False, maxshape=(None, 1))
        # dtype = np.dtype("S32")
        model_name_array = root.zeros("model_name", shape=(0, 1), chunks=(chunks_size, 1), dtype='S32',
                                    overwrite=False, maxshape=(None, 1))
        collision_array = root.zeros("collision", shape=(0, 1), chunks=(chunks_size, 1), dtype='bool',
                                    overwrite=False, maxshape=(None, 1))
        log_prob_array = root.zeros("log_prob", shape=(0, 1), chunks=(chunks_size, 1), dtype='float32',
                                    overwrite=False, maxshape=(None, 1))
        
        obs_group = root.create_group('observations')
        for key in obs_keys:
            if key not in obs_group:
                obs_group.create_group(key)

        infos_group = root.create_group('infos')
        for key in ['lidar_timestamp', 'pose_timestamp', 'action_timestamp']:
            if key not in infos_group:
                infos_group.create_group(key)
    # Create an extendible array named "observations"

    # extract from pickle array

    args.append= False
    for file in all_files:
        actions = []
        observations_group = []
        
        obs_lists = {key: [] for key in obs_keys}
        rewards = []
        dones = []
        truncates = []
        timesteps = []
        model_names = []
        log_probs = []
        collisions = []
        raw_actions = []
        infos = {key: [] for key in ['lidar_timestamp', 'pose_timestamp', 'action_timestamp']}
        print(f"Processing {file}")
        with open(os.path.join(args.input_folder, file), 'rb') as f:
            while True:
                try:
                    action, obs, reward, done, truncated, log_prob, timestep, model_name, collision, raw_action, info = pkl.load(f)
                    actions.append(action[0])
                    raw_actions.append(raw_action[0])
                    for key, value in obs.items():
                        # print(key)
                        if key == 'lidar_occupancy':
                            obs_lists[key].append(value) #TODO! fix this inconsistency
                        else:
                            obs_lists[key].append(value[0])
                    for key, value in info.items():
                        #print("key")
                        #print(value)
                        infos[key].append(value)
                    rewards.append(reward)
                    dones.append(done)
                    truncates.append(truncated)
                    timesteps.append(timestep)
                    model_names.append(model_name)
                    collisions.append(collision)
                    log_probs.append(log_prob)

                except EOFError:
                    print("Done loading data")
                    break

        print(f"Number of timesteps: {len(timesteps)}")
        # print keys
        for key in obs_lists:
            obs_lists[key] = np.array(obs_lists[key])

        for key, array in obs_lists.items():
            if args.append:
                root['observations'][key].append(array)
            else:
                root['observations'][key] = array
        for key in infos:
            infos[key] = np.array(infos[key])
        for key, array in infos.items():
            if args.append:
                root['infos'][key].append(array)
            else:
                root['infos'][key] = array
        
        
        if args.append:
            root["actions"].append(np.array(actions))
            root["raw_actions"].append(np.array(raw_actions))
            root["rewards"].append(np.array(rewards))
            root["done"].append(np.array(dones))
            root["truncated"].append(np.array(truncates))
            root["timestep"].append(np.array(timesteps))
            root["model_name"].append(np.array(model_names))
            root["collision"].append(np.array(collisions))
            root["log_prob"].append(np.array(log_probs))
            print(np.array(model_names)[-1])
            print(root["model_name"][-1])
        else:
            actions = np.array(actions)
            raw_actions = np.array(raw_actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            truncated = np.array(truncates)
            timesteps = np.array(timesteps)
            model_names = np.array(model_names)
            #print(model_names)
            #print(model_names[-1])
            # model_names = np.char.add(model_names, "_____")
            root["actions"] = actions
            root["raw_actions"] = raw_actions
            root["rewards"] = rewards
            root["done"] = dones
            root["truncated"] = truncated
            root["timestep"] = timesteps
            root["model_name"] = model_names
            root["collision"] = np.array(collisions)
            root["log_prob"] = np.array(log_probs)
        args.append= True
            
if __name__ == "__main__":
    main(args)