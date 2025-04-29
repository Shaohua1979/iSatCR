import argparse
import yaml
import random
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


args = parse_args()
config = load_config(args.config)

# config = load_config('train_NewDQN.yaml')

random.seed(config['general']['random_seed'])
torch.manual_seed(config['general']['random_seed'])
np.random.seed(config['general']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['general']['random_seed'])

from RL_environment_for_computing import SatelliteEnv
from Base_Agents import DDQN_Agent, ShuffleEx, cal_agent_dim, PPO_Agent, DQN_Agent

phase = config['general']['phase']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = config['agent']['mode']
if mode in ['Pure_DQN', "New_DQN", "Pure_PPO","New_PPO","Weak_DQN"]:
    state_dim, action_dim, state_mask = cal_agent_dim(neighbors_dim= config['agent']['neighbors_dim'],
                                                      edges_dim= config['agent']['edges_dim'],
                                                      distance_dim= config['agent']['distance_dim'],
                                                      mission_dim= config['agent']['mission_dim'],
                                                      current_dim= config['agent']['current_dim'],
                                                      action_dim=config['agent']['action_dim'])
    if 'DQN' in mode:
        if 'Weak' in mode:
            Agent = DQN_Agent
        else:
            Agent = DDQN_Agent
    elif 'PPO' in mode:
        Agent = PPO_Agent
    agent = Agent(state_dim=state_dim,
                       hidden_dim=config['agent']['hidden_dim'],
                       action_dim=action_dim,
                       buffer_length=config['agent']['buffer_length'],
                       batch_size=config['agent']['batch_size'],
                       gamma=config['agent']['gamma'],
                       device=device,
                       q_mask=config['agent']['q_mask'],
                       activation=config['agent']['activation'],
                       hidden_layers=config['agent']['hidden_layers'],
                       dueling = config['agent']['dueling'],
                       learning_rate=config['agent']['learning_rate'],
                       repeat=config['agent']['repeat'],
                       shuffle_func=ShuffleEx(state_mask).shuffle if config['agent']['shuffle'] else None)
    if phase != 'train':
        agent.load_model(config['agent']['model_path'])
else:
    agent = None

env = SatelliteEnv(mode=config['agent']['mode'],
                   select_mode=config['general']['select_mode'],
                   q_net=agent.online_net if agent else None,
                   epsilon=config['general']['epsilon'],
                   reward_factors=config['general']['reward_factors'],
                   device=device,
                   mission_possibility=config['environment']['mission_possibility'],
                   poisson_rate=config['environment']['poisson_rate'],
                   packet_frequency=config['environment']['packet_frequency'],
                   computing_demand_factor=config['environment']['computing_demand_factor'],
                   computing_demand_factor_2=config['environment']['computing_demand_factor_2'],
                   size_after_computing_factor=config['environment']['size_after_computing_factor'],
                   size_after_computing_1=config['environment']['size_after_computing_1'],
                   begin_time=config['general']['begin_time'],
                   end_time=None,
                   time_stride=config['general']['time_stride'],
                   tle_filepath=config['environment']['tle_filepath'],
                   SOD_file_path=config['environment']['SOD_file_path'],
                   mean_interval_time=config['environment']['mean_interval_time'],
                   memory=config['environment']['memory'],
                   computing_ability=config['environment']['computing_ability'],
                   transmission_rate=config['environment']['transmission_rate'],
                   downlink_rate=config['environment']['downlink_rate'],
                   downstream_delays=config['environment']['downstream_delays'],
                   packet_size_range=config['environment']['packet_size_range'],
                   state_update_period=config['environment']['state_update_period'],
                   print_cycle=config['general']['print_cycle'],
                   del_cycle=config['environment']['del_cycle'],
                   visualize=config['environment']['visualize'],
                   print_info=config['environment']['print_info'],
                   show_detail=config['environment']['show_detail'],
                   save_log=config['environment']['save_log'],
                   random_edges_del=config['environment']['random_edges_del'],
                   random_nodes_del=config['environment']['random_nodes_del'],
                   update_cycle=config['environment']['update_cycle'],
                   save_training_data=config['environment']['save_training_data'],
                   elevation_angle=config['environment']['elevation_angle'],
                   pole=config['environment']['pole'])

begin_time = config['general']['begin_time']
time_stride = config['general']['time_stride']
rounds = config['general']['rounds']
skip_time = config['general']['skip_time']
duration = config['general']['duration']
epsilon = config['general']['epsilon']
min_epsilon = config['general']['min_epsilon']
epsilon_decay = config['general']['epsilon_decay']

if phase != 'train':
    epsilon = 0

for k in range(rounds):
    env.reset(begin_time)
    for t in range(int(duration / time_stride)):
        experiences = env.step(epsilon)
        # env.render()
        if phase == 'train' and agent:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            agent.update(experiences)
            if (t + 1) % int(config['agent']['update_cycle']) == 0:
                if 'DQN' in mode:
                    agent.target_update()
                agent.save_model(config['agent']['model_path'])
    if phase == 'test':
        env.show_satellite_computing_time()
    begin_time = env.add_time_to_str(begin_time, skip_time)
