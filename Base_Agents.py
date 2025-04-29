from collections import deque
import torch
from torch import nn
import numpy as np
import os

def get_activation(act_type: str):
    if act_type == 'LeakyRelu':
        return nn.LeakyReLU()
    elif act_type == 'Relu':
        return nn.ReLU()
    elif act_type == 'PRelu':
        return nn.PReLU()
    else:
        return nn.Identity()


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, activation: str = 'LeakyRelu',
                 hidden_layers: int = 2, dueling = False, scale = 1.):
        super(QNetwork, self).__init__()

        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation)
        self.dueling = dueling
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

        self.scale = scale

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.mid_acts = nn.ModuleList([get_activation(activation) for _ in range(hidden_layers)])

    def forward(self, observation):
            x = self.in_layer(observation)
            x = self.act(x)

            for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
                x = mid_layer(x)
                x = mid_act(x)

            if self.dueling:
                value = self.value_stream(x)
                advantages = self.advantage_stream(x)
                x = value + (advantages - advantages.mean(dim=1, keepdim=True))
            else:
                x = self.out_layer(x)

            if self.scale > 1:
                x *= self.scale
            return x


class DDQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)
            curr_q = self.online_net(state)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            next_q = self.online_net(next_state)
            next_q_1 = next_q[:, :self.q_mask].max(dim=1)[0]
            next_q_2 = next_q.max(dim=1)[0]
            next_q = mark * next_q_1 + (1 - mark) * next_q_2

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        print("Target network updated")

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))
            self.target_net.load_state_dict(torch.load(file_path))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

class DQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            next_q = self.online_net(next_state).max(dim=1)[0]

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))

    def target_update(self):
        pass

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


class PPO_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, scale= 1e2).to(device)
        self.critic_net = QNetwork(state_dim, hidden_dim, 1, activation, hidden_layers, dueling).to(device)

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer_actor = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

        self.eps_clip=0.1
        self.max_grad_norm = 0.5

    def update(self, experiences):

        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            action, old_log_prob = [a[0] for a in action], [a[1] for a in action]
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            old_log_prob = torch.tensor(old_log_prob, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            with torch.no_grad():
                next_state = self.critic_net(next_state).squeeze()

            action_prob = self.online_net(state)

            mask = torch.ones_like(action_prob)
            mask[:, -1] = 0
            action_prob_1 = action_prob.masked_fill(mask == 0, float('-inf'))
            action_prob_1 = torch.nn.functional.softmax(action_prob_1, dim=-1)
            dist_1 = torch.distributions.Categorical(action_prob_1)

            action_prob = torch.nn.functional.softmax(action_prob, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action_log_prob = dist.log_prob(action)
            action_log_prob_1 = dist_1.log_prob(action)
            action_log_prob = action_log_prob_1 * mark + action_log_prob * (1-mark)

            state_value = self.critic_net(state).squeeze()

            advantages = reward + self.gamma * next_state * (1 - done) - state_value.detach()
            ratios = torch.exp(action_log_prob - old_log_prob.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            critic_loss = nn.functional.mse_loss(state_value, reward + self.gamma * next_state * (1 - done))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.online_net.state_dict(), file_path + '/actor.pth')
        torch.save(self.critic_net.state_dict(), file_path + '/critic.pth')

    def load_model(self,file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path + '/actor.pth', map_location=self.device))
            self.critic_net.load_state_dict(torch.load(file_path + '/critic.pth', map_location=self.device))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

def shuffle_neighbors(neighbor_states, other_states,action):
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    if action < 4:
        new_action = int(np.where(indices == action)[0])
    else:
        new_action = action
    return np.concatenate([new_state, other_states]), new_action


class ShuffleEx:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, state, action):
        return shuffle_neighbors(state[:self.shuffle_mask], state[self.shuffle_mask:], action)


def cal_agent_dim(neighbors_dim: int, edges_dim: int, distance_dim: int, mission_dim: int, current_dim: int,
                  action_dim: int):
    return neighbors_dim + edges_dim + distance_dim + mission_dim + current_dim, action_dim, -(
            mission_dim + current_dim)
