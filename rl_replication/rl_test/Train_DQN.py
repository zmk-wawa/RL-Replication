import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Model import Model
from collections import deque
import random
from frozen_lake import FrozenEnv
from CONFIG import FOG_NUM, R_RANK_NUM, UN_RS

cur_capacity_dict = {1: 1000, 2: 2000, 3: 2500, 4: 1500}
ori_capacity_dict = {1: 3000, 2: 5000, 3: 5500, 4: 4500}
distance = [0, 60, 137, 64]
max_min_dict = {1: [[3, 2], [-1, 9], [6, 3], [1, 1]],
                2: [[-1, 9], [5, 4], [5, 4], [-1, 9]],
                3: [[-1, 9], [-1, 9], [-1, 9], [5, 3]],
                4: [[-1, 9], [5, 2], [-1, 9], [-1, 9]],
                5: [[-1, 9], [4, 2], [-1, 9], [-1, 9]],
                6: [[7, 4], [-1, 9], [-1, 9], [-1, 9]],
                7: [[-1, 9], [-1, 9], [-1, 9], [6, 3]],
                8: [[-1, 9], [2, 2], [5, 3], [-1, 9]]}
r_rank_mem_rate_dict = {1: [0.30000000000000004, 0.0, 0.8, 0.1],
                        2: [0.0, 0.5, 0.5, 0.0],
                        3: [0.0, 0.0, 0.0, 0.5],
                        4: [0.0, 0.5, 0.0, 0.0],
                        5: [0.0, 0.4, 0.0, 0.0],
                        6: [0.7000000000000001, 0.0, 0.0, 0.0],
                        7: [0.0, 0.0, 0.0, 0.6000000000000001],
                        8: [0.0, 0.2, 0.5, 0.0]}


# Parameters
use_cuda = True
episode_limit = 10
target_update_delay = 2  # update target net every target_update_delay episodes
test_delay = 10
learning_rate = 1e-4
epsilon = 1  # initial epsilon
min_epsilon = 0.1
epsilon_decay = 0.9 / 2.5e3
gamma = 0.99
memory_len = 10000

env = FrozenEnv(1,0.9,50,0.999,cur_capacity_dict,ori_capacity_dict,max_min_dict,distance)
n_features = len(env.observation_space.high)
n_actions = env.action_space.n

memory = deque(maxlen=memory_len)
# each memory entry is in form: (state, action, env_reward, next_state)
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
policy_net = Model(n_features, n_actions).to(device)
target_net = Model(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


def get_states_tensor(sample, states_idx):
    sample_len = len(sample)
    states_tensor = torch.empty((sample_len, n_features), dtype=torch.float32, requires_grad=False)

    features_range = range(n_features)
    for i in range(sample_len):
        for j in features_range:
            states_tensor[i, j] = sample[i][states_idx][j].item()

    return states_tensor



def state_reward(state, env_reward):
    return env_reward - (abs(state[0]) + abs(state[2])) / 2.5


def get_action(state, e=min_epsilon):
    if random.random() < e:
        # 随机
        action = random.randint(0, FOG_NUM * R_RANK_NUM - 1)
    else:
        # print("打印")
        # print(state)
        # print(np.shape(state))
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = policy_net(state).argmax().item()

    return action


def fit(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)
    train_ds = TensorDataset(inputs, labels)
    train_dl = DataLoader(train_ds, batch_size=5)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0.0

    for x, y in train_dl:
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    return total_loss / len(inputs)


def optimize_model(train_batch_size=100):
    train_batch_size = min(train_batch_size, len(memory))
    train_sample = random.sample(memory, train_batch_size)

    state = get_states_tensor(train_sample, 0)
    next_state = get_states_tensor(train_sample, 3)

    q_estimates = policy_net(state.to(device)).detach()
    next_state_q_estimates = target_net(next_state.to(device)).detach()
    next_actions = policy_net(next_state.to(device)).argmax(dim=1)

    for i in range(len(train_sample)):
        next_action = next_actions[i].item()
        q_estimates[i][train_sample[i][1]] = (state_reward(next_state[i], train_sample[i][2]) +
                                              gamma * next_state_q_estimates[i][next_action].item())

    fit(policy_net, state, q_estimates)


def train_one_episode():
    global epsilon
    current_state, _ = env.reset(cur_capacity_dict.copy())
    done = False
    score = 0
    reward = 0
    while not done:
        action = get_action(current_state,epsilon)
        # print(env.step(action))
        next_state, env_reward, done = env.step(action)
        if next_state is None:
            continue

        memory.append((current_state, action, env_reward, next_state))
        current_state = next_state
        score += env_reward
        reward += state_reward(next_state, env_reward)

        optimize_model(100)

        epsilon -= epsilon_decay

    return score, reward


def test():
    # print("测试开始")
    state,_ = env.reset(cur_capacity_dict.copy())
    # print("重置的状态",np.shape(state))
    done = False
    score = 0
    reward = 0

    actions = []

    while not done:
        # print("测试中",state)
        action = get_action(state)
        next_state, env_reward, done = env.step(action)

        if next_state is None:
            continue
        score += env_reward
        actions.append(action)
        reward += state_reward(next_state, env_reward)
        state = next_state.copy()

    return score, reward,actions


def main():
    best_test_reward = -10000
    best_new_actions = []

    for i in range(episode_limit):
        score, reward = train_one_episode()

        # print(f'Episode {i + 1}: score: {score} - reward: {reward}')

        if i % target_update_delay == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        if (i + 1) % test_delay == 0:
            test_score, test_reward,test_actions = test()
            # print(f'测试 Episode {i + 1}: test score: {test_score} - test reward: {test_reward},test_actions:{test_actions}')
            if test_reward > best_test_reward:
                best_new_actions = []
                # print('New best test reward. Saving model')
                best_test_reward = test_reward
                for  a in test_actions:
                    best_new_actions.append([a % FOG_NUM + 1, a // FOG_NUM + 1])
                torch.save(policy_net.state_dict(), 'policy_net.pth')

    if episode_limit % test_delay != 0:
        test_score, test_reward,test_actions = test()
        # print(f'测试 Episode {episode_limit}: test score: {test_score} - test reward: {test_reward},test_actions:{test_actions}')
        if test_reward > best_test_reward:
            # print('新的New best test reward. Saving model')
            best_test_reward = test_reward
            best_new_actions = test_actions.copy()
            torch.save(policy_net.state_dict(), 'policy_net.pth')

    print(f'最终best test reward: {best_test_reward},actions:{best_new_actions}')


if __name__ == '__main__':
    main()
