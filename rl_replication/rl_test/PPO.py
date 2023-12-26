from logging import critical
from math import dist
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# 1. PPO
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []  # Implement the memories to List
        self.actions = []
        self.rewards = []
        self.vals = []  # Critic Calculate
        self.probs = []  #
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)  # Start to make batch
        indices = np.arange(n_states, dtype=np.int64)  # Store in indecies
        np.random.shuffle(indices)  # For Stochastic
        batches = [indices[i:i + self.batch_size] for i in batch_start]  # get whole batches from i

        return \
            np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches

    def store_memory(self, state, action, probs, vals, reward, done):  # store the memories
        self.states.append(state)  #
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):  # clear the Memories
        self.states = []  # at the end of every trajectory
        self.actions = []
        self.probs = []
        self.rewards = []
        self.vals = []
        self.dones = []


# 2. ActorNetwork
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(ActorNetwork, self).__init__()
        print(f"Actor : {input_dims}")
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")  # Actor Model file
        self.actor = nn.Sequential(  # Fully Connected Layers
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)  # Actor Probability sum = 1
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)  # Alpha = Learning Rate
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)  # Wanna know about state so we put it to actor Sequential
        dist = Categorical(dist)  # Calculating Series of Probabilities that we're going to use
        return dist  # draw from a distribution to get our actual action
        # And then we can use that to get the log probabilites for the
        # Calculation of the ratio of the two probabilities in our update
        # for our learning function

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)  # Torch provide Save Function

    def load_checkpoint(self):  # Torch provide Load Function
        self.load_state_dict(T.load(self.checkpoint_file))


# 3. CriticNetwork
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir="tmp/ppo"):  # Almost same as ActorNetwork
        super(CriticNetwork, self).__init__()
        print(f"Critic : {input_dims}")
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")  # Critic Model file
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)  # Single Value output // ActorNetwork have a
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# 4. Agent:
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 N=2048, n_epochs=10):
        print(f"Agent : {type(input_dims)}")
        print(f"Action_n : {n_actions}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("---saving models---")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("---loading model---")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(
            self.actor.device)  # Observation of the Current State of the Env as Input
        # print(f"state :{state}")
        dist = self.actor(
            state)  # And then we'll get our distribution for choosing an 'action' from ActorNetwork by giving "State"
        action = dist.sample()  # Get Action from distribution.Sample() [torch.distributions.categorical.Categorical.sample]
        value = self.critic(state)  # We need to get a Value of That Particular 'State'

        probs = T.squeeze(dist.log_prob(action)).item()  # cap the log to selected action from dist.sample()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value  # Little bit a difference from other Models (takes 3 values)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)  # start to calculating Advanetages

            for t in range(len(reward_arr) - 1):  # for each time step to len(reward)-1,
                # not going to overwirte our goal beyond the bounds of our array
                discount = 1.
                a_t = 0
                for k in range(t,
                               len(reward_arr) - 1):  # PPO Paper Algorithm 1, K Part (Multiple Epochs For Policy Updating)
                    # PPO Paper formula (11)
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[
                    batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()