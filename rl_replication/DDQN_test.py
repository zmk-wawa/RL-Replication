import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    batch_size = 32
    num_episodes = 500
    convergence_threshold = 195  # 收敛阈值，即平均奖励达到195时认为已经收敛
    rewards = []  # 用于记录每个训练周期的平均奖励
    convergence_count = 0  # 连续达到收敛阈值的次数

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0

        for time in range(500):
            # env.renren'dder()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}".format(episode, num_episodes, time))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        rewards.append(episode_reward)
        if episode_reward >= convergence_threshold:
            convergence_count += 1
        else:
            convergence_count = 0

        if convergence_count >= 10:
            print("DDQN converged.")
            break

        agent.epsilon *= agent.epsilon_decay

    plot_convergence(rewards, convergence_threshold)


def plot_convergence(rewards, threshold):
    plt.plot(rewards)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title('Convergence Rate')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(['Reward', 'Convergence Threshold'])
    plt.show()


if __name__ == "__main__":
    main()