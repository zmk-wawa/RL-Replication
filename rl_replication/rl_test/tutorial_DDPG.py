"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_DDPG.py --train/test

"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl

from pendulum import PendulumEnv
from CONFIG import FOG_NUM, R_RANK_NUM, UN_RS

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

RANDOMSEED = 1              # random seed

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 10000     # size of replay buffer
BATCH_SIZE = 32             # update batchsize

MAX_EPISODES = 20          # total number of episodes for training
MAX_EP_STEPS = 20          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 3                     # control exploration

###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, a_dim, s_dim, a_bound):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            #注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        #建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # 选择动作，把s带进入，输出a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()


    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_target.hdf5', self.critic_target)




if __name__ == '__main__':
    
    #初始化环境
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

    env = PendulumEnv(1, 0.9, 50, 0.999, cur_capacity_dict, ori_capacity_dict, max_min_dict, distance)
    # reproducible，设置随机种子，为了能够重现
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    #定义状态空间，动作空间，动作幅度范围
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    # print('s_dim',s_dim)
    # print('a_dim',a_dim)

    #用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound)

    #训练部分：
    if args.train:  # train
        
        reward_buffer = []      #用于记录每个EP的reward，统计变化
        t0 = time.time()        #统计时间
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s,_ = env.reset(cur_capacity_dict.copy())
            ep_reward = 0       #记录当前EP的reward
            for j in range(MAX_EP_STEPS):
                # Add exploration noise
                a = ddpg.choose_action(s)       #这里很简单，直接用actor估算出a动作

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪
                a = np.clip(np.random.normal(a, VAR), -2, 2)
                a = int(np.floor(((a + 2) / 4) * 31))  # 线性映射到 0 到 31 范围上，并向下取整
                print("第",j,"次",a)
                # 与环境进行互动
                s_, r, done = env.step(a)

                if s_ is None:
                    continue

                # 保存s，a，r，s_
                ddpg.store_transition(s, a, r / 10, s_)

                # 第一次数据满了，就可以开始学习
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()

                #输出数据记录
                s = s_.copy()
                ep_reward += r  #记录当前EP的总reward
                if j == MAX_EP_STEPS - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward,
                            time.time() - t1
                        ), end=''
                    )

            # test
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s,_ = env.reset(cur_capacity_dict.copy())
                ep_reward = 0
                for j in range(MAX_EP_STEPS):

                    a = ddpg.choose_action(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
                    a = np.clip(np.random.normal(a, VAR), -2, 2)
                    a = int(np.floor(((a + 2) / 4) * 31))  # 线性映射到 0 到 31 范围上，并向下取整
                    print("动作",a)
                    s_, r, done= env.step(a)
                    if s_ is None:
                        continue
                    s = s_.copy()
                    ep_reward += r
                    if j == MAX_EP_STEPS - 1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                i, MAX_EPISODES, ep_reward,
                                time.time() - t1
                            )
                        )

                        reward_buffer.append(ep_reward)


        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()

    # test
    actions = []
    s,_ = env.reset(cur_capacity_dict.copy())
    while True:
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, VAR), -2, 2)
        a = int(np.floor(((a + 2) / 4) * 31))  # 线性映射到 0 到 31 范围上，并向下取整
        s_, r, done = env.step(a)

        if s_ is None:
            continue
        actions.append([a%FOG_NUM + 1, a//FOG_NUM + 1])
        print(s_,r,done,a)
        s = s_.copy()
        if done:
            break
    print("最后的结果",actions)

