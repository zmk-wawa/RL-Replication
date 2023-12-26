"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_PPO.py --train/test

"""
import argparse
import os
import time

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pendulum import PendulumEnv
from CONFIG import FOG_NUM, R_RANK_NUM, UN_RS


import tensorlayer as tl
# from tensorlayer.files.utils import _load_weights_from_hdf5_group_in_order
from tensorlayer import logging

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

RANDOMSEED = 1  # random seed

EP_MAX = 20  # total number of episodes for training
EP_LEN = 10  # total number of steps for each episode
GAMMA = 0.9  # reward discount   折扣因子
A_LR = 0.0001  # learning rate for actor   学习率
C_LR = 0.0002  # learning rate for critic
BATCH = 32  # update batchsize   用前一轮的数据训练多少次当前数据
A_UPDATE_STEPS = 10  # actor update steps  多少步后更新actor和critic(一般平均Reward)
C_UPDATE_STEPS = 10  # critic update steps
S_DIM, A_DIM = 3, 1  # state dimension, action dimension   state与action 的向量维度
EPS = 1e-8  # epsilon



# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),
    # Clipped surrogate objective, find this is better  PPO2   clips是相似度 前一个参数与当前的比值0.8~1.2
][1]  # choose the method for optimization


###############################  PPO  ####################################


class PPO(object):
    '''
    PPO 类
    '''

    def __init__(self):

        # 构建critic网络：
        # 输入state，输出V值
        # tfs是TensorFlow的placeholder，定义了输入的形状为[None, S_DIM]
        tfs = tl.layers.Input([None, S_DIM], tf.float32, 'state')
        l1 = tl.layers.Dense(100, tf.nn.relu)(tfs)
        v = tl.layers.Dense(1)(l1)
        self.critic = tl.models.Model(tfs, v)
        self.critic.train()

        # 构建actor网络：
        # actor有两个 actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入时state，输出是描述动作分布的mu和sigma
        self.actor = self._build_anet('pi', trainable=True)
        self.actor_old = self._build_anet('oldpi', trainable=False)
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    def a_train(self, tfs, tfa, tfadv):
        '''
        更新策略网络(policy network)
        '''
        # 输入时s，a，td-error。这个和AC是类似的。
        tfs = np.array(tfs, np.float32)  # state
        tfa = np.array(tfa, np.float32)  # action
        tfadv = np.array(tfadv, np.float32)  # td-error

        with tf.GradientTape() as tape:

            # 【敲黑板】这里是重点！！！！
            # 我们需要从两个不同网络，构建两个正态分布pi，oldpi。
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下，同样输出a的概率的比值
            # 除以(oldpi.prob(tfa) + EPS)，其实就是做了import-sampling。怎么解释这里好呢
            # 本来我们是可以直接用pi.prob(tfa)去跟新的，但为了能够更新多次，我们需要除以(oldpi.prob(tfa) + EPS)。
            # 在AC或者PG，我们是以1,0作为更新目标，缩小动作概率到1or0的差距
            # 而PPO可以想作是，以oldpi.prob(tfa)出发，不断远离（增大or缩小）的过程。
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            # 这个的意义和带参数更新是一样的。
            surr = ratio * tfadv

            # 我们还不能让两个分布差异太大。
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            # PPO2：
            # 很直接，就是直接进行截断。
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(ratio * tfadv,  # surr
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        '''
        更新Critic网络
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)  # tfdc_r可以理解为PG中就是G，通过回溯计算。只不过这PPO用TD而已。

        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v  # 就是我们说的td-error    R-b
            closs = tf.reduce_mean(tf.square(advantage))

        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        '''
        计算advantage，也就是td-error   时序差分误差：当前状态下的回报值减去估计的价值函数值
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)  # advantage = r - gamma * V(s_)
        return advantage.numpy()

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)

        self.update_old_pi()
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        #### PPO1比较复杂:
        # 动态调整参数 adaptive KL penalty
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution

        #### PPO2比较简单，直接就进行a_train更新:
        # clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        # 更新 critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

    def _build_anet(self, name, trainable):
        '''
        Build policy network
        :param name: name
        :param trainable: trainable flag
        :return: policy network
        '''
        # 连续动作型问题，输出mu和sigma。
        tfs = tl.layers.Input([None, S_DIM], tf.float32, name + '_state')
        l1 = tl.layers.Dense(100, tf.nn.relu, name=name + '_l1')(tfs)

        a = tl.layers.Dense(A_DIM, tf.nn.tanh, name=name + '_a')(l1)
        mu = tl.layers.Lambda(lambda x: x * 2, name=name + '_lambda')(a)

        sigma = tl.layers.Dense(A_DIM, tf.nn.softplus, name=name + '_sigma')(l1)

        model = tl.models.Model(tfs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model


    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''

        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor(s)  # 通过actor计算出分布的mu和sigma
        pi = tfp.distributions.Normal(mu, sigma)  # 用mu和sigma构建正态分布
        a = tf.squeeze(pi.sample(1), axis=0)[0]  # 根据概率分布随机出动作
        a = a*10
        print("初始a的值为", a)
        a = np.clip(a, 0, 31)
        print("选出的a的值为",a)
        return int(np.floor(a))  # 最后sample动作，并进行裁剪。

    def get_v(self, s):
        '''
        计算value值。
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]  # 要和输入的形状对应。
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/ppo_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ppo_actor_old.hdf5', self.actor_old)
        tl.files.save_weights_to_hdf5('model/ppo_critic.hdf5', self.critic)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        self.self_load_hdf5_in_order('model/ppo_actor.hdf5', self.actor)
        self.self_load_hdf5_in_order('model/ppo_actor_old.hdf5', self.actor_old)
        self.self_load_hdf5_in_order('model/ppo_critic.hdf5', self.critic)

    def self_load_hdf5_in_order(self,filepath, network):
        f = h5py.File(filepath, 'r')
        try:
            layer_names = [n.encode('utf-8') for n in f.attrs["layer_names"]]
        except Exception:
            raise NameError(
                "The loaded hdf5 file needs to have 'layer_names' as attributes. "
                "Please check whether this hdf5 file is saved from TL."
            )

        if len(network.all_layers) != len(layer_names):
            logging.warning(
                "Number of weights mismatch."
                "Trying to load a saved file with " + str(len(layer_names)) + " layers into a model with " +
                str(len(network.all_layers)) + " layers."
            )

        self._load_weights_from_hdf5_group_in_order(f, network.all_layers)

        f.close()
        logging.info("[*] Load %s SUCCESS!" % filepath)

    def _load_weights_from_hdf5_group_in_order(self,f, layers):
        layer_names = [n.encode('utf8') for n in f.attrs["layer_names"]]

        for idx, name in enumerate(layer_names):
            g = f[name]
            layer = layers[idx]
            if isinstance(layer, tl.models.Model):
                self._load_weights_from_hdf5_group_in_order(g, layer.all_layers)
            elif isinstance(layer, tl.layers.ModelLayer):
                self._load_weights_from_hdf5_group_in_order(g, layer.model.all_layers)
            elif isinstance(layer, tl.layers.LayerList):
                self._load_weights_from_hdf5_group_in_order(g, layer.layers)
            elif isinstance(layer, tl.layers.Layer):
                weight_names = [n.encode('utf8') for n in g.attrs['weight_names']]
                for iid, w_name in enumerate(weight_names):
                    tl.files.assign_tf_variable(layer.all_weights[iid], np.asarray(g[w_name]))
            else:
                raise Exception("Only layer or model can be saved into hdf5.")
            if idx == len(layers) - 1:
                break

if __name__ == '__main__':

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

    env = PendulumEnv(1,0.9,50,0.999,cur_capacity_dict,ori_capacity_dict,max_min_dict,distance)

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO()


    if args.train:
        all_ep_r = []

        # 更新流程：
        # 最多EP_MAX次 即1000次
        for ep in range(EP_MAX):
            s,_ = env.reset(cur_capacity_dict.copy())  # 初始化，重新玩
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            t = 0
            while True:  # in one episode 在一次游戏中，最多执行EP_Len次（防止卡死）
                # env.render()
                # TODO: 如果在当前可靠性等级下，fog_id无终端 -> 重新选择
                # TODO: 选择的是当前fog_id  这些都需要在choose_action中排除
                a = ppo.choose_action(s)
                s_, r, done = env.step(a)
                if s_ is None:
                    continue
                t += 1
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # 对奖励进行归一化。有时候会挺有用的。所以我们说说，奖励是个主观的东西。
                s = s_
                ep_r += r

                # # N步更新的方法，每BATCH步了就可以进行一次更新
                if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                    v_s_ = ppo.get_v(s_)  # 计算n步中最后一个state的v_s_

                    # 和PG一样，向后回溯计算。
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    # 所以这里的br并不是每个状态的reward，而是通过回溯计算的V值
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br)
                if done:
                    break

            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

        ppo.save_ckpt()

    # test  训练 ，当可靠性满足后，done == True即可
    ppo.load_ckpt()
    print('开始test')

    actions = []

    s,_ = env.reset(cur_capacity_dict.copy())
    while True:
        # env.render()
        a = ppo.choose_action(s)
        s_, r, done = env.step(a)
        if s_ is None:
            continue
        actions.append([a % FOG_NUM + 1, a // FOG_NUM + 1])
        print("第次",s,r,done)
        s = s_.copy()
        if done:
            print("选出",actions)
            break
