import tensorflow as tf
import tensorlayer as tl
from collections import deque
import numpy as np
import random

from ddqn_env import CartPoleEnv
from CONFIG import FOG_NUM, R_RANK_NUM

TRAIN_NUM = 40

INFER_NUM = 10

# CUR_DICT = {}


class Double_DQN():
    def __init__(self, d_size, d_r, local_fog_id, cur_d_u_r, cur_capacity_dict, ori_capacity_dict, distance, max_min_dict):
        '''
        初始化
        :param d_size: 数据大小
        :param d_r: 数据可靠性
        :param local_fog_id: 本地备份的fog_id
        :param cur_d_u_r: 当前数据达到的不可靠率（本地备份一次）
        :param cur_capacity_dict: 各个雾节点的剩余存储容量
        :param ori_capacity_dict: 各个雾节点的最初存储容量
        :param distance: 本地雾节点到其他雾节点的距离
        :param max_min_dict: 各个可靠性等级下，各个雾节点拥有的终端容量的最大和最小等级
        '''
        # cur_capacity_dict = {1: 1000, 2: 2000, 3: 2500, 4: 1500}
        # ori_capacity_dict = {1: 3000, 2: 5000, 3: 5500, 4: 4500}
        # distance = [0, 60, 137, 64]
        # max_min_dict = {1: [[3, 2], [-1, 9], [6, 3], [1, 1]],
        #                 2: [[-1, 9], [5, 4], [5, 4], [-1, 9]],
        #                 3: [[-1, 9], [-1, 9], [-1, 9], [5, 3]],
        #                 4: [[-1, 9], [5, 2], [-1, 9], [-1, 9]],
        #                 5: [[-1, 9], [4, 2], [-1, 9], [-1, 9]],
        #                 6: [[7, 4], [-1, 9], [-1, 9], [-1, 9]],
        #                 7: [[-1, 9], [-1, 9], [-1, 9], [6, 3]],
        #                 8: [[-1, 9], [2, 2], [5, 3], [-1, 9]]}
        # r_rank_mem_rate_dict = {1: [0.30000000000000004, 0.0, 0.8, 0.1],
        #                         2: [0.0, 0.5, 0.5, 0.0],
        #                         3: [0.0, 0.0, 0.0, 0.5],
        #                         4: [0.0, 0.5, 0.0, 0.0],
        #                         5: [0.0, 0.4, 0.0, 0.0],
        #                         6: [0.7000000000000001, 0.0, 0.0, 0.0],
        #                         7: [0.0, 0.0, 0.0, 0.6000000000000001],
        #                         8: [0.0, 0.2, 0.5, 0.0]}

        self.env = CartPoleEnv(local_fog_id, cur_d_u_r, d_size, d_r, cur_capacity_dict, ori_capacity_dict,
                                max_min_dict, distance)  # 定义环境
        self.input_dim = self.env.observation_space.shape[0]  # 定义网络的输入形状，这里就是输入S

        self.cur_capacity_dict = cur_capacity_dict
        # self.r_rank_mem_rate = r_rank_mem_rate
        # self.max_min = max_min

        # 建立两个网络
        self.Q_network = self.get_model()  # 建立一个Q网络
        self.Q_network.train()  # 在tensorlayer要指定这个网络用于训练。
        self.target_Q_network = self.get_model()  # 创建一个target_Q网络
        self.target_Q_network.eval()  # 这个网络指定为不用于更新。

        ## epsilon-greedy相关参数
        self.epsilon = 1.0  # epsilon大小，随机数大于epsilon，则进行开发；否则，进行探索。
        self.epsilon_decay = 0.995  # 减少率：epsilon会随着迭代而更新，每次会乘以0.995
        self.epsilon_min = 0.01  # 小于最小epsilon就不再减少了。

        # 其余超参数
        self.memory = deque(maxlen=2000)  # 队列，最大值是2000
        self.batch = 10
        self.gamma = 0.95  # 折扣率
        self.learning_rate = 1e-3  # 学习率
        self.opt = tf.optimizers.Adam(self.learning_rate)  # 优化器
        # self.is_rend = False  # 默认不渲染，当达到一定次数后，开始渲染。

    # dueling DQN只改了网络架构。
    def get_model(self):
        # 第一部分
        input = tl.layers.Input(shape=[None, self.input_dim])
        h1 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(input)
        h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(h1)
        # # 第二部分
        svalue = tl.layers.Dense(FOG_NUM*R_RANK_NUM, )(h2)
        # 第三部分
        avalue = tl.layers.Dense(FOG_NUM*R_RANK_NUM, )(h2)  # 计算avalue
        mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)  # 用Lambda层，计算avg(a)
        advantage = tl.layers.ElementwiseLambda(lambda x, y: x - y)([avalue, mean])  # a - avg(a)

        output = tl.layers.ElementwiseLambda(lambda x, y: x + y)([svalue, avalue])
        return tl.models.Model(inputs=input, outputs=output)

    def update_epsilon(self):
        '''
        用于更新epsilon
            除非已经epsilon_min还小，否则比每次都乘以减少率epsilon_decay。
        '''
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_Q(self):
        '''
        Q网络学习完之后，需要把参数赋值到target_Q网络
        '''
        for i, target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)

    def remember(self, s, a, s_, r, done):
        '''
        把数据放入到队列中保存。
        '''
        data = (s, a, s_, r, done)
        self.memory.append(data)

    def process_data(self):

        # 从队列中，随机取出一个batch大小的数据。
        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        # 原始DQN的target
        '''
        target_Q = np.max(self.target_Q_network(np.array(s_,dtype='float32')))  #计算下一状态最大的Q值
        target = target_Q * self.gamma + r
        '''
        # [敲黑板]
        # 计算Double的target
        y = self.Q_network(np.array(s, dtype='float32'))
        y = y.numpy()
        Q1 = self.target_Q_network(np.array(s_, dtype='float32'))
        Q2 = self.Q_network(np.array(s_, dtype='float32'))
        next_action = np.argmax(Q2, axis=1)

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                # [敲黑板]
                # next_action是从Q_network计算出来的最大Q值的动作
                # 但输出的，是target_Q_network中的next_action的Q值。
                # 可以理解为：一个网络提议案，另外一个网络进行执行
                target = r + self.gamma * Q1[i][next_action[i]]
            target = np.array(target, dtype='float32')

            # print(y,i,a)
            # y 就是更新目标。
            y[i][a] = target
        return s, y

    def update_Q_network(self):
        '''
        更新Q_network，最小化target和Q的距离
        '''
        s, y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s, dtype='float32'))
            loss = tl.cost.mean_squared_error(Q, y)  # 最小化target和Q的距离
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.Q_network.trainable_weights))
        return loss

    def get_action(self, s):
        '''
        用epsilon-greedy的方式求动作。
        '''
        # 先随机一个数，如果比epsilon大，那么，就输出最大Q值的动作。
        if np.random.rand() >= self.epsilon:
            q = self.Q_network(np.array(s, dtype='float32').reshape([-1, self.input_dim]))
            a = np.argmax(q)
            # print("贪心得到动作", a % FOG_NUM + 1, q)
            return a
        # 否则，随机一个动作输出。
        else:
            a = random.randint(0, FOG_NUM * R_RANK_NUM - 1)
            # print("随机得到动作",a,a % FOG_NUM+1,a // FOG_NUM+1)
            return a

    ## 开始训练
    def train(self, episode):
        step = 0
        for ep in range(episode):
            # print("重置前",self.cur_capacity_dict)
            s, _ = self.env.reset(self.cur_capacity_dict.copy())  # 重置初始状态s
            # print("重置后", self.cur_capacity_dict)
            total_reward = 0
            total_loss = []

            # print("开始第",str(ep),"次游戏")

            while True:
                # 进行游戏
                a = self.get_action(s)
                s_, r, done, _p, _ = self.env.step(a)
                # print(s_ is None)
                # time.sleep(1)
                if s_ is None:
                    continue
                total_reward += r
                step += 1

                # 保存s, a, s_, r, done
                self.remember(s, a, s_, r, done)
                s = s_

                # 如果数据足够，那么就开始更新
                if len(self.memory) > self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss)
                    if (step + 1) % 5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                # 如果到最终状态，就打印一下成绩如何
                if done:
                    # print('EP:%i,  total_rewards:%f,   epsilon:%f, loss:%f' % (
                    # ep, total_reward, self.epsilon, np.mean(loss)))
                    break

    def test(self):
        '''
        :return: 返回的是[[],...,[]],[id,r_rank]
        '''
        reward_result_dict = {}
        for i in range(INFER_NUM):
            result = []
            reward = 0
            s, _ = self.env.reset(self.cur_capacity_dict)  # 重置初始状态s
            while True:
                # 进行游戏
                a = self.get_action(s)
                s_, r, done, _p, _ = self.env.step(a)
                if s_ is None:
                    continue
                reward += r
                result.append([a % FOG_NUM + 1, a // FOG_NUM + 1])
                # 如果到最终状态，就打印一下成绩如何
                if done:
                    # print("最后结果",i,result)
                    reward_result_dict[reward] = result
                    break
        # print('test结果：',reward_result_dict)
        max_reward = max(reward_result_dict.keys())
        return reward_result_dict[max_reward]


# 通过ddqn得到每次备份的策略
def ddqn_backup_strategy(d_size, d_r, local_fog_id, cur_d_r, cur_capacity_dict, ori_capacity_dict, distance,
                         max_min_dict):
    ddqn = Double_DQN(d_size, d_r, local_fog_id, cur_d_r, cur_capacity_dict, ori_capacity_dict, distance, max_min_dict)
    ddqn.train(TRAIN_NUM)

    return ddqn.test()
