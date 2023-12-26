import numpy as np

from gym import spaces
from CONFIG import FOG_NUM, R_RANK_NUM, UN_RS

SMALL_ZERO = 1e-6


class PendulumEnv():



    def __init__(self, fog_id, ori_un_reliability, d_size, d_reliability, fog_id_cur_capacity_dict,
                 fog_id_ori_capacity_dict, fog_c_r_r_mem_rank_max_min_dict, f2f_distance):
        '''

        :param fog_id:
        :param ori_un_reliability: 本地备份后的不可靠率
        :param d_size:
        :param d_reliability:
        :param fog_id_cur_capacity_dict:
        :param fog_id_ori_capacity_dict:
        :param fog_c_r_r_mem_rank_max_min_dict:
        :param f2f_distance: 到本地fog_id的距离
        '''
        self.state = None
        self.fog_id = fog_id  # 当前的fog_id
        self.ori_un_r = ori_un_reliability  # 本地备份后的不可靠性
        self.d_size = d_size
        self.d_reliability = d_reliability

        self.fog_mem_rate_dict = {}
        self.fog_id_cur_capacity_dict = fog_id_cur_capacity_dict
        self.fog_id_ori_capacity_dict = fog_id_ori_capacity_dict

        self.fog_c_r_r_mem_rank_max_min_dict = fog_c_r_r_mem_rank_max_min_dict
        self.f2f_distance = f2f_distance  # 这里应该是个一维数组，即针对于self.fog_id的

        for i in range(1, FOG_NUM + 1):
            self.fog_mem_rate_dict[i] = self.fog_id_cur_capacity_dict[i] / self.fog_id_ori_capacity_dict[i]

        self.action_space = spaces.Box(
            low=0, high=FOG_NUM*R_RANK_NUM, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=np.array([0, 0.0, 1]), high=np.array([np.inf, self.ori_un_r, np.inf]),
                                         dtype=np.float32)


    def step(self, a):
        if not self.is_valid(a):
            return None, 0, False

        assert self.state is not None, "调用step函数前需要先reset"

        # print("开始step: 动作为",a)
        # print("原状态为: ",self.state)
        max_distance, cur_un_r, cur_backup_num = self.state

        fog_id = a % FOG_NUM + 1  # 所选fog_id
        r_rank = a // FOG_NUM + 1  # 要存的终端可靠性等级

        temp_distance = self.f2f_distance[fog_id - 1]  # 两个节点距离

        # 更新state
        temp_un_r = cur_un_r * UN_RS[r_rank - 1]  # 更新数据不可靠率
        self.fog_id_cur_capacity_dict[fog_id] -= self.d_size  # 更新内存利用情况
        self.fog_mem_rate_dict[fog_id] = self.fog_id_cur_capacity_dict[fog_id] / self.fog_id_ori_capacity_dict[fog_id]

        # print(temp_distance,max_distance,temp_un_r,cur_backup_num)
        self.state = np.array([int(max(max_distance, temp_distance)), temp_un_r, int(cur_backup_num + 1)])

        # print("更新后状态为：", self.state)

        # TODO: 计算cost
        # 当前fog_id的内存利用率在其他中的占比，归一化得出一个数
        # 当前时延
        # 1. 内存剩余率 越大越优先

        max_mem_rate = max(self.fog_mem_rate_dict.values())
        min_mem_rate = min(self.fog_mem_rate_dict.values())

        mem_normalized = (self.fog_mem_rate_dict[fog_id] - min_mem_rate) / (
                max_mem_rate - min_mem_rate + SMALL_ZERO)

        # 2.存储资源消耗量  backup num  （最多6次备份即可） 越小越好
        backup_rate = (cur_backup_num + 1) / 6

        # 3.时延（长度） 最大时延  越小越优先
        max_length = max(self.f2f_distance)
        min_length = min(self.f2f_distance)
        time_normalized = (self.state[0] - min_length) / (max_length - min_length + 1)

        cost = 1.5 * backup_rate + time_normalized - mem_normalized

        # print("-cost为:", -cost)

        # 可靠性满足，则返回的done为True即可
        if temp_un_r <= 1 - self.d_reliability:
            # 可靠性已经满足
            return self._get_obs(), -cost, True
        # 未满足，继续
        return self._get_obs(), -cost, False

    def reset(self, fog_id_cur_capacity_dict):
        self.fog_id_cur_capacity_dict = fog_id_cur_capacity_dict

        # 初始状态为只有本地存储了，所以时延为0，不可靠率为本地不可靠率，备份了1份
        self.state = np.array([int(0), self.ori_un_r, int(1)])
        return self._get_obs(), {}

    def render(self):
        print("不展示hh")

    def close(self):
        i = 1

    # # 将self.state构建成状态空间
    def _get_obs(self):
        return np.array([max(0, int(self.state[0])), max(0.0, min(1.0, self.state[1])), max(0, int(self.state[2]))])

    def is_valid(self, a):
        err_msg = f"{a!r} ({type(a)}) 动作不合法"

        temp_max_min = self.fog_c_r_r_mem_rank_max_min_dict[a // FOG_NUM + 1][a % FOG_NUM]
        # print(a,temp_max_min)

        if temp_max_min[0] < 0:
            # 当前雾节点没有该可靠性等级
            return False

        for i in range(1, R_RANK_NUM + 1):
            if self.fog_c_r_r_mem_rank_max_min_dict[i][FOG_NUM - 1][0] > 1:
                # 最后一个节点还没有全满
                if temp_max_min[0] <= 1:
                    # 当前终端容量在500-1000之间
                    return False
        # 最后一个终端接近全满
        return True

