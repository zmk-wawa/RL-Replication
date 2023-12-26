# -*- coding: utf-8 -*-
# @Time    : 2023-02-11 14:52
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : SubCloud.py
import pickle
import random
import socket
import threading
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

import CONFIG
import Method
import Communicator
from Diagram.Dijkstra import dijkstra
from de import ga_train

######## ddqn ########
from ddqn import ddqn_backup_strategy


class subCloud:
    def __init__(self, sub_cloud_id):
        self.cloud_socket = None
        self.sub_cloud_addr = (Method.gen_ip(), CONFIG.CLOUD_ADDR[sub_cloud_id - 1])
        self.fog_ids = []  # 所管辖的fogs的id
        self.fog_sockets_dict = {}  # fog_id -> fog_socket

        # dcaba
        self.fog_total_mem_rate = {}

        # method 2
        self.last_r_rank = 0

        self.fog_id_ori_capacity_dict = {}  # id -> ori_capacity
        self.fog_id_cur_capacity_dict = {}  # id -> cur_capacity
        # fog_id -> fog所拥有的各个可靠性档次的终端内存利用率(1为没有或者已满)和最大内存与最小内存等级
        # eg：
        # 可靠性为1档的终端中：fog1内存利用率为0.3，fog2当前内存利用率为0.6..
        self.fog_c_r_r_mem_rate_dict = {}
        # eg: 1 -> [[6,2],[5,1]...] 可靠性为1档终端中：fog1的终端最大内存为6档，最小为2档;fog2最大5档，最小1档
        self.fog_c_r_r_mem_rank_max_min_dict = {}
        self.f2f_distance, temp_route = dijkstra(CONFIG.FOG_NUM, CONFIG.FOG_CONN_EDGE_NUM)  # 两两雾节点之间的最小传播时延

        # method 1
        self.fog_c_r_r_ids_dict = {}  # fog_id -> fog所拥有的各个可靠性档次的终端id们
        self.fog_c_c_r_ids_dict = {}  # fog_id -> fog所拥有的各个容量档次的终端id们

        self.c_r_fs_dict = {}  # c_rank 1~8 -> 该rank下的fog们
        self.r_r_fs_dict = {}  # r_rank 1~8 -> 该rank下的fog们

        self.f_ids_utilization = {}

        # 初始化c_r_fs_dict 和 r_r_fs_dict
        for i in range(0, CONFIG.C_RANK_NUM + 1):
            self.c_r_fs_dict[i] = []

        for i in range(1, CONFIG.R_RANK_NUM + 1):
            self.r_r_fs_dict[i] = []

        self.fog_socket_config()

        self.tell_fog_rs_nums()

        print("初始值：", self.fog_c_r_r_mem_rank_max_min_dict)

        # print(self.cloud_socket)
        # print(self.f2f_distance)

    # 开始备份前与各个普通fog的socket连接配置
    def fog_socket_config(self):
        # 初始化一些参数
        for i in range(1, CONFIG.R_RANK_NUM + 1):
            fog_num = CONFIG.FOG_NUM
            self.fog_c_r_r_mem_rank_max_min_dict[i] = []
            self.fog_c_r_r_mem_rate_dict[i] = []
            while fog_num > 0:
                self.fog_c_r_r_mem_rank_max_min_dict[i].append([])
                self.fog_c_r_r_mem_rate_dict[i].append([])
                fog_num -= 1

        self.cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 防止启动时端口被占用
        self.cloud_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定并监听
        self.cloud_socket.bind(self.sub_cloud_addr)
        self.cloud_socket.listen(CONFIG.FOG_NUM + 10)

        fog_sockets = []
        while len(fog_sockets) < CONFIG.FOG_NUM:
            conn_socket, conn_addr = self.cloud_socket.accept()
            fog_sockets.append(conn_socket)

        # 统计各个fog的信息和其终端信息
        while len(self.fog_sockets_dict) < CONFIG.FOG_NUM:
            msg = Communicator.recvMsg(fog_sockets[0])
            # msg = fog_sockets[0].recv(CONFIG.MAX_TCP_SIZE)
            if msg:
                temp_data = pickle.loads(msg)
                fog_id = temp_data['fog_id']
                self.fog_ids.append(fog_id)
                self.fog_sockets_dict[fog_id] = fog_sockets[0]
                del fog_sockets[0]

        for i in self.fog_sockets_dict.keys():
            self.fog_total_mem_rate[i] = 1.0

    # 开始备份前与各个普通fog的连接后初始信息的接收确认
    def handle_fog_config_msg(self, temp_data):
        fog_id = temp_data['fog_id']
        self.fog_ids.append(fog_id)
        self.fog_c_r_r_ids_dict[fog_id] = temp_data['c_r_r_ids']
        self.fog_c_c_r_ids_dict[fog_id] = temp_data['c_c_r_ids']
        self.fog_id_cur_capacity_dict[fog_id] = temp_data['initial_mem']
        self.fog_id_ori_capacity_dict[fog_id] = temp_data['initial_mem']

        f_r_rate = temp_data['r_r_mem_rates']
        f_r_mem_max_min = temp_data['max_min']

        for i in range(1, CONFIG.R_RANK_NUM + 1):
            if len(self.fog_c_r_r_ids_dict[fog_id][i]) > 0:
                self.r_r_fs_dict[i].append(fog_id)

        for i in range(0, CONFIG.C_RANK_NUM + 1):
            if len(self.fog_c_c_r_ids_dict[fog_id][i]) > 0:
                self.c_r_fs_dict[i].append(fog_id)

        for i in range(1, CONFIG.R_RANK_NUM + 1):
            self.fog_c_r_r_mem_rate_dict[i][fog_id - 1] = f_r_rate[i]
            self.fog_c_r_r_mem_rank_max_min_dict[i][fog_id - 1] = f_r_mem_max_min[i]

    # 计算节点负载：0.9*mem_rate+0.1*distance/100
    def cal_f_loads(self, mem_rate, source_fog_id, target_fog_id):
        return mem_rate * CONFIG.MEM_RATE_LOAD - CONFIG.DISTANCE_LOAD * self.f2f_distance[source_fog_id - 1][
            target_fog_id - 1] / 100

    # 更新
    def update1_fog_c_rank(self, temp_data):
        fog_id = temp_data['fog_id']
        c_c_rank_ids = temp_data['c_c_rank_ids']
        self.fog_c_c_r_ids_dict[fog_id] = c_c_rank_ids
        up_c_rank = temp_data['up_c_rank']
        down_c_rank = temp_data['down_c_rank']
        if len(c_c_rank_ids[up_c_rank]) == 1:
            # 由0变1
            self.c_r_fs_dict[up_c_rank].append(fog_id)
        if len(c_c_rank_ids[down_c_rank]) == 0:
            # 由1变0
            self.c_r_fs_dict[down_c_rank].remove(fog_id)

    def update2_fog_c_rank(self, temp_data):
        fog_id = temp_data['fog_id']
        f_r_rate = temp_data['r_r_mem_rates']
        f_r_mem_max_min = temp_data['max_min']
        for i in range(1, CONFIG.R_RANK_NUM + 1):
            self.fog_c_r_r_mem_rate_dict[i][fog_id - 1] = f_r_rate[i]
            self.fog_c_r_r_mem_rank_max_min_dict[i][fog_id - 1] = f_r_mem_max_min[i]
        # print("****** 更新了 *****",temp_data,self.fog_c_r_r_mem_rank_max_min_dict)

    def update_dcaba_fog_c_rank(self, temp_data):
        fog_id = temp_data['fog_id']
        new_rate = temp_data['f_total_mem_rate']

        self.fog_total_mem_rate[fog_id] = new_rate

        # 按照fog_total_mem_rate从大到小的顺序排序
        new_order = sorted(self.fog_total_mem_rate.items(), key=lambda x: x[1], reverse=True)

        if new_order[0][0] == fog_id:
            # 源雾节点是最大的
            return new_order[1][0]
        else:
            return new_order[0][0]

    def update_fog_mem_rate_after_backup(self, d_size, selected_fog_ids):
        for fog_id in selected_fog_ids:
            self.fog_id_cur_capacity_dict[fog_id] -= d_size

    ######## RL ########
    def update_fog_cur_capacity_after_ddqn(self, fog_ids, data_c):
        for fog_id in fog_ids:
            self.fog_id_cur_capacity_dict[fog_id] -= (data_c * len(fog_ids[fog_id]))

    def receive_msg_from_fog(self, fog_id):
        while True:
            msg = Communicator.recvMsg(self.fog_sockets_dict[fog_id])
            # msg = self.fog_sockets_dict[fog_id].recv(CONFIG.MAX_TCP_SIZE)
            # print(msg)
            if msg:
                temp_thread = threading.Thread(target=self.handle_fog_msg, args=(msg,))
                temp_thread.start()

    # 决策
    def tell_fog_rs_nums(self):
        pool = ThreadPoolExecutor(max_workers=CONFIG.FOG_NUM)

        pool.map(self.receive_msg_from_fog, list(self.fog_sockets_dict.keys()))

    # method 1 随机选两个（再加上自身其中终端不能重复，但可以存在同一雾节点下）
    def select_by_method1(self, from_fog_id, c_rank):
        temp_c = []
        if c_rank == CONFIG.C_RANK_NUM:
            # 最大等级的数据，那只能试试看能不能存了
            temp_c = self.c_r_fs_dict[c_rank]
        else:
            for i in range(c_rank + 1, CONFIG.C_RANK_NUM + 1):
                temp_c += self.c_r_fs_dict[i]

        # print(temp_c)
        if len(temp_c) < 2:
            return -1
        return random.sample(temp_c, 2)

    # 自己设计的方法，要排序
    def select_by_method2(self, data_r, data_c_rank, source_c_r_rank, source_fog_id):
        # F_LOAD = a * (CPU_Usage / CPU_Max) + b * (Mem_Usage / Mem_Max) +
        # c * (Net_Usage / Net_Max) + d * (H_Res / H_Res_Max)
        # 其中 a + b + c + d = 1
        # CPU_Usage / CPU_Max：表示服务器的CPU使用率；
        # Mem_Usage / Mem_Max：表示服务器的内存使用率；
        # Net_Usage / Net_Max：表示服务器的网络带宽使用率；
        # H_Res / H_Res_Max：表示服务器的资源使用情况。
        # 不执行任务时，CPU使用率和服务器网络带宽使用率都为1

        now_un_r = CONFIG.UN_RS[source_c_r_rank - 1]
        data_un_r = 1 - data_r
        selected_ids = []
        selected_r_ranks = []
        if self.last_r_rank < CONFIG.R_RANK_NUM:
            for i in range(self.last_r_rank + 1, CONFIG.R_RANK_NUM + 1):
                max_rate = -1000
                max_fog_id = []
                temp_i_rate = self.fog_c_r_r_mem_rate_dict[i]
                temp_i_max_mins = self.fog_c_r_r_mem_rank_max_min_dict[i]
                # print(i,' ',temp_i_rate,temp_i_max_mins)
                if i == source_c_r_rank:
                    # 当前可靠性等级下，暂且不要选择源雾节点
                    for k in range(0, len(temp_i_max_mins)):
                        if k + 1 == source_fog_id:
                            continue
                        if temp_i_max_mins[k][0] > data_c_rank:
                            if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                                max_fog_id = [k + 1]
                            elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_fog_id.append(k + 1)
                else:
                    # print('max_min',temp_i_max_mins)
                    for k in range(0, len(temp_i_max_mins)):
                        # print('k: ',k)
                        if temp_i_max_mins[k][0] > data_c_rank:
                            if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                                max_fog_id = [k + 1]
                            elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_fog_id.append(k + 1)
                # print(max_fog_id)
                if len(max_fog_id) > 0:
                    selected_id = random.choice(max_fog_id)
                    selected_ids.append(selected_id)
                    selected_r_ranks.append(i)
                    now_un_r *= CONFIG.UN_RS[i - 1]
                    if now_un_r <= data_un_r:
                        self.last_r_rank = i
                        # print('fogs:', selected_ids,selected_r_ranks)
                        return selected_ids, selected_r_ranks

        while True:
            cur_len = len(selected_ids)
            for i in range(1, CONFIG.R_RANK_NUM + 1):
                max_rate = -1000
                max_fog_id = []
                temp_i_rate = self.fog_c_r_r_mem_rate_dict[i]
                temp_i_max_mins = self.fog_c_r_r_mem_rank_max_min_dict[i]
                if i == source_c_r_rank:
                    for k in range(0, len(temp_i_max_mins)):
                        if k + 1 == source_fog_id:
                            continue
                        if temp_i_max_mins[k][0] > data_c_rank:
                            if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                                max_fog_id = [k + 1]
                            elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_fog_id.append(k + 1)
                else:
                    for k in range(0, len(temp_i_max_mins)):
                        if temp_i_max_mins[k][0] > data_c_rank:
                            if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                                max_fog_id = [k + 1]
                            elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                                max_fog_id.append(k + 1)
                if len(max_fog_id) > 0:
                    selected_id = random.choice(max_fog_id)
                    selected_ids.append(selected_id)
                    selected_r_ranks.append(i)
                    now_un_r *= CONFIG.UN_RS[i - 1]
                    if now_un_r <= data_un_r:
                        self.last_r_rank = i
                        # print('fogs:', selected_ids,selected_r_ranks)
                        return selected_ids, selected_r_ranks

            if len(selected_ids) == cur_len:
                # 没有筛选出来什么
                return [], []

    # dcaba 找出除了自身雾节点外，负载最小的那个雾节点，在其中选择两个终端设备
    def select_by_method_dcaba(self, fog_id, new_rate):
        # 更新源雾节点的total_mem_rate
        self.fog_total_mem_rate[fog_id] = new_rate

        # 按照fog_total_mem_rate从大到小的顺序排序
        new_order = sorted(self.fog_total_mem_rate.items(), key=lambda x: x[1], reverse=True)

        if new_order[0][0] == fog_id:
            # 源雾节点是最大的
            return new_order[1][0]
        else:
            return new_order[0][0]

    def select_by_method_clb(self, fog_id):
        # print('开始选')
        # print(self.f2f_distance)
        temp_id = fog_id - 1
        # print(self.f2f_distance[temp_id])
        temp_distance = sorted(self.f2f_distance[temp_id])
        print(temp_distance)
        min1 = temp_distance[1]  # 最小的应该是本地，为0，所以跳过
        min2 = temp_distance[2]
        print(fog_id, ' Method：',
              {self.f2f_distance[temp_id].index(min1) + 1: 1, self.f2f_distance[temp_id].index(min2) + 1: 1})
        return {self.f2f_distance[temp_id].index(min1) + 1: 1, self.f2f_distance[temp_id].index(min2) + 1: 1}

    def select_by_method_aco_vmm(self, fog_id):
        temp_fog_dis_mem_loads = {}  # 除了源雾节点外，其他雾节点的total_mem_rate - 其他雾节点到源雾节点distance
        for i in self.fog_total_mem_rate.keys():
            if i == fog_id:
                continue
            temp_fog_dis_mem_loads[i] = self.cal_f_loads(self.fog_total_mem_rate[i], fog_id, i)

        # loads 从大到小 优先取大的，负载高/距离近
        sorted_fog_dis_mem_loads = sorted(temp_fog_dis_mem_loads.items(), key=lambda x: x[1], reverse=True)
        return {sorted_fog_dis_mem_loads[0][0]: 1, sorted_fog_dis_mem_loads[1][0]: 1}

    ######### IS2023 优化 ##########
    # dcaba 改成在edge侧存储
    # dcaba 找出除了自身雾节点外，负载最小的两个雾节点
    def select_by_method3_dcaba(self, fog_id, new_rate):
        # 更新源雾节点的total_mem_rate
        self.fog_total_mem_rate[fog_id] = new_rate

        # 按照fog_total_mem_rate从大到小的顺序排序
        new_order = sorted(self.fog_total_mem_rate.items(), key=lambda x: x[1], reverse=True)

        if new_order[0][0] == fog_id:
            # 源雾节点是最大的
            return {new_order[1][0]: 1, new_order[2][0]: 1}
        else:
            return {new_order[0][0]: 1, new_order[1][0]: 1}

    ############ RL-RANDOM ############
    # 随机选，直至满足可靠性
    def select_by_RL_RANDOM(self, data_r, data_cur_r):
        data_cur_un_r = 1 - data_cur_r
        result = []
        while 1 - data_r < data_cur_un_r:
            a = random.randint(0, CONFIG.FOG_NUM * CONFIG.R_RANK_NUM - 1)
            select_fog_id = a % CONFIG.FOG_NUM + 1  # 所选fog_id
            r_rank = a // CONFIG.FOG_NUM + 1  # 要存的终端可靠性等级
            if self.fog_c_r_r_mem_rank_max_min_dict[r_rank][select_fog_id - 1][0] < 1:
                continue
            result.append([select_fog_id,r_rank])
            data_cur_un_r *= CONFIG.UN_RS[r_rank - 1]
            # print(a,data_cur_un_r,1-data_r)

        return result

    ############ RL ############
    def select_by_method_ddqn(self, temp_type, data_r, data_c, fog_id, data_cur_r):
        rl_result = {}
        if temp_type == CONFIG.F2S_ASK_FOR_ANSWER_RL_DDQN_TO_STORE:
            rl_result = ddqn_backup_strategy(data_c, data_r, fog_id, 1 - data_cur_r,
                                             self.fog_id_cur_capacity_dict.copy(),
                                             self.fog_id_ori_capacity_dict, self.f2f_distance[fog_id - 1],
                                             self.fog_c_r_r_mem_rank_max_min_dict)
        else:
            if temp_type == CONFIG.F2S_ASK_FOR_ANSWER_GA_TO_STORE:
                method = CONFIG.GA_METHOD
            elif temp_type == CONFIG.F2S_ASK_FOR_ANSWER_DE_TO_STORE:
                method = CONFIG.DE_METHOD
            elif temp_type == CONFIG.F2S_ASK_FOR_ANSWER_PSO_TO_STORE:
                method = CONFIG.PSO_METHOD
            elif temp_type == CONFIG.F2S_ASK_FOR_ANSWER_RANDOM_TO_STORE:
                method = CONFIG.RANDOM_METHOD
            else:
                method = CONFIG.SA_METHOD

            if method == CONFIG.RANDOM_METHOD:
                rl_result = self.select_by_RL_RANDOM(data_r, data_cur_r)
                # print(rl_result)
            else:
                rl_result = ga_train(data_c, data_r, 1 - data_cur_r, self.fog_id_cur_capacity_dict.copy(),
                                     self.fog_id_ori_capacity_dict, self.f2f_distance[fog_id - 1],
                                     self.fog_c_r_r_mem_rank_max_min_dict, method)

        fog_ids = {}
        for a in rl_result:
            if a[0] in fog_ids:
                fog_ids[a[0]].append(a[1])
            else:
                fog_ids[a[0]] = [a[1]]
        rl_result = None
        # print("策略：",fog_ids)
        # 选完后更新一下内存空间
        self.update_fog_cur_capacity_after_ddqn(fog_ids, data_c)
        return fog_ids

    def handle_fog_msg(self, msg):
        temp_data = pickle.loads(msg)
        # print(temp_data)
        if temp_data['type'] == CONFIG.F2S_UPDATE_C_RANK_NUMS1:
            self.update1_fog_c_rank(temp_data)
        elif temp_data['type'] == CONFIG.F2S_UPDATE_C_RANK_NUMS2:
            # print('有更新', temp_data)
            self.update2_fog_c_rank(temp_data)
            # print('更新完')
        elif temp_data['type'] == CONFIG.F2S_HAVE_CLIENT_FULL:
            self.handle_have_client_full()
        elif temp_data['type'] == CONFIG.F2S_CONFIG:
            self.handle_fog_config_msg(temp_data)
        ####### RL_DDQN #######
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_RL_DDQN_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        ####### GA #######
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_GA_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        ####### DE #######
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_DE_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        ####### PSO #######
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_PSO_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        ####### SA #######
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_SA_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_RANDOM_TO_STORE:
            self.handle_f2s_ask_for_answer_ddqn_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER2_TO_STORE:
            self.handle_f2s_ask_for_answer2_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_DRPS_TO_STORE:
            self.handle_f2s_ask_for_answer2_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_DCABA_TO_STORE:
            self.handle_f2s_ask_for_answer_dcaba_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_CLB_TO_STORE:
            self.handle_f2s_ask_for_answer_clb_to_store(temp_data)
        elif temp_data['type'] == CONFIG.F2S_ASK_FOR_ANSWER_ACO_VMM_TO_STORE:
            self.handle_f2s_ask_for_answer_aco_vmm_to_store(temp_data)

    ############ RL/GA/DE/PSO/SA ############
    def handle_f2s_ask_for_answer_ddqn_to_store(self, temp_data):
        temp_type = temp_data['type']
        # print(temp_type)
        temp_data['type'] = CONFIG.S2F_TELL_ANSWER2_TO_STORE
        fog_id = temp_data['from_fog_id']
        data_c = temp_data['capacity']  # data的大小
        data_r = temp_data['reliability']  # data的可靠性要求
        source_c_r_rank = temp_data['cur_r_rank']  # 本地备份的终端的可靠性等级
        temp_data['store_fog_ids'] = self.select_by_method_ddqn(temp_type, data_r, data_c, fog_id,
                                                                CONFIG.R_RANKS[source_c_r_rank - 1])

        temp_data['protect'] = 1
        temp_data['start_time'] = datetime.now()
        Communicator.sendMsg(self.fog_sockets_dict[fog_id], pickle.dumps(temp_data))

    # 自己的方法的备份策略选择
    def handle_f2s_ask_for_answer2_to_store(self, temp_data):
        temp_data['type'] = CONFIG.S2F_TELL_ANSWER2_TO_STORE
        fog_id = temp_data['from_fog_id']
        data_c_rank = Method.convert_c_to_rank(temp_data['capacity'])
        data_r = temp_data['reliability']
        source_c_r_rank = temp_data['cur_r_rank']  # 本地备份的终端的可靠性等级
        if temp_data['method'] == CONFIG.METHOD2_STORE:
            f_ids, f_id_r_ranks = self.select_by_method2(data_r, data_c_rank, source_c_r_rank, fog_id)
        else:
            # DRPS
            f_ids, f_id_r_ranks = self.select_by_method3_drps(data_r, data_c_rank, source_c_r_rank, fog_id)
        store_fog_ids_r_rs_dict = {}
        for i in range(0, len(f_ids)):
            if f_ids[i] in store_fog_ids_r_rs_dict:
                store_fog_ids_r_rs_dict[f_ids[i]].append(f_id_r_ranks[i])
            else:
                store_fog_ids_r_rs_dict[f_ids[i]] = [f_id_r_ranks[i]]
        temp_data['store_fog_ids'] = store_fog_ids_r_rs_dict
        temp_data['protect'] = 1
        Communicator.sendMsg(self.fog_sockets_dict[fog_id], pickle.dumps(temp_data))
        # self.fog_sockets_dict[fog_id].sendall(pickle.dumps(temp_data))

    # dcaba方法的备份策略选择
    def handle_f2s_ask_for_answer_dcaba_to_store(self, temp_data):
        temp_data['type'] = CONFIG.S2F_TELL_ANSWER1_TO_STORE
        fog_id = temp_data['from_fog_id']
        new_rate = temp_data['new_total_rate']
        if temp_data['method'] == CONFIG.DCABA_STORE:
            select_fog = self.select_by_method_dcaba(fog_id, new_rate)
            source_fog_ids = {select_fog: 2}  # 不同的fog_id -> 备份数目
        else:
            source_fog_ids = self.select_by_method3_dcaba(fog_id, new_rate)
            # print(temp_data['index'], source_fog_ids)
        temp_data['store_fog_ids'] = source_fog_ids
        Communicator.sendMsg(self.fog_sockets_dict[fog_id], pickle.dumps(temp_data))
        # self.fog_sockets_dict[fog_id].sendall(pickle.dumps(temp_data))

    # clb方法的备份策略选择
    def handle_f2s_ask_for_answer_clb_to_store(self, temp_data):
        method = temp_data['method']
        if method == CONFIG.CLB_STORE:
            temp_data['type'] = CONFIG.S2F_TELL_ANSWER1_TO_STORE
            fog_id = temp_data['from_fog_id']
            source_fog_ids = self.select_by_method_clb(fog_id)  # 不同的fog_id -> 备份数目
        elif method == CONFIG.CLB_STORE3:
            temp_data['type'] = CONFIG.S2F_TELL_ANSWER1_TO_STORE
            fog_id = temp_data['from_fog_id']
            source_fog_ids = self.select_by_method_clb(fog_id)  # 不同的fog_id -> 备份数目

        temp_data['store_fog_ids'] = source_fog_ids
        Communicator.sendMsg(self.fog_sockets_dict[fog_id], pickle.dumps(temp_data))
        # self.fog_sockets_dict[fog_id].sendall(pickle.dumps(temp_data))

    # aco_vmm方法的备份策略选择
    def handle_f2s_ask_for_answer_aco_vmm_to_store(self, temp_data):
        temp_data['type'] = CONFIG.S2F_TELL_ANSWER1_TO_STORE
        fog_id = temp_data['from_fog_id']
        data_c_rank = Method.convert_c_to_rank(temp_data['capacity'])
        source_fog_ids = self.select_by_method_aco_vmm(fog_id)
        # print(temp_data['index'],source_fog_ids)
        temp_data['store_fog_ids'] = source_fog_ids
        Communicator.sendMsg(self.fog_sockets_dict[fog_id], pickle.dumps(temp_data))
        # self.fog_sockets_dict[fog_id].sendall(pickle.dumps(temp_data))

    # 有其中一个终端满了，那么所有雾节点终端均停止收发包，整个测试结束
    def handle_have_client_full(self):
        data = {'type': CONFIG.S2F_ASK_FOR_C_UTILIZATION, 'protect': 1}
        for i in self.fog_sockets_dict.keys():
            Communicator.sendMsg(self.fog_sockets_dict[i], pickle.dumps(data))

    ########## IS2023 ##########
    # dcaba 找出除了自身雾节点外，负载最小的那个雾节点，在其中选择两个终端设备
    # 所以dcaba 不需要重写
    # clb 多副本
    def select_by_method3_drps(self, data_r, data_c_rank, source_c_r_rank, source_fog_id):
        # 大于0.95 为高可靠性要求
        # 小于等于0.95 为低可靠性要求

        now_un_r = CONFIG.UN_RS[source_c_r_rank - 1]
        data_un_r = 1 - data_r
        selected_ids = []
        selected_r_ranks = []

        if data_r > 0.95:
            # 高可靠性要求
            high_r_rank_list = [5, 6, 7, 8]
            r_ranks = random.sample(high_r_rank_list, 2)
        else:
            low_r_rank_list = [1, 2, 3, 4]
            r_ranks = random.sample(low_r_rank_list, 2)

        for i in r_ranks:
            temp_selected_id = self.select_max_fog_id_in_r_rank(i, source_c_r_rank, source_fog_id, data_c_rank)
            if temp_selected_id < 0:
                continue
            selected_ids.append(temp_selected_id)
            selected_r_ranks.append(i)
            now_un_r *= CONFIG.UN_RS[i - 1]

        if now_un_r <= data_un_r:
            return selected_ids, selected_r_ranks
        else:
            return selected_ids, selected_r_ranks

    def select_max_fog_id_in_r_rank(self, i, source_c_r_rank, source_fog_id, data_c_rank):
        max_rate = -1000
        max_fog_id = []
        temp_i_rate = self.fog_c_r_r_mem_rate_dict[i]
        temp_i_max_mins = self.fog_c_r_r_mem_rank_max_min_dict[i]

        if i == source_c_r_rank:
            # 当前可靠性等级下，暂且不要选择源雾节点
            for k in range(0, len(temp_i_max_mins)):
                if k + 1 == source_fog_id:
                    continue
                if temp_i_max_mins[k][0] > data_c_rank:
                    if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                        max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                        max_fog_id = [k + 1]
                    elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                        max_fog_id.append(k + 1)
        else:
            # print('max_min',temp_i_max_mins)
            for k in range(0, len(temp_i_max_mins)):
                # print('k: ',k)
                if temp_i_max_mins[k][0] > data_c_rank:
                    if max_rate < self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                        max_rate = self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1)
                        max_fog_id = [k + 1]
                    elif max_rate == self.cal_f_loads(temp_i_rate[k], source_fog_id, k + 1):
                        max_fog_id.append(k + 1)
        # print(max_fog_id)
        if len(max_fog_id) > 0:
            selected_id = random.choice(max_fog_id)
            return selected_id

        return -1
