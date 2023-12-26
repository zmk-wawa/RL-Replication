# -*- coding: utf-8 -*-
# @Time    : 2023-02-11 14:52
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : Fog.py
import operator
import pickle
import random
import socket
import threading
import time

import pandas as pd

import CONFIG
import Method

from concurrent.futures import ThreadPoolExecutor

from Diagram.SearchPath import Path

import Communicator
from collections import Counter


class fog:
    def __init__(self, fog_id, fog_addr, client_num, sub_cloud_id):
        print(fog_addr)
        self.pool = None
        self.fog_socket = None
        self.fog_id = fog_id
        self.fog_addr = fog_addr
        self.client_num = client_num
        self.sub_cloud_id = sub_cloud_id

        self.initial_c_c_dict = {}  # c_id -> capacity 最初的终端的可用容量 用来统计总内存利用率
        self.cur_c_c_dict = {}  # 用来统计总内存利用率 c_id -> cur_capacity

        # dcaba
        self.cur_total_mem = 0
        self.initial_total_mem = 0

        self.cs_r_ranks_dict = {}  # c_id -> reliability_rank  可靠性等级 注：等级均为int型
        self.cs_c_ranks_dict = {}  # c_id -> capacity_rank 容量等级 注：等级均为int型

        self.client_sockets_dict = {}  # client_id -> client_socket
        self.c_r_rank_c_ids_dict = {}  # r_rank -> c_ids
        self.c_c_rank_c_ids_dict = {}  # c_rank -> c_ids

        # 仅在分级的方法中用到
        self.c_r_rank_mem_rate = {}  # r_rank -> rate 内存使用率
        self.c_r_rank_c_rank_ids_dict = {}  # r_rank -> [[],[],[]...]各个r_rank下的各个等级的c_rank下的c_ids
        self.c_r_r_mem_rank_max_min_dict = {}  # r_rank -> [max,min]
        self.c_r_r_initial_sum_mem = {}  # r_rank -> initial_sum_capacity_sum 各个可靠性等级的终端的原始总内存
        self.c_r_r_cur_sum_mem = {}  # r_rank -> cur_sum_capacity 各个可靠性等级的终端的当前总内存

        self.f2f_c_sockets_dict = {}  # 该雾节点的fog_client们的socket
        self.f2f_s_sockets_dict = {}  # 该雾节点去连对应fog_server，而自己产生的socket  fog_server_id -> 对应client的socket

        self.sub_cloud_socket = None  # 和子云通信的socket

        self.have_store_index = {}  # 已经存好的但是Client那边还没确认，index -> (fog_id,num)
        self.wait_store_ack_cs = {}  # 发送了存储请求，等待ack确认 (index) -> [ids]
        self.wait_store_ack_cs_rs = {}  # 主要在三副本时用到 (index) -> [rs]
        self.wait_store_ack_fogs = {}  # 向fogs发送了存储请求，完成存储后删一笔 index -> [fogs]
        self.wait_store_ack_is_not_all_store = []  # 失败的地方 [index,...]
        self.store_cs_r = {}  # 最终备份的终端当时的cur_r们 index -> [cur_rs]
        self.already_store_fog_id = {}  # 已经收集到的备份 index ->

        self.path = Path()

        self.which_method = -1

        # 初始化c_c_rank_nums 和 c_r_rank_nums
        for i in range(0, CONFIG.C_RANK_NUM + 1):
            self.c_c_rank_c_ids_dict[i] = []

        for i in range(1, CONFIG.R_RANK_NUM + 1):
            self.c_r_rank_c_ids_dict[i] = []

        self.f2c_f2f_sockets_config(client_num)

        self.cal_c_r_r_initial_sum_mem()
        print(self.c_r_rank_mem_rate)
        print(self.c_r_r_mem_rank_max_min_dict)
        print(self.cs_c_ranks_dict)
        print(self.cs_r_ranks_dict)
        print(self.c_c_rank_c_ids_dict)
        self.f2SC_sockets_config()
        time.sleep(1)

        self.send_sc_config()


        print('初始',self.c_r_rank_c_rank_ids_dict)

        self.receive_msg()
        # self.cal_c_r_r_initial_sum_mem()

    ## 初始统计管辖终端的总容量
    def cal_c_r_r_initial_sum_mem(self):
        for i in range(1, CONFIG.R_RANK_NUM + 1):
            self.c_r_r_initial_sum_mem[i] = 0
            self.c_r_r_cur_sum_mem[i] = 0
            self.c_r_rank_mem_rate[i] = 1.0
            # max , min
            self.c_r_r_mem_rank_max_min_dict[i] = [-1, CONFIG.C_RANK_NUM + 1]
            self.c_r_rank_c_rank_ids_dict[i] = []
            # 0等级代表存满
            for j in range(0, CONFIG.C_RANK_NUM + 1):
                self.c_r_rank_c_rank_ids_dict[i].append([])

        for i in self.cs_r_ranks_dict.keys():
            r_rank = self.cs_r_ranks_dict[i]
            c_rank = self.cs_c_ranks_dict[i]
            c_rank_max = self.c_r_r_mem_rank_max_min_dict[r_rank][0]
            c_rank_min = self.c_r_r_mem_rank_max_min_dict[r_rank][1]

            self.initial_total_mem += self.initial_c_c_dict[i]
            self.c_r_rank_c_rank_ids_dict[r_rank][c_rank].append(i)
            if c_rank_max < c_rank:
                self.c_r_r_mem_rank_max_min_dict[r_rank][0] = c_rank
            if c_rank_min > c_rank:
                self.c_r_r_mem_rank_max_min_dict[r_rank][1] = c_rank

            self.c_r_r_initial_sum_mem[r_rank] += self.initial_c_c_dict[i]
            self.c_r_r_cur_sum_mem[r_rank] += self.initial_c_c_dict[i]

        self.cur_total_mem = self.initial_total_mem

    # 各个socket开启，身份认证
    def f2c_f2f_sockets_config(self, client_num):
        f2f_c_sockets_num = CONFIG.FOG_NUM - self.fog_id

        self.fog_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 防止启动时端口被占用
        self.fog_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定并监听
        self.fog_socket.bind(self.fog_addr)
        self.fog_socket.listen(client_num + f2f_c_sockets_num + 10)

        # 获取所有应得的client/fog的连接
        temp_sockets = []
        while len(temp_sockets) < self.client_num + f2f_c_sockets_num:
            # 接收并存储addr与socket
            conn_socket, conn_addr = self.fog_socket.accept()
            temp_sockets.append(conn_socket)

        # 判断连接是fog还是Client 以及 fog client的id
        while len(temp_sockets) > 0 & len(self.client_sockets_dict) + len(
                self.f2f_c_sockets_dict) < client_num + f2f_c_sockets_num:
            msg = Communicator.recvMsg(temp_sockets[0])
            temp_data = pickle.loads(msg)
            # print(temp_data)
            if temp_data['role'] == CONFIG.FOG_ROLE:
                # print(temp_data['id'])
                self.f2f_c_sockets_dict[temp_data['id']] = temp_sockets[0]
            elif temp_data['role'] == CONFIG.CLIENT_ROLE:
                # print(temp_data)
                c_id = temp_data['id']
                self.client_sockets_dict[c_id] = temp_sockets[0]
                self.cs_r_ranks_dict[c_id] = temp_data['r_rank']
                self.cs_c_ranks_dict[c_id] = temp_data['c_rank']
                self.initial_c_c_dict[c_id] = temp_data['c']  # 原来的capacity
                self.cur_c_c_dict[c_id] = temp_data['c']
                # 顺带统计拥有各个等级的终端的个数
                self.c_r_rank_c_ids_dict[temp_data['r_rank']].append(c_id)
                self.c_c_rank_c_ids_dict[temp_data['c_rank']].append(c_id)
            else:
                print('信息出错')
            del temp_sockets[0]

        # 发送作为fog_client的TCP连接请求
        if self.fog_id > 1:
            for i in range(1, self.fog_id):
                temp_f2f_s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                temp_f2f_s_socket.connect(
                    (Method.gen_ip(), CONFIG.FOG_ADDRS[i - 1]))  # print('请求fog', i + 1, '成功addr', CONFIG.FOG_ADDRS[i])
                self.f2f_s_sockets_dict[i] = temp_f2f_s_socket

        # 作为fog_client向对应Server发送自己的身份信息
        msg = pickle.dumps({'role': CONFIG.FOG_ROLE, 'id': self.fog_id})
        for i in self.f2f_s_sockets_dict.keys():
            # print('我要向fog', str(i), '发送自己的信息')
            Communicator.sendMsg(self.f2f_s_sockets_dict[i], msg)
            # self.f2f_s_sockets_dict[i].sendall(msg)

    # 与子云的socket配置
    def f2SC_sockets_config(self):
        # fog向sub_Cloud发信息，汇报自己和所拥有的device的情况
        self.sub_cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sub_cloud_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sub_cloud_socket.connect((Method.gen_ip(), CONFIG.CLOUD_ADDR[self.sub_cloud_id - 1]))

        temp_data = {'fog_id': self.fog_id}
        Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))


    def send_sc_config(self):
        msg = pickle.dumps(
            {'type': CONFIG.F2S_CONFIG, 'fog_id': self.fog_id, 'c_r_r_ids': self.c_r_rank_c_ids_dict,
             'c_c_r_ids': self.c_c_rank_c_ids_dict,
             'r_r_mem_rates': self.c_r_rank_mem_rate, 'max_min': self.c_r_r_mem_rank_max_min_dict,'initial_mem':self.initial_total_mem})
        # print('要发给subCloud了')
        Communicator.sendMsg(self.sub_cloud_socket, msg)
        # self.sub_cloud_socket.sendall(msg)

    # 对每一个client cfog sfog配置一个线程,用于轮询等待接收数据
    def receive_msg(self):
        self.pool = ThreadPoolExecutor(
            max_workers=self.client_num + len(self.f2f_c_sockets_dict) + len(self.f2f_s_sockets_dict) + 1)

        self.pool.submit(self.receive_msg_from_subC)
        self.pool.map(self.receive_msg_from_c, list(self.client_sockets_dict.keys()))
        self.pool.map(self.receive_msg_from_cfog, list(self.f2f_c_sockets_dict.keys()))
        self.pool.map(self.receive_msg_from_sfog, list(self.f2f_s_sockets_dict.keys()))


    # 轮询接收来自client_id的信息（会运行在一个线程中）
    def receive_msg_from_c(self, client_id):
        while True:
            msg = Communicator.recvMsg(self.client_sockets_dict[client_id])
            # msg = self.client_sockets_dict[client_id].recv(CONFIG.MAX_TCP_SIZE)
            if msg:
                # self.pool.submit(self.handle_client_msg, msg)
                temp_thread = threading.Thread(target=self.handle_client_msg, args=(msg,))
                temp_thread.start()

    # 轮询接收来自cfog_id的信息（会运行在一个线程中）
    def receive_msg_from_cfog(self, cfog_id):
        while True:
            msg = Communicator.recvMsg(self.f2f_c_sockets_dict[cfog_id])
            if msg:
                temp_thread = threading.Thread(target=self.handle_fog_msg, args=(msg,))
                temp_thread.start()

    # 轮询接收来自sfog_id的信息（会运行在一个线程中）
    def receive_msg_from_sfog(self, sfog_id):
        while True:
            msg = Communicator.recvMsg(self.f2f_s_sockets_dict[sfog_id])
            if msg:
                temp_thread = threading.Thread(target=self.handle_fog_msg, args=(msg,))
                temp_thread.start()

    # 轮询接收来自subC的信息（会运行在一个线程中）
    def receive_msg_from_subC(self):
        print('开启了sub线程')
        while True:
            msg = Communicator.recvMsg(self.sub_cloud_socket)
            if msg:
                temp_thread = threading.Thread(target=self.handle_sub_cloud_msg, args=(msg,))
                temp_thread.start()


    def update1_client_c_r_one_client_full(self, c_id, cur_c_rank, cur_c, cur_r_rank):
        # 先在fog中更新终端当前的容量
        self.c_r_r_cur_sum_mem[cur_r_rank] -= (self.cur_c_c_dict[c_id] - cur_c)
        self.cur_total_mem -= (self.cur_c_c_dict[c_id] - cur_c)
        self.c_r_rank_mem_rate[cur_r_rank] = self.c_r_r_cur_sum_mem[cur_r_rank] / self.c_r_r_initial_sum_mem[cur_r_rank]
        self.cur_c_c_dict[c_id] = cur_c
        temp_c_c_rank = None
        temp_c_c_rank = self.cs_c_ranks_dict[c_id]
        if temp_c_c_rank == cur_c_rank:
            # 容量等级未发生改变
            return
        else:
            self.cs_c_ranks_dict[c_id] = cur_c_rank
            self.c_c_rank_c_ids_dict[temp_c_c_rank].remove(c_id)
            self.c_c_rank_c_ids_dict[cur_c_rank].append(c_id)
            # 有终端存满，告知子云节点
            if cur_c_rank == 0:
                # 有终端存满
                print('满了满了，有终端满了', c_id)
                data = {'type': CONFIG.F2S_HAVE_CLIENT_FULL, 'fog_id': self.fog_id}
                Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(data))
                self.sub_cloud_socket.sendall(pickle.dumps(data))
                return
            if len(self.c_c_rank_c_ids_dict[temp_c_c_rank]) == 0 or len(self.c_c_rank_c_ids_dict[cur_c_rank]) == 1:
                # fog管辖的某个容量等级的终端数目由1变0，或由0变1，要告知子云节点更新
                temp_data = {'fog_id': self.fog_id, 'c_c_rank_ids': self.c_c_rank_c_ids_dict,
                             'type': CONFIG.F2S_UPDATE_C_RANK_NUMS1, 'up_c_rank': cur_c_rank,
                             'down_c_rank': temp_c_c_rank}
                Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
                # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))

            # print('更新后：', self.c_c_rank_c_ids_dict)

    def update2_client_c_r_one_client_full(self, c_id, cur_c_rank, cur_c, cur_r_rank, last_c_rank):
        # 先在fog中更新终端当前的容量
        self.c_r_r_cur_sum_mem[cur_r_rank] -= (self.cur_c_c_dict[c_id] - cur_c)
        self.c_r_rank_mem_rate[cur_r_rank] = self.c_r_r_cur_sum_mem[cur_r_rank] / self.c_r_r_initial_sum_mem[cur_r_rank]
        self.cur_c_c_dict[c_id] = cur_c
        temp_c_c_rank = None
        temp_c_c_rank = last_c_rank

        # print(self.c_r_rank_mem_rate)
        if temp_c_c_rank == cur_c_rank:
            # 容量等级未发生改变
            return
        else:
            print('改变了',self.c_r_rank_c_rank_ids_dict)
            # max/min发生改变时，告知子云节点
            if c_id in self.c_r_rank_c_rank_ids_dict[cur_r_rank][temp_c_c_rank]:
                self.c_r_rank_c_rank_ids_dict[cur_r_rank][temp_c_c_rank].remove(c_id)
            # 容量等级发生改变
            if cur_c_rank == 0:
                print('满了满了，有终端满了', c_id)
                self.c_c_rank_c_ids_dict[0].append(c_id)
                # 已存满,告知子云节点
                data = {'type': CONFIG.F2S_HAVE_CLIENT_FULL, 'fog_id': self.fog_id}
                Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(data))
                # self.sub_cloud_socket.sendall(pickle.dumps(data))
                return

            # cur_r_rank可靠性等级下容量等级为cur_c_rank的终端数目由无变有了
            if len(self.c_r_rank_c_rank_ids_dict[cur_r_rank][cur_c_rank]) == 0:
                if self.c_r_r_mem_rank_max_min_dict[cur_r_rank][1] == temp_c_c_rank:
                    # 原来是最小容量档次，那就尾没了，改变
                    self.c_r_r_mem_rank_max_min_dict[cur_r_rank][1] = cur_c_rank
                    temp_data = {'fog_id': self.fog_id, 'type': CONFIG.F2S_UPDATE_C_RANK_NUMS2,
                                 'r_r_mem_rates': self.c_r_rank_mem_rate, 'max_min': self.c_r_r_mem_rank_max_min_dict}
                    Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
                    # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))

            self.c_r_rank_c_rank_ids_dict[cur_r_rank][cur_c_rank].append(c_id)
            # cur_r_rank可靠性等级下容量等级为temp_c_c_rank的终端数目由有变无了
            if len(self.c_r_rank_c_rank_ids_dict[cur_r_rank][temp_c_c_rank]) == 0:
                # 该变化终端原来的容量档次没有终端了
                if self.c_r_r_mem_rank_max_min_dict[cur_r_rank][0] == temp_c_c_rank:
                    # 原来是最大容量档次，那就头没了，改变
                    for i in range(temp_c_c_rank - 1, 0, -1):
                        # 倒序遍历，找最大的节点
                        if len(self.c_r_rank_c_rank_ids_dict[cur_r_rank][i]) > 0:
                            self.c_r_r_mem_rank_max_min_dict[cur_r_rank][0] = i
                            break
                    temp_data = {'fog_id': self.fog_id, 'type': CONFIG.F2S_UPDATE_C_RANK_NUMS2,
                                 'r_r_mem_rates': self.c_r_rank_mem_rate, 'max_min': self.c_r_r_mem_rank_max_min_dict}
                    print('更新了：', temp_data)
                    Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
                    # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))
            # print('更新后：', self.c_c_rank_c_ids_dict, self.c_r_rank_c_rank_ids_dict)

    def handle_sub_cloud_msg(self, msg):
        temp_data = pickle.loads(msg)
        # print(temp_data)
        if temp_data['type'] == CONFIG.S2F_TELL_ANSWER1_TO_STORE:
            self.handle_s2f_ask_for_answer1_to_store(temp_data)
        elif temp_data['type'] == CONFIG.S2F_ASK_FOR_C_UTILIZATION:
            self.handle_s2f_ask_for_c_utilization()
        elif temp_data['type'] == CONFIG.S2F_TELL_ANSWER2_TO_STORE:
            self.handle_s2f_ask_for_answer2_to_store(temp_data)

    def handle_fog_msg(self, msg):
        # print(msg)
        temp_data = pickle.loads(msg)
        # print(temp_data)
        if temp_data['type'] == CONFIG.F2F_STORE:
            # 有雾节点告知本雾节点要选人备份了
            self.handle_f2f_store(temp_data)
        elif temp_data['type'] == CONFIG.F2F_STORE_ACK:
            # 其他雾节点回复该雾节点，备份完成
            self.handle_f2f_store_ack(temp_data)
        elif temp_data['type'] == CONFIG.F2F_STORE_ACK_ERROR:
            self.handle_f2f_store_ack(temp_data)

    def handle_client_msg(self, msg):
        temp_data = pickle.loads(msg)
        # print(temp_data)
        if temp_data['type'] == CONFIG.C2F_STORE_ACK:
            # client对于备份请求的回复（要Client备份，备份好了，确认）
            self.handle_c2f_store_ack(temp_data)

        ######## RL ########
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_RL_DDQN:
            self.which_method = CONFIG.METHOD2
            temp_data['method'] = CONFIG.RL_DDQN_METHOD
            self.handle_c2f_upload2_baseline(temp_data)
        ###### GA ######
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_GA:
            self.which_method = CONFIG.METHOD5
            temp_data['method'] = CONFIG.GA_METHOD
            self.handle_c2f_upload2_baseline(temp_data)

        ###### DE ######
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_DE:
            self.which_method = CONFIG.METHOD3
            temp_data['method'] = CONFIG.DE_METHOD
            self.handle_c2f_upload2_baseline(temp_data)

        ###### PSO ######
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_PSO:
            self.which_method = CONFIG.METHOD4
            temp_data['method'] = CONFIG.PSO_METHOD
            self.handle_c2f_upload2_baseline(temp_data)

        ###### SA ######
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_SA:
            self.which_method = CONFIG.METHOD1
            temp_data['method'] = CONFIG.SA_METHOD
            self.handle_c2f_upload2_baseline(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD_RANDOM:
            self.which_method = CONFIG.METHOD6
            temp_data['method'] = CONFIG.RANDOM_METHOD
            self.handle_c2f_upload2_baseline(temp_data)

        ####### 多副本 IS2023优化 #######
        elif temp_data['type'] == CONFIG.C2F_UPLOAD3_BASELINE:
            self.which_method = CONFIG.ORIGIN_STORE3
            temp_data['method'] = CONFIG.ORIGIN_STORE3
            self.handle_c2f_upload1_baseline(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD3_DCABA:
            self.which_method = CONFIG.DCABA_STORE3
            temp_data['method'] = CONFIG.DCABA_STORE3
            self.handle_c2f_upload1_dcaba(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD3_CLB:
            self.which_method = CONFIG.CLB_STORE3
            temp_data['method'] = CONFIG.CLB_STORE3
            self.handle_c2f_upload1_clb(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD3_ACO_VMM:
            self.which_method = CONFIG.ACO_VMM_STORE3
            temp_data['method'] = CONFIG.ACO_VMM_STORE3
            self.handle_c2f_upload1_aco_vmm(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD3_DRPS:
            self.which_method = CONFIG.DRPS_STORE3
            temp_data['method'] = CONFIG.DRPS_STORE3
            self.handle_c2f_upload2_baseline(temp_data)



        elif temp_data['type'] == CONFIG.C2F_UPLOAD1_BASELINE:
            self.which_method = CONFIG.THREE_STORE
            temp_data['method'] = CONFIG.THREE_STORE
            self.handle_c2f_upload1_baseline(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD2_BASELINE:
            temp_data['method'] = CONFIG.METHOD2_STORE
            self.which_method = CONFIG.METHOD2_STORE
            self.handle_c2f_upload2_baseline(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD1_DCABA:
            temp_data['method'] = CONFIG.DCABA_STORE
            self.which_method = CONFIG.DCABA_STORE
            self.handle_c2f_upload1_dcaba(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD1_CLB:
            temp_data['method'] = CONFIG.CLB_STORE
            self.which_method = CONFIG.CLB_STORE
            self.handle_c2f_upload1_clb(temp_data)
        elif temp_data['type'] == CONFIG.C2F_UPLOAD1_ACO_VMM:
            temp_data['method'] = CONFIG.ACO_VMM_STORE
            self.which_method = CONFIG.ACO_VMM_STORE
            self.handle_c2f_upload1_aco_vmm(temp_data)



    # 按照实际地图计算出的最优路径，向下一个fog传递信息
    def send_msg_to_next_fog(self, to_fog_id, msg):
        if to_fog_id > self.fog_id:
            # f2f中self是Server角色,next是Client
            Communicator.sendMsg(self.f2f_c_sockets_dict[to_fog_id], msg)
            # self.f2f_c_sockets_dict[next_path].sendall(msg)
        else:
            Communicator.sendMsg(self.f2f_s_sockets_dict[to_fog_id], msg)
            # self.f2f_s_sockets_dict[next_path].sendall(msg)

    # 在本地随机选择一个终端存储(三副本备份中的一环）
    def select_client_by_three_copy(self, temp_data):
        num = temp_data['num']
        capacity = temp_data['capacity']
        source_c_id = temp_data['client_id']
        c_rank = Method.convert_c_to_rank(capacity)
        # 选出能存的client集合
        temp_avail = []

        for i in self.cs_c_ranks_dict.keys():
            if self.cs_c_ranks_dict[i] > c_rank:
                temp_avail.append(i)

        if source_c_id in temp_avail:
            temp_avail.remove(source_c_id)

        if len(temp_avail) < num:
            # 没有合适容量的终端
            print('没有合适容量的终端')
            data = {'index': temp_data['index'], 'type': CONFIG.F2F_STORE_ACK_ERROR, 'from_fog_id': self.fog_id,
                    'to_fog_id': temp_data['from_fog_id'], 'client_id': temp_data['client_id'],
                    'hash_id': temp_data['hash_id'], 'store_fog_ids': temp_data['store_fog_ids'],
                    'start_time': temp_data['start_time'], 'reliability': temp_data['reliability'],
                    'capacity': temp_data['capacity'], 'method': CONFIG.THREE_STORE, 'is_store': False}
            self.send_msg_to_next_fog(data['to_fog_id'], pickle.dumps(temp_data))
            return
        # print('在选可用终端时，符合条件的终端为：', temp_avail)
        selected_ids = random.sample(temp_avail, num)
        # print('选择终端 ', selected_id)
        temp_tuple = temp_data['index']
        self.wait_store_ack_cs[temp_tuple] = selected_ids  # 等待接收client确认
        for selected_id in selected_ids:
            # print('给终端发', selected_id, temp_data)
            Communicator.sendMsg(self.client_sockets_dict[selected_id], pickle.dumps(temp_data))
        # self.client_sockets_dict[selected_id].sendall(pickle.dumps(temp_data))
        # self.wait_store_ack_cs.append(selected_id)  # 等待接收client确认

    # 根据self的方法选择终端（ 可靠性等级+容量 选择容量最大的）
    def select_client_by_method2(self, temp_data):

        data_c = temp_data['capacity']
        temp_tuple = temp_data['index']
        target_r_ranks = temp_data['target_r_rank']
        total_num = len(target_r_ranks)
        have_selected_ids = []

        r_rank_counter = Counter(target_r_ranks)

        data_c_rank = Method.convert_c_to_rank(data_c)

        # 选取规定可靠性等级下的，空闲容量最大的终端
        for target_r_rank, target_r_rank_num in r_rank_counter.items():
            target_c_rank_ids = self.c_r_rank_c_rank_ids_dict[target_r_rank]

            for i in range(CONFIG.C_RANK_NUM, data_c_rank - 1, -1):
                temp_ids = target_c_rank_ids[i]

                if len(temp_ids) > 0:
                    could_num = min(len(temp_ids), target_r_rank_num)
                    selected_ids = random.sample(temp_ids, could_num)
                    have_selected_ids.extend(selected_ids)
                    target_r_rank_num -= could_num
                    if target_r_rank_num == 0:
                        break

        if len(have_selected_ids) == total_num:
            self.wait_store_ack_cs[temp_tuple] = have_selected_ids
            for i in have_selected_ids:
                Communicator.sendMsg(self.client_sockets_dict[i], pickle.dumps(temp_data))
        else:
            print('未选到合适的终端', temp_data)
            data = {'index': temp_data['index'], 'type': CONFIG.F2F_STORE_ACK_ERROR, 'from_fog_id': self.fog_id,
                    'to_fog_id': temp_data['from_fog_id'], 'client_id': temp_data['client_id'],
                    'hash_id': temp_data['hash_id'], 'store_fog_ids': temp_data['store_fog_ids'],
                    'start_time': temp_data['start_time'], 'reliability': temp_data['reliability'],
                    'capacity': temp_data['capacity'], 'method': temp_data['method'], 'is_store': False}
            if temp_data['from_fog_id'] == self.fog_id:
                self.handle_f2f_store_ack(data)
            else:
                self.send_msg_to_next_fog(data['to_fog_id'], pickle.dumps(data))

    # 选出本地负载最小的终端
    def select_client_by_clb(self,temp_data):
        max_c_rank = CONFIG.C_RANK_NUM
        for i in range(CONFIG.C_RANK_NUM, 0, -1):
            if len(self.c_c_rank_c_ids_dict[i]) > 0:
                max_c_rank = i
                break
        temp_max_id = -1
        temp_max_capacity = -1
        # print(self.c_c_rank_c_ids_dict[max_c_rank])
        for i in self.c_c_rank_c_ids_dict[max_c_rank]:
            if self.cur_c_c_dict[i] > temp_max_capacity:
                temp_max_id = i
                temp_max_capacity = self.cur_c_c_dict[i]
        # print([temp_max_id])
        temp_tuple = temp_data['index']
        self.wait_store_ack_cs[temp_tuple] = [temp_max_id]  # 等待接收client确认
        Communicator.sendMsg(self.client_sockets_dict[temp_max_id], pickle.dumps(temp_data))


    # 是要存储数据的目的/中转雾节点,收到了f2f_store的消息
    def handle_f2f_store(self, temp_data):
        # print(temp_data['from_fog_id'],'来的消息开始选终端')
        #  是目的雾节点，选择Client存储数据
        temp_data['type'] = CONFIG.F2C_STORE
        if temp_data['method'] == CONFIG.THREE_STORE:
            self.select_client_by_three_copy(temp_data)
        elif temp_data['method'] == CONFIG.RL_DDQN_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.GA_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.DE_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.PSO_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.SA_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.RANDOM_METHOD:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.METHOD2_STORE:
            self.select_client_by_method2(temp_data)
        elif temp_data['method'] == CONFIG.DCABA_STORE:
            self.select_client_by_three_copy(temp_data)

        elif temp_data['method'] == CONFIG.CLB_STORE:
            self.select_client_by_clb(temp_data)
        elif temp_data['method'] == CONFIG.ACO_VMM_STORE:
            self.select_client_by_three_copy(temp_data)

        elif temp_data['method'] == CONFIG.DCABA_STORE3:
            self.select_client_by_three_copy(temp_data)

    def handle_c2f_store_ack(self, temp_data):
        temp_tuple = temp_data['index']
        self.wait_store_ack_cs[temp_tuple].remove(temp_data['copy_client_id'])
        # self.cur_c_c[temp_data['copy_client_id']] = temp_data['cur_c']
        # 更新fog中有关该client的配置
        if temp_data['method'] == CONFIG.THREE_STORE:
            # 先记录备份的终端的可靠性，以备后续验证是否满足可靠性
            if temp_tuple in self.wait_store_ack_cs_rs:
                self.wait_store_ack_cs_rs[temp_tuple].append(temp_data['cur_r'])
            else:
                self.wait_store_ack_cs_rs[temp_tuple] = [temp_data['cur_r']]

            self.update1_client_c_r_one_client_full(temp_data['copy_client_id'], temp_data['cur_c_rank'],
                                                    temp_data['cur_c'],
                                                    temp_data['cur_r_rank'])
        else:
            if temp_tuple in self.wait_store_ack_cs_rs:
                self.wait_store_ack_cs_rs[temp_tuple].append(temp_data['cur_r'])
            else:
                self.wait_store_ack_cs_rs[temp_tuple] = [temp_data['cur_r']]

            # print(temp_data)
            self.update2_client_c_r_one_client_full(temp_data['copy_client_id'], temp_data['cur_c_rank'],
                                                    temp_data['cur_c'],
                                                    temp_data['cur_r_rank'], temp_data['last_c_rank'])
        # 给源雾节点发送备份完成的信息
        # temp_data['from_fog_id'] = self.fog_id
        temp_data['type'] = CONFIG.F2F_STORE_ACK
        if temp_tuple not in self.wait_store_ack_cs or len(self.wait_store_ack_cs[temp_tuple]) == 0:
            # 此时当前雾节点上的要求备份的信息已经完成
            if temp_tuple in self.wait_store_ack_cs:
                del self.wait_store_ack_cs[temp_tuple]
                temp_data['cur_rs'] = self.wait_store_ack_cs_rs[temp_tuple]
                temp_data['is_store'] = True
                if temp_data['to_fog_id'] == self.fog_id:
                    # 备份在源雾节点上了
                    self.handle_f2f_store_ack(temp_data)
                else:
                    # print('发送确认', temp_data)
                    self.send_msg_to_next_fog(temp_data['to_fog_id'], pickle.dumps(temp_data))

    # 收到了要备份节点的已备份的回复
    def handle_f2f_store_ack(self, temp_data):
        # print('信息', temp_data, '已到达目的地', self.fog_addr)
        # 根据元组查询waiting list  比如要等两个fog确认，那收到一个划掉一个
        from_fog_id = temp_data['from_fog_id']
        temp_tuple = temp_data['index']
        # print('初始wait_store_ack_fogs：', self.wait_store_ack_fogs)
        for tuple in self.wait_store_ack_fogs.keys():
            if operator.eq(tuple, temp_tuple):
                # print('already: ',self.already_store_fog_id)
                if not temp_data['is_store']:
                    print('有error', temp_data)
                    self.wait_store_ack_is_not_all_store.append(temp_tuple)
                else:
                    self.store_cs_r[tuple].extend(temp_data['cur_rs'])
                if tuple in self.already_store_fog_id:
                    self.already_store_fog_id[tuple].append(from_fog_id)
                else:
                    self.already_store_fog_id[tuple] = [from_fog_id]
                if len(self.wait_store_ack_fogs[tuple]) == len(self.already_store_fog_id[tuple]):
                    # 此时说明关于temp_tuple的备份，各个雾节点均已完成并返回确认
                    max_distance_time = 0
                    for i in temp_data['store_fog_ids']:
                        temp_distance_time = self.path.get_virtual_time(self.fog_id, i)
                        if temp_distance_time > max_distance_time:
                            max_distance_time = temp_distance_time
                    max_distance_time = max_distance_time / 100
                    num = len(self.store_cs_r[tuple])

                    data = {'index': temp_data['index'], 'type': CONFIG.F2C_STORE_ACK1,
                            'fogs': temp_data['store_fog_ids'], 'hash_id': temp_data['hash_id'],
                            'start_time': temp_data['start_time'], 'distance_time': max_distance_time,
                            'data_r': temp_data['reliability'], 'cur_rs': self.store_cs_r[tuple],
                            'data_c': temp_data['capacity'], 'method': CONFIG.THREE_STORE, 'is_enough':0 , 'num': num, 'protect': 1}
                    if temp_data['method'] == CONFIG.METHOD2_STORE:
                        data['type'] = CONFIG.F2C_STORE_ACK2
                        data['method'] = CONFIG.METHOD2_STORE
                    if temp_data['method'] == CONFIG.DRPS_STORE3:
                        data['type'] = CONFIG.F2C_STORE_ACK2
                        data['method'] = CONFIG.DRPS_STORE3
                    if tuple in self.wait_store_ack_is_not_all_store:
                        # 有其中存在存储失败的节点
                        data['type'] = CONFIG.F2C_STORE_ACK_ERROR
                        self.wait_store_ack_is_not_all_store.remove(tuple)

                    # print('关于这个数据的缓存已完毕，tuple为', tuple, ' ,确切信息为：', data)
                    self.have_store_index[tuple] = data
                    Communicator.sendMsg(self.client_sockets_dict[temp_data['client_id']], pickle.dumps(data))
                    # print('已发送')
                    # self.client_sockets_dict[tuple[0]].sendall(pickle.dumps(data))
                    if tuple in self.wait_store_ack_fogs:
                        # 还没被删掉
                        del self.wait_store_ack_fogs[tuple]
                    if tuple in self.store_cs_r:
                        del self.store_cs_r[tuple]
                break

    def handle_s2f_ask_for_c_utilization(self):
        data = {'type': CONFIG.F2C_FOG_ALL_FULL, 'protect': 1}
        for i in self.client_sockets_dict.values():
            Communicator.sendMsg(i, pickle.dumps(data))

        print('发送完毕，收到了统计利用率的请求')
        c_ids = list(self.initial_c_c_dict)
        initial_cs = list(self.initial_c_c_dict.values())
        cur_cs = list(self.cur_c_c_dict.values())

        dateframe = pd.DataFrame({'c_id': c_ids, 'initial_capacity': initial_cs, 'current_capacity': cur_cs})
        dateframe.to_csv(CONFIG.RESULT_FOLDER + str(self.which_method) + '/result/fog' + str(self.fog_id) + '.csv', mode='a', header=True, index=False, sep=',')

    def handle_s2f_ask_for_answer1_to_store(self, temp_data):
        fogs = temp_data['store_fog_ids']
        temp_data['store_fog_ids'] = list(fogs.keys())
        if fogs == -1:
            # 没有容量合适的节点,备份失败
            print('子云没有选出合适的雾节点，失败')
            data = {'index': temp_data['index'], 'type': CONFIG.F2C_STORE_ACK_ERROR, 'hash_id': temp_data['hash_id'],
                    'data_c': temp_data['capacity'], 'protect': 1}
            Communicator.sendMsg(self.client_sockets_dict[temp_data['client_id']], pickle.dumps(data))
            # self.client_sockets_dict[temp_data['client_id']].sendall(pickle.dumps(data))
        else:
            temp_data['type'] = CONFIG.F2F_STORE
            temp_tuple = temp_data['index']
            self.wait_store_ack_fogs[temp_tuple] = []
            self.store_cs_r[temp_tuple] = []
            # print('self.wait_store_ack_fogs为', self.wait_store_ack_fogs)
            for i in fogs.keys():
                self.wait_store_ack_fogs[temp_tuple].append(i)
            for i in fogs.keys():
                temp_data['to_fog_id'] = i
                temp_data['num'] = fogs[i]
                if self.fog_id == i:
                    # print('本地备份')
                    self.handle_f2f_store(temp_data)
                else:
                    temp_data['type'] = CONFIG.F2F_STORE
                    # print('给fog', i, ' 发备份请求', temp_data)
                    self.send_msg_to_next_fog(i, pickle.dumps(temp_data))
                # time.sleep(1)

    def handle_s2f_ask_for_answer2_to_store(self, temp_data):
        fogs = temp_data['store_fog_ids']
        temp_data['store_fog_ids'] = list(fogs.keys())
        if len(fogs) == 0:
            # 没有容量合适的节点,备份失败
            print('子云没有选出合适的雾节点，失败')
            data = {'index': temp_data['index'], 'type': CONFIG.F2C_STORE_ACK_ERROR, 'hash_id': temp_data['hash_id'],
                    'data_c': temp_data['capacity'], 'protect': 1}
            Communicator.sendMsg(self.client_sockets_dict[temp_data['client_id']], pickle.dumps(data))
            # self.client_sockets_dict[temp_data['client_id']].sendall(pickle.dumps(data))
        else:
            temp_tuple = temp_data['index']
            self.wait_store_ack_fogs[temp_tuple] = []
            self.store_cs_r[temp_tuple] = []
            # print('self.wait_store_ack_fogs为', self.wait_store_ack_fogs)
            for i in fogs.keys():
                self.wait_store_ack_fogs[temp_tuple].append(i)

            for i in fogs.keys():
                temp_data['to_fog_id'] = i
                temp_data['target_r_rank'] = fogs[i]
                if i == self.fog_id:
                    # print('本地备份')
                    temp_data['type'] = CONFIG.F2C_STORE
                    self.select_client_by_method2(temp_data)
                else:
                    temp_data['type'] = CONFIG.F2F_STORE
                    # print('给fog发 ', temp_data)
                    self.send_msg_to_next_fog(i, pickle.dumps(temp_data))


    ### baseline

    def select_by_method1_baseline(self):
        fogs = []
        source_fog_ids = {}
        for i in range(2):
            fogs.append(random.randint(1, CONFIG.FOG_NUM))
        for i in range(2):
            if fogs[i] in source_fog_ids:
                source_fog_ids[fogs[i]] += 1
            else:
                source_fog_ids[fogs[i]] = 1
        return source_fog_ids

    # 处理来自终端的origin备份请求
    def handle_c2f_upload1_baseline(self, temp_data):
        c_id = temp_data['client_id']
        # self.cur_c_c[c_id] = temp_data['cur_c']
        self.update1_client_c_r_one_client_full(c_id, temp_data['cur_c_rank'], temp_data['cur_c'],
                                                temp_data['cur_r_rank'])

        temp_data['from_fog_id'] = self.fog_id
        if temp_data['method'] == CONFIG.THREE_STORE:
            # 采用三副本的方式来存
            # 告诉子云，请求给予备份节点,答案在子云给了消息才会处理
            temp_data['store_fog_ids'] = self.select_by_method1_baseline()
        elif temp_data['method'] == CONFIG.ORIGIN_STORE3:
            temp_data['store_fog_ids'] = self.select_by_method3_baseline(temp_data['reliability'])
        # print(temp_data['store_fog_ids'])
        self.handle_s2f_ask_for_answer1_to_store(temp_data)
        # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))

    # 处理来自终端的self备份请求
    def handle_c2f_upload2_baseline(self, temp_data):
        c_id = temp_data['client_id']
        self.update2_client_c_r_one_client_full(c_id, temp_data['cur_c_rank'], temp_data['cur_c'],
                                                temp_data['cur_r_rank'],
                                                temp_data['last_c_rank'])
        if temp_data['is_enough'] == 1:
            # 已满足
            data = {'index': temp_data['index'], 'type': CONFIG.F2C_STORE_ACK2, 'fogs': [],
                    'hash_id': temp_data['hash_id'],
                    'start_time': temp_data['start_time'], 'distance_time': 0, 'cur_rs': [], 'num': 0, 'is_enough': 1}
            Communicator.sendMsg(self.client_sockets_dict[c_id], pickle.dumps(data))
        else:
            # 未满足 要选出合适fog
            # print('要选')
            temp_data['from_fog_id'] = self.fog_id
            if temp_data['method'] == CONFIG.METHOD2_STORE:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER2_TO_STORE
            elif temp_data['method'] == CONFIG.RL_DDQN_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_RL_DDQN_TO_STORE
            elif temp_data['method'] == CONFIG.GA_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_GA_TO_STORE
            elif temp_data['method'] == CONFIG.DE_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_DE_TO_STORE
            elif temp_data['method'] == CONFIG.PSO_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_PSO_TO_STORE
            elif temp_data['method'] == CONFIG.SA_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_SA_TO_STORE
            elif temp_data['method'] == CONFIG.RANDOM_METHOD:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_RANDOM_TO_STORE
            else:
                temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_DRPS_TO_STORE
            Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))

    # 处理来自终端的3-dcaba 备份请求
    # DCABA： 三副本，除了自身外，选择负载最小的雾节点（契合分组概念），在这个负载最小的雾节点中选择两个
    def handle_c2f_upload1_dcaba(self, temp_data):
        c_id = temp_data['client_id']
        # self.cur_c_c[c_id] = temp_data['cur_c']
        self.update1_client_c_r_one_client_full(c_id, temp_data['cur_c_rank'], temp_data['cur_c'], temp_data['cur_r_rank'])
        # temp_data['type'] = CONFIG.F2F_STORE
        # 告诉子云，请求给予备份节点,答案在子云给了消息才会处理
        temp_data['from_fog_id'] = self.fog_id
        temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_DCABA_TO_STORE
        temp_data['new_total_rate'] = self.cur_total_mem / self.initial_total_mem
        Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
        # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))

    # 处理来自终端的3-clb 备份请求
    def handle_c2f_upload1_clb(self, temp_data):
        c_id = temp_data['client_id']
        # self.cur_c_c[c_id] = temp_data['cur_c']
        self.update1_client_c_r_one_client_full(c_id, temp_data['cur_c_rank'], temp_data['cur_c'], temp_data['cur_r_rank'])
        # temp_data['type'] = CONFIG.F2F_STORE
        # 采用三副本的方式来存
        # temp_data['method'] = CONFIG.CLB_STORE
        # 告诉子云，请求给予备份节点,答案在子云给了消息才会处理
        temp_data['from_fog_id'] = self.fog_id
        temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_CLB_TO_STORE
        Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
        # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))

    # 处理来自终端的3-aco_vmm 备份请求
    def handle_c2f_upload1_aco_vmm(self, temp_data):
        c_id = temp_data['client_id']
        # self.cur_c_c[c_id] = temp_data['cur_c']
        self.update1_client_c_r_one_client_full(c_id, temp_data['cur_c_rank'], temp_data['cur_c'], temp_data['cur_r_rank'])
        # temp_data['type'] = CONFIG.F2F_STORE
        # 采用三副本的方式来存
        # temp_data['method'] = CONFIG.ACO_VMM_STORE
        # 告诉子云，请求给予备份节点,答案在子云给了消息才会处理
        temp_data['from_fog_id'] = self.fog_id
        temp_data['type'] = CONFIG.F2S_ASK_FOR_ANSWER_ACO_VMM_TO_STORE
        Communicator.sendMsg(self.sub_cloud_socket, pickle.dumps(temp_data))
        # self.sub_cloud_socket.sendall(pickle.dumps(temp_data))


    ###### 多副本下的四种方案 ######
    # 因为处理都一样，关键在于select选择不一样，select只有origin是在普通雾节点随机选择
    # 多-origin
    def select_by_method3_baseline(self,data_r):
        fogs = []
        source_fog_ids = {}

        client_num = 1

        while 1-data_r <= pow(1-0.8,client_num):
            client_num += 1

        for i in range(2):
            fogs.append(random.randint(1, CONFIG.FOG_NUM))
        for i in range(2):
            if fogs[i] in source_fog_ids:
                source_fog_ids[fogs[i]] += 1
            else:
                source_fog_ids[fogs[i]] = 1
        return source_fog_ids