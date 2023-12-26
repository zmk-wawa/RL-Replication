# -*- coding: utf-8 -*-
# @Time    : 2023-02-11 14:51
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : Client.py
import os.path
import pickle
import socket
import threading
from datetime import datetime

import Method
import Communicator
from CONFIG import PATH
from DataHash import gen_hash
from CONFIG import CLIENT_ROLE
import CONFIG


class client():
    result_success = None
    result_error = None
    success_file = None
    error_file = None

    def __init__(self, id, reliability, capacity, fog_addr, fog_id):
        """
        :param id: client的id
        :param reliability: 可靠性，变动了会通知所属fog节点
        :param capacity: 剩余内存容量
        :param fog_addr: 所属雾节点地址
        :param fog_id: 所属雾节点id 创建空间时用
        :return:
        """
        # 初始设置
        self.folder_path = None
        # print('client', id, '属于fog ', fog_id)
        self.id = id
        self.reliability = reliability
        self.reliability_rank = Method.convert_r_to_rank(reliability)
        self.capacity = capacity
        self.capacity_rank = Method.convert_c_to_rank(capacity)
        self.fog_addr = (Method.gen_ip(), fog_addr)
        self.fog_id = fog_id
        self.isFull = False
        self.isBusy = False

        # self.wait_store_indexs = []  # waiting的tuple们[indexs]
        self.cur_waiting_index = -1

        self.fog_is_all_full = False


        # 构建Socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 请求与fog的连接
        self.client_socket.connect(self.fog_addr)

        msg = pickle.dumps(
            {'role': CLIENT_ROLE, 'id': self.id, 'r_rank': self.reliability_rank, 'c_rank': self.capacity_rank,
             'c': self.capacity, 'protect': 1})
        Communicator.sendMsg(self.client_socket, msg)
        # self.client_socket.sendall(msg)

        # 开辟空间
        self.mkdir()

    def mkdir(self):
        self.folder_path = PATH + 'fog' + str(self.fog_id) + '/C' + str(self.id)

        folder = os.path.exists(self.folder_path)

        if not folder:
            # 文件夹不存在，创建文件夹
            os.makedirs(self.folder_path)

    def create_file(self, data, ori_hash_id=0):
        if ori_hash_id == 0:
            hash_id = gen_hash(data)
        else:
            hash_id = ori_hash_id
        # TODO:应该有个对文件是否存在的判断，存在则data追加在文件后面这样
        return hash_id

    # 3-origin
    def upload_data1_baseline(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD1_BASELINE, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        return self.capacity

    # self
    def upload_data2_baseline(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        last_c_rank = self.capacity_rank
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        temp_data = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
                     'reliability': reliability,
                     'type': CONFIG.C2F_UPLOAD2_BASELINE, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
                     'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'last_c_rank': last_c_rank,
                     'protect': 1}
        # self.wait_store_indexs.append(index)
        if self.reliability >= reliability:
            temp_data['is_enough'] = 1
        else:
            temp_data['is_enough'] = 0
        # 把对象序列化为string类型
        send_msg = pickle.dumps(temp_data)
        # print('msg: ', msg)
        # print('发数据',temp_data)
        Communicator.sendMsg(self.client_socket, send_msg)
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 3-dcaba
    def upload_data1_dcaba(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD1_DCABA, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 3-clb
    def upload_data1_clb(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD1_CLB, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 3-aco-vmm
    def upload_data1_aco_vmm(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD1_ACO_VMM, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity


    ###### 多副本下的DCABA、CLB、ACO_VMM与随机副本 ######

    # 多-origin
    def upload_data3_baseline(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD3_BASELINE, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        return self.capacity

    # 多-dcaba
    def upload_data3_dcaba(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD3_DCABA, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 多-clb
    def upload_data3_clb(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD3_CLB, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 多-aco-vmm
    def upload_data3_aco_vmm(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        msg = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
               'reliability': reliability,
               'type': CONFIG.C2F_UPLOAD3_ACO_VMM, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
               'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'protect': 1}
        # 把对象序列化为string类型
        # self.wait_store_indexs.append(index)
        send_msg = pickle.dumps(msg)
        # print('msg: ', msg)
        Communicator.sendMsg(self.client_socket, send_msg)
        # t = threading.Timer(2.0,self.clock_handle_package_error,args=(index,1,))
        # t.start()
        # self.client_socket.sendall(send_msg)
        return self.capacity


    ######## RL ########
    def upload_data_rl_ddqn(self,data,capacity,reliability,index):
        last_c_rank = self.capacity_rank
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        temp_data = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
                     'reliability': reliability,
                     'type': CONFIG.C2F_UPLOAD_RL_DDQN, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
                     'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'last_c_rank': last_c_rank,
                     'protect': 1}
        # self.wait_store_indexs.append(index)
        if self.reliability >= reliability:
            temp_data['is_enough'] = 1
        else:
            temp_data['is_enough'] = 0
        # 把对象序列化为string类型
        send_msg = pickle.dumps(temp_data)
        # print('msg: ', msg)
        # print('发数据',temp_data)
        Communicator.sendMsg(self.client_socket, send_msg)
        # self.client_socket.sendall(send_msg)
        return self.capacity

    def upload_data_gas(self,data,capacity,reliability,index,method):
        last_c_rank = self.capacity_rank
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        temp_data = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
                     'reliability': reliability,
                     'type': CONFIG.C2F_UPLOAD_GA, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
                     'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'last_c_rank': last_c_rank,
                     'protect': 1}
        if method == CONFIG.DE_METHOD:
            temp_data['type'] = CONFIG.C2F_UPLOAD_DE
        elif method == CONFIG.PSO_METHOD:
            temp_data['type'] = CONFIG.C2F_UPLOAD_PSO
        elif method == CONFIG.SA_METHOD:
            temp_data['type'] = CONFIG.C2F_UPLOAD_SA
        elif method == CONFIG.RANDOM_METHOD:
            temp_data['type'] = CONFIG.C2F_UPLOAD_RANDOM


        if self.reliability >= reliability:
            temp_data['is_enough'] = 1
        else:
            temp_data['is_enough'] = 0
        # 把对象序列化为string类型
        send_msg = pickle.dumps(temp_data)
        # print('msg: ', msg)
        # print('发数据',temp_data)
        Communicator.sendMsg(self.client_socket, send_msg)
        # self.client_socket.sendall(send_msg)
        return self.capacity

    # 持续接收来自fog的消息
    def receive_msg_from_fog(self):
        while True:
            msg = Communicator.recvMsg(self.client_socket)
            temp_thread = threading.Thread(target=self.handle_fog_msg, args=(msg,))
            temp_thread.start()

    # 更新自己当前的容量
    def update_cur_c(self, data_c):
        self.capacity -= data_c
        # self.capacity_rank = Method.convert_c_to_rank(self.capacity)
        temp_c_rank = Method.convert_c_to_rank(self.capacity)
        if temp_c_rank != self.capacity_rank:
            print('等级发生了改变',self.capacity,data_c,self.capacity_rank)

        self.capacity_rank = temp_c_rank
        if self.capacity_rank == 0:
            self.isFull = True
            self.fog_is_all_full = True

    # 判断这次备份的数据是否满足数据可靠性要求
    def judge_is_reliable(self, data_r, had_rs):
        target = 1 - data_r
        real = 1 - self.reliability
        # print('备份的r:', had_rs, '与', self.reliability, 'data_r', data_r)
        for i in had_rs:
            real = real * (1 - i)
        if real <= target:
            # print('满足')
            return '1'
        else:
            # print('未满足')
            return '0'

    ######## 以下为对收到的信息进行处理的函数 ########

    # 处理来自fog的信息
    def handle_fog_msg(self, msg):
        temp_data = pickle.loads(msg)
        # print(temp_data)
        if temp_data['type'] == CONFIG.F2C_STORE:
            # 也就是说我（终端）要把数据存在本地QAQ
            # print(temp_data)
            self.handle_f2c_store(temp_data)
        elif temp_data['type'] == CONFIG.F2C_STORE_ACK1:
            # 我之前提了备份请求，现在备份完毕了，告知我一声，好耶！
            self.handle_f2c_store_success(temp_data)
        elif temp_data['type'] == CONFIG.F2C_STORE_ACK2:
            self.handle_f2c_store_success(temp_data)
        elif temp_data['type'] == CONFIG.F2C_STORE_ACK_ERROR:
            # print('我是client ', self.id, ' ,我之前要求的备份请求，备份失败QAQ')
            self.handle_f2c_store_error(temp_data)
        elif temp_data['type'] == CONFIG.F2C_FOG_ALL_FULL:
            self.fog_is_all_full = True


    # fog端的数据备份请求
    def handle_f2c_store(self, temp_data):
        # print('开始备份')
        last_c_rank = self.capacity_rank

        if last_c_rank == 1:
            print("我是备选上的，好小了")

        data_c = temp_data['capacity']
        hash_id = self.create_file(temp_data['data'], temp_data['hash_id'])

        self.update_cur_c(data_c)
        # capacity reliability 都是数据的容量和可靠性要求 cur_r_rank 为自己的可靠性要求等级
        response_msg = {'index': temp_data['index'], 'copy_client_id': self.id, 'hash_id': hash_id,
                        'type': CONFIG.C2F_STORE_ACK,
                        'cur_c_rank': self.capacity_rank, 'store_fog_ids': temp_data['store_fog_ids'],
                        'to_fog_id': temp_data['from_fog_id'], 'reliability': temp_data['reliability'],
                        'client_id': temp_data['client_id'], 'start_time': temp_data['start_time'],
                        'cur_r_rank': self.reliability_rank, 'cur_r': self.reliability, 'cur_c': self.capacity,
                        'last_c_rank': last_c_rank, 'method': temp_data['method'], 'is_store': True,
                        'from_fog_id': self.fog_id,'capacity':data_c, 'protect': 1}
        # print('备份好了', response_msg)
        Communicator.sendMsg(self.client_socket, pickle.dumps(response_msg))
        # self.client_socket.sendall(pickle.dumps(response_msg))

    # fog端的数据备份确认的消息 success与error两种
    # 完成备份了，但可能有可靠性没满足的情况
    def handle_f2c_store_success(self, temp_data):
        # print("备份好了",备份好了temp_data)
        # print('wait',self.wait_store_indexs)
        start_time = temp_data['start_time']
        # print(temp_data)
        end_time = datetime.now()
        temp = (end_time - start_time).total_seconds()
        sum_time = temp + temp_data['distance_time'] * 2
        # is_reliable = 0
        # print(temp_data)
        if temp_data['type'] == CONFIG.F2C_STORE_ACK2:
            print('ok')
            if temp_data['is_enough'] == 1:
                is_reliable = 1
            else:
                is_reliable = self.judge_is_reliable(temp_data['data_r'], temp_data['cur_rs'])
                if is_reliable == '0':
                    print("未满足可靠性")
        else:
            print('ok')
            is_reliable = self.judge_is_reliable(temp_data['data_r'], temp_data['cur_rs'])
            print(is_reliable, self.fog_id, temp_data['fogs'])
        data = [self.fog_id, self.id, temp_data['hash_id'], temp_data['fogs'], sum_time, temp_data['num'] + 1,
                is_reliable]
        client.result_success.writerow(data)
        client.success_file.flush()
        self.cur_waiting_index = temp_data['index']



    # 空间无法满足 备份未完成
    def handle_f2c_store_error(self, temp_data):
        # print(temp_data)
        data = [self.fog_id, self.id, temp_data['hash_id'], temp_data['data_c']]
        client.result_error.writerow(data)
        client.error_file.flush()
        self.cur_waiting_index = temp_data['index']


    def is_fog_all_full(self):
        return self.fog_is_all_full

    # DRPS 备份三份或者一份
    def upload_data3_drps(self, data, capacity, reliability, index):
        # data要先存在自己本地，再请求fog: 查看可靠性是否达标，不达标再选择存副本，然后生成关于data的索引
        last_c_rank = self.capacity_rank
        hash_id = self.create_file(data)
        self.update_cur_c(capacity)
        # 生成msg 即hash_id和data
        temp_data = {'index': index, 'client_id': self.id, 'hash_id': hash_id, 'data': data, 'capacity': capacity,
                     'reliability': reliability,
                     'type': CONFIG.C2F_UPLOAD3_DRPS, 'cur_c_rank': self.capacity_rank, 'cur_c': self.capacity,
                     'start_time': datetime.now(), 'cur_r_rank': self.reliability_rank, 'last_c_rank': last_c_rank,
                     'protect': 1}
        # self.wait_store_indexs.append(index)
        if self.reliability >= reliability:
            temp_data['is_enough'] = 1
        else:
            temp_data['is_enough'] = 0
        # 把对象序列化为string类型
        send_msg = pickle.dumps(temp_data)
        # print('msg: ', msg)
        # print('发数据',temp_data)
        Communicator.sendMsg(self.client_socket, send_msg)
        # self.client_socket.sendall(send_msg)
        return self.capacity