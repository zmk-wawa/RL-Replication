# -*- coding: utf-8 -*-
# @Time    : 2023-02-23 19:27
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : Method.py
import random
from bisect import bisect

import numpy as np
import numpy.random
from scipy import stats as stats

import CONFIG

from CONFIG import R_RANKS, C_RANKS, R_LOWER, R_UPPER, R_MU, R_SIGMA, FOG_NUM, CLIENT_NUM, C_LOWER, C_UPPER, GB_TO_MB
import socket


def convert_r_to_rank(reliability):
    breakpoints = R_RANKS[1], R_RANKS[2], R_RANKS[3], R_RANKS[4], R_RANKS[5], R_RANKS[6], R_RANKS[7]
    ranks = '12345678'
    return int(ranks[bisect(breakpoints, reliability)])


# 单位为MB
def convert_c_to_rank(capacity):
    if capacity <= data_c_list[2]:
        return 0
    breakpoints = C_RANKS[1], C_RANKS[2], C_RANKS[3], C_RANKS[4], C_RANKS[5], C_RANKS[6], C_RANKS[7]
    ranks = '12345678'
    return int(ranks[bisect(breakpoints, capacity)])


def create_reliability(num=FOG_NUM * CLIENT_NUM, lower=R_LOWER, upper=R_UPPER, mu=R_MU, sigma=R_SIGMA):
    # X表示含有最大最小值约束的正态分布
    # N表示不含最大最小值约束的正态分布
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
    # N = stats.norm(loc=mu, scale=sigma)  # 无区间限制的随机数
    return X.rvs(num)  # 取其中的5个数，赋值给a；a为array类型


# 随机的数据可靠性要求
def random_data_reliability(lower=R_LOWER, upper=R_UPPER, mu=R_MU + 0.2, sigma=R_SIGMA):
    while True:
        x = np.random.normal(mu, sigma)
        if lower < x < upper:
            return x


# 单位统一化为MB
# lower=C_RANKS[len(C_RANKS) // 2], upper=C_RANKS[len(C_RANKS) - 1]
def create_capacity(fog_num=FOG_NUM, size=CLIENT_NUM, lower=C_RANKS[len(C_RANKS) // 2], upper=C_RANKS[len(C_RANKS) - 1]):
    sum_fs = {}
    cut = fog_num // 4

    mid = (upper + lower) / 2

    for i in range(1, cut + 1):
        sum_fs[i] = np.random.randint(mid - lower, upper, size)

    for i in range(cut + 1, cut + cut + 1):
        sum_fs[i] = np.random.randint(lower, upper, size)

    for i in range(cut + cut + 1, fog_num + 1 - cut):
        sum_fs[i] = np.random.randint(lower, mid + lower, size)

    for i in range(fog_num + 1 - cut, fog_num + 1):
        sum_fs[i] = np.random.randint(lower, upper, size)
    return sum_fs


# baseline TST方法
data_c_list = [10, 30, 50]  # 单位MB
data_c_weight_list = [0.5, 0.3, 0.2]

data_r_list = [0.9, 0.93, 0.95, 0.99, 0.999]


# 生成baseline数据的大小
def create_data_capacity_baseline():
    return random.choices(data_c_list, data_c_weight_list)[0]


# 生成数据的可靠性要求（数据只在这个list中取）
def create_data_reliability_baseline():
    return random.choice(data_r_list)


# 超算上自动获得ip
def gen_ip():
    # hostname = socket.gethostname()
    # ip =  socket.gethostbyname(hostname)
    # # return '127.0.0.1'
    # return ip
    return '127.0.0.1'
