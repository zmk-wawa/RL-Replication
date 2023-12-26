# 差分进化算法
import numpy as np

from sko.GA import GA
from sko.DE import DE
from sko.PSO import PSO
from sko.SA import SA
from sko.ACA import ACA_TSP

from CONFIG import FOG_NUM, R_RANK_NUM, UN_RS,GA_METHOD,DE_METHOD,PSO_METHOD,SA_METHOD

cur_capacity_dict = {}
ori_capacity_dict = {}
distance = []
max_min_dict = {}
fog_mem_rate_dict = {}

long_num = [0, 1]  # 最长距离&备份次数

SMALL_ZERO = 1e-6


def obj_func(p):
    fog_id = np.round(p[0]).astype(int)
    r_rank = np.round(p[1]).astype(int)
    temp_distance = distance[fog_id - 1]  # 两个节点距离
    max_mem_rate = max(fog_mem_rate_dict.values())
    min_mem_rate = min(fog_mem_rate_dict.values())
    temp_distance = max(long_num[0], temp_distance)

    # 1. 内存剩余率 越大越优先
    mem_normalized = (fog_mem_rate_dict[fog_id] - min_mem_rate) / (
            max_mem_rate - min_mem_rate + SMALL_ZERO)
    if max_min_dict[r_rank][fog_id - 1][0] <= 2:
        mem_normalized = -50
    if max_min_dict[r_rank][fog_id - 1][0] < 0:
        mem_normalized = -10000

    # 2.存储资源消耗量  backup num  （最多6次备份即可） 越小越好
    backup_rate = (long_num[1] + 1) / 6

    # 3.时延（长度） 最大时延  越小越优先
    max_length = max(distance)
    min_length = min(distance)
    time_normalized = (temp_distance - min_length) / (max_length - min_length + 1)

    return  1.5 * backup_rate + time_normalized - mem_normalized


constraint_ueq = [
    lambda p: -max_min_dict[np.round(p[1]).astype(int)][np.round(p[0]).astype(int) - 1][0]
]


def ga_train(d_size, d_r, cur_u_r, cur_c_dict, ori_c_dict, temp_distance, temp_max_min_dict,method):
    global distance
    global cur_capacity_dict
    global ori_capacity_dict
    global max_min_dict

    distance = None
    cur_capacity_dict = None
    ori_capacity_dict = None
    max_min_dict = None

    distance = temp_distance
    cur_capacity_dict = cur_c_dict
    ori_capacity_dict = ori_c_dict
    max_min_dict = temp_max_min_dict

    result = []
    un = cur_u_r

    for i in range(1, FOG_NUM + 1):
        fog_mem_rate_dict[i] = cur_c_dict[i] / ori_c_dict[i]

    while True:
        ga = 0
        if method == GA_METHOD:
            ga = GA(func=obj_func, n_dim=2, size_pop=50, max_iter=200, prob_mut=0.001, lb=[1, 1], ub=[FOG_NUM, R_RANK_NUM],
                    constraint_ueq = constraint_ueq,precision=1)
        elif method == DE_METHOD:
            ga = DE(func=obj_func, n_dim=2, size_pop=50, max_iter=200, lb=[1, 1], ub=[FOG_NUM, R_RANK_NUM],
                    constraint_ueq = constraint_ueq)
        elif method == PSO_METHOD:
            ga = PSO(func=obj_func, n_dim=2, max_iter=200, lb=[1, 1], ub=[FOG_NUM, R_RANK_NUM],
                    constraint_ueq = constraint_ueq)
        elif method == SA_METHOD:
            ga = SA(func=obj_func,x0 = [1,1],lb=[1, 1], ub=[FOG_NUM, R_RANK_NUM])

        x, y = ga.run()

        fog_id = np.round(x[0]).astype(int)
        r_rank = np.round(x[1]).astype(int)
        result.append([fog_id, r_rank])
        un = un * UN_RS[r_rank - 1]
        if un < 1 - d_r:
            break
        cur_capacity_dict[fog_id] -= d_size
        fog_mem_rate_dict[fog_id] = cur_capacity_dict[fog_id] / ori_capacity_dict[fog_id]
        long_num[0] = max(long_num[0], distance[fog_id - 1])
        long_num[1] += 1

    distance = None
    cur_capacity_dict = None
    ori_capacity_dict = None
    max_min_dict = None

    return result


# def ha():
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
    for i in range(20):
        result = ga_train(500, 0.9999, 0.1, cur_capacity_dict, ori_capacity_dict, distance, max_min_dict,SA_METHOD)
        print(i,result)
