# -*- coding: utf-8 -*-
# @Time    : 2023-02-14 19:54
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : SearchPath.py


from Diagram.Dijkstra import dijkstra
from CONFIG import FOG_NUM
from CONFIG import FOG_CONN_EDGE_NUM
from CONFIG import FOG_ADDRS


class Path:

    def __init__(self):
        (self.DISTANCE, self.ROUTE) = dijkstra(FOG_NUM, FOG_CONN_EDGE_NUM)

    def get_virtual_time(self, from_fog_id, to_fog_id):
        if from_fog_id < to_fog_id:
            return self.DISTANCE[from_fog_id - 1][to_fog_id - 1]
        else:
            return self.DISTANCE[to_fog_id - 1][from_fog_id - 1]

    # 默认from != to
    def find_path_from_to(self, from_fog_id, to_fog_id):
        # 参数是id int类型
        if from_fog_id < to_fog_id:

            key = str(from_fog_id) + '-' + str(to_fog_id)

            # print(from_fog, '到', to_fog, '的route为', ROUTE[key])
            return int(self.ROUTE[key][1])
            # return self.DISTANCE[from_fog_id - 1][to_fog_id - 1], self.ROUTE[key]
        else:
            # 3-1的就是1-3的反过来
            key = str(to_fog_id) + '-' + str(from_fog_id)
            ori_route = self.ROUTE[key]
            route = []
            for i in range(len(ori_route) - 1, -1, -1):
                route.append(ori_route[i])

            # print(from_fog, '到', to_fog, '的route为', route)
            return int(route[1])
            # return self.DISTANCE[to_fog_id - 1][from_fog_id - 1], route
