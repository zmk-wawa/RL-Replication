# -*- coding: utf-8 -*-
# @Time    : 2023-02-12 22:33
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : TestMethod.py

import CONFIG
from Fog import fog
import Method


def fog_run(fog_id):
    print('fog ', fog_id)
    print((Method.gen_ip(), CONFIG.FOG_ADDRS[fog_id - 1]))
    temp_fog = fog(fog_id, (Method.gen_ip(), CONFIG.FOG_ADDRS[fog_id - 1]), CONFIG.CLIENT_NUM, 1)
    # temp_fog.receive_msg()


