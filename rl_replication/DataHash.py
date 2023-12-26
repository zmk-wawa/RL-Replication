# -*- coding: utf-8 -*-
# @Time    : 2023-02-11 21:47
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : DataHash.py

"""
    把缓存的数据 生成hash值
"""
import random


def gen_hash(data):
    ## 根据data生成msg
    return random.randint(1, 50)
