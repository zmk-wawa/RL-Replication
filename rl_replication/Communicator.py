# -*- coding: utf-8 -*-
# @Time    : 2023-03-10 21:42
# @Author  : Mengke Zheng
# @Email   : 18307110476@fudan.edu.cn
# @File    : Communicator.py
import pickle
import socket
import struct
import sys


# 发数据封装
def sendMsg(to_socket, msg):
    msg_len = len(msg)
    to_socket.sendall(struct.pack('>I', msg_len))
    end = b'K\x06u.'
    if msg[:1] == b'\x00':
        msg = msg[4:]
        msg += end
        print('发送方数据不对了哦', pickle.loads(msg))
    to_socket.sendall(msg)
    # print('!!!!发送数据',pickle.loads(msg))

# 收数据封装
def recvMsg(from_socket):
    # 长度为4的第一个
    msg_len = struct.unpack('>I',from_socket.recv(4))[0]
    msg = from_socket.recv(msg_len,socket.MSG_WAITALL)
    end = b'K\x06u.'
    if msg[:1] == b'\x00':
        msg = msg[4:]
        msg += end
        print('篡改数据了哦',pickle.loads(msg))
    # print('???收到数据', pickle.loads(msg))
    if len(msg) < msg_len:
        print('少了少了！！！！！！')
    if len(msg) > msg_len:
        print('粘包！！！！！！')
    return msg
