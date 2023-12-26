# PATH = '/public/home/ssct005t/project/zmk/device/20/test/clientSpace/'
# RESULT_FOLDER = '/public/home/ssct005t/project/zmk/device/20/test/result'

PATH = '/public/home/ssct005t/project/zmk/main/80/test/clientSpace/'
RESULT_FOLDER = '/public/home/ssct005t/project/zmk/main/80/test/result'

# /public/home/ssct005t/project/zmk/device/4/test
# PATH = 'clientSpace/'
# RESULT_FOLDER = '/home/data/ZMK/rl_replication/result'

MEM_RATE_LOAD = 0.9
DISTANCE_LOAD = 0.1

SUPPLEMENTARY_DATA = 'g'

FOG_ROLE = 1
CLIENT_ROLE = 2

CLIENT_NUM = 80  # 一个Fog对应几个client
FOG_NUM = 4

MAX_TCP_SIZE = 1024  # fog-client间上传/下载请求
GB_TO_MB = 1024

# TCP监听端口
# CLOUD_ADDR = [(SELF_IP, 8000)]  # 1
CLOUD_ADDR = [50000]

FOG_ADDRS = [58035, 58005, 58015, 58025, 58045, 58055, 58065, 58075, 58085, 58095, 58105,
             58115, 58125, 58135, 58145, 58155, 58165, 58175, 58185, 58195, 58205]

# 0.8~1.0 正态分布，按照概率均等来划分，分为8档
R_RANKS = [0.800, 0.854, 0.873, 0.887, 0.900, 0.913, 0.927, 0.946]
UN_RS = [0.200, 0.146, 0.127, 0.113, 0.1, 0.087, 0.073, 0.054]
R_RANK_NUM = len(R_RANKS)
R_MU = 0.9
R_SIGMA = 0.04
R_LOWER = 0.8
R_UPPER = 1.0

# 64~128 均匀分布 分8档
# C_RANKS = [64, 72, 80, 88, 96, 104, 112, 120]
# 8~72 均匀分布 8档
# C_RANKS = [1, 8, 16, 24, 32, 40, 48, 56, 64]
# C_RANKS = [50, 80, 160, 256, 512, 1024, 2048, 5096, 10192]
# C_RANKS = [500, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536]
# C_RANK_NUM = 8

C_RANKS = [50, 500, 8192, 16384, 24576, 32768, 40960, 49152, 57344]
C_RANK_NUM = len(C_RANKS) - 1

# C_LOWER = 64
# C_UPPER = 128
C_LOWER = C_RANKS[1]
C_UPPER = C_RANKS[len(C_RANKS)-1]


# 4
ADJACENCY_MATRIX = [[0, 60, 0, 64],
                    [60, 0, 0, 77],
                    [0, 0, 0, 73],
                    [64, 77, 73, 0]]
# ADJACENCY_MATRIX = [[0, 60, 0, 64],
#                     [64, 77, 73, 0],
#                     [0, 0, 0, 73],
#                     [60, 0, 0, 77]]
FOG_CONN_EDGE_NUM = 8

# # 8
# ADJACENCY_MATRIX = [[0, 0, 34, 0, 31, 0, 0, 36],
#                     [0, 0, 31, 0, 0, 33, 0, 0],
#                     [34, 31, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 36, 0, 35, 36],
#                     [31, 0, 0, 36, 0, 35, 0, 0],
#                     [0, 33, 0, 0, 35, 0, 0, 0],
#                     [0, 0, 0, 35, 0, 0, 0, 0],
#                     [36, 0, 0, 36, 0, 0, 0, 0]]
# FOG_CONN_EDGE_NUM = 18

# # 12
# ADJACENCY_MATRIX = [[0, 0, 26, 24, 0, 0, 0, 28, 23, 30, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 22, 0, 28, 0, 27, 28],
#                     [26, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [24, 0, 27, 0, 0, 0, 0, 0, 25, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 26, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 24, 0, 29, 0, 0],
#                     [0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [28, 0, 0, 0, 0, 24, 0, 0, 22, 0, 0, 27],
#                     [23, 28, 0, 25, 0, 0, 0, 22, 0, 29, 0, 0],
#                     [30, 0, 0, 0, 27, 29, 0, 0, 29, 0, 0, 25],
#                     [0, 27, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 28, 0, 0, 0, 0, 0, 27, 0, 25, 0, 0]]
# FOG_CONN_EDGE_NUM = 38

# # 16
# ADJACENCY_MATRIX = [[0, 0, 17, 16, 0, 0, 0, 18, 15, 19, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 15, 0, 18, 0, 17, 18, 17, 0, 0, 0, 0, 0, 0, 0],
#                     [17, 15, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0],
#                     [16, 0, 0, 0, 17, 17, 0, 0, 20, 20, 18, 0, 0, 0, 0, 0],
#                     [0, 18, 0, 17, 0, 0, 0, 15, 0, 0, 19, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 17, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 18],
#                     [0, 17, 0, 0, 0, 0, 0, 15, 0, 16, 0, 0, 18, 0, 0, 19],
#                     [18, 18, 0, 0, 15, 16, 15, 0, 0, 20, 19, 0, 0, 0, 16, 0],
#                     [15, 17, 16, 20, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 20, 20],
#                     [19, 0, 0, 20, 0, 0, 16, 20, 0, 0, 0, 17, 0, 0, 0, 0],
#                     [0, 0, 0, 18, 19, 0, 0, 19, 0, 0, 0, 17, 0, 17, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 19, 17, 17, 0, 0, 19, 0, 18],
#                     [0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 19, 0, 0, 20, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 16, 20, 0, 0, 0, 0, 20, 0, 0],
#                     [0, 0, 0, 0, 0, 18, 19, 0, 20, 0, 0, 18, 0, 0, 0, 0]]
# FOG_CONN_EDGE_NUM = 72

# # 20
# ADJACENCY_MATRIX = [[0, 0, 9, 8, 0, 0, 0, 9, 8, 10, 0, 0, 0, 0, 0, 0, 8, 0, 9, 0],
#                     [0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
#                     [9, 9, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0, 10, 10, 9, 0, 0, 0, 0, 0],
#                     [8, 9, 0, 0, 0, 0, 8, 0, 0, 10, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
#                     [0, 9, 0, 0, 0, 0, 0, 0, 0, 9, 8, 0, 8, 0, 0, 9, 0, 0, 10, 0],
#                     [0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 8, 0, 0, 0, 10, 0, 0, 10, 10],
#                     [0, 0, 0, 8, 0, 10, 0, 0, 9, 0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 10],
#                     [9, 0, 0, 0, 0, 10, 0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 10, 0, 10, 0],
#                     [8, 0, 9, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8],
#                     [10, 0, 9, 10, 9, 0, 0, 9, 0, 0, 10, 0, 8, 0, 8, 9, 8, 0, 0, 0],
#                     [0, 0, 0, 0, 8, 0, 0, 0, 0, 10, 0, 9, 0, 8, 0, 0, 0, 0, 10, 0],
#                     [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9, 0, 9, 0, 0, 8, 9, 8, 8, 0],
#                     [0, 0, 10, 0, 8, 0, 0, 0, 0, 8, 0, 9, 0, 10, 10, 0, 0, 0, 0, 0],
#                     [0, 0, 10, 0, 0, 0, 9, 10, 0, 0, 8, 0, 10, 0, 9, 0, 0, 0, 8, 0],
#                     [0, 0, 9, 0, 0, 0, 0, 0, 0, 8, 0, 0, 10, 9, 0, 9, 10, 0, 0, 0],
#                     [0, 0, 0, 0, 9, 10, 9, 0, 0, 9, 0, 8, 0, 0, 9, 0, 0, 9, 8, 0],
#                     [8, 0, 0, 8, 0, 0, 0, 10, 8, 8, 0, 9, 0, 0, 10, 0, 0, 0, 0, 10],
#                     [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 0],
#                     [9, 0, 0, 0, 10, 10, 0, 10, 0, 0, 10, 8, 0, 8, 0, 8, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 10, 10, 0, 8, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0]]
# FOG_CONN_EDGE_NUM = 124

P_MAX = 0.25
P_MIN = 0.25

FOGS_CLIENTS = [[10, 13, 14, 17, 18, 21, 36],
                [5, 6, 11, 16, 22, 26, 30, 33, 35, 37, 40],
                [1, 2, 3, 7, 8, 9, 12, 15, 20, 24, 25, 28, 29, 31, 32, 34, 38, 39],
                [4, 19, 23, 27]]

# msg type

# {'client_id': , 'data': ,'type': ,'hash_id': ,'capacity': data的要求,'reliability': data的要求,'start_time': ,
#  'cur_c':当前终端的容量,'cur_c_rank': ,'cur_r_rank': }
C2F_UPLOAD1 = 5

# 自己的方法
# {'client_id': , 'data': ,'type': ,'hash_id': ,'capacity': data的要求,'reliability': data的要求,'start_time': ,
#  'cur_c':当前终端的容量,'cur_c_rank': ,'cur_r_rank': ，'is_enough': 只在本地备份是否就已经满足可靠性要求 0/1,'last_c_rank': }
C2F_UPLOAD2 = 17

# fog收到上传请求后，询问subCloud三副本备份的策略
# C2F_UPLOAD信息 + {'from_fog_id':自己的id}
# {'id': , 'r': rank,'c': , 'method': ,'type':}
F2S_ASK_FOR_ANSWER1_TO_STORE = 9

# {'type': ,'fogs': }
S2F_TELL_ANSWER1_TO_STORE = 14

F2S_ASK_FOR_ANSWER2_TO_STORE = 19

# {'type': ,'fogs': }
S2F_TELL_ANSWER2_TO_STORE = 20

# {'to_fog_id': ,'from_fog_id': ,'store_fog_ids': (,'target_r_rank': to_fog_id的要求r_rank method2)}
# + C2F_UPLOAD的信息(type要更改一下)
F2F_STORE = 1

# F2F_STORE中的信息,type要更改
F2C_STORE = 3  # F通知C存储在自己本地

# {'copy_client_id': (本地id)备份的client_id,'hash_id': ,'type': ,'to_fog_id': 即源fog，要回复,'cur_c_rank': ,
# 'reliability': 数据可靠性要求,'store_fog_ids': ,'client_id': 原来的client_id 即发出备份请求的，'start_time': ,
# 'cur_r_rank': , 'cur_r': ,'cur_c': ,'method'}
C2F_STORE_ACK = 4  # C回复F关于缓存的请求

# C2F_STORE_ACK中的信息 + {'type': ,'from_fog_id': ,}
F2F_STORE_ACK = 2  # 回复的应该是 索引(在三副本时还有reliability)

# {'type': ,'fogs': .'hash_id': ,'start_time': ,'distance_time': ,'data_r': ,'cur_rs':}
F2C_STORE_ACK1 = 6  # F告诉C已经保存完毕

# {'type': ,'fogs': ,'hash_id': ,'start_time': ,'distance_time': ,'is_enough': ,'data_r': ,'cur_rs':}
F2C_STORE_ACK2 = 18

# 在让sub选择雾节点备份时，无法选出来
# {'type': ,'hash_id': }
F2C_STORE_ACK_ERROR = 7

# 方法1的 {'fog_id': , 'c_c_rank_nums': ,'type': ,'up_c_rank': ,'down_c_rank': }
F2S_UPDATE_C_RANK_NUMS1 = 8

# 方法2的
F2S_UPDATE_C_RANK_NUMS2 = 15

# 有终端资源耗尽了 {'type': ,'fog_id':}
F2S_HAVE_CLIENT_FULL = 10

# 出现资源耗尽的情况，停止备份，查看现在的资源利用情况
S2F_ASK_FOR_C_UTILIZATION = 11

# F告知S自己的终端空间使用情况
F2S_TELL_C_UTILIZATION = 12

F2S_HEARTBEAT_RATE = 13

# F对S的初始信息确认
F2S_CONFIG = 16

# 告知源fog节点，自己无法备份，因为空间不够
F2F_STORE_ACK_ERROR = 22

F2C_FOG_ALL_FULL = 23


# 备份请求类型
C2F_UPLOAD1_BASELINE = 30

C2F_UPLOAD2_BASELINE = 31

C2F_ASK_IS_SUCCESS = 35

C2F_UPLOAD1_DCABA = 36

C2F_UPLOAD1_CLB = 37

C2F_UPLOAD1_ACO_VMM = 38

# method
THREE_STORE = 1

METHOD2_STORE = 2

DCABA_STORE = 3

CLB_STORE = 4

ACO_VMM_STORE = 5


# 回复类型
F2S_ASK_FOR_ANSWER_DCABA_TO_STORE = 40

F2S_ASK_FOR_ANSWER_CLB_TO_STORE = 41

F2S_ASK_FOR_ANSWER_ACO_VMM_TO_STORE = 42

F2S_ASK_FOR_ANSWER_DRPS_TO_STORE = 60



# IS2023 优化  DCABA、CLB、ACO-VMM与三副本变成多副本 -> 默认终端可靠性为最低可靠性0.8
C2F_UPLOAD3_BASELINE = 50

C2F_UPLOAD3_DCABA = 51

C2F_UPLOAD3_CLB = 52

C2F_UPLOAD3_ACO_VMM = 53

C2F_UPLOAD3_DRPS = 55




ORIGIN_STORE3 = 10

DCABA_STORE3 = 13

CLB_STORE3 = 14

ACO_VMM_STORE3 = 15

DRPS_STORE3 = 16



####### RL #######
F2S_ASK_FOR_ANSWER_RL_DDQN_TO_STORE = 101

C2F_UPLOAD_RL_DDQN = 102

RL_DDQN_METHOD = 20

###### GA ######
F2S_ASK_FOR_ANSWER_GA_TO_STORE = 104

C2F_UPLOAD_GA = 105

GA_METHOD = 21

###### DE ######
F2S_ASK_FOR_ANSWER_DE_TO_STORE = 107

C2F_UPLOAD_DE = 108

DE_METHOD = 22

###### PSO ######
F2S_ASK_FOR_ANSWER_PSO_TO_STORE = 110

C2F_UPLOAD_PSO = 111

PSO_METHOD = 23


###### SA ######
F2S_ASK_FOR_ANSWER_SA_TO_STORE = 113

C2F_UPLOAD_SA = 114

SA_METHOD = 24

###### Random ######
F2S_ASK_FOR_ANSWER_RANDOM_TO_STORE = 116

C2F_UPLOAD_RANDOM = 117

RANDOM_METHOD = 25



###### result ######
METHOD1 = 1  #SA
METHOD2 = 2  #DDQN
METHOD3 = 3  #DE
METHOD4 = 4  #PSO
METHOD5 = 5  #GA
METHOD6 = 6  #RANDOM
