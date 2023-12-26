import os
import random
import threading
import time
import csv

import CONFIG as CONFIG
import Method as Method
from Client import client


# capacity = []

if __name__ == '__main__':

    clients = []
    thread = []

    data1 = 's'
    data2 = 'h'

    reliabilitys = Method.create_reliability()
    capacitys = Method.create_capacity()
    print(reliabilitys)
    print(capacitys)
    index = 0

    if not os.path.exists(CONFIG.RESULT_FOLDER + str(CONFIG.METHOD4)):
        os.makedirs(CONFIG.RESULT_FOLDER + str(CONFIG.METHOD4))
    client.success_file = open(CONFIG.RESULT_FOLDER + str(CONFIG.METHOD4) + '/result_success.csv', "a", newline='')
    print('生成成功的输出结果文件')
    client.result_success = csv.writer(client.success_file)
    client.result_success.writerow(['f_id', 'c_id', 'hash_id', 'store_fogs', 'time', 'backup_num', 'is_reliable'])
    client.success_file.flush()

    client.error_file = open(CONFIG.RESULT_FOLDER + str(CONFIG.METHOD4) + '/result_error.csv', "a", newline='')
    print('生成失败的输出结果文件')
    client.result_error = csv.writer(client.error_file)
    client.result_error.writerow(['f_id', 'c_id', 'hash_id', 'data_c'])
    client.error_file.flush()

    if not os.path.exists(CONFIG.RESULT_FOLDER + str(CONFIG.METHOD4)+'/result'):
        os.mkdir(CONFIG.RESULT_FOLDER+ str(CONFIG.METHOD4)+'/result')
    total_c_num = CONFIG.CLIENT_NUM * CONFIG.FOG_NUM

    time.sleep(100)

    while index < total_c_num:
        fog_id = index % CONFIG.FOG_NUM + 1
        c = index // CONFIG.FOG_NUM
        temp_client = client(index + 1, reliabilitys[index], capacitys[fog_id][c], CONFIG.FOG_ADDRS[index % CONFIG.FOG_NUM],
                             fog_id)
        clients.append(temp_client)
        temp_thread = threading.Thread(target=temp_client.receive_msg_from_fog)
        thread.append(temp_thread)
        temp_thread.start()
        index += 1

    time.sleep(50)

    for i in range(0, 1000000000000000):
        c_id = random.randint(1, total_c_num)
        # r = random.uniform(0.8, 1.0)
        r = Method.create_data_reliability_baseline()
        # print('内存：',temp_c.capacity)
        temp_c = clients[c_id - 1]
        c = Method.create_data_capacity_baseline()
        temp_c.upload_data_gas(data2, c, r, i, CONFIG.PSO_METHOD)
        print(i, '  ', c_id)
        while temp_c.cur_waiting_index != i:
            continue
        if temp_c.is_fog_all_full():
            print('有雾节点满了')
            break

    print('打印东西')

    for i in thread:
        i.join()
