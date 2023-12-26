import CONFIG

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    j = 6
    str1 = ""
    for i in range(1, CONFIG.FOG_NUM + 1):
        str1 += "nohup python3 s" + str(
            i) + ".py > log" + str(j) + "/s" + str(i) + ".log 2>&1 &\n"
    str1 += "nohup python3 sub_cloud.py > log" + str(
        j) + "/sub_cloud.log 2>&1 &\n"
    str1 += "python3 c" + str(
        j) + ".py > log" + str(j) + "/c" + str(j) + ".log 2>&1\n"
    print(str1)
    # for j in range(6 ,7):
    #     str1 = "#!/bin/bash\n"
    #     str1 += "echo \""+str(j)+"\"\n"
    #     str1 += "source ~/software/anaconda3/etc/profile.d/conda.sh\n"
    #     str1 += "conda activate zmk1\n"
    #     str1 += "sleep 20\n"
    #     for i in range(1, CONFIG.FOG_NUM + 1):
    #         str1 += "nohup python3 /public/home/ssct005t/project/zmk/device/20/test/s" + str(i) + ".py > /public/home/ssct005t/project/zmk/device/20/test/log" + str(j) + "/s" + str(i) + ".log 2>&1 &\n"
    #     str1 += "sleep 10\n"
    #     str1 += "nohup python3 /public/home/ssct005t/project/zmk/device/20/test/sub_cloud.py > /public/home/ssct005t/project/zmk/device/20/test/log" + str(
    #             j) + "/sub_cloud.log 2>&1 &\n"
    #     str1 += "python3 /public/home/ssct005t/project/zmk/device/20/test/c" + str(
    #         j) + ".py > /public/home/ssct005t/project/zmk/device/20/test/log" + str(j) + "/c" + str(j) + ".log 2>&1\n"
    #
    #     with open("run" + str(j) + ".sh", 'w') as file:
    #         file.write(str1)
    #         file.flush()

    # for j in range(6, 7):
    #     str1 = "#!/bin/bash\n"
    #     str1 += "echo \""+str(j)+"\"\n"
    #     str1 += "source ~/software/anaconda3/etc/profile.d/conda.sh\n"
    #     str1 += "conda activate zmk1\n"
    #     str1 += "sleep 20\n"
    #     for i in range(1, CONFIG.FOG_NUM + 1):
    #         str1 += "nohup python3 /public/home/ssct005t/project/zmk/main/160/test/s" + str(i) + ".py > /public/home/ssct005t/project/zmk/main/160/test/log" + str(j) + "/s" + str(i) + ".log 2>&1 &\n"
    #     str1 += "sleep 10\n"
    #     str1 += "nohup python3 /public/home/ssct005t/project/zmk/main/160/test/sub_cloud.py > /public/home/ssct005t/project/zmk/main/160/test/log" + str(
    #             j) + "/sub_cloud.log 2>&1 &\n"
    #     str1 += "python3 /public/home/ssct005t/project/zmk/main/160/test/c" + str(
    #         j) + ".py > /public/home/ssct005t/project/zmk/main/160/test/log" + str(j) + "/c" + str(j) + ".log 2>&1\n"
    #
    #
    #     with open("run" + str(j) + ".sh", 'w') as file:
    #         file.write(str1)
    #         file.flush()

    # cur_capacity_dict = {1: 1000, 2: 2000, 3: 2500, 4: 1500}
    # ori_capacity_dict = {1: 3000, 2: 5000, 3: 5500, 4: 4500}
    # distance = [0, 60, 137, 64]
    # max_min_dict = {1: [[3, 2], [-1, 9], [6, 3], [1, 1]],
    #                 2: [[-1, 9], [5, 4], [5, 4], [-1, 9]],
    #                 3: [[-1, 9], [-1, 9], [-1, 9], [5, 3]],
    #                 4: [[-1, 9], [5, 2], [-1, 9], [-1, 9]],
    #                 5: [[-1, 9], [4, 2], [-1, 9], [-1, 9]],
    #                 6: [[7, 4], [-1, 9], [-1, 9], [-1, 9]],
    #                 7: [[-1, 9], [-1, 9], [-1, 9], [6, 3]],
    #                 8: [[-1, 9], [2, 2], [5, 3], [-1, 9]]}
    # r_rank_mem_rate_dict = {1: [0.30000000000000004, 0.0, 0.8, 0.1],
    #                         2: [0.0, 0.5, 0.5, 0.0],
    #                         3: [0.0, 0.0, 0.0, 0.5],
    #                         4: [0.0, 0.5, 0.0, 0.0],
    #                         5: [0.0, 0.4, 0.0, 0.0],
    #                         6: [0.7000000000000001, 0.0, 0.0, 0.0],
    #                         7: [0.0, 0.0, 0.0, 0.6000000000000001],
    #                         8: [0.0, 0.2, 0.5, 0.0]}
    #
    # print(ddqn_backup_strategy(30,0.990,1,0.99,cur_capacity_dict,ori_capacity_dict,distance,max_min_dict))

    # print(dijkstra())

    # 将离散动作转换为坐标形式
    # action_coord = (action // 8 + 1, action % 8 + 1)
    # print(action_coord)
    # print(action)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
