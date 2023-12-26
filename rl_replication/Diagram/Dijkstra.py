from CONFIG import FOG_NUM
from CONFIG import FOG_CONN_EDGE_NUM
from CONFIG import ADJACENCY_MATRIX


def find_path(prior_matrix, i, j):
    '''
    :param prior_matrix: 记录从所有初始节点到终止节点的最短路径所需要经过的上一个节点
    :param i:当前的起始节点 i+1为真实节点编号
    :param j:当前的终止节点 +1为真实节点编号
    :return:返回最短路径所经过的节点
    '''
    if prior_matrix[i][j] == j:
        return '%d' % (i + 1) + '-' + '%d' % (j + 1)
    else:
        return find_path(prior_matrix, i, prior_matrix[i][j]) + find_path(prior_matrix, prior_matrix[i][j], j)


def dijkstra(num_node=FOG_NUM, num_vertice=FOG_CONN_EDGE_NUM):
    vertice_list = []

    # print('请输入连接的边(eg: 1 2 1 表示fog1--fog2，权重为1):')
    for i in range(num_node):
        for j in range(i + 1, num_node):
            if ADJACENCY_MATRIX[i][j] == 0:
                continue
            temp_line = [i + 1, j + 1, ADJACENCY_MATRIX[i][j]]
            vertice_list.append(temp_line)

    distance_matrix = [[float('inf') for i in range(num_node)] for j in range(num_node)]

    for i in range(num_node):
        distance_matrix[i][i] = 0

    for vertice in vertice_list:
        distance_matrix[vertice[0] - 1][vertice[1] - 1] = vertice[2]
        distance_matrix[vertice[1] - 1][vertice[0] - 1] = vertice[2]

        # 无向图的距离路径矩阵为对称矩阵

    # print(distance_matrix)

    prior_matrix = [[0 for i in range(num_node)] for j in range(num_node)]
    # 初始化prior矩阵

    for p in range(num_node):
        for q in range(num_node):
            prior_matrix[p][q] = q
    # print(prior_matrix)

    # 从for循环的角度也可以看出，floyd算法的时间复杂度是O(n**3)
    for k in range(num_node):
        # 将无向图中的当前节点加入进来，判断以当前节点为中介节点后，最短路径是否发生变换
        for i in range(num_node):
            for j in range(num_node):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    # 更新距离矩阵中的数值
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    prior_matrix[i][j] = prior_matrix[i][k]

    # print('各个顶点对之间的最短路径：')
    # print(prior_matrix)
    # print(distance_matrix)

    rount = {}
    for i in range(num_node):
        # print('\n')
        for j in range(i + 1, num_node):
            temp_route = []

            temp_route = find_path(prior_matrix, i, j)

            if temp_route.count('>') == 1:  # 如果从初始节点i到终止节点j并不需要任何的中间节点，则直接输出字符串
                display_line = temp_route
            else:
                # 此时 find_path 函数返回字符串具有中间的overlap，需要去重
                # print(temp_route)
                output_str = temp_route.split('-')
                display_line = ''
                display_line += '%d' % (i + 1)
                for t in range(1, len(output_str) - 1):
                    # output_str[t] = output_str[t][0:int(len(output_str) / 2)]
                    display_line += '-' + output_str[t][0:int(len(output_str[t]) / 2)]
                display_line += '-%d' % (j + 1)
            # print('%d-%d: distance:%d' % (i + 1, j + 1, distance_matrix[i][j]), 'route:', display_line)

            temp_key = str(i + 1) + '-' + str(j + 1)
            rount[temp_key] = display_line.split('-')

    return distance_matrix, rount
