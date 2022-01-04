from random import random

import networkx as nx



def generate_cnot_err(architecture):
    if architecture == "ibm_q20_tokyo":
        vertices = [i for i in range(20)]
        edges = connect_vertices_as_grid(5, 4, vertices)
        cross_edges = [
            (1, 7), (2, 8),
            (3, 5), (4, 6),
            (6, 12), (7, 13),
            (8, 10), (9, 11),
            (11, 17), (12, 18),
            (13, 15), (14, 16)
        ]
        edges.extend([(vertices[v1], vertices[v2]) for v1, v2 in cross_edges])
        # 基于edges生成无向图
        G = nx.Graph()
        # G.add_edges_from(edges)
        # print(G.nodes)
        # print(G.edges)

        # 添加权重属性 CNOT-err
        for edge in edges:
            while True:
                weight = random()
                if 0.005 < weight < 0.03:
                    G.add_edge(edge[0], edge[1], weight=weight)
                    break

        # print(G.edges)
        # print(G.edges.data())

        # 获取 每个边, 以及每个边的权重
        # 将边和权重写入distance字典
        distances = {}
        for edge in edges:
            begin = edge[0]
            end = edge[1]
            w = G[begin][end]["weight"]
            distances[(begin, end)] = w
            distances[(end, begin)] = w

        # print(distances)
        return distances


def connect_vertices_as_grid(width, height, vertices):  # 以width=3, height=3, vertices=9为例
    # 如果输入的长*宽 != 总顶点数, 输出错误提示
    if len(vertices) != width * height:
        raise KeyError("To make a grid, you need vertices exactly equal to width*height, but got %d=%d*%d." % (
            len(vertices), width, height))

    # 将顶点练成一条线 列表中的每个元素都是元组 [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    edges = connect_vertices_in_line(vertices)

    # 生成[水平]行: ['012', '345', '678']
    horizontal_lines = [vertices[i * width: (i + 1) * width] for i in range(height)]  # [range(0, 4), range(4, 8), range(8, 12), range(12, 16)]

    # horizontal_lines = [range(0, 3), range(3, 6), range(6, 9)] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # horizontal_lines[1:] = [range(3, 6), range(6, 9)] = [(3, 4, 5), (6, 7, 8)]
    # line1 = range(0, 3)
    # line2 = range(3, 6)
    for line1, line2 in zip(horizontal_lines, horizontal_lines[1:]):

        # line1[:-1] 取line1的索引0~倒数第二索引位
        # line2[1:0] 取line2的索引1~最后一位
        new_edges = [(v1, v2) for v1, v2 in zip(line1[:-1], reversed(line2[1:]))]  # 第一次 line1[:-1]=[0, 1], reversed(line2[1:] = [5, 4]
        edges.extend(new_edges)  # 第一次 new_edges = [(0, 5), (1, 4)]
    return edges  # [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (0, 5), (1, 4), (3, 8), (4, 7)]

#  生成每个顶点之间连线的列表:[('0', '1'), ('1', '2'), ('2', '3')...]
def connect_vertices_in_line(vertices):
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]



if __name__ == '__main__':
    generate_cnot_err("ibm_q20_tokyo")