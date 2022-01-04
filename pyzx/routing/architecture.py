# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2019 - Aleks Kissinger, John van de Wetering,
#                      and Arianne Meijer-van de Griend

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from ..graph.graph import Graph
# from pyzx.graph.base import BaseGraph # TODO fix the right graph import - one of many - right backend etc

import numpy as np

SQUARE = "square"
LINE = "line"
FULLY_CONNNECTED = "fully_connected"
CIRCLE = "circle"
IBM_QX2 = "ibm_qx2"
IBM_QX3 = "ibm_qx3"
IBM_QX4 = "ibm_qx4"
IBM_QX5 = "ibm_qx5"
IBMQ_16_MELBOURNE = "ibmq_16_melbourne"
IBM_Q20_TOKYO = "ibm_q20_tokyo"
RIGETTI_16Q_ASPEN = "rigetti_16q_aspen"
RIGETTI_8Q_AGAVE = "rigetti_8q_agave"

architectures = [SQUARE, CIRCLE, FULLY_CONNNECTED, LINE, IBM_QX4, IBM_QX2, IBM_QX3, IBM_QX5, IBMQ_16_MELBOURNE,
                 IBM_Q20_TOKYO,
                 RIGETTI_8Q_AGAVE, RIGETTI_16Q_ASPEN]

dynamic_size_architectures = [FULLY_CONNNECTED, LINE, CIRCLE, SQUARE]

# debug = True  # 调试时设置成True,原来是False
debug = False


class Architecture():
    def __init__(self, name, coupling_graph=None, coupling_matrix=None, backend=None):
        """
        Class that represents the architecture of the qubits to be taken into account when routing. 表示路由时要考虑的量子位的体系结构的类

        :param coupling_graph: a PyZX Graph representing the architecture, optional  代表架构的PyZX图，可选
        :param coupling_matrix: a 2D numpy array representing the adjacency of the qubits, from which the Graph is created, optional  表示从其创建图的qubits邻接的2D numpy数组，可选
        :param backend: The PyZX Graph backend to be used when creating it from the adjacency matrix, optional  从邻接矩阵创建时使用的PyZX Graph后端，可选
        """
        self.name = name
        if coupling_graph is None:
            # 如果图为空,就生成空图
            self.graph = Graph(backend=backend)
        else:
            # 否则, 将现有的图赋值给当前类
            self.graph = coupling_graph

        if coupling_matrix is not None:
            # build the architecture graph
            n = coupling_matrix.shape[0]
            self.vertices = self.graph.add_vertices(n)
            edges = [(self.vertices[row], self.vertices[col]) for row in range(n) for col in range(n) if
                     coupling_matrix[row, col] == 1]
            self.graph.add_edges(edges)
        else:
            # 矩阵为空, 将图的顶点赋值给一个vertices列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            self.vertices = [v for v in self.graph.vertices()]

        self.pre_calc_distances()
        # print("self.distances:",self.distances)  #任意两点之间的距离和具体路径

        self.qubit_map = [i for i, v in enumerate(self.vertices)]
        # print("self.qubit_map:",self.qubit_map) #[0, 1, 2, 3, 4]

        self.n_qubits = len(self.vertices)  # 5
        # print("self.n_qubits:",self.n_qubits)

    def pre_calc_distances(self):
        # upper视为有向图, full视为无向图
        self.distances = {"upper": [self.floyd_warshall(until, upper=True) for until, v in enumerate(self.vertices)],
                          # until 和 v 是对 self.vertices 取的枚举: [(0, 0), (1, 1) ... (15, 15)]
                          "full": [self.floyd_warshall(until, upper=False) for until, v in
                                   enumerate(self.vertices)]}  # full表示,只找src>tgt的路径

    def to_quil_device(self):
        # Only required here
        import networkx as nx
        from pyquil.device import NxDevice
        edges = [edge for edge in self.graph.edges() if edge[0] in self.vertices]
        topology = nx.from_edgelist(edges)
        device = NxDevice(topology)
        return device

    def visualize(self, filename=None):
        import networkx as nx
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        g = nx.Graph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from(self.graph.edges())
        nx.draw(g, with_labels=True, font_weight='bold')
        if filename is None:
            filename = self.name + ".png"
        plt.savefig(filename)

    def floyd_warshall(self, exclude_excl, upper=True):
        # Floyd算法又称为插点法,是一种利用动态规划的思想寻找给定的加权图中多源点之间最短路径的算法
        # 可以生成当前图中, 每两个顶点之间的最短路径
        """
        Implementation of the Floyd-Warshall algorithm to calculate the all-pair distances in a given graph
        :param exclude_excl: index up to which qubit should be excluded from the distances  应该从distances中排除的量子位的索引
        :param upper: whether use bidirectional edges or only ordered edges (src, tgt) such that src > tgt, default True # 默认使用双向边
        :return: a dict with for each pair of qubits in the graph, a tuple with their distance and the corresponding shortest path
        """
        # https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
        distances = {}

        # 如果upper为真, vertices取until~最后一个 , 否则, vertices取0~until个
        # 第一次until=0, 取0~最后一个
        vertices = self.vertices[exclude_excl:] if upper else self.vertices[:exclude_excl + 1]

        for edge in self.graph.edges():
            src, tgt = self.graph.edge_st(edge)  # 接收给定边的源点和目标点组成的元组
            if src in vertices and tgt in vertices:
                # 将边的数据,以 (源点, 终点):(1, [(源点,终点)]) 的格式保存到 distances 字典中
                if upper:
                    # distances[(src, tgt)] = (1, [(src, tgt)])
                    # distances[(tgt, src)] = (1, [(tgt, src)])

                    # if src == 0 and tgt == 1:
                    #     distances[(src, tgt)] = (0.00715, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00715, [(tgt, src)])
                    # if src == 1 and tgt == 2:
                    #     distances[(src, tgt)] = (0.033, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.033, [(tgt, src)])
                    # if src == 2 and tgt == 3:
                    #     distances[(src, tgt)] = (0.0335, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.0335, [(tgt, src)])
                    # if src == 3 and tgt == 4:
                    #     distances[(src, tgt)] = (0.00831, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00831, [(tgt, src)])
                    # if src == 4 and tgt == 5:
                    #     distances[(src, tgt)] = (0.0238, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.0238, [(tgt, src)])
                    # if src == 5 and tgt == 6:
                    #     distances[(src, tgt)] = (0.016, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.016, [(tgt, src)])
                    # if src == 6 and tgt == 7:
                    #     distances[(src, tgt)] = (0.0225, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.0225, [(tgt, src)])
                    # if src == 7 and tgt == 8:
                    #     distances[(src, tgt)] = (0.0192, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.0192, [(tgt, src)])
                    # if src == 0 and tgt == 5:
                    #     distances[(src, tgt)] = (0.00485, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00485, [(tgt, src)])
                    # if src == 1 and tgt == 4:
                    #     distances[(src, tgt)] = (0.00747, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00747, [(tgt, src)])
                    # if src == 3 and tgt == 8:
                    #     distances[(src, tgt)] = (0.00826, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00826, [(tgt, src)])
                    # if src == 4 and tgt == 7:
                    #     distances[(src, tgt)] = (0.00779, [(src, tgt)])
                    #     distances[(tgt, src)] = (0.00779, [(tgt, src)])

                    # ibm_qx2 本次实验错误率文件, 20200507
                    if src == 0 and tgt == 1:
                        distances[(src, tgt)] = (0.01828, [(src, tgt)])
                        distances[(tgt, src)] = (0.01828, [(tgt, src)])
                    if src == 1 and tgt == 2:
                        distances[(src, tgt)] = (0.02383, [(src, tgt)])
                        distances[(tgt, src)] = (0.02383, [(tgt, src)])
                    if src == 2 and tgt == 3:
                        distances[(src, tgt)] = (0.0212, [(src, tgt)])
                        distances[(tgt, src)] = (0.0212, [(tgt, src)])
                    if src == 3 and tgt == 4:
                        distances[(src, tgt)] = (0.01667, [(src, tgt)])
                        distances[(tgt, src)] = (0.01667, [(tgt, src)])
                    if src == 0 and tgt == 2:
                        distances[(src, tgt)] = (0.02512, [(src, tgt)])
                        distances[(tgt, src)] = (0.02512, [(tgt, src)])
                    if src == 2 and tgt == 4:
                        distances[(src, tgt)] = (0.01386, [(src, tgt)])
                        distances[(tgt, src)] = (0.01386, [(tgt, src)])

                    # ibm_qx20_tokyo 模拟错误率
                    # edges = [(0, 1), (0, 9), (1, 2), (1, 8), (1, 7), (2, 3), (2, 7), (2, 8), (3, 4), (3, 6), (3, 5), (4, 5), (4, 6), (5, 6), (5, 14), (6, 7), (6, 13), (6, 12), (7, 8), (7, 12), (7, 13), (8, 9), (8, 11), (8, 10), (9, 10), (9, 11), (10, 11), (10, 19), (11, 12), (11, 18), (11, 17), (12, 13), (12, 17), (12, 18), (13, 14), (13, 16), (13, 15), (14, 15), (14, 16), (15, 16), (16, 17), (17, 18), (18, 19)]
                    # d = {(0, 1): 0.014207527638399275, (1, 0): 0.014207527638399275, (1, 2): 0.024290178457425093, (2, 1): 0.024290178457425093, (2, 3): 0.027168787351182666, (3, 2): 0.027168787351182666, (3, 4): 0.013706785185861659, (4, 3): 0.013706785185861659, (4, 5): 0.029119061527025036, (5, 4): 0.029119061527025036, (5, 6): 0.00787828680396152, (6, 5): 0.00787828680396152, (6, 7): 0.009063153966809345, (7, 6): 0.009063153966809345, (7, 8): 0.014442076833968032, (8, 7): 0.014442076833968032, (8, 9): 0.010813366654275902, (9, 8): 0.010813366654275902, (9, 10): 0.005096951898053481, (10, 9): 0.005096951898053481, (10, 11): 0.023286190583792887, (11, 10): 0.023286190583792887, (11, 12): 0.02413382052201607, (12, 11): 0.02413382052201607, (12, 13): 0.006597365209280004, (13, 12): 0.006597365209280004, (13, 14): 0.026225576032290032, (14, 13): 0.026225576032290032, (14, 15): 0.009411644623279036, (15, 14): 0.009411644623279036, (15, 16): 0.02033301700041823, (16, 15): 0.02033301700041823, (16, 17): 0.009226216756734296, (17, 16): 0.009226216756734296, (17, 18): 0.011682597952483387, (18, 17): 0.011682597952483387, (18, 19): 0.01220832829466012, (19, 18): 0.01220832829466012, (0, 9): 0.010800035815830222, (9, 0): 0.010800035815830222, (1, 8): 0.026420019573295206, (8, 1): 0.026420019573295206, (2, 7): 0.005737052087316896, (7, 2): 0.005737052087316896, (3, 6): 0.005647303849380836, (6, 3): 0.005647303849380836, (5, 14): 0.026953063111803233, (14, 5): 0.026953063111803233, (6, 13): 0.02935246179456441, (13, 6): 0.02935246179456441, (7, 12): 0.02284138606686259, (12, 7): 0.02284138606686259, (8, 11): 0.014650617673114597, (11, 8): 0.014650617673114597, (10, 19): 0.012848053617932731, (19, 10): 0.012848053617932731, (11, 18): 0.013150872568784333, (18, 11): 0.013150872568784333, (12, 17): 0.017264775497292106, (17, 12): 0.017264775497292106, (13, 16): 0.025300458539239123, (16, 13): 0.025300458539239123, (1, 7): 0.008366877726535549, (7, 1): 0.008366877726535549, (2, 8): 0.025989727023136067, (8, 2): 0.025989727023136067, (3, 5): 0.013379099609787004, (5, 3): 0.013379099609787004, (4, 6): 0.017063609942954305, (6, 4): 0.017063609942954305, (6, 12): 0.009770069110481061, (12, 6): 0.009770069110481061, (7, 13): 0.013902018293443952, (13, 7): 0.013902018293443952, (8, 10): 0.007881462071904877, (10, 8): 0.007881462071904877, (9, 11): 0.006206668868140808, (11, 9): 0.006206668868140808, (11, 17): 0.028524596489507892, (17, 11): 0.028524596489507892, (12, 18): 0.011519143095856887, (18, 12): 0.011519143095856887, (13, 15): 0.006980592807724562, (15, 13): 0.006980592807724562, (14, 16): 0.02693989151391718, (16, 14): 0.02693989151391718}
                    # for e in edges:
                    #     src = e[0]
                    #     tgt = e[1]
                    #     weight = d[(src, tgt)]
                    #     distances[(src, tgt)] = (weight, [(src, tgt)])
                    #     distances[(tgt, src)] = (weight, [(tgt, src)])

                elif src > tgt:
                    distances[(src, tgt)] = (1, [(src, tgt)])
                else:
                    distances[(tgt, src)] = (1, [(tgt, src)])

        # 自己到自己的路径为0
        for v in vertices:
            distances[(v, v)] = (0, [])

        # i, v0 取 0,0 | 1,1 | 2,2 | ... | 8,8
        # 当upper为True时, j, v1 取 0,0 | 1,1 | 2,2 | ... | 8,8;
        # 当upper为False时, 取 0,0 | 1,1 | 2,2 | ... | i,i
        # 当upper为True时, v2 取 0, 1, ... , 8;
        # 当upper为False时, 取 0, 1, ... , i + j
        # 如果 (v0, v1) 和 (v1, v2) 都在路径集合里, 但 (v0, v2) 或者 (v0, v2)在集合中但它的权值大于 (v0, v1) 加上 (v1, v2) 的权值:
        # 则更新字典中key为(v0, v2)的记录 --> (权值, [边的元组集])
        for i, v0 in enumerate(vertices):
            for j, v1 in enumerate(vertices if upper else vertices[:i + 1]):
                for v2 in vertices if upper else vertices[: i + j + 1]:
                    if (v0, v1) in distances.keys():
                        if (v1, v2) in distances.keys():
                            if (v0, v2) not in distances.keys() or distances[(v0, v2)][0] > distances[(v0, v1)][0] + \
                                    distances[(v1, v2)][0]:
                                distances[(v0, v2)] = (distances[(v0, v1)][0] + distances[(v1, v2)][0],
                                                       distances[(v0, v1)][1] + distances[(v1, v2)][1])
                                if upper:
                                    distances[(v2, v0)] = (distances[(v0, v1)][0] + distances[(v1, v2)][0],
                                                           distances[(v2, v1)][1] + distances[(v1, v0)][1])
        return distances

    def steiner_tree(self, start, nodes, upper=True):
        """
        给定体系结构，根量子位和应存在的其他量子位，近似斯坦纳树。这是通过使用预先计算的所有对最短距离和创建最小生成树的Prim算法完成的
        Approximates the steiner tree given the architecture, a root qubit and the other qubits that should be present.
        This is done using the pre-calculated all-pairs shortest distance and Prim's algorithm for creating a minimum spanning tree
        :param start: The index of the root qubit to be used
        :param nodes: The indices of the other qubits that should be present in the steiner tree
        :param upper: Whether the steiner tree is used for creating an upper triangular matrix or a full reduction.
        :yields: First yields all edges from the tree top-to-bottom, finished with None, then yields all edges from the tree bottom-up, finished with None.
        """
        # 通过计算所有对的最短路径，然后求解顶点子集及其各自的最短路径的最小生成树来近似
        # Approximated by calculating the all-pairs shortest paths and then solving the mininum spanning tree over the subset of vertices and their respective shortest paths.
        # https://en.wikipedia.org/wiki/Steiner_tree_problem#Approximating_the_Steiner_tree

        # The all-pairs shortest paths are pre-calculated and the mimimum spanning tree is solved with Prim's algorithm
        # 预先计算所有对的最短路径，并使用Prim算法求解最小生成树
        # https://en.wikipedia.org/wiki/Prim%27s_algorithm

        # returns an iterator that walks the steiner tree, yielding (adj_node, leaf) pairs. If the walk is finished, it yields None
        # 返回遍历斯坦纳树的迭代器，产生（adj_node，叶子）对。如果步行完成，则无
        state = [start, [n for n in nodes]]
        debug and print("state:", state)

        root = start
        # TODO deal with qubit mapping
        vertices = [root]
        edges = []
        debug and print(f"root : {root}, upper : {upper}, nodes : {nodes}")

        distances = self.distances["upper"][root] if upper else self.distances["full"][root]
        # print("distances:",distances)  # 弗洛伊德算法生成的多源最短路径的字典集合

        # distances = self.distances["upper"][root] if upper else self.distances["full"][root]
        # if upper is False:
        #     temp_distances = distances.copy()
        #
        #     # 遍历字典, 获取当前字典的key
        #     # 判断key中的顶点大小, 如果 key[0] < key[1] 则删除当前键值对
        #     for key in temp_distances:
        #         path = temp_distances[key]
        #         for item in path[1]:
        #             if item[0] < item[1]:
        #                 # 删除存在由小到大的路径
        #                 # 如果key不在字典中, 则pass
        #                 distances_keys = distances.keys()
        #                 for k in distances_keys:
        #                     if k == key:
        #                         del distances[key]
        #                         break
        #                     # 删除否则当前键值对
        #                     else:
        #                         continue

        steiner_pnts = []

        while nodes != []:
            options = [(node, v, *distances[(v, node)]) for node in nodes for v in (vertices + steiner_pnts) if
                       (v,
                        node) in distances.keys()]  # 连接vertices和steiner_pnts列表，v为遍历该列表的临时变量，node为遍历nodes列表的临时变量；以node和v为变量，遍历distances，获取对应的node到v的路径列表
            # distances中包含距离的值和具体路径，distances[(v, node)]
            # 拆包
            # 单星号 * 用于对列表LIST或元组tuple中的元素进行取出（unpacke）
            # 双星号 ** 可将字典里的“值”取出
            debug and print("options:", options)

            best_option = min(options, key=lambda x: x[2])
            debug and print("best_option:", best_option)

            debug and print("Adding to tree: vertex ", best_option[0], "Edges ", best_option[3])

            vertices.append(best_option[0])
            edges.extend(best_option[3])

            steiner = [v for edge in best_option[3] for v in edge if
                       v not in vertices]  # 将在distances中,但不在vertices中的顶点加入steiner列表, 数据会重复
            debug and print(steiner)

            steiner_pnts.extend(steiner)  # 将steiner加入到steiner_pnts列表中

            nodes.remove(best_option[0])  # 将当前node从nodes列表中移除

        edges = set(edges)  # remove duplicates，变成集合，去重复

        if debug:
            print("edges:", edges)
            print("vertices:", vertices)
            print("steiner points:", steiner_pnts)

        # First go through the tree to find and remove zeros
        state += [[e for e in edges], [v for v in vertices], [s for s in steiner_pnts]]
        debug and print("stat:", state)

        vs = {root}
        n_edges = len(edges)
        yielded_edges = set()
        debug_count = 0
        yield_count = 0
        warning = 0

        while len(yielded_edges) < n_edges:
            es = [e for e in edges for v in vs if e[0] == v]
            debug and print("es:", es)

            old_vs = [v for v in vs]
            yielded = False

            for edge in es:
                yield edge
                # 如果你看不懂生成器函数，也就是带有yield关键字的函数，那么你可以这样去理解：
                # 在函数开始处，加入 result = list()；
                # 将每个 yield 表达式 yield expr 替换为 result.append(expr)；
                # 在函数末尾处，加入 return result。
                # 也就是说，yield的本质功能还是返回了一个可供迭代的列表。
                vs.add(edge[1])
                if edge in yielded_edges:
                    print("DOUBLE yielding! - should not be possible!")
                yielded_edges.add(edge)
                yielded = True
                yield_count += 1
            [vs.remove(v) for v in old_vs]
            if not yielded:
                debug and print("leaf!")
                debug_count += 1
                if debug_count > len(vertices):
                    print("infinite loop!", warning)
                    warning += 1
            if yield_count > len(edges):
                print("Yielded more edges than existing... This should not be possible!", warning)
                warning += 1
            if warning > 5:
                print(state, yielded_edges)
                # input("note it down")
                break
        yield None

        # Walk the tree bottom up to remove all ones.
        yield_count = 0
        while len(edges) > 0:
            # find leaf nodes:
            debug and print(vertices, steiner_pnts, edges)

            vs_to_consider = [vertex for vertex in vertices if vertex not in [e0 for e0, e1 in edges]] + \
                             [vertex for vertex in steiner_pnts if vertex not in [e0 for e0, e1 in edges]]
            debug and print("vs_to_consider:", vs_to_consider)

            yielded = False
            for v in vs_to_consider:
                # Get the edge that is connected to this leaf node
                for edge in [e for e in edges if e[1] == v]:
                    yield edge
                    edges.remove(edge)
                    yielded = True
                    yield_count += 1
                    # yield map(lambda i: self.qubit_map[i], edge)
            if not yielded:
                print("Infinite loop!", warning)
                warning += 1
            if yield_count > n_edges:
                print("Yielded more edges than existing again... This should not be possible!!", warning)
                warning += 1
            if warning > 10:
                print(state, edges, yield_count)
                # input("Note it down!")
                break
        yield None


def dynamic_size_architecture_name(base_name, n_qubits):
    return str(n_qubits) + "q-" + base_name  # "16" + "q-" + "square" ---> '16q-square'


#  生成每个顶点之间连线的列表:[('0', '1'), ('1', '2'), ('2', '3')...]
def connect_vertices_in_line(vertices):
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def connect_vertices_as_grid(width, height, vertices):  # 以width=3, height=3, vertices=9为例
    # 如果输入的长*宽 != 总顶点数, 输出错误提示
    if len(vertices) != width * height:
        raise KeyError("To make a grid, you need vertices exactly equal to width*height, but got %d=%d*%d." % (
            len(vertices), width, height))

    # 将顶点练成一条线 列表中的每个元素都是元组 [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    edges = connect_vertices_in_line(vertices)

    # 生成[水平]行: ['012', '345', '678']
    horizontal_lines = [vertices[i * width: (i + 1) * width] for i in
                        range(height)]  # [range(0, 4), range(4, 8), range(8, 12), range(12, 16)]

    # horizontal_lines = [range(0, 3), range(3, 6), range(6, 9)] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # horizontal_lines[1:] = [range(3, 6), range(6, 9)] = [(3, 4, 5), (6, 7, 8)]
    # line1 = range(0, 3)
    # line2 = range(3, 6)
    for line1, line2 in zip(horizontal_lines, horizontal_lines[1:]):
        # line1[:-1] 取line1的索引0~倒数第二索引位
        # line2[1:0] 取line2的索引1~最后一位
        new_edges = [(v1, v2) for v1, v2 in
                     zip(line1[:-1], reversed(line2[1:]))]  # 第一次 line1[:-1]=[0, 1], reversed(line2[1:] = [5, 4]
        edges.extend(new_edges)  # 第一次 new_edges = [(0, 5), (1, 4)]
    return edges  # [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (0, 5), (1, 4), (3, 8), (4, 7)]


def create_line_architecture(n_qubits, backend=None, **kwargs):
    graph = Graph(backend=backend)
    vertices = graph.add_vertices(n_qubits)
    edges = connect_vertices_in_line(vertices)
    graph.add_edges(edges)
    name = dynamic_size_architecture_name(LINE, n_qubits)
    return Architecture(name=name, coupling_graph=graph, backend=backend, **kwargs)


def create_circle_architecture(n_qubits, backend=None, **kwargs):
    graph = Graph(backend=backend)
    vertices = graph.add_vertices(n_qubits)
    edges = connect_vertices_in_line(vertices)
    edges.append((vertices[-1], vertices[0]))
    graph.add_edges(edges)
    name = dynamic_size_architecture_name(CIRCLE, n_qubits)
    return Architecture(name=name, coupling_graph=graph, backend=backend, **kwargs)


# **kwargs  存放字典的多值参数
def create_square_architecture(n_qubits, backend=None, **kwargs):
    # No floating point errors
    sqrt_qubits = 0
    # 求出正方型的边长
    for n in range(n_qubits):
        if n_qubits == n ** 2:
            sqrt_qubits = n
        if n ** 2 > n_qubits:
            break  # 量子位数不符合正方形架构
    if sqrt_qubits == 0:
        raise KeyError("Sqaure architecture requires a square number of qubits, but got " + str(n_qubits))
    graph = Graph(backend=backend)  # 实例化一个空的图
    vertices = graph.add_vertices(n_qubits)  # 生成顶点集, range(0, 16)
    # 通过connect_vertices_as_grid()方法, 生成边的列表集合, 元素类型是元组
    # edges = [(0, 1), ... , (14, 15), |(0, 7), (1, 6), (2, 5), |(4, 11), (5, 10), (6, 9), |(8, 15), (9, 14), (10, 13)]
    edges = connect_vertices_as_grid(sqrt_qubits, sqrt_qubits, vertices)  # 生成顶点之间边的列表
    graph.add_edges(edges)
    """
    graph = {0: {1: 1, 7: 1}, 
             1: {0: 1, 2: 1, 6: 1}, 
             2: {1: 1, 3: 1, 5: 1}, 
             3: {2: 1, 4: 1}, 
             4: {3: 1, 5: 1, 11: 1}, 
             5: {4: 1, 6: 1, 2: 1, 10: 1}, 
             6: {5: 1, 7: 1, 1: 1, 9: 1}, 
             7: {6: 1, 8: 1, 0: 1}, 
             8: {7: 1, 9: 1, 15: 1}, 
             9: {8: 1, 10: 1, 6: 1, 14: 1}, 
             10: {9: 1, 11: 1, 5: 1, 13: 1}, 
             11: {10: 1, 12: 1, 4: 1}, 
             12: {11: 1, 13: 1}, 
             13: {12: 1, 14: 1, 10: 1}, 
             14: {13: 1, 15: 1, 9: 1}, 
             15: {14: 1, 8: 1}}
    """
    name = dynamic_size_architecture_name(SQUARE, n_qubits)
    return Architecture(name=name, coupling_graph=graph, backend=backend, **kwargs)


"""
def create_9q_square_architecture(**kwargs):
    m = np.array([
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0]
    ])
    return Architecture(name=SQUARE_9Q, coupling_matrix=m, **kwargs)

def create_5q_line_architecture(**kwargs):
    m = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    return Architecture(name=LINE_5Q, coupling_matrix=m, **kwargs)
"""


def create_ibm_qx2_architecture(**kwargs):
    m = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    return Architecture(IBM_QX2, coupling_matrix=m, **kwargs)


def create_ibm_qx4_architecture(**kwargs):
    m = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    return Architecture(IBM_QX4, coupling_matrix=m, **kwargs)


def create_ibm_qx3_architecture(**kwargs):
    m = np.array([
        # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],  # 14
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 15
    ])
    return Architecture(IBM_QX3, coupling_matrix=m, **kwargs)


def create_ibm_qx5_architecture(**kwargs):
    m = np.array([
        # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 3
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 11
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 12
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 13
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 14
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 15
    ])
    return Architecture(IBM_QX5, coupling_matrix=m, **kwargs)


"""
graph = {0: {1: 1, 15: 1}, 
         1: {0: 1, 2: 1, 14: 1}, 
         2: {1: 1, 3: 1, 13: 1}, 
         3: {2: 1, 4: 1, 12: 1}, 
         4: {3: 1, 5: 1, 11: 1}, 
         5: {4: 1, 6: 1, 10: 1}, 
         6: {5: 1, 7: 1, 9: 1}, 
         7: {6: 1, 8: 1}, 
         8: {7: 1, 9: 1}, 
         9: {6: 1, 8: 1, 10: 1}, 
         10: {5: 1, 9: 1, 11: 1}, 
         11: {4: 1, 10: 1, 12: 1}, 
         12: {3: 1, 11: 1, 13: 1}, 
         13: {2: 1, 12: 1, 14: 1}, 
         14: {1: 1, 13: 1, 15: 1}, 
         15: {0: 1, 14: 1}}
"""


def create_ibmq_16_melbourne_architecture(**kwargs):
    m = np.array([
        # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 0
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 9
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 10
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 11
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 12
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 14

    ])
    return Architecture(IBMQ_16_MELBOURNE, coupling_matrix=m, **kwargs)


def create_ibm_q20_tokyo_architecture(backend=None, **kwargs):
    graph = Graph(backend=backend)
    vertices = graph.add_vertices(20)
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
    graph.add_edges(edges)
    return Architecture(name=IBM_Q20_TOKYO, coupling_graph=graph, backend=backend, **kwargs)


"""
graph = {0: {1: 1, 9: 1}, 
         1: {0: 1, 2: 1, 8: 1, 7: 1},
         2: {1: 1, 3: 1, 7: 1, 8: 1}, 
         3: {2: 1, 4: 1, 6: 1, 5: 1}, 
         4: {3: 1, 5: 1, 6: 1}, 
         5: {4: 1, 6: 1, 14: 1, 3: 1}, 
         6: {5: 1, 7: 1, 3: 1, 13: 1, 4: 1, 12: 1}, 
         7: {6: 1, 8: 1, 2: 1, 12: 1, 1: 1, 13: 1}, 
         8: {7: 1, 9: 1, 1: 1, 11: 1, 2: 1, 10: 1}, 
         9: {8: 1, 10: 1, 0: 1, 11: 1}, 
         10: {9: 1, 11: 1, 19: 1, 8: 1}, 
         11: {10: 1, 12: 1, 8: 1, 18: 1, 9: 1, 17: 1}, 
         12: {11: 1, 13: 1, 7: 1, 17: 1, 6: 1, 18: 1}, 
         13: {12: 1, 14: 1, 6: 1, 16: 1, 7: 1, 15: 1}, 
         14: {13: 1, 15: 1, 5: 1, 16: 1}, 
         15: {14: 1, 16: 1, 13: 1}, 
         16: {15: 1, 17: 1, 13: 1, 14: 1}, 
         17: {16: 1, 18: 1, 12: 1, 11: 1}, 
         18: {17: 1, 19: 1, 11: 1, 12: 1}, 
         19: {18: 1, 10: 1}}
"""


def create_rigetti_16q_aspen_architecture(backend=None, **kwargs):
    graph = Graph(backend=backend)
    vertices = graph.add_vertices(16)
    edges = connect_vertices_in_line(vertices)
    extra_edges = [(0, 7), (8, 15), (15, 0)]
    edges += [(vertices[v1], vertices[v2]) for v1, v2 in extra_edges]
    graph.add_edges(edges)
    return Architecture(RIGETTI_16Q_ASPEN, coupling_graph=graph, backend=backend, **kwargs)


"""
graph ={0: {1: 1, 7: 1, 15: 1}, 
        1: {0: 1, 2: 1}, 
        2: {1: 1, 3: 1}, 
        3: {2: 1, 4: 1}, 
        4: {3: 1, 5: 1}, 
        5: {4: 1, 6: 1}, 
        6: {5: 1, 7: 1}, 
        7: {6: 1, 8: 1, 0: 1}, 
        8: {7: 1, 9: 1, 15: 1}, 
        9: {8: 1, 10: 1}, 
        10: {9: 1, 11: 1}, 
        11: {10: 1, 12: 1}, 
        12: {11: 1, 13: 1}, 
        13: {12: 1, 14: 1}, 
        14: {13: 1, 15: 1}, 
        15: {14: 1, 8: 1, 0: 1}}
"""


def create_rigetti_8q_agave_architecture(**kwargs):
    m = np.array([
        [0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ])
    return Architecture(RIGETTI_8Q_AGAVE, coupling_matrix=m, **kwargs)


def create_fully_connected_architecture(n_qubits=None, **kwargs):
    if n_qubits is None:
        print("Warning: size is not given for the fully connected architecuture, using 9 as default.")
        n_qubits = 9
    m = np.ones(shape=(n_qubits, n_qubits))
    for i in range(n_qubits):
        m[i][i] = 0
    name = dynamic_size_architecture_name(FULLY_CONNNECTED, n_qubits)
    return Architecture(name, coupling_matrix=m, **kwargs)


def create_architecture(name, **kwargs):
    # Source Rigetti architectures: https://www.rigetti.com/qpu # TODO create the architectures from names in pyquil.list_quantum_computers() <- needs mapping
    # Source IBM architectures: http://iic.jku.at/files/eda/2018_tcad_mapping_quantum_circuit_to_ibm_qx.pdf​
    # IBM architectures are currently ignoring CNOT direction.
    if isinstance(name, Architecture):
        return name
    if name == SQUARE:
        return create_square_architecture(**kwargs)
    elif name == LINE:
        return create_line_architecture(**kwargs)
    elif name == FULLY_CONNNECTED:
        return create_fully_connected_architecture(**kwargs)
    elif name == CIRCLE:
        return create_circle_architecture(**kwargs)
    elif name == IBM_QX2:
        return create_ibm_qx2_architecture(**kwargs)
    elif name == IBM_QX3:
        return create_ibm_qx3_architecture(**kwargs)
    elif name == IBM_QX4:
        # print("architecture:",name) #ibm_qx4，两者能匹配出相等
        return create_ibm_qx4_architecture(**kwargs)
    elif name == IBM_QX5:
        return create_ibm_qx5_architecture(**kwargs)
    elif name == IBMQ_16_MELBOURNE:
        return create_ibmq_16_melbourne_architecture(**kwargs)
    elif name == IBM_Q20_TOKYO:
        return create_ibm_q20_tokyo_architecture(**kwargs)
    elif name == RIGETTI_16Q_ASPEN:
        return create_rigetti_16q_aspen_architecture(**kwargs)
    elif name == RIGETTI_8Q_AGAVE:
        return create_rigetti_8q_agave_architecture(**kwargs)
    else:
        raise KeyError("name " + str(name) + "not recognized as architecture name. Please use one of", *architectures)


def colored_print_9X9(np_array):
    """
    Prints a 9x9 numpy array with colors representing their distance in a 9x9 square architecture
    :param np_array:  the array
    """
    if np_array.shape == (9, 9):
        CRED = '\033[91m '
        CEND = '\033[0m '
        CGREEN = '\33[32m '
        CYELLOW = '\33[33m '
        CBLUE = '\33[34m '
        CWHITE = '\33[37m '
        CVIOLET = '\33[35m '
        color = [CBLUE, CGREEN, CVIOLET, CYELLOW, CRED]
        layout = [[0, 1, 2, 3, 2, 1, 2, 3, 4],
                  [1, 0, 1, 2, 1, 2, 3, 2, 3],
                  [2, 1, 0, 1, 2, 3, 4, 3, 2],
                  [3, 2, 1, 0, 1, 2, 3, 2, 1],
                  [2, 1, 2, 1, 0, 1, 2, 1, 2],
                  [1, 2, 3, 2, 1, 0, 1, 2, 3],
                  [2, 3, 4, 3, 2, 1, 0, 1, 2],
                  [3, 2, 3, 2, 1, 2, 1, 0, 1],
                  [4, 3, 2, 1, 2, 3, 2, 1, 0]]
        for i, l in enumerate(layout):
            print('[', ', '.join([(color[c] + '1' if v == 1 else CWHITE + '0') for c, v in zip(l, np_array[i])]), CEND,
                  ']')
    else:
        print(np_array)


if __name__ == '__main__':
    sys.path.append('..')
    n_qubits = 25
    for name in dynamic_size_architectures:
        arch = create_architecture(name, n_qubits=n_qubits)
        arch.visualize()

    arch = create_architecture(IBM_Q20_TOKYO)
    arch.visualize()
