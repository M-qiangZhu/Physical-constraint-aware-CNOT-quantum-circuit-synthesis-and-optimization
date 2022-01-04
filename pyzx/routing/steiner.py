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

from . import architecture, cnot_err_simulator
from ..linalg import Mat2

# debug = True  #True  #修改成了true，以便看到调试数据，原来是False
debug = False

# ibmq_5_yorktown 架构 20210508 distances
distances = {(0, 1): 0.018282915538629818,
             (1, 0): 0.018282915538629818,
             (1, 2): 0.02382597319762883,
             (2, 1): 0.02382597319762883,
             (2, 3): 0.021199360577361798,
             (3, 2): 0.021199360577361798,
             (3, 4): 0.016671016328373228,
             (4, 3): 0.016671016328373228,
             (0, 2): 0.025117409134446322,
             (2, 0): 0.025117409134446322,
             (2, 4): 0.013859579352843793,
             (4, 2): 0.013859579352843793, }


# ibmq_qx20_tokyo 架构 模拟 CNOT-error
# distances = cnot_err_simulator.generate_cnot_err("ibm_q20_tokyo")
# print(distances)


def WPL(edges_list, distances):
    wpl = 1
    for edge in edges_list:
        x = 1 - distances[edge]
        wpl *= x
    return wpl


def steiner_gauss(matrix, architecture, full_reduce=False, x=None, y=None, permutation=None):
    # 基于steiner方式的高斯消元实现
    """
    Performs Gaussian elimination that is constraint bij the given architecture
    :param matrix: PyZX Mat2 matrix to be reduced
    :param architecture: The Architecture object to conform to
    :param full_reduce: Whether to fully reduce or only create an upper triangular form  是否完全缩小或仅创建上三角形状
    :param x: 
    :param y: 
    :return: Rank of the given matrix  给定矩阵的 秩
    """
    debug and print("steiner_guass is called.")
    debug and print(matrix)
    debug and print("architecture:", architecture.name)
    debug and print("full_reduce:", full_reduce)
    debug and print("x:", x)
    debug and print("y:", y.matrix)  # 单位阵
    debug and print("permutation:", permutation)

    if permutation is None:
        permutation = [i for i in range(len(matrix.data))]
        debug and print("permutation:", permutation)
    else:
        matrix = Mat2([matrix.data[i] for i in permutation])

    # print(matrix)
    # print("*" * 50)

    def row_add(c0, c1):
        debug and print("-" * 60)
        matrix.row_add(c0, c1)  # 将矩阵中c0行加到c1行上，让c1行所处理的列变为1

        c0 = permutation[c0]
        debug and print("起始行c0：", c0)

        c1 = permutation[c1]
        debug and print("目标行c1：", c1)

        debug and print("Reducing", c0, c1)

        debug and print("x[运算前]：", x)
        if x != None: x.row_add(c0, c1)
        debug and print("x[运算后]：", x)

        debug and print("y[运算前]：", y.matrix)
        if y != None: y.col_add(c1, c0)
        debug and print("y[运算后]：\n", y.matrix)

    def steiner_reduce(col, root, nodes, upper):

        # 计数
        cnts = 0

        # 记录消除的边
        edges_list = []

        steiner_tree = architecture.steiner_tree(root, nodes, upper)
        # steiner_tree = architecture.my_steiner_tree(root, nodes, upper)
        # print(f"steiner_tree :")
        # for i in range(0, len(steiner_tree) - 1):
        #     print(i)

        # Remove all zeros 移去所有的0,将steiner点置为1
        next_check = next(steiner_tree)  # 类似return edge
        debug and print("next_check:", next_check)

        debug and print("<-----------------------Step 1: remove zeros-------------------------->")
        if upper:
            zeros = []
            while next_check is not None:
                s0, s1 = next_check
                if matrix.data[s0][col] == 0:  # s1 is a new steiner point or root = 0
                    zeros.append(next_check)  # 获取叶子结点的边
                next_check = next(steiner_tree)
                debug and print("next_check again:", next_check)
            while len(zeros) > 0:
                s0, s1 = zeros.pop(-1)
                if matrix.data[s0][col] == 0:
                    # 判断函数, 不一定是s0, s1, 也有可能执行另一行和s1, 需要判断对当前行产生的1的代价

                    row_add(s1, s0)
                    edges_list.append((s1, s0))
                    cnts += 1
                    debug and print("传递1后的当前矩阵为:")
                    debug and print(matrix.data)
                    debug and print(f"a[{s0}][{col}]={matrix.data[s0][col]},\na[{s1}][{col}]={matrix.data[s1][col]}")

        else:
            debug and print("deal with zero root")
            if next_check is not None and matrix.data[next_check[0]][col] == 0:  # root is zero
                print("WARNING : Root is 0 => reducing non-pivot column", matrix.data)
            debug and print("Step 1: remove zeros", [r[c] for r in matrix.data])
            while next_check is not None:
                s0, s1 = next_check
                if matrix.data[s1][col] == 0:  # s1 is a new steiner point
                    row_add(s0, s1)
                    edges_list.append((s0, s1))
                    cnts += 1
                    debug and print("传递1后的当前矩阵为:")
                    debug and print(f"a[{s0}][{col}]={matrix.data[s0][col]},\na[{s1}][{col}]={matrix.data[s1][col]}")
                    # debug and print(matrix.data)
                next_check = next(steiner_tree)

        # Reduce stuff
        debug and print("<<----------------------Step 2: remove ones-------------------------->>")
        next_add = next(steiner_tree)
        while next_add is not None:
            s0, s1 = next_add
            row_add(s0, s1)
            edges_list.append((s0, s1))
            cnts += 1
            debug and print("消去1后的当前矩阵为:")
            debug and print(matrix.data)
            next_add = next(steiner_tree)
            # next() 返回迭代器的下一个项目。
            debug and print(next_add)
        debug and print("<<<--------------------Step 3: -------------------------->>>")
        # print(f"edges_list: {edges_list}")
        return cnts, edges_list

    rows = matrix.rows()
    debug and print("rows:", rows)
    cols = matrix.cols()
    debug and print("cols:", cols)
    p_cols = []
    pivot = 0

    # 记录最终线路中cnot门数量
    cnot_cnts = 0
    edges_list = []

    for c in range(cols):
        debug and print(f"下三角第{c}列:")
        if pivot < rows:
            nodes = [r for r in range(pivot, rows) if pivot == r or matrix.data[r][c] == 1]  #
            debug and print("nodes:", nodes)
            cnot_cnts, edges = steiner_reduce(c, pivot, nodes, True)  # 递归调用
            edges_list = edges_list + edges
            if matrix.data[pivot][c] == 1:
                p_cols.append(c)
                pivot += 1

    debug and print("Upper triangle form: ")
    debug and print(matrix.data)

    rank = pivot
    debug and print(f"p_cols: {p_cols}")

    if full_reduce:
        pivot -= 1
        for c in reversed(p_cols):
            debug and print(f"上三角第{c}列:")
            debug and print(pivot, [r[c] for r in matrix.data])

            nodes = [r for r in range(0, pivot + 1) if r == pivot or matrix.data[r][c] == 1]
            debug and print("nodes:", nodes)
            if len(nodes) > 1:
                cnot_cnts, edges = steiner_reduce(c, pivot, nodes, False)
                edges_list = edges_list + edges

            pivot -= 1

    # print(f"rank : {rank}")
    # print(f"cnot_cnts : {cnot_cnts}")
    # 打印最终的路径
    # print(f"edges_list = {edges_list}")

    # 计算带权路径总长
    wpl = WPL(edges_list, distances)
    # print(f"wpl = {wpl}")

    return rank, wpl
    # return rank
