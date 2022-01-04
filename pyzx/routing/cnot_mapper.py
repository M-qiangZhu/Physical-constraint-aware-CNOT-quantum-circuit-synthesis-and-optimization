import csv
import sys, os
from multiprocessing import Process

import pandas as pd
import threading

if __name__ == '__main__':
    sys.path.append('..')
import numpy as np

try:
    from pandas import DataFrame
except:
    DataFrame = None
    if __name__ == '__main__':
        print("Warning: could not import pandas. No performance data will be exported.")
import time

from ..linalg import Mat2
from .architecture import create_fully_connected_architecture, create_architecture, dynamic_size_architectures
from ..parity_maps import CNOT_tracker
from ..machine_learning import GeneticAlgorithm
from ..utils import make_into_list
from .steiner import steiner_gauss

debug = True  # 设置成True,原来是False
# debug = False

# ELIMINATION MODES:
GAUSS_MODE = "gauss"
STEINER_MODE = "steiner"
GENETIC_STEINER_MODE = "genetic_steiner"
GENETIC_GAUSS_MODE = "genetic_gauss"

elim_modes = [STEINER_MODE, GAUSS_MODE, GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE]
genetic_elim_modes = [GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE]
no_genetic_elim_modes = [STEINER_MODE, GAUSS_MODE]

# COMPILE MODES
QUIL_COMPILER = "quilc"
NO_COMPILER = "not_compiled"

compiler_modes = [QUIL_COMPILER, NO_COMPILER]


def cnot_fitness_func(mode, matrix, architecture, row=True, col=True, full_reduce=True, **kwargs):
    # fitness_func =  cnot_fitness_func  函数入口，计算适应度值

    """
    Creates and returns a fitness function to be used for the genetic algorithm that uses CNOT gate count as fitness.

    :param mode: The type of Gaussian elimination to be used
    :param matrix: A Mat2 parity map to route.
    :param architecture: The architecture to take into account when routing
    :param row: Whether to find a row permutation
    :param col: Whether to find a column permutation
    :param full_reduce: Whether to fully reduce the matrix, thus rebuild the full circuit.
    :return: A fitness function that calculates the number of gates needed for a given permutation.
    """
    n_qubits = len(matrix.data)

    def fitness_func(permutation):
        row_perm = permutation if row else np.arange(len(matrix.data))
        col_perm = permutation if col else np.arange(len(matrix.data[0]))
        circuit = CNOT_tracker(n_qubits)
        mat = Mat2([[matrix.data[r][c] for c in col_perm] for r in row_perm])
        # print("mode:",mode) #mode: steiner
        # type_flag = "消除类型"
        gauss(mode, mat, architecture=architecture, y=circuit, full_reduce=full_reduce, **kwargs)  # y是一个单位矩阵,用来跟踪消除的步骤
        # 多次调用STEINER_MODE的原因

        return circuit.count_cnots()

    return fitness_func


def gauss(mode, matrix, architecture=None, permutation=None, **kwargs):  # 可变参数，带遗传的，不带遗传的
    """
    Performs gaussian elimination of type mode on Mat2 matrix on the given architecture, if needed.
    :param mode: Type of Gaussian elimination to be used
    :param matrix: Mat2 matrix to run the algorithm on
    :param architecture: Device architecture to take into account [optional]
    :param kwargs: Other arguments that can be given to the Mat2.gauss() function or parameters for the genetic algorithm.
    :return: The rank of the matrix. Mat2 matrix is transformed.
    """
    if mode == GAUSS_MODE:
        # TODO - adjust to get the right gate locations for the given permutation.        
        if permutation is not None:
            # print("\033[91m Warning: Permutation parameter with Gauss-Jordan elimination is not yet supported, it can be optimized with permutated_gauss(). \033[0m ")
            # return matrix.gauss(**kwargs)
            # Broken code that tries to implement this.
            matrix = Mat2([matrix.data[i] for i in permutation])
            old_x, old_y = None, None
            if "x" in kwargs:
                old_x = kwargs["x"]
            if "y" in kwargs:
                old_y = kwargs["y"]
            n_qubits = len(matrix.data)
            x = CNOT_tracker(n_qubits)
            kwargs["x"] = x
            kwargs["y"] = None
            rank = matrix.gauss(**kwargs)
            for gate in x.gates:
                c = permutation[gate.control]
                t = permutation[gate.target]
                if old_x != None: old_x.row_add(c, t)
                if old_y != None: old_y.col_add(t, c)
            return rank
        else:
            return matrix.gauss(**kwargs)

    elif mode == STEINER_MODE:
        # print("这是一次steiner消除")
        if architecture is None:
            print(
                "\033[91m Warning: Architecture is not given, assuming fully connected architecture of size matrix.shape[0]. \033[0m ")
            architecture = create_fully_connected_architecture(len(matrix.data))

        # print("Calling STEINER_MODE.....")  #被执行多次
        # print("matrix:",matrix)
        # print("architecture:",architecture.name)
        # print("permutation:",permutation)

        return steiner_gauss(matrix, architecture, permutation=permutation, **kwargs)

    elif mode == GENETIC_STEINER_MODE:
        print("Calling GENETIC_STEINER_MODE .....")
        perm, cnots, rank, wpl= permutated_gauss(matrix, STEINER_MODE, architecture=architecture, permutation=permutation,
                                             **kwargs)
        now_thread = threading.current_thread()

        print(f"---------{now_thread.name}-----------")
        print("result perm:", perm)
        print(f"result cnots: {cnots}")
        print(f"result rank: {rank}")
        print(f"result wpl: {wpl}")

        # print(f"type_flag: {type_flag}, result cnots: {cnots}")
        # print(f"type_flag: {type_flag}, result rank: {rank}")

        """
        记录最后的cnot门数
        """
        result_conts = str(cnots)
        result_wpl = str(wpl)
        rows = [result_conts, now_thread.name]
        now_process = str(os.getpid())


        # 多进程门数记录
        with open('../scripts/tokyo_parallel_result_conts.csv', 'a+', encoding='utf-8') as f:
            # f_csv = csv.writer(f)
            # f_csv.writerows(rows)
            f.write(result_conts)
            f.write(",")
            f.write(result_wpl)
            f.write(",")
            # f.write(now_thread.name)
            f.write(now_process)
            f.write('\n')


        #
        # # 串行门数记录
        # with open('../scripts/serial_result_conts.csv', 'a+',
        #           encoding='utf-8') as f:
        #     # f_csv = csv.writer(f)
        #     # f_csv.writerows(rows)
        #     f.write(result_conts)
        #     f.write(",")
        #     f.write(now_thread.name)
        #     f.write('\n')


    elif mode == GENETIC_GAUSS_MODE:
        perm, cnots, rank, wpl = permutated_gauss(matrix, GAUSS_MODE, architecture=architecture, permutation=permutation,
                                             **kwargs)


def permutated_gauss(matrix, mode=None, architecture=None, population_size=100, crossover_prob=0.8, mutate_prob=0.2,
                     n_iterations=100,  # 原来是: population_size=30, n_iterations=50
                     row=True, col=True, full_reduce=True, fitness_func=None, x=None, y=None, **kwargs):
    """
    Finds an optimal permutation of the matrix to reduce the number of CNOT gates.
    #找到矩阵的最优排序一次减少CNOT门数
    :param matrix: Mat2 matrix to do gaussian elimination over
    :param population_size: For the genetic algorithm
    :param crossover_prob: For the genetic algorithm
    :param mutate_prob: For the genetic algorithm
    :param n_iterations: For the genetic algorithm
    :param row: If the rows should be permutated
    :param col: If the columns should be permutated
    :param full_reduce: Whether to do full gaussian reduction
    :return: Best permutation found, list of CNOTS corresponding to the elimination.
    """

    if fitness_func is None:
        fitness_func = cnot_fitness_func(mode, matrix, architecture, row=row, col=col, full_reduce=full_reduce,
                                         **kwargs)
        # 仅返回一个函数入口

    optimizer = GeneticAlgorithm(population_size, crossover_prob, mutate_prob, fitness_func)
    # 创建优化器

    permsize = len(matrix.data) if row else len(matrix.data[0])  # 5
    debug and print("permsize:", permsize)

    # 循环很多次？调用了STEINER_MODE
    # print("ok1!")
    best_permutation = optimizer.find_optimimum(permsize, n_iterations, continued=True)
    # graph_now = architecture.graph.graph
    # best_permutation = get_heuristic_initial_permutation(matrix, graph_now)
    # 通过遗传算法找到的最好排序
    # print("best_permutation:", best_permutation)

    # print("ok2!")  #此处往后只输出一次
    debug and print("best_permutation:", best_permutation)

    n_qubits = len(matrix.data)  # matrix根据best_permutation发生了改变
    row_perm = best_permutation if row else np.arange(len(matrix.data))
    debug and print("row_permn:", row_perm)

    col_perm = best_permutation if col else np.arange(len(matrix.data[0]))
    debug and print("col_perm:", col_perm)

    if y is None:  # y是目标线路，对应单位矩阵
        circuit = CNOT_tracker(n_qubits)
    else:
        circuit = y

    debug and print("matrix:", matrix.data)
    mat = Mat2([[matrix.data[r][c] for c in col_perm] for r in row_perm])
    # 将matrix都按新的序列重新排序

    circuit.row_perm = row_perm
    debug and print("circuit.row_perm:", circuit.row_perm)

    circuit.col_perm = col_perm  # circuit根据best_permutaiton发生了改变
    debug and print("circuit.col_perm:", circuit.col_perm)  # 序列都相同，但每次都不一样

    # 此时得到了最优的排序后的矩阵，初始布局结束
    # debug and print("mat:", mat)
    # debug and print("x:", x)
    # debug and print("y:", y.matrix)
    print("mat:")
    print(mat)
    # print("x:", x)
    # print("y:", y.matrix)

    rank, wpl = gauss(mode, mat, architecture, x=x, y=circuit, full_reduce=full_reduce, **kwargs)
    # rank= gauss(mode, mat, architecture, x=x, y=circuit, full_reduce=full_reduce, **kwargs)
    # 执行一次，STEINER_MODE

    return best_permutation, circuit.count_cnots(), rank, wpl
    # return best_permutation, circuit.count_cnots(), rank


def count_cnots_mat2(mode, matrix, compile_mode=None, architecture=None, n_compile=1, store_circuit_as=None, **kwargs):
    if compile_mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        circuit = PyQuilCircuit(architecture)
    else:
        circuit = CNOT_tracker(matrix.data.shape[0])
    mat = Mat2(np.copy(matrix.data))
    gauss(mode, mat, architecture=architecture, y=circuit, **kwargs)
    return count_cnots_circuit(compile_mode, circuit, n_compile, store_circuit_as)


def count_cnots_circuit(mode, circuit, n_compile=1, store_circuit_as=None):
    count = -1
    if mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        if isinstance(circuit, PyQuilCircuit):
            count = sum([circuit.compiled_cnot_count() for i in range(n_compile)]) / n_compile
    elif mode == NO_COMPILER:
        count = circuit.count_cnots()
    if store_circuit_as is not None:
        with open(store_circuit_as, 'w') as f:
            f.write(circuit.to_qasm())
    return count


def create_dest_filename(original_file, population=None, iteration=None, crossover_prob=None, mutation_prob=None,
                         index=None):
    pop_ext = "" if population is None else "pop" + str(population)
    iter_ext = "" if iteration is None else "iter" + str(iteration)
    crosover_ext = "" if crossover_prob is None else "crossover" + str(crossover_prob)
    mutation_ext = "" if mutation_prob is None else "mutate" + str(mutation_prob)
    index_ext = "" if index is None else "(" + str(index) + ")"
    filename = os.path.basename(original_file)
    base_file, extension = os.path.splitext(filename)
    new_filename = '_'.join([part for part in [base_file, pop_ext, iter_ext, crosover_ext, mutation_ext, index_ext] if
                             part != ""]) + extension
    return new_filename


def get_metric_header():
    metrics = CNOT_tracker.get_metric_names()
    return ["id", "architecture", "mode", "index", "population", "n_iterations", "crossover", "mutation"] + metrics + [
        "time", "destination_file"]


def make_metrics(circuit, id, architecture_name, mode, dest_file=None, population=None, iteration=None,
                 crossover_prob=None, mutation_prob=None, passed_time=None, index=None):
    result = circuit.gather_metrics()
    result["id"] = id
    result["mode"] = mode
    result["architecture"] = architecture_name
    result["population"] = population
    result["n_iterations"] = iteration
    result["crossover"] = crossover_prob
    result["mutation"] = mutation_prob
    result["time"] = passed_time
    result["index"] = index
    result["destination_file"] = dest_file
    return result


# 批量处理cnot线路映射
def batch_map_cnot_circuits(source, modes, architectures, n_qubits=None, populations=30, iterations=15,
                            crossover_probs=0.8,
                            mutation_probs=0.5, dest_folder=None, metrics_file=None, n_compile=1):
    modes = make_into_list(modes)  # 将modes转换为列表
    debug and print("modes:", modes)  # ['genetic_steiner']  # 将debug设为True, 进入调试输出

    architectures = make_into_list(architectures)
    debug and print("architectures:", architectures)  # ['ibm_qx4']

    populations = make_into_list(populations)
    debug and print("populations:", populations)

    iterations = make_into_list(iterations)
    debug and print("iterations:", iterations)

    crossover_probs = make_into_list(crossover_probs)
    debug and print("crossover_probs:", crossover_probs)

    mutation_probs = make_into_list(mutation_probs)
    debug and print("mutation_probs:", mutation_probs)

    # 处理源文件
    if os.path.isfile(source):  # 判断某一对象(需提供绝对路径)是否为文件
        source, file = os.path.split(source)  # 把路径分割成 路径名 和 文件名，返回一个元组
        files = [file]
    else:
        # print("os.listdir:",os.listdir(source)) 同下面
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    # os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表
    debug and print("处理的源文件：", files)

    if not os.path.exists(source):
        raise IOError("Folder does not exist: " + source)  # 文件夹不存在
    if dest_folder is None:
        dest_folder = source  # 目标文件夹
    else:
        os.makedirs(dest_folder, exist_ok=True)  # 递归创建目录
    # 可以一次创建多级目录，哪怕中间目录不存在也能正常的（替你）创建
    # 如果exists_ok为False(默认值)，则如果目标目录已存在，则引发OSError错误。

    arch_iter = []
    circuits = {}
    metrics = []

    # 生成对应的架构的网格
    for architecture in architectures:
        if architecture in dynamic_size_architectures:
            # dynamic_size_architectures = [FULLY_CONNNECTED, LINE, CIRCLE, SQUARE]
            if n_qubits is None:
                # n_qubits为空时报错
                raise KeyError("Number of qubits not specified for architecture" + architecture)
            else:
                n_qubits = make_into_list(n_qubits)
                arch_iter.extend([create_architecture(architecture, n_qubits=q) for q in n_qubits])
                # extend()用于在列表末尾一次性追加另一个序列中的多个值
        else:
            arch_iter.append(create_architecture(architecture))

    # 遍历架构
    for architecture in arch_iter:
        circuits[architecture.name] = {}
        for mode in modes:
            if mode == QUIL_COMPILER:
                n_compile_list = range(n_compile)
            else:
                n_compile_list = [None]

            # 输出结果文件夹的设置
            new_dest_folder = os.path.join(dest_folder, architecture.name, mode)  # (resultdata, '9q-square', 'steiner')
            # 创建目录
            os.makedirs(new_dest_folder, exist_ok=True)

            if mode in genetic_elim_modes:
                # genetic_elim_modes = [GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE] = [genetic_steiner, genetic_gauss]
                # elim_modes = [STEINER_MODE, GAUSS_MODE, GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE]
                pop_iter = populations
                iter_iter = iterations
                crossover_iter = crossover_probs
                mutation_iter = mutation_probs
                circuits[architecture.name][mode] = {}
            else:
                if mode == QUIL_COMPILER:  # 'quilc'
                    circuits[architecture.name][mode] = []
                pop_iter = [None]
                iter_iter = [None]
                crossover_iter = [None]
                mutation_iter = [None]

            for population in pop_iter:
                for iteration in iter_iter:
                    for crossover_prob in crossover_iter:
                        for mutation_prob in mutation_iter:
                            for file in files:
                                if os.path.splitext(file)[
                                    1].lower() == ".qasm":  # 分割路径中的文件名与拓展名；默认返回(fname,fextension)元组，可做分片操作

                                    origin_file = os.path.join(source, file)  # 把目录和文件名合成一个路径
                                    for i in n_compile_list:
                                        # 生成目标文件名
                                        dest_filename = create_dest_filename(origin_file, population, iteration,
                                                                             crossover_prob, mutation_prob, i)
                                        # 合成目标文件路径
                                        dest_file = os.path.join(dest_folder, architecture.name, mode, dest_filename)
                                        try:
                                            start_time = time.time()  # 开始计时，单个线路的映射
                                            print("Calling map_cnot_circuit....")
                                            circuit = map_cnot_circuit(origin_file, architecture, mode=mode,
                                                                       dest_file=dest_file,
                                                                       population=population, iterations=iteration,
                                                                       crossover_prob=crossover_prob,
                                                                       mutation_prob=mutation_prob)
                                            # print(circuit.matrix)
                                            # print(type(circuit))
                                            end_time = time.time()  # 结束计时

                                            if metrics_file is not None:
                                                metrics.append(
                                                    make_metrics(circuit, origin_file, architecture.name, mode,
                                                                 dest_file, population, iteration, crossover_prob,
                                                                 mutation_prob, end_time - start_time, i))

                                            if mode in genetic_elim_modes:
                                                circuits[architecture.name][mode][
                                                    (population, iteration, crossover_prob, mutation_prob)] = circuit
                                            elif mode == QUIL_COMPILER:
                                                circuits[architecture.name][mode].append(circuit)
                                            else:
                                                circuits[architecture.name][mode] = circuit
                                        except KeyError as e:  # Should only happen with quilc
                                            if mode == QUIL_COMPILER:
                                                print("\033[31mCould not compile", origin_file, "into", dest_file,
                                                      end="\033[0m\n")
                                            else:
                                                raise e

    if len(metrics) > 0 and DataFrame is not None:
        df = DataFrame(metrics)
        if os.path.exists(metrics_file):  # append to the file - do not overwrite!
            df.to_csv(metrics_file, columns=get_metric_header(), header=False, index=False, mode='a')
        else:
            df.to_csv(metrics_file, columns=get_metric_header(), index=False)
    return circuits


def get_heuristic_initial_permutation(old_matrix, graph_now):
    # 拷贝一个新的Mat2矩阵
    new_matrix = old_matrix.copy()
    # 获得矩阵数据
    matrix_data = old_matrix.data
    # 记录每行1的权重和的列表 第一列权重为8, 二列为7, 以此类推
    sum_list = []
    # 统计下三角每行权重之和
    for i in range(0, len(matrix_data)):
        row_sum = 0
        for j in range(0, i + 1):
            # print(f"matrix_data[{i}][{j}] : {matrix_data[i][j]}")
            row_sum += matrix_data[i][j] * (len(matrix_data) - j)
        # 每统计完一行, 添加进sum_list
        sum_list.append(row_sum)
        print(f"row_sum : {row_sum}")
        print(f"sum_list : {sum_list}")
        print("-" * 50)

    # 初始化权值索引排序列表
    index_order = []
    # 一次找到最大值的索引， 传入index_order中
    for i in range(0, len(sum_list)):
        # 获得最大值的索引
        max_index = sum_list.index(max(sum_list))
        # 将最大值的索引存入index_order中
        index_order.append(max_index)
        # 将当前最大值赋值为-1, 保证找最大值时不会找到当前索引
        sum_list[max_index] = -1
    print(index_order)

    # 获取graph中每个key的value长度
    valid_vertex_list = []
    for key in graph_now:
        print(f"{key} : {len(graph_now[key])}")
        # 将得到的长度赋值给valid_vertex_list
        valid_vertex_list.append(len(graph_now[key]))
    print(valid_vertex_list)

    # 初始化初始映射列表
    initial_mapping = [0] * len(graph_now)
    print(initial_mapping)

    # 进行初始映射
    # 进入循环
    for i in range(0, len(initial_mapping)):
        # 从valid_vertex_list找到最大值的索引，将当前位置的值置为0
        get_index = valid_vertex_list.index(max(valid_vertex_list))
        valid_vertex_list[get_index] = 0
        print(f"valid_vertex_list : {valid_vertex_list}")
        # 取出 index_order 对应i的索引位置的值, 赋值到initial_mapping中对应索引get_index位置
        initial_mapping[get_index] = index_order[i]
        print(f"initial_mapping : {initial_mapping}")
        print(f"order_list : {index_order}")
        print("*" * 50)

    return initial_mapping


class GuessThread(threading.Thread):
    """
    多线程执行消除
    """

    def __init__(self, type_flag, matrix, circuit, mode, architecture, population, crossover_prob, mutation_prob,
                 iterations, dest_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_flag = type_flag
        self.matrix = matrix
        self.circuit = circuit
        self.mode = mode
        self.architecture = architecture
        self.population = population
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.iterations = iterations
        self.dest_file = dest_file

    def run(self):
        start_time = time.time()  # 开始计时
        compiled_circuit = CNOT_tracker(self.circuit.n_qubits)  # 目标线路即为单位矩阵

        if self.mode in no_genetic_elim_modes:
            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit)
        elif self.mode in genetic_elim_modes:
            # debug and print("original matrix:",matrix) #原始线路所对应的矩阵
            # debug and print("original y:",compiled_circuit.matrix) #单位阵

            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit,
                  population_size=self.population, crossover_prob=self.crossover_prob, mutate_prob=self.mutation_prob,
                  n_iterations=self.iterations)
        elif self.mode == QUIL_COMPILER:
            from pyzx.pyquil_circuit import PyQuilCircuit
            compiled_circuit = PyQuilCircuit.from_CNOT_tracker(self.circuit, self.architecture)
            compiled_circuit.compile()

        if self.dest_file is not None:
            compiled_qasm = compiled_circuit.to_qasm()  # 映射后的线路变成QASM格式
            with open(self.dest_file, "w") as f:
                f.write(compiled_qasm)  # 讲QASM线路写入文件
        end_time = time.time()  # 开始结束
        t = end_time - start_time
        s = str(t)
        now_thread = threading.current_thread()
        # rows = [s, now_thread.name]
        print(f"type: {self.type_flag}, time: {end_time - start_time}, now_thread_name : {now_thread.name}")

        # 写入每个线程耗时
        with open('run_times.csv', 'a+', encoding='utf-8') as f:
            f.write(s)
            f.write(",")
            f.write(now_thread.name)
            f.write(",")
            f.write(self.type_flag)
            f.write('\n')
        # thread_time = [s]
        # now_thread = [threading.current_thread().name]
        #
        # # 字典中的key值即为csv中列名
        # data = pd.DataFrame({'time': thread_time, 'now_thread': now_thread})
        #
        # # 将DataFrame存储为csv,index表示是否显示行名，default=True
        # data.to_csv("run_times.csv", index=False, sep=',')

        return compiled_circuit


class GuessProcess(Process):
    """
        多进程执行消除
    """

    def __init__(self, type_flag, matrix, circuit, mode, architecture, population, crossover_prob, mutation_prob,
                 iterations, dest_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_flag = type_flag
        self.matrix = matrix
        self.circuit = circuit
        self.mode = mode
        self.architecture = architecture
        self.population = population
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.iterations = iterations
        self.dest_file = dest_file

    def run(self):
        start_time = time.time()  # 开始计时
        compiled_circuit = CNOT_tracker(self.circuit.n_qubits)  # 目标线路即为单位矩阵

        if self.mode in no_genetic_elim_modes:
            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit)
        elif self.mode in genetic_elim_modes:
            # debug and print("original matrix:",matrix) #原始线路所对应的矩阵
            # debug and print("original y:",compiled_circuit.matrix) #单位阵

            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit,
                  population_size=self.population, crossover_prob=self.crossover_prob, mutate_prob=self.mutation_prob,
                  n_iterations=self.iterations)
        elif self.mode == QUIL_COMPILER:
            from pyzx.pyquil_circuit import PyQuilCircuit
            compiled_circuit = PyQuilCircuit.from_CNOT_tracker(self.circuit, self.architecture)
            compiled_circuit.compile()

        if self.dest_file is not None:
            compiled_qasm = compiled_circuit.to_qasm()  # 映射后的线路变成QASM格式
            with open(self.dest_file, "w") as f:
                f.write(compiled_qasm)  # 讲QASM线路写入文件
        end_time = time.time()  # 开始结束
        t = end_time - start_time
        s = str(t)
        # now_thread = threading.current_thread()
        now_process = str(os.getpid())
        # rows = [s, now_thread.name]
        print(f"type: {self.type_flag}, time: {end_time - start_time}, now_process : {now_process}")

        # 写入每个进程耗时
        with open('parallel_run_times.csv', 'a+', encoding='utf-8') as f:
            f.write(s)
            f.write(",")
            # f.write(now_thread.name)
            f.write(now_process)
            f.write(",")
            f.write(self.type_flag)
            f.write('\n')
        # thread_time = [s]
        # now_thread = [threading.current_thread().name]
        #
        # # 字典中的key值即为csv中列名
        # data = pd.DataFrame({'time': thread_time, 'now_thread': now_thread})
        #
        # # 将DataFrame存储为csv,index表示是否显示行名，default=True
        # data.to_csv("run_times.csv", index=False, sep=',')

        return compiled_circuit




def rotate_matrix(matrix):
    # 获取矩阵
    x = matrix.data
    # 获得矩阵的边长
    n = len(x)
    # print(n)
    # 获取矩阵的元素类型
    # print(x.dtype)
    # 获取矩阵对应的列表, x仍为numpy类型的数组类型
    y = x.tolist()

    # 将列表中的每一项中的元素反转
    for i in range(0, len(y)):
        y[i].reverse()

    # 首尾依次交换
    start = 0
    end = n - 1
    while start < end:
        tmp = y[start]
        y[start] = y[end]
        y[end] = tmp
        start += 1
        end -= 1

    # 将旋转好的列表还原为numpy类型的矩阵
    total_matrix = np.array(y)
    # 返回新的矩阵, 用于赋值给matrix.data
    return total_matrix


"""
9q   population=30  iterations=15
16q  population=50  iterations=100
20q  population=100  iterations=100
"""
def map_cnot_circuit(file, architecture, mode=GENETIC_STEINER_MODE, dest_file=None, population=30, iterations=15,
                     crossover_prob=0.8, mutation_prob=0.2, **kwargs):
    if type(architecture) == type(""):
        architecture = create_architecture(architecture)

    circuit = CNOT_tracker.from_qasm_file(file)  # 从文件读量子线路
    print(f"初始门数: {len(circuit.gates)}")
    matrix = circuit.matrix
    print("matrix.data: ")
    print(matrix.data)
    print("matrix : ")
    print(matrix)

    # 获取新的矩阵, 用于赋值给matrix.data
    r_matrix = rotate_matrix(matrix)

    # 一个存放四种矩阵的列表
    matrix_list = []
    # 先化为上三角型, 主对角线以下全为0, 每次新的赋值一定要使用copy方法
    matrix_first_up = matrix.copy()
    # 加入列表
    matrix_list.append(matrix_first_up)
    # 转置矩阵先化为上三角型
    matrix_first_up_transpose = matrix_first_up.transpose().copy()
    # 加入列表
    matrix_list.append(matrix_first_up_transpose)

    # 记原理矩阵的data
    old_data = matrix.data
    # 旋转矩阵
    matrix.data = r_matrix
    print("rotate_matrix.data: ")
    print(matrix.data)
    # 等价于原矩阵先化为下三角型, 原矩阵主对角线以上全为0
    matrix_first_down = matrix.copy()
    # 加入列表
    matrix_list.append(matrix_first_down)
    # 等价于原矩阵转置矩阵先化为下三角型,
    matrix_first_down_transpose = matrix_first_down.transpose().copy()
    # 加入列表
    matrix_list.append(matrix_first_down_transpose)
    # 恢复matrix的data
    matrix.data = old_data

    # print("matrix.data")
    # print(matrix.data)
    # print("读取的线路生成的矩阵如下:")
    # print(matrix)
    # print('matrix.transpose:')
    # print(matrix.transpose())

    # *****************获取matrix对应的numpy矩阵*******************
    # print(type(matrix.data))
    # print(matrix.data.tolist())
    # print(np.sum(matrix.data, axis=0))

    # *************************************通过权值进行初始映射******************************************
    """
    先获取architecture的distances,
    通过分析matrix, 
    print("当前架构跑出的floyd算法结果如下:")
    for k in architecture.distances:
        print(f"{k} : ")
        for i in architecture.distances[k]:
            print(i)
    """


    # 原来的guess代码，实现串行代码
    # compiled_circuit = CNOT_tracker(circuit.n_qubits)  # 目标线路即为单位矩阵
    #
    # if mode in no_genetic_elim_modes:
    #     gauss(mode, matrix, architecture, full_reduce=True, y=compiled_circuit, **kwargs)
    # elif mode in genetic_elim_modes:
    #     # debug and print("original matrix:",matrix) #原始线路所对应的矩阵
    #     # debug and print("original y:",compiled_circuit.matrix) #单位阵
    #     start_time = time.time()  # 开始计时
    #     for i in range(0, len(matrix_list)):
    #         gauss(mode, matrix_list[i], architecture, full_reduce=True, y=compiled_circuit,
    #                  population_size=population, crossover_prob=crossover_prob, mutate_prob=mutation_prob,
    #                  n_iterations=iterations, **kwargs)
    #     end_time = time.time()  # 开始结束
    #     t = end_time - start_time
    #     s = str(t)
    #     print(f"运行耗时: {s}")
    #     with open('serial_run_times.csv', 'a+', encoding='utf-8') as f:
    #         f.write(s)
    #         f.write('\n')
    # elif mode == QUIL_COMPILER:
    #     from pyzx.pyquil_circuit import PyQuilCircuit
    #     compiled_circuit = PyQuilCircuit.from_CNOT_tracker(circuit, architecture)
    #     compiled_circuit.compile()
    #
    # if dest_file is not None:
    #     compiled_qasm = compiled_circuit.to_qasm()  # 映射后的线路变成QASM格式
    #     with open(dest_file, "w") as f:
    #         f.write(compiled_qasm)  # 讲QASM线路写入文件
    #
    # return compiled_circuit



    # 已下是多线程代码

    s_time = time.time()  # 开始计时

    p1 = GuessProcess("matrix_first_up", matrix_first_up, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p2 = GuessProcess("matrix_first_up_transpose", matrix_first_up_transpose, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p3 = GuessProcess("matrix_first_down", matrix_first_down, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p4 = GuessProcess("matrix_first_down_transpose", matrix_first_down_transpose, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)


    # p1.start()
    p2.start()
    # p3.start()
    # p4.start()


    # 比较出耗时短的线路,作为返回值返回


    # p1.join()
    p2.join()
    # p3.join()
    # p4.join()

    e_time = time.time()  # 计时结束
    t = e_time - s_time
    s = str(t)

    # 写入进程总的耗时
    with open('total_parallel_run_times.csv', 'a+', encoding='utf-8') as f:
        f.write("并行时间")
        f.write(",")
        f.write(s)
        f.write('\n')


