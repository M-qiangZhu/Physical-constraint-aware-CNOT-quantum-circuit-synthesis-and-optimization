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

debug = False

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
        gauss(mode, mat, architecture=architecture, y=circuit, full_reduce=full_reduce, **kwargs)  

        return circuit.count_cnots()

    return fitness_func


def gauss(mode, matrix, architecture=None, permutation=None, **kwargs):  
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
        if architecture is None:
            print(
                "\033[91m Warning: Architecture is not given, assuming fully connected architecture of size matrix.shape[0]. \033[0m ")
            architecture = create_fully_connected_architecture(len(matrix.data))

        # print("Calling STEINER_MODE.....")  
        # print("matrix:",matrix)
        # print("architecture:",architecture.name)
        # print("permutation:",permutation)

        return steiner_gauss(matrix, architecture, permutation=permutation, **kwargs)

    elif mode == GENETIC_STEINER_MODE:
        print("Calling GENETIC_STEINER_MODE .....")
        perm, cnots, rank = permutated_gauss(matrix, STEINER_MODE, architecture=architecture, permutation=permutation,
                                             **kwargs)
        now_thread = threading.current_thread()
        print(f"---------{now_thread.name}-----------")
        print("result perm:", perm)
        print(f"result cnots: {cnots}")
        print(f"result rank: {rank}")

        # print(f"type_flag: {type_flag}, result cnots: {cnots}")
        # print(f"type_flag: {type_flag}, result rank: {rank}")

   
        result_conts = str(cnots)
        rows = [result_conts, now_thread.name]

        now_process = str(os.getpid())

        
        with open('/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/pyzx/scripts/result_conts.csv', 'a+', encoding='utf-8') as f:
            # f_csv = csv.writer(f)
            # f_csv.writerows(rows)
            f.write(result_conts)
            f.write(",")
            # f.write(now_thread.name)
            f.write(now_process)
            f.write('\n')



        # with open('/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/pyzx/scripts/serial_result_conts.csv', 'a+',
        #           encoding='utf-8') as f:
        #     # f_csv = csv.writer(f)
        #     # f_csv.writerows(rows)
        #     f.write(result_conts)
        #     f.write(",")
        #     f.write(now_thread.name)
        #     f.write('\n')

        return rank

    elif mode == GENETIC_GAUSS_MODE:
        perm, cnots, rank = permutated_gauss(matrix, GAUSS_MODE, architecture=architecture, permutation=permutation,
                                             **kwargs)
        return rank


def permutated_gauss(matrix, mode=None, architecture=None, population_size=100, crossover_prob=0.8, mutate_prob=0.2,
                     n_iterations=100,  # 原来是: population_size=30, n_iterations=50
                     row=True, col=True, full_reduce=True, fitness_func=None, x=None, y=None, **kwargs):
    """
    Finds an optimal permutation of the matrix to reduce the number of CNOT gates.
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

    optimizer = GeneticAlgorithm(population_size, crossover_prob, mutate_prob, fitness_func)

    permsize = len(matrix.data) if row else len(matrix.data[0])  # 5
    debug and print("permsize:", permsize)

    # print("ok1!")
    best_permutation = optimizer.find_optimimum(permsize, n_iterations, continued=True)
    graph_now = architecture.graph.graph
    # best_permutation = get_heuristic_initial_permutation(matrix, graph_now)
    # print("best_permutation:", best_permutation)

    debug and print("best_permutation:", best_permutation)

    n_qubits = len(matrix.data) 
    row_perm = best_permutation if row else np.arange(len(matrix.data))
    debug and print("row_permn:", row_perm)

    col_perm = best_permutation if col else np.arange(len(matrix.data[0]))
    debug and print("col_perm:", col_perm)

    if y is None:  
        circuit = CNOT_tracker(n_qubits)
    else:
        circuit = y

    debug and print("matrix:", matrix.data)
    mat = Mat2([[matrix.data[r][c] for c in col_perm] for r in row_perm])

    circuit.row_perm = row_perm
    debug and print("circuit.row_perm:", circuit.row_perm)

    circuit.col_perm = col_perm 
    debug and print("circuit.col_perm:", circuit.col_perm)  

    # debug and print("mat:", mat)
    # debug and print("x:", x)
    # debug and print("y:", y.matrix)
    print("mat:")
    print(mat)
    # print("x:", x)
    # print("y:", y.matrix)
    rank = gauss(mode, mat, architecture, x=x, y=circuit, full_reduce=full_reduce, **kwargs)
    # 执行一次，STEINER_MODE

    return best_permutation, circuit.count_cnots(), rank


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


def batch_map_cnot_circuits(source, modes, architectures, n_qubits=None, populations=30, iterations=15,
                            crossover_probs=0.8,
                            mutation_probs=0.5, dest_folder=None, metrics_file=None, n_compile=1):
    modes = make_into_list(modes)  
    debug and print("modes:", modes)  # ['genetic_steiner']  

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

    if os.path.isfile(source):  
        source, file = os.path.split(source)  
        files = [file]
    else:
        # print("os.listdir:",os.listdir(source)) 
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    if not os.path.exists(source):
        raise IOError("Folder does not exist: " + source)  
    if dest_folder is None:
        dest_folder = source  
    else:
        os.makedirs(dest_folder, exist_ok=True)  

    arch_iter = []
    circuits = {}
    metrics = []

    for architecture in architectures:
        if architecture in dynamic_size_architectures:
            # dynamic_size_architectures = [FULLY_CONNNECTED, LINE, CIRCLE, SQUARE]
            if n_qubits is None:
                raise KeyError("Number of qubits not specified for architecture" + architecture)
            else:
                n_qubits = make_into_list(n_qubits)
                arch_iter.extend([create_architecture(architecture, n_qubits=q) for q in n_qubits])
        else:
            arch_iter.append(create_architecture(architecture))

    for architecture in arch_iter:
        circuits[architecture.name] = {}
        for mode in modes:
            if mode == QUIL_COMPILER:
                n_compile_list = range(n_compile)
            else:
                n_compile_list = [None]

            new_dest_folder = os.path.join(dest_folder, architecture.name, mode)  # (resultdata, '9q-square', 'steiner')

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
                                    1].lower() == ".qasm":  

                                    origin_file = os.path.join(source, file)  
                                    for i in n_compile_list:
                                        dest_filename = create_dest_filename(origin_file, population, iteration,
                                                                             crossover_prob, mutation_prob, i)
                                        dest_file = os.path.join(dest_folder, architecture.name, mode, dest_filename)
                                        try:
                                            start_time = time.time()  
                                            print("Calling map_cnot_circuit....")
                                            circuit = map_cnot_circuit(origin_file, architecture, mode=mode,
                                                                       dest_file=dest_file,
                                                                       population=population, iterations=iteration,
                                                                       crossover_prob=crossover_prob,
                                                                       mutation_prob=mutation_prob)
                                            # print(circuit.matrix)
                                            # print(type(circuit))
                                            end_time = time.time()  

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
    new_matrix = old_matrix.copy()
    matrix_data = old_matrix.data
    sum_list = []
    for i in range(0, len(matrix_data)):
        row_sum = 0
        for j in range(0, i + 1):
            # print(f"matrix_data[{i}][{j}] : {matrix_data[i][j]}")
            row_sum += matrix_data[i][j] * (len(matrix_data) - j)
        sum_list.append(row_sum)
        print(f"row_sum : {row_sum}")
        print(f"sum_list : {sum_list}")
        print("-" * 50)

    index_order = []

    for i in range(0, len(sum_list)):
        max_index = sum_list.index(max(sum_list))
        index_order.append(max_index)
        sum_list[max_index] = -1
    print(index_order)

    valid_vertex_list = []
    for key in graph_now:
        print(f"{key} : {len(graph_now[key])}")
        valid_vertex_list.append(len(graph_now[key]))
    print(valid_vertex_list)

    initial_mapping = [0] * len(graph_now)
    print(initial_mapping)

    for i in range(0, len(initial_mapping)):
        get_index = valid_vertex_list.index(max(valid_vertex_list))
        valid_vertex_list[get_index] = 0
        print(f"valid_vertex_list : {valid_vertex_list}")
        initial_mapping[get_index] = index_order[i]
        print(f"initial_mapping : {initial_mapping}")
        print(f"order_list : {index_order}")
        print("*" * 50)

    return initial_mapping


class GuessThread(threading.Thread):
    """
    Multi-threading execution elimination
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
        start_time = time.time()  
        compiled_circuit = CNOT_tracker(self.circuit.n_qubits)  

        if self.mode in no_genetic_elim_modes:
            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit)
        elif self.mode in genetic_elim_modes:

            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit,
                  population_size=self.population, crossover_prob=self.crossover_prob, mutate_prob=self.mutation_prob,
                  n_iterations=self.iterations)
        elif self.mode == QUIL_COMPILER:
            from pyzx.pyquil_circuit import PyQuilCircuit
            compiled_circuit = PyQuilCircuit.from_CNOT_tracker(self.circuit, self.architecture)
            compiled_circuit.compile()

        if self.dest_file is not None:
            compiled_qasm = compiled_circuit.to_qasm()  
            with open(self.dest_file, "w") as f:
                f.write(compiled_qasm)  
        end_time = time.time()  
        t = end_time - start_time
        s = str(t)
        now_thread = threading.current_thread()
        # rows = [s, now_thread.name]
        print(f"type: {self.type_flag}, time: {end_time - start_time}, now_thread_name : {now_thread.name}")

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
        # data = pd.DataFrame({'time': thread_time, 'now_thread': now_thread})
        #
        # data.to_csv("run_times.csv", index=False, sep=',')

        return compiled_circuit


class GaussProcess(Process):
    """
        Multi-process execution elimination
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
        start_time = time.time()  
        compiled_circuit = CNOT_tracker(self.circuit.n_qubits)  

        if self.mode in no_genetic_elim_modes:
            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit)
        elif self.mode in genetic_elim_modes:
            gauss(self.mode, self.matrix, self.architecture, full_reduce=True, y=compiled_circuit,
                  population_size=self.population, crossover_prob=self.crossover_prob, mutate_prob=self.mutation_prob,
                  n_iterations=self.iterations)
        elif self.mode == QUIL_COMPILER:
            from pyzx.pyquil_circuit import PyQuilCircuit
            compiled_circuit = PyQuilCircuit.from_CNOT_tracker(self.circuit, self.architecture)
            compiled_circuit.compile()

        if self.dest_file is not None:
            compiled_qasm = compiled_circuit.to_qasm()  
            with open(self.dest_file, "w") as f:
                f.write(compiled_qasm)  

        end_time = time.time()  
        t = end_time - start_time
        s = str(t)
        now_process = str(os.getpid())
        # print(f"type: {self.type_flag}, time: {end_time - start_time}, now_process : {now_process}")

        with open('run_times.csv', 'a+', encoding='utf-8') as f:
            f.write(s)
            f.write(",")
            f.write(now_process)
            f.write(",")
            f.write(self.type_flag)
            f.write('\n')

        # data = pd.DataFrame({'time': thread_time, 'now_thread': now_thread})
        # 
        # data.to_csv("run_times.csv", index=False, sep=',')

        return compiled_circuit




def rotate_matrix(matrix):
    x = matrix.data
    n = len(x)
    # print(n)
    # print(x.dtype)
    y = x.tolist()

    for i in range(0, len(y)):
        y[i].reverse()

    start = 0
    end = n - 1
    while start < end:
        tmp = y[start]
        y[start] = y[end]
        y[end] = tmp
        start += 1
        end -= 1

    total_matrix = np.array(y)

    return total_matrix


def map_cnot_circuit(file, architecture, mode=GENETIC_STEINER_MODE, dest_file=None, population=30, iterations=15,
                     crossover_prob=0.8, mutation_prob=0.2, **kwargs):
    if type(architecture) == type(""):
        architecture = create_architecture(architecture)

    circuit = CNOT_tracker.from_qasm_file(file)  
    print(f"initial gates: {len(circuit.gates)}")
    matrix = circuit.matrix
    print("matrix.data: ")
    print(matrix.data)
    print("matrix : ")
    print(matrix)

    r_matrix = rotate_matrix(matrix)

    matrix_list = []
    matrix_first_up = matrix.copy()
    matrix_list.append(matrix_first_up)
    matrix_first_up_transpose = matrix_first_up.transpose().copy()
    matrix_list.append(matrix_first_up_transpose)

    old_data = matrix.data
    matrix.data = r_matrix
    print("rotate_matrix.data: ")
    print(matrix.data)
    matrix_first_down = matrix.copy()
    matrix_list.append(matrix_first_down)
    matrix_first_down_transpose = matrix_first_down.transpose().copy()
    matrix_list.append(matrix_first_down_transpose)
    matrix.data = old_data



    """
    compiled_circuit = CNOT_tracker(circuit.n_qubits)  

    if mode in no_genetic_elim_modes:
        gauss(mode, matrix, architecture, full_reduce=True, y=compiled_circuit, **kwargs)
    elif mode in genetic_elim_modes:
        # debug and print("original matrix:",matrix) 
        # debug and print("original y:",compiled_circuit.matrix) 
        start_time = time.time()  
        for i in range(0, len(matrix_list)):
            gauss(mode, matrix_list[i], architecture, full_reduce=True, y=compiled_circuit,
                     population_size=population, crossover_prob=crossover_prob, mutate_prob=mutation_prob,
                     n_iterations=iterations, **kwargs)
        end_time = time.time()  
        t = end_time - start_time
        s = str(t)
        print(f"serial times: {s}")
        with open('serial_run_times.csv', 'a+', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')
    elif mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        compiled_circuit = PyQuilCircuit.from_CNOT_tracker(circuit, architecture)
        compiled_circuit.compile()

    if dest_file is not None:
        compiled_qasm = compiled_circuit.to_qasm()  
        with open(dest_file, "w") as f:
            f.write(compiled_qasm)  

    return compiled_circuit
    """

    p1 = GaussProcess("matrix_first_up", matrix_first_up, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p2 = GaussProcess("matrix_first_up_transpose", matrix_first_up_transpose, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p3 = GaussProcess("matrix_first_down", matrix_first_down, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    p4 = GaussProcess("matrix_first_down_transpose", matrix_first_down_transpose, circuit, mode, architecture, population, crossover_prob, mutation_prob, iterations, dest_file, **kwargs)
    
    p_list = [p1, p2, p3, p4]

    start_time = time.time() 

    for i in range(0, len(p_list)):
        p_list[i].start()

    for i in range(0, len(p_list)):
        p_list[i].join() 

    end_time = time.time()  
    t = end_time - start_time
    s = str(t)
    print(f"parallel times: {s}")
    with open('run_times.csv', 'a+', encoding='utf-8') as f:
        f.write("parallel times")
        f.write(',')
        f.write(s)
        f.write('\n')


