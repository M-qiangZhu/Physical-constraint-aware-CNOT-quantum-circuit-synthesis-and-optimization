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
import os
import time

if __name__ == '__main__':
    print("Please call this as python -m pyzx mapper ...")
    exit()

from ..linalg import Mat2
from ..routing.architecture import architectures, SQUARE
from ..routing.cnot_mapper import STEINER_MODE, QUIL_COMPILER, batch_map_cnot_circuits, elim_modes, compiler_modes
from ..utils import make_into_list, restricted_float

description = "Compiles given qasm files or those in the given folder to a given architecture."

import argparse

# 用于将命令行字符串解析为Python对象的对象
# prog-程序的名称（默认：sys.argv [0]）
# description-程序功能的描述
parser = argparse.ArgumentParser(prog="pyzx mapper", description=description)
parser.add_argument("QASM_source", nargs='+', help="The QASM file or folder with QASM files to be routed.")  # 用来映射的QASM文件或存有QASM文件的文件夹
parser.add_argument("-m", "--mode", nargs='+', dest="mode", default=STEINER_MODE,
                    help="The mode specifying how to route. choose 'all' for using all modes.",
                    choices=elim_modes + [QUIL_COMPILER, "all"])
parser.add_argument("-a", "--architecture", nargs='+', dest="architecture", default=SQUARE, choices=architectures,
                    help="Which architecture it should run compile to.")  # 默认架构: SQUARE
parser.add_argument("-q", "--qubits", nargs='+', default=None, type=int,
                    help="The number of qubits for the fully connected architecture.")
# parser.add_argument("-f", "--full_reduce", dest="full_reduce", default=1, type=int, choices=[0,1], help="Full reduce")
parser.add_argument("--population", nargs='+', default=30, type=int,
                    help="The population size for the genetic algorithm.")
parser.add_argument("--iterations", nargs='+', default=15, type=int,
                    help="The number of iterations for the genetic algorithm.")
parser.add_argument("--crossover_prob", nargs='+', default=0.8, type=restricted_float,
                    help="The crossover probability for the genetic algorithm. Must be between 0.0 and 1.0.")
parser.add_argument("--mutation_prob", nargs='+', default=0.2, type=restricted_float,
                    help="The mutation probability for the genetic algorithm. Must be between 0.0 and 1.0.")
parser.add_argument("--perm", default="both", choices=["row", "col", "both"], help="Whether to find a single optimal permutation that permutes the rows, columns or both with the genetic algorithm.")
parser.add_argument("--destination",
                    help="Destination file or folder where the compiled circuit should be stored. Otherwise the source folder is used.")  # 应在其中存储已编译电路的目标文件或文件夹。否则，将使用源文件夹
parser.add_argument("--metrics_csv", default=None,
                    help="The location to store compiling metrics as csv, if not given, the metrics are not calculated. Only used when the source is a folder")
# 将编译度量存储为csv的位置，如果没有给出，则不计算度量。只在源为文件夹时使用
parser.add_argument("--n_compile", default=1, type=int,
                    help="How often to run the Quilc compiler, since it is not deterministic.")
parser.add_argument("--subfolder", default=None, type=str, nargs="+",
                    help="Possible subfolders from the main QASM source to compile from. Less typing when source folders are in the same folder. Can also be used for subfiles.")

# debug = True
debug = False

def main(args):

    debug and print("cont2cnot_args:", args)
    # print("测试ipython是否可以动态保存")

    args = parser.parse_args(args)

    if args.metrics_csv is not None and os.path.exists(args.metrics_csv):

        # print("executed")  #没有执行到
        delete_csv = None
        text = input(
            "The given metrics file [%s] already exists. Do you want to overwrite it? (Otherwise it is appended) [y|n]" % args.metrics_csv)

        if text.lower() in ['y', "yes"]:  # 转换字符串中所有大写字符为小写。
            delete_csv = True
        elif text.lower() in ['n', 'no']:
            delete_csv = False

        while delete_csv is None:
            text = input("Please answer yes or no.")
            if text.lower() in ['y', "yes"]:
                delete_csv = True
            elif text.lower() in ['n', 'no']:
                delete_csv = False

        if delete_csv:
            os.remove(args.metrics_csv)
            # 删除指定路径的文件。如果指定的路径是一个目录，将抛出OSError。

    debug and print(args.QASM_source)  #['QASM_source', 'testdata']
    # print(type(args.QASM_source))
    sources = make_into_list(args.QASM_source)  # 将args.QASM_source表示成列表
    # print(sources)

    if args.subfolder is not None:  # 未执行到，源数据含有子文件夹
        sources = [os.path.join(source, subfolder) for source in sources for subfolder in args.subfolder if os.path.isdir(source)]  # 列表解析
        # Remove non existing paths

    sources = [source for source in sources if
               os.path.exists(source) or print("Warning, skipping non-existing source:", source)]  # sources列表被清空
    # os.path.exists()判断括号里的文件是否存在的意思，括号内的可以是文件路径,存在返回True
    debug and print("sources:", sources) #sources: ['testdata']

    debug and print("args.mode:", args.mode)  # args.mode: steiner
    if "all" in args.mode:
        mode = elim_modes + [QUIL_COMPILER]
    else:
        mode = args.mode

    all_circuits = []
    for source in sources:
        debug and print("Mapping qasm files in path:", source)

        circuits = batch_map_cnot_circuits(source, mode, args.architecture, n_qubits=args.qubits,
                                           populations=args.population,
                                           iterations=args.iterations,
                                           crossover_probs=args.crossover_prob, mutation_probs=args.mutation_prob,
                                           dest_folder=args.destination, metrics_file=args.metrics_csv,
                                           n_compile=args.n_compile)


        all_circuits.extend(circuits)  # 将circuits列表中的内容添加到all_circuits末尾

    # 自行添加
    # return circuits

