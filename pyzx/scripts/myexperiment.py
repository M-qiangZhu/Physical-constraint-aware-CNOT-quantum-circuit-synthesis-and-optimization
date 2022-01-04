import re
import threading
import time
import pandas as pd
from pyzx.scripts import cnot2cnot


# 处理文件中的线路
def processor(architecture, gates, mode='steiner'):
    # 生成文件全名
    selected_file = architecture + "_" + gates

    # 分割输入架构
    list_str = architecture.split("_")
    # 遍历str获取架构名分解后含数字的项
    for i in list_str:
        if not i.isalpha():
            target = i
    # 从数字项中获取量子位数
    q = re.sub("\D", "", target)  # 在字符串中找到非数字的字符（正则表达式中 '\D' 表示非数字），并用 “” 替换，然后返回的就是只剩下数字的字符串

    # 获取架构类型
    for i in list_str:
        if i == "square":
            architecture_type = "square"
        if i == "aspen":
            architecture_type = "rigetti_16q_aspen"
        if i == "qx5":
            architecture_type = "ibm_qx5"
        if i == "melbourne":
            architecture_type = "ibmq_16_melbourne"
        if i == "tokyo":
            architecture_type = "ibm_q20_tokyo"
        if i == "qx2":
            architecture_type = "ibm_qx2"

        # set population 和 iterations
    if q == '9' or '2':
        population = "30"
        iterations = "15"
    if q == '16' or q == '5':
        population = "50"
        iterations = "100"
    if q == '20':
        population = "100"
        iterations = "100"

    # 获取文件中的量子门数
    gates_dir = gates

    for i in range(0, 20):
        # 获取当前需处理的文件名
       
        if architecture == "9q_square" or "16q_square":
            file_name = f"../../circuits/steiner/{q}qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "rigetti_16q_aspen":
            file_name = f"../../circuits/steiner/16qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "ibm_qx5":
            file_name = f"../../circuits/steiner/16qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "ibm_q20_tokyo":
            file_name = f"../../circuits/steiner/20qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "ibm_qx2":
            file_name = f"../../circuits/steiner/5qubits/{gates_dir}/Original{i}.qasm"

        print("-" * 50 + f"开始处理{selected_file}的第{i}号个文件" + "-" * 50)
        start_time = time.time()  # 开始计时

        # 开始映射
        cnot2cnot.main(["QASM_source", file_name,
                        "--mode", mode,
                        "--architecture", architecture_type,
                        "--destination", "resultdata",
                        "--qubits", q,
                        "--population", population,
                        "--iterations", iterations])

        end_time = time.time()  # 结束计时
        print("运行耗时: {:.6f} 秒".format(end_time - start_time))


        s = end_time - start_time
        s = str(s)

        print("#" * 60 + "结束" + "#" * 60)
        print()  # 间隔



# 主函数
# 记录需要遍历的架构
tg_dict = {"9q_square": [3, 5, 10, 20, 30],
           "16q_square": [4, 8, 16, 32, 64, 128, 256],
           "rigetti_16q_aspen": [4, 8, 16, 32, 64, 128, 256],
           "ibm_qx5": [4, 8, 16, 32, 64, 128, 256],
           "ibm_q20_tokyo": [4, 8, 16, 32, 64, 128, 256]}

# processor(architecture, gates)
# 遍历字典的key, 遍历value值
# for k in tg_dict:
#     list_gates = tg_dict[k]
#     architecture = k
#     for gate in list_gates:
#         gates = str(gate)
#         processor(architecture, gates)
#         print("*" * 120)


if __name__ == '__main__':

    # save parallel computing data
    open("tokyo_parallel_result_conts.csv", 'w', encoding='utf-8').close()
    open("parallel_run_times.csv", 'w', encoding='utf-8').close()
    open("total_parallel_run_times.csv", 'w', encoding='utf-8').close()

    for gate in [4, 8, 16, 32, 64, 128, 256]:
        gates = str(gate)
        processor("ibm_q20_tokyo", gates, "genetic_steiner")
        print("*" * 120)


    for gate in [15, 20, 40, 80, 100]:
        gates = str(gate)
        processor("ibm_qx2", gates, "genetic_steiner")
        print("*" * 120)

    """ ******************************************************************************************** """
    # save serial computing data
    # open("serial_result_conts.csv", 'w', encoding='utf-8').close()
    # open("serial_run_times.csv", 'w', encoding='utf-8').close()
    # series test
    # for gate in [4, 8, 16, 32, 64, 128, 256]:
    #     gates = str(gate)
    #     processor("ibm_q20_tokyo", gates, "genetic_steiner")
    #     print("*" * 120)

    """ ******************************************************************************************** """
    # second test
    # take the average value of multiple operations, cycle 40 times
    # for i in range(0, 40):
    #     print("+" * 30 + f"第{i + 1}次循环" + "+" * 30)
    #     processor("ibm_q20_tokyo", "256", 'genetic_steiner')

    """ ******************************************************************************************** """
    # first test
    # file_name = "/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/16/16q-square/steiner/Original0.qasm"
    # file_name = "/Users/kungfu/Desktop/Original0.qasm"
    # mode = "genetic_steiner"
    # architecture_type = "square"
    # q = "16"
    # start_time = time.time()
    # cnot2cnot.main(["QASM_source", file_name,
    #                 "--mode", mode,
    #                 "--architecture", architecture_type,
    #                 "--destination", "resultdata",
    #                 "--qubits", q])
    # end_time = time.time()
    # print("times: {:.4f} sec".format(end_time - start_time))
    # print("#" * 60 + "over" + "#" * 60)
    # print()
