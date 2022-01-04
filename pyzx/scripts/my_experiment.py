import re
import threading
import time
import pandas as pd
from pyzx.scripts import cnot2cnot



def processor(architecture, gates, mode='steiner'):

    selected_file = architecture + "_" + gates

    list_str = architecture.split("_")

    for i in list_str:
        if not i.isalpha():
            target = i

    q = re.sub("\D", "", target)  

    # get architecture_type
    for i in list_str:
        if i == "square":
            architecture_type = "square"
        if i == "aspen":
            architecture_type = "rigetti_16q_aspen"
        if i == "qx5":
            architecture_type = "ibm_qx5"
        if i == "tokyo":
            architecture_type = "ibm_q20_tokyo"

    # set population and iterations
    if q == '9':
        population = "30"
        iterations = "15"
    if q == '16' or q == '5':
        population = "50"
        iterations = "100"
    if q == '20':
        population = "100"
        iterations = "100"

    # get gates
    gates_dir = gates

    for i in range(0, 20):
        # get file name, use your file path to change the directory
        """
        9q-square :           /Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/9qubits/3/Original0.qasm
        16q-square :          /Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/4/Original0.qasm
        rigetti-16q-aspen:    /Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/4/rigetti_16q_aspen/steiner/Original0.qasm
        ibm-qx5 :             /Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/4/ibm_qx5/steiner/Original0.qasm
        ibm-q20-tokyo :       /Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/20qubits/4/ibm_q20_tokyo/steiner/Original0.qasm
        """
        if architecture == "9q_square" or "16q_square":
            file_name = f"/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/{q}qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "rigetti_16q_aspen":
            file_name = f"/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "ibm_qx5":
            file_name = f"/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/{gates_dir}/Original{i}.qasm"
        if architecture == "ibm_q20_tokyo":
            file_name = f"/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/20qubits/{gates_dir}/Original{i}.qasm"

        print("-" * 50 + f"start processing the {i} file of {selected_file}" + "-" * 50)
        start_time = time.time()

        # start mappinng
        cnot2cnot.main(["QASM_source", file_name,
                        "--mode", mode,
                        "--architecture", architecture_type,
                        "--destination", "resultdata",
                        "--qubits", q,
                        "--population", population,
                        "--iterations", iterations])

        end_time = time.time() 
        print("times: {:.6f} sec".format(end_time - start_time))


        s = end_time - start_time
        s = str(s)
        # s = s[0:8]

        # final_time = [s]
        # now_thread = [threading.current_thread().name]
        #
        # data = pd.DataFrame({'time': final_time, 'now_thread': now_thread})
        #
        # data.to_csv("run_times.csv", index=False, sep=',')

        now_thread = threading.current_thread()
        with open('run_times.csv', 'a+', encoding='utf-8') as f:
            f.write(s)
            f.write(",")
            f.write(now_thread.name)
            f.write('\n')

        print("#" * 60 + "over" + "#" * 60)
        print() 

# save the schema that needs to be traversed
# tg_dict = {"9q_square": [3, 5, 10, 20, 30],
#            "16q_square": [4, 8, 16, 32, 64, 128, 256],
#            "rigetti_16q_aspen": [4, 8, 16, 32, 64, 128, 256],
#            "ibm_qx5": [4, 8, 16, 32, 64, 128, 256],
#            "ibm_q20_tokyo": [4, 8, 16, 32, 64, 128, 256]}


# processor(architecture, gates)
# iterate through the key of the dictionary, iterate through the value
# for k in tg_dict:
#     list_gates = tg_dict[k]
#     architecture = k
#     for gate in list_gates:
#         gates = str(gate)
#         processor(architecture, gates)
#         print("*" * 120)


if __name__ == '__main__':


    # save parallel computing data
    open("result_conts.csv", 'w', encoding='utf-8').close()
    open("run_times.csv", 'w', encoding='utf-8').close()

    processor("9q_square", "3", "genetic_steiner")
    # processor("9q_square", "5", "genetic_steiner")
    # processor("9q_square", "10", "genetic_steiner")
    # processor("9q_square", "20", "genetic_steiner")
    # processor("9q_square", "30", "genetic_steiner")

    # processor("16q_square", "4", "genetic_steiner")
    # processor("16q_square", "8", "genetic_steiner")
    # processor("16q_square", "16", "genetic_steiner")
    # processor("16q_square", "32", "genetic_steiner")
    # processor("16q_square", "64", "genetic_steiner")
    # processor("16q_square", "128", "genetic_steiner")
    # processor("16q_square", "256", "genetic_steiner")

    # processor("rigetti_16q_aspen", "4", "genetic_steiner")
    # processor("rigetti_16q_aspen", "8", "genetic_steiner")
    # processor("rigetti_16q_aspen", "16", "genetic_steiner")
    # processor("rigetti_16q_aspen", "32", "genetic_steiner")
    # processor("rigetti_16q_aspen", "64", "genetic_steiner")
    # processor("rigetti_16q_aspen", "128", "genetic_steiner")
    # processor("rigetti_16q_aspen", "256", "genetic_steiner")

    # processor("ibm_qx5", "4", "genetic_steiner")
    # processor("ibm_qx5", "8", "genetic_steiner")
    # processor("ibm_qx5", "16", "genetic_steiner")
    # processor("ibm_qx5", "32", "genetic_steiner")
    # processor("ibm_qx5", "64", "genetic_steiner")
    # processor("ibm_qx5", "128", "genetic_steiner")
    # processor("ibm_qx5", "256", "genetic_steiner")

    # processor("ibm_q20_tokyo", "4", "genetic_steiner")
    # processor("ibm_q20_tokyo", "8", "genetic_steiner")
    # processor("ibm_q20_tokyo", "16", "genetic_steiner")
    # processor("ibm_q20_tokyo", "32", "genetic_steiner")
    # processor("ibm_q20_tokyo", "64", "genetic_steiner")
    # processor("ibm_q20_tokyo", "128", "genetic_steiner")
    # processor("ibm_q20_tokyo", "256", "genetic_steiner")


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
    #     print("+" * 30 + f"the {i + 1}th loop" + "+" * 30)
    #     processor("ibm_q20_tokyo", "256", 'genetic_steiner')


    """ ******************************************************************************************** """
    # first test
    # file_name = "/Users/kungfu/Desktop/pyzx-steiner_decomp_annotation/circuits/steiner/16qubits/16/16q-square/steiner/Original0.qasm"
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
