import pandas as pd
import xlwt

def read_execle(fpath):
    io = pd.io.excel.ExcelFile(fpath)
    data = pd.read_excel(io, sheet_name=None, header=None, names=['gates', 'accuracy', 'thread'])
    return data


def read_csv(fpath):
    data = pd.read_csv(fpath, header=None, names=['gates', 'accuracy', 'thread'])
    return data


def get_parallels_max(data, gates):
    # 保存每组的最大值
    df_max_list = []
    # 每次获取4行数据
    for i in range(0, 20):
        # 取当前4行中的最大值
        df = data[4 * i: 4 * (i + 1)]
        df_max = df['accuracy'].max()
        df_max_list.append(df_max)
        print(df)
        print(df_max)

    # 将 df_max_list 写入 DataFrame
    parallels_max = pd.DataFrame(data=df_max_list, columns=['max'])
    print(parallels_max)

    # 将 DataFrame 写入excel
    parallels_max.to_excel(f'./get_parallels_max/tokyo_parallels_max_{gates}.xls', sheet_name=str(gates), index=False)
    # with pd.ExcelWriter('parallels_max.xls') as writer:
    #     parallels_max.to_excel(writer, encoding='utf-8', sheet_name=str(gates))

if __name__ == '__main__':

    qx2_gates_list = [2, 4, 5, 8, 10, 15, 20, 30, 40, 80, 100, 200]
    tokyo_gates_list = [4, 8, 16, 32, 64, 128, 256]

    parallels_max_list = []
    # gates_list = [2]
    for gates in tokyo_gates_list:
        # 读取数据
        fpath = "parallel_result/tokyo_parallel_result_conts_" + str(gates) + ".csv"
        data = read_csv(fpath)
        print(data)
        print(type(data))

        get_parallels_max(data, gates)








