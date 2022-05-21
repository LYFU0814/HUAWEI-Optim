import numpy as np
root_dir = "./"
# 输出文件格式
index = 0
RealTeamID = "hw16273433"
OutputResultFileName = root_dir + "Result_{}_{}.csv".format(index, RealTeamID)
# 输入文件格式
input_file = root_dir + "InputData{}.csv".format(index)
scenario_file = root_dir + "ScenarioData{}.csv".format(index)
# 训练参数
N_sample = 10  # 样例数
N_cell = 4  # 小区个数 / 样例
# 一个小区的属性
N_UE = 90      # 用户个数 / 小区
N_TX = 64      # 发射天线数 / 小区
N_RB = 15      # 频谱个数
# 用户属性
N_RX = 1   # 天线数 / 用户
# 天线属性
sigma = 1

sample_n_row = N_cell * N_UE * N_RB
sample_n_col = N_TX * 2 * N_cell

