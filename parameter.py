import numpy as np

root_dir = "./"
# 输出文件格式
index = 0
RealTeamID = "hw16273433"
OutputResultFileName = root_dir + "Result_{}_{}.csv".format(index, RealTeamID)
# 输入文件格式
input_file = root_dir + "InputData{}.csv".format(index)
scenario_file = root_dir + "ScenarioData{}.csv".format(index)
param = np.array(np.loadtxt(scenario_file, dtype=int, usecols=0))
# 训练参数
N_sample = param[6]  # 样例数, default: 10
N_cell = param[0]  # 小区个数 / 样例, default: 4
# 一个小区的属性
N_UE = param[3]      # 用户个数 / 小区, default: 90
N_TX = param[1]      # 发射天线数 / 小区, default: 64
N_RB = param[2]      # 频谱个数, default: 15
N_layer = param[5]   # 每个RB上的最大用户数, default: 6
# 用户属性
N_RX = param[4]   # 天线数 / 用户, default: 1
# 天线属性
sigma = param[7]   # 底噪功率, default: 1

sample_n_row = N_cell * N_UE * N_RB
sample_n_col = N_TX * 2 * N_cell

