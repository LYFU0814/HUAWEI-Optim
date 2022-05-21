import time

import numpy as np
import pandas as pd
import parameter
from parameter import *
import os

# input0 = np.genfromtxt("InputData0.csv", delimiter=" ", dtype=float)
# input0 = np.array(np.loadtxt("InputData0.csv", dtype=str, delimiter=' ', usecols=1, encoding='utf-8'))
# print(input0.shape)


readSize = 0
def readSample(line_count=sample_n_row, file=input_file):
    global readSize
    fileSize = os.path.getsize(file)
    sample = []
    with open(input_file, 'rb') as fp:  # to use seek from end, must use mode 'rb'
        offset = readSize  # initialize offset
        while offset < fileSize:
            fp.seek(offset, os.SEEK_SET)  # 移动指针
            line = fp.readline()
            readSize += len(line)
            offset += len(line)
            line_count -= 1
            values = np.fromstring(line.decode().strip(), dtype=float, sep=' ')
            sample.append(values.reshape(int(len(values) / 2), 2))
            if line_count == 0:
                break
    return np.array(sample)


for i in range(1, 11):
    cur = time.time()
    sample = readSample()
    print("sample shape : " + str(sample.shape))
    print("sample[0] shape : " + str(sample[0].shape))
    end = time.time()
    print("Time : " + str(end - cur))


# print(readSample(1))


def readScenario():
    pass


def useful_signal_power(k):
    # x_mk
    pass


def getSINR(k):
    a = np.array([[complex(1, -1), 3], [2, complex(1, 1)]])
    print(a)
    print("矩阵2的范数")
    print(np.linalg.norm(a, ord=2))  # 计算矩阵2的范数
    print("矩阵1的范数")
    print(np.linalg.norm(a, ord=1))  # 计算矩阵1的范数
    print("矩阵无穷的范数")
    print(np.linalg.norm(a, ord=np.inf))


def start():
    for sample_id in range(1, N_sample):
        sample = readSample()  # 行：360个用户，4个小区，15条信道；列：64*4条信道
        village = []
        village.append([])  # 单小区频谱数
        village[0][0] = [0.5, 0, 0.4, 0.1]  # 频谱分配结果 {id:w}
        k, m, n = 1, 2, 3
        #   RB数：1 <= m <= 15    小区数：1 <= n <= 4  四个小区的所有RB， 用户数：1 <= k <= 90
        #  h : 某个用户接收64个信道的信道系数，是个1*64的向量（也就是一个小区范围内），sample一行有4个小区的h
        #  w ：信道分配给用户k的加权系数矢量 64 * 1
        #  pk：用户功率值
        # sample[(cell-1)*90] ~ sample[cell*90-1]
        h = village[n-1][m-1][k-1] = sample[0] # 1*64

