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
            sample.append(np.fromstring(line.decode().strip(), dtype=float, sep=' '))
            if line_count == 0:
                break
    return sample


for i in range(1, 11):
    cur = time.time()
    sample = readSample()
    print(len(sample))
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
        sample = readSample(sample_id)
        RB = []  # 单小区频谱数
        RB[0] = {1: 0.5, 2: 0.4, 3: 0, 10: 0.1}  # 频谱分配结果 {id:w}
