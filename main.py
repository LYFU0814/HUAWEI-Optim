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

def readChannel(sample, rb_num=N_RB, cell_num=N_cell, ue_num=N_UE, tx_num=N_TX):
    sample = np.array(sample)
    h = sample.reshape((((rb_num, cell_num, ue_num, cell_num, 2*tx_num))))
    return h

def generateChMtx(): #生成样例的信道矩阵
    cur = time.time()
    sample = readSample()
    print("sample shape : " + str(sample.shape))
    print("sample[0] shape : " + str(sample[0].shape))
    h=readChannel(sample)
    print(len(sample))
    end = time.time()
    print("Time : " + str(end - cur))

    #h_k_m_n = h[m][n][k][n]
    #RB-Cell-UE-Cell
    # print(h[14][3][89][3])
    return h

# print(readSample(1))


def readScenario():
    pass


def useful_signal_power(k):
    # x_mk
    pass


def getSINR(phi, h):
    # a = np.array([[complex(1, -1), 3], [2, complex(1, 1)]])
    # print(a)
    # print("矩阵2的范数")
    # print(np.linalg.norm(a, ord=2))  # 计算矩阵2的范数
    # print("矩阵1的范数")
    # print(np.linalg.norm(a, ord=1))  # 计算矩阵1的范数
    # print("矩阵无穷的范数")
    # print(np.linalg.norm(a, ord=np.inf))

    SINR = [] #所有用户的SINR
    for cell in range(1, N_cell):
        for rb in range(1, N_RB):
            ues = phi[rb][cell]
            H_m_n = []
            for k in list(ues.keys()):
                h_k_m_n = h[rb][cell][k][cell]
                H_m_n.append(h_k_m_n)
            H_m_n = np.array(H_m_n)
            H_m_n_H = np.transpose(np.conjugate(H_m_n))
            temp = np.linalg.inv(np.dot(H_m_n, H_m_n_H))
            w_m_n = np.dot(H_m_n_H, temp) / (np.linalg.norm(np.dot(H_m_n_H, temp)))

            for k, p_k in ues.items():
                idx_k = list(ues.keys()).index(k)
                #有用信号功率
                CalcSignal = ((abs(np.dot(H_m_n[idx_k].reshape(1,128), np.transpose(w_m_n)[idx_k].reshape(128, 1)) * np.sqrt(p_k)))**2)[0][0]
                #配对用户间干扰
                CalcMuInterf = 0
                for l, p_l in ues.items():
                    if l == k:
                        continue
                    else:
                        idx_l = list(ues.keys()).index(k)
                        CalcMuInterf += ((abs(np.dot(H_m_n[idx_l].reshape(1,128), np.transpose(w_m_n)[idx_l].reshape(128, 1)) * np.sqrt(p_k)))**2)[0][0]
                #邻小区同RB上所有配对用户的小区间干扰信号功率
                CalcCellInterf = 0
                for cell_interf in range(1, N_cell):
                    if cell_interf == cell:
                        continue
                    else:
                        ues_cell_interf = phi[rb][cell_interf]
                        H_m_n2 = []
                        for k in list(ues_cell_interf.keys()):
                            h_k_m_n2 = h[rb][cell_interf][k][cell_interf]
                            H_m_n2.append(h_k_m_n2)
                        H_m_n2 = np.array(H_m_n2)
                        H_m_n_H2 = np.transpose(np.conjugate(H_m_n2))
                        temp2 = np.linalg.inv(np.dot(H_m_n2, H_m_n_H2))
                        w_m_n2 = np.dot(H_m_n_H2, temp2) / (np.linalg.norm(np.dot(H_m_n_H2, temp2)))

                        for l2, p_l2 in ues.items():
                            idx_l2 = list(ues_cell_interf.keys()).index(l2)
                            CalcCellInterf += ((abs(np.dot(H_m_n2[idx_l2].reshape(1, 128),
                                                     np.transpose(w_m_n2)[idx_l2].reshape(128, 1)) * np.sqrt(p_l2))) ** 2)[ 0][0]
                #SINR
                CalcSinr = CalcSignal / (CalcMuInterf + CalcCellInterf + sigma)
                SINR.append(CalcSinr)

        print()
    return SINR


def start():
    for sample_id in range(1, N_sample+1):
        h = generateChMtx()

        phi = [] #最终分配结果矩阵,行表示RB,列表示Cell
        for r in range (0, N_RB):
            phi.append([])
            for c in range(0, N_cell):
                phi[r].append({1: 0.3, 2: 0.4, 3: 0.2, 10: 0.1})  # 频谱分配结果 {id:w}

        SINR = getSINR(phi, h)
        print()


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


        break


start()