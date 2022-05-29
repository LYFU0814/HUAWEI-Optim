import time

import numpy as np
from parameter import *
import os
from balanceO import balanceO
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
            values = np.fromstring(line.decode().strip(), dtype=float, sep=' ').reshape(N_TX*N_cell, 2).T
            combine = values[0] + 1j * values[1]
            sample.append(combine)
            if line_count == 0:
                break
    return np.array(sample)


def readChannel(sample, rb_num=N_RB, cell_num=N_cell, ue_num=N_UE, tx_num=N_TX):
    # sample = np.array(sample)
    h = sample.reshape((rb_num, cell_num, ue_num, cell_num, tx_num))
    return h


def generateChMtx():  # 生成样例的信道矩阵
    cur = time.time()
    read_sample = readSample()
    print("read_sample shape :" + str(read_sample.shape))
    sample = readChannel(read_sample)  # 行：360个用户，4个小区，15条信道；列：64*4条信道
    print("sample shape :" + str(sample.shape))
    end = time.time()
    print("ReadTime : " + str(end - cur))
    #   RB数：1 <= m <= 15    小区数：1 <= n <= 4  四个小区的所有RB， 用户数：1 <= k <= 90
    #  h : 某个用户接收64个信道的信道系数，是个1*64的向量（也就是一个小区范围内），sample一行有4个小区的h
    #  w ：信道分配给用户k的加权系数矢量 64 * 1
    #  pk：用户功率值
    # h_k_m_n = h[m][n][k][n]
    # RB-Cell-UE-Cell
    # print(h[14][3][89][3])
    print(sample[0][0][0][1][0])
    return sample


# print(readSample(1))


def readScenario():
    pass


def useful_signal_power(k):
    # x_mk
    pass


def getSINR(phi, h):

    SINR = []  # 所有用户的SINR 90个
    for cell in range(0, N_cell):  # n
        for rb in range(0, N_RB):  # m  # k 用户
            ues = phi[rb][cell]
            H_m_n = []
            for j in range(0, N_layer):
                k = phi[rb][cell][rb][j][0]
                h_k_m_n = h[rb][cell][j][cell]
                H_m_n.append(h_k_m_n)
            H_m_n = np.array(H_m_n)
            H_m_n_H = np.transpose(np.conjugate(H_m_n))
            temp = np.linalg.inv(np.dot(H_m_n, H_m_n_H))
            w_m_n = np.dot(H_m_n_H, temp) / (np.linalg.norm(np.dot(H_m_n_H, temp)))  # 默认ord=fro

            for idx_k in range(0, N_layer):
                k, p_k = phi[rb][cell][rb][idx_k][0], phi[rb][cell][rb][idx_k][1]
                # idx_k = list(ues.keys()).index(k)
                # 有用信号功率
                CalcSignal = ((abs(np.dot(H_m_n[idx_k].reshape(1, N_TX),
                                          np.transpose(w_m_n)[idx_k].reshape(N_TX, 1)) * np.sqrt(p_k))) ** 2)[0][0]
                # 配对用户间干扰
                CalcMuInterf = 0
                for idx_l in range(0, N_layer):
                    l, p_l = phi[rb][cell][rb][idx_l][0], phi[rb][cell][rb][idx_l][1]
                    if idx_l == idx_k:
                        continue
                    else:
                        CalcMuInterf += ((abs(np.dot(H_m_n[idx_k].reshape(1, N_TX),
                                                     np.transpose(w_m_n)[idx_l].reshape(N_TX, 1)) * np.sqrt(p_l))) ** 2)[0][0]
                # 邻小区同RB上所有配对用户的小区间干扰信号功率
                CalcCellInterf = 0
                for cell_interf in range(0, N_cell):
                    if cell_interf == cell:
                        continue
                    else:
                        ues_cell_interf = phi[rb][cell_interf][rb]
                        H_m_n2 = []
                        for j in range(0, N_layer):  # 用户标号是从哪里开始
                            k = ues_cell_interf[j][0]
                            h_k_m_n2 = h[rb][cell_interf][j][cell_interf]
                            H_m_n2.append(h_k_m_n2)
                        H_m_n2 = np.array(H_m_n2)
                        H_m_n_H2 = np.transpose(np.conjugate(H_m_n2))
                        temp2 = np.linalg.inv(np.dot(H_m_n2, H_m_n_H2))
                        w_m_n2 = np.dot(H_m_n_H2, temp2) / (np.linalg.norm(np.dot(H_m_n_H2, temp2)))

                        for idx_l2 in range(0, N_layer):
                            l2, p_l2 = ues_cell_interf[idx_l2][0], ues_cell_interf[idx_l2][1]
                            CalcCellInterf += ((abs(np.dot(H_m_n2[idx_k].reshape(1, N_TX),
                                                             np.transpose(w_m_n2)[idx_l2].reshape(N_TX, 1)) * np.sqrt(p_l2))) ** 2)[0][0]
                # SINR
                CalcSinr = CalcSignal / (CalcMuInterf + CalcCellInterf + sigma)
                SINR.append(CalcSinr)

        # print()
    return SINR

def init():
    phi = []  # 最终分配结果矩阵, 行表示RB, 列表示Cell
    user = np.arange(0, N_UE * N_cell).reshape((N_cell, N_RB, 6))
    for rb in range(0, N_RB):
        phi.append([])
        for cell in range(0, N_cell):
            res = []
            for rbi in range(0, N_RB):
                resu = []
                alc = [round(np.random.random(), 3) for i in range(0, 5)]
                alc.append(0)
                alc = sorted(alc)
                alc.append(1)
                for i in range(1, 7):
                    resu.append((user[cell][rbi][i-1], round(alc[i] - alc[i-1], 3)))
                res.append(resu)
            phi[rb].append(res)

            # phi[rb].append([[(i, 0.166) for i in user[cell][rbi]] for rbi in range(0, N_RB)])  # 频谱分配结果 {id:p}
            # phi[rb].append(dict.fromkeys(range(rb * 6, rb * 6 + 6), 1/6))  # 频谱分配结果 {id:p}
            #
    return phi

def start():
    # sample[(cell-1)*90] ~ sample[cell*90-1]
    for sample_id in range(1, N_sample + 1):
        h = generateChMtx()

        phi = init()
        SINR = getSINR(phi, h)
        SINR_mtx = np.array(SINR).reshape((4, 15, 6))



        for i in range(100):
            for cell in range(0, N_cell):
                # SINR_mtx = np.array(SINR).reshape((4, 15, 6))
                # min = np.min(SINR_mtx[0][0])
                # print(min)
                balanceO(phi, SINR_mtx[0], (15, 6, 0), cell)
                # SINR = getSINR(phi, h)
                # print(phi[0][0][0])

            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    max_idx = np.argmax(SINR_mtx[cell][rb])
                    min_idx = np.argmin(SINR_mtx[cell][rb])
                    if max_idx == min_idx:
                        min_idx = np.random.choice([i for i in range(0, N_layer) if i != max_idx])

                    phi[rb][cell][rb][min_idx] = (phi[rb][cell][rb][min_idx][0], round(phi[rb][cell][rb][min_idx][1] + 0.005, 3))
                    phi[rb][cell][rb][max_idx] = (phi[rb][cell][rb][max_idx][0], round(phi[rb][cell][rb][max_idx][1] - 0.005, 3))


            # for i in range(15):
            #     print(phi[i][0][i])

            SINR = getSINR(phi, h)
            SINR_mtx = np.array(SINR).reshape((4, 15, 6))

            min = np.min(SINR_mtx[0][0])
            print(min)

        for i in range(15):
            print(phi[i][0][i])
           # break
        break

start()
