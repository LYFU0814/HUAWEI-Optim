import time

# import numpy as np
import numpy as np

from parameter import *
import os
# from balanceO import balanceO
# input0 = np.genfromtxt("InputData0.csv", delimiter=" ", dtype=float)
# input0 = np.array(np.loadtxt("InputData0.csv", dtype=str, delimiter=' ', usecols=1, encoding='utf-8'))
# print(input0.shape)


readSize = 0

def my_dot(arrA, arrB, bit=32):
    for i in range(0, arrA.shape[0]):
        arrA[i] = np.round(arrA[i].real, bit) + 1j*np.round(arrA[i].imag, bit)
    for i in range(0, arrB.shape[0]):
        arrB[i] = np.round(arrB[i].real, bit) + 1j*np.round(arrB[i].imag, bit)
    return np.dot(np.array(arrA), np.array(arrB))

def readSample(line_count=sample_n_row * N_sample, file=input_file):
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


def writeResult(result):
    with open(OutputResultFileName, mode='w', encoding='utf-8') as file:
        # file.truncate()
        for sample_id in range(0, N_sample):
            phi = result[sample_id]
            #  write 60 row
            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    res_str = ""
                    for tup in phi[rb][cell][rb]:
                        res_str += str(tup[0]) + " "
                    res_str = res_str.strip(" ")
                    file.write(res_str + "\n")

        for sample_id in range(0, N_sample):
            phi = result[sample_id]
            #  write 60 * 6 row
            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    # res_str = ""
                    for tup in phi[rb][cell][rb]:
                        res_str = "" + str(tup[1])
                        res_str = res_str.strip()
                        file.write(res_str + "\n")


def readChannel(sample, rb_num=N_RB, cell_num=N_cell, ue_num=N_UE, tx_num=N_TX):
    # sample = np.array(sample)
    h = sample.reshape((rb_num, cell_num, ue_num, cell_num, tx_num))

    return h


def generateChMtx(sample_all, sample_id):  # 生成样例的信道矩阵
    cur = time.time()
    read_sample = sample_all[(sample_id - 1) * sample_n_row: sample_id * sample_n_row]
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
            phi_m_n = phi[rb][cell][rb]
            H_m_n = []
            for j in range(0, len(phi_m_n)):
                k = int(phi_m_n[j][0] - cell * N_UE) - 1
                h_k_m_n = h[rb][cell][k][cell]
                H_m_n.append(h_k_m_n)
            H_m_n = np.array(H_m_n)
            H_m_n_H = np.transpose(np.conjugate(H_m_n))
            temp = np.linalg.pinv(np.matmul(H_m_n, H_m_n_H))
            numerator = np.matmul(H_m_n_H, temp)
            w_m_n = numerator / np.linalg.norm(numerator)  # 默认ord=fro

            for idx_k in range(0, len(phi_m_n)):
                k, p_k = phi_m_n[idx_k][0], phi_m_n[idx_k][1]
                # idx_k = list(ues.keys()).index(k)
                # 有用信号功率
                CalcSignal = ((abs(np.matmul(H_m_n[idx_k].reshape(1, N_TX),
                                          np.transpose(w_m_n)[idx_k].reshape(N_TX, 1)) * np.sqrt(p_k))) ** 2)[0][0]
                # CalcSignal = np.linalg.norm((np.dot(H_m_n[idx_k].reshape(1, N_TX),
                #                           np.transpose(w_m_n)[idx_k].reshape(N_TX, 1)) * np.sqrt(p_k)), 2)

                # 配对用户间干扰
                CalcMuInterf = 0
                for idx_l in range(0, len(phi_m_n)):
                    l, p_l = phi[rb][cell][rb][idx_l][0], phi[rb][cell][rb][idx_l][1]
                    if idx_l == idx_k:
                        continue
                    else:
                        # tmph = H_m_n[idx_k]
                        # tmp_weight2 = np.transpose(w_m_n)[idx_l]
                        CalcMuInterf += ((abs(my_dot(H_m_n[idx_k].reshape(1, N_TX),
                                                     np.transpose(w_m_n)[idx_l].reshape(N_TX, 1)) * np.sqrt(p_l)))** 2)[0][0]
                        # CalcMuInterf += np.linalg.norm((np.dot(H_m_n[idx_k].reshape(1, N_TX),
                        #                              np.transpose(w_m_n)[idx_l].reshape(N_TX, 1)) * np.sqrt(p_l)), 2)

                # 邻小区同RB上所有配对用户的小区间干扰信号功率
                CalcCellInterf = 0
                for cell_interf in range(0, N_cell):
                    if cell_interf == cell:
                        continue
                    else:
                        ues_cell_interf = phi[rb][cell_interf][rb]
                        H_m_n2_w = []
                        H_m_n2_h = []
                        for j2 in range(0, len(phi_m_n)):  # 用户标号是从哪里开始
                            # k = ues_cell_interf[j][0]
                            k2 = int(ues_cell_interf[j2][0] - cell_interf * N_UE) - 1
                            h_k_m_n2 = h[rb][cell_interf][k2][cell_interf]
                            h_k_m_n2_h = h[rb][cell][k2][cell_interf]
                            H_m_n2_w.append(h_k_m_n2)
                            H_m_n2_h.append(h_k_m_n2_h)
                        H_m_n2_w = np.array(H_m_n2_w)
                        H_m_n2_h = np.array(H_m_n2_h)
                        H_m_n_H2_w = np.transpose(np.conjugate(H_m_n2_w))
                        temp2 = np.linalg.inv(np.dot(H_m_n2_w, H_m_n_H2_w))
                        w_m_n2 = np.dot(H_m_n_H2_w, temp2) / (np.linalg.norm(np.dot(H_m_n_H2_w, temp2)))

                        for idx_l2 in range(0, N_layer):
                            l2, p_l2 = ues_cell_interf[idx_l2][0], ues_cell_interf[idx_l2][1]
                            # CalcCellInterf += ((abs(np.dot(H_m_n2[idx_k].reshape(1, N_TX),
                            #                                  np.transpose(w_m_n2)[idx_l2].reshape(N_TX, 1)) * np.sqrt(p_l2))) ** 2)[0][0]
                            CalcCellInterf += (np.linalg.norm((np.dot((H_m_n2_h[idx_k].reshape(1, N_TX)),
                                                             (np.transpose(w_m_n2)[idx_l2].reshape(N_TX, 1))) * np.sqrt(p_l2)), 2))**2
                            # print()
                # SINR
                CalcSinr = CalcSignal / (CalcMuInterf + CalcCellInterf + sigma)
                SINR.append(CalcSinr)

        # print()
    return SINR

def init():
    phi = []  # 最终分配结果矩阵, 行表示RB, 列表示Cell
    user = np.arange(1, N_UE * N_cell + 1).reshape((N_cell, N_RB, 6))
    for rb in range(0, N_RB):
        phi.append([])
        for cell in range(0, N_cell):
            res = []
            for rbi in range(0, N_RB):
                resu = []
                # alc = [round(((1 - 0.005) * np.random.random() + 0.005), 3) for i in range(0, 5)] #random.random()生成[0,1)区间
                alc = [((1 - 0.005) * np.random.random() + 0.005) for i in range(0, 5)] #random.random()生成[0,1)区间
                alc.append(0)
                alc = sorted(alc)
                alc.append(1)
                for i in range(1, 7):
                    resu.append((user[cell][rbi][i-1], round(alc[i] - alc[i-1] + 0.001, 3)))
                    # resu.append((user[cell][rbi][i-1],1/6))
                res.append(resu)
            phi[rb].append(res)

            # phi[rb].append([[(i, 0.166) for i in user[cell][rbi]] for rbi in range(0, N_RB)])  # 频谱分配结果 {id:p}
            # phi[rb].append(dict.fromkeys(range(rb * 6, rb * 6 + 6), 1/6))  # 频谱分配结果 {id:p}
            #
    return phi

def balanceO(phi, SINR_mtx, cell):
    total = SINR_mtx.sum(axis=1)
    row1, row2 = total.argmax(), total.argmin()
    # row1 = total.argmin()  # rb
    # row2 = np.array([total[i] for i in range(0, len(total)) if i != row1]).argmin()  # rb
    col1 = np.argmax(SINR_mtx[row1])
    col2 = np.argmin(SINR_mtx[row2])
    phi[row1][cell][row1][col1], phi[row2][cell][row2][col2] = (phi[row2][cell][row2][col2][0],phi[row1][cell][row1][col1][1]), \
                                                               (phi[row1][cell][row1][col1][0],phi[row2][cell][row2][col2][1])


def start():
    beg = time.time()
    # sample[(cell-1)*90] ~ sample[cell*90-1]
    result = []  # 10 * phi
    sample_all = readSample()
    for sample_id in range(1, N_sample + 1):
        h = generateChMtx(sample_all, sample_id)

        phi = init()
        SINR = getSINR(phi, h)
        SINR_mtx = np.array(SINR).reshape((4, 15, 6))

        for i in range(20):
            for cell in range(0, N_cell):
                # SINR_mtx = np.array(SINR).reshape((4, 15, 6))
                # min = np.min(SINR_mtx[0][0])
                # print(min)
                balanceO(phi, SINR_mtx[cell], cell)
                # SINR = getSINR(phi, h)
                # print(phi[0][0][0])

            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    max_idx = np.argmax(SINR_mtx[cell][rb])
                    min_idx = np.argmin(SINR_mtx[cell][rb])
                    if max_idx == min_idx:
                        min_idx = np.random.choice([i for i in range(0, N_layer) if i != max_idx])

                    if phi[rb][cell][rb][max_idx][1] - 0.02 > 0:
                        # phi[rb][cell][rb][min_idx] = (phi[rb][cell][rb][min_idx][0], round(phi[rb][cell][rb][min_idx][1] + 0.01, 3))
                        # phi[rb][cell][rb][max_idx] = (phi[rb][cell][rb][max_idx][0], round(phi[rb][cell][rb][max_idx][1] - 0.01, 3))
                        phi[rb][cell][rb][min_idx] = (phi[rb][cell][rb][min_idx][0], phi[rb][cell][rb][min_idx][1] + 0.01)
                        phi[rb][cell][rb][max_idx] = (phi[rb][cell][rb][max_idx][0], phi[rb][cell][rb][max_idx][1] - 0.01)


            SINR = getSINR(phi, h)
            SINR_mtx = np.array(SINR).reshape((4, 15, 6))

            min = np.min(SINR_mtx)
            print(min)

        for i in range(15):
            print(phi[i][0][i])
        # break

        result.append(phi)

    writeResult(result)
    end = time.time()
    print("total time: " ,end - beg)
start()
