import copy
import time

import numpy as np
from numpy import transpose, dot, squeeze, conjugate, sqrt
from numpy.linalg import pinv, norm

from parameter import *
import os

readSize = 0


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
            values = np.fromstring(line.decode().strip(), dtype=float, sep=' ').reshape(N_TX * N_cell, 2).T
            combine = values[0] + 1j * values[1]
            sample.append(combine)
            if line_count == 0:
                break
    return np.array(sample)


def writeResult(result):
    with open(OutputResultFileName, mode='w', encoding='utf-8') as file:
        file.truncate()
        for sample_id in range(0, N_sample):
            phi = result[sample_id]
            #  write 60 row
            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    res_str = ""
                    for tup in phi[cell][rb]:
                        res_str += str(tup[0]) + " "
                    res_str = res_str.strip(" ")
                    file.write(res_str + "\n")

        for sample_id in range(0, N_sample):
            phi = result[sample_id]
            #  write 60 * 6 row
            for cell in range(0, N_cell):
                for rb in range(0, N_RB):
                    for tup in phi[cell][rb]:
                        res_str = "" + str(tup[1])
                        res_str = res_str.strip()
                        file.write(res_str + "\n")


def readChannel(sample, rb_num=N_RB, cell_num=N_cell, ue_num=N_UE, tx_num=N_TX):
    h = sample.reshape((rb_num, ue_num * cell_num, cell_num, tx_num))
    return h


def generateChMtx(sample_all, sample_id):  # 生成样例的信道矩阵
    cur = time.time()
    read_sample = sample_all[(sample_id - 1) * sample_n_row: sample_id * sample_n_row]
    # print("read_sample shape :" + str(read_sample.shape))
    sample = readChannel(read_sample)  # 行：360个用户，4个小区，15条信道；列：64*4条信道
    # print("sample shape :" + str(sample.shape))
    end = time.time()
    # print("ReadTime : " + str(end - cur))
    #   RB数：1 <= m <= 15    小区数：1 <= n <= 4  四个小区的所有RB， 用户数：1 <= k <= 90
    #  h : 某个用户接收64个信道的信道系数，是个1*64的向量（也就是一个小区范围内），sample一行有4个小区的h
    #  w ：信道分配给用户k的加权系数矢量 64 * 1
    #  pk：用户功率值
    return sample


def getSINR(UePairResult, InputH):  # rb cell rb 15 * 4 * 15
    CalcWeightSet = np.empty((N_cell, N_RB, N_TX, N_layer), dtype=complex)
    for CellIdx in range(0, N_cell):
        for RbIdx in range(0, N_RB):
            MuH = []
            UePairSet = UePairResult[CellIdx][RbIdx]
            for UeIdx in range(0, len(UePairSet)):
                UeID = UePairSet[UeIdx][0] - 1
                TmpH = transpose(squeeze(InputH[RbIdx][UeID][CellIdx]))
                MuH.append(TmpH)
            MuH = np.array(MuH)
            TmpMatrix = dot(transpose(conjugate(MuH)), pinv(dot(MuH, transpose(conjugate(MuH)))))
            CalcWeightSet[CellIdx][RbIdx] = TmpMatrix / norm(TmpMatrix)  # 4 * 15 * 64 * 6

    AllUeSinrSet = np.empty((N_cell * N_UE, 1), dtype=float)
    for CellIdx in range(0, N_cell):
        for RbIdx in range(0, N_RB):
            UePairSet = UePairResult[CellIdx][RbIdx]
            for UeIdx in range(0, len(UePairSet)):
                UeID = UePairSet[UeIdx][0] - 1
                TmpH = transpose(squeeze(InputH[RbIdx][UeID][CellIdx]))
                TmpWeight = CalcWeightSet[CellIdx][RbIdx][:, UeIdx]
                CalcSignal = (abs(dot(TmpH, TmpWeight)) ** 2) * UePairSet[UeIdx][1]
                CalcMuInterf = 0
                for UeIdx2 in range(0, len(UePairSet)):
                    if UeIdx == UeIdx2:
                        continue
                    UeID2 = UePairSet[UeIdx2][0] - 1
                    TmpWeight2 = CalcWeightSet[CellIdx][RbIdx][:, UeIdx2]
                    CalcMuInterf += (abs(dot(TmpH.reshape(1, N_TX), TmpWeight2.reshape(N_TX, 1))) ** 2) * \
                                    UePairSet[UeIdx2][1]

                CalcCellInterf = 0
                for CellIdx2 in range(0, N_cell):
                    if CellIdx == CellIdx2:
                        continue
                    UePairSet2 = UePairResult[CellIdx2][RbIdx]
                    for UeIdx3 in range(0, len(UePairSet2)):
                        UeID3 = UePairSet2[UeIdx3][0] - 1
                        # TmpH2 = transpose(squeeze(InputH[RbIdx][UeID][CellIdx2]))) [CellIdx][RbIdx][UeID][CellIdx]
                        TmpH2 = transpose(squeeze(InputH[RbIdx][UeID][CellIdx2]))
                        TmpWeight3 = CalcWeightSet[CellIdx2][RbIdx][:, UeIdx3]
                        CalcCellInterf += (norm(
                            dot(TmpH2.reshape(1, N_TX), TmpWeight3.reshape(N_TX, 1)) * sqrt(UePairSet[UeIdx3][1]),
                            2)) ** 2
                CalcSinr = CalcSignal / (CalcMuInterf + CalcCellInterf + sigma)
                AllUeSinrSet[UeID] = CalcSinr
    return AllUeSinrSet


def init():
    phi = []  # 最终分配结果矩阵, 行表示RB, 列表示Cell
    user = np.arange(1, N_UE * N_cell + 1).reshape((N_cell, N_RB, 6))
    for cell in range(0, N_cell):  # 4
        phi.append([])
        for rb in range(0, N_RB):  # 15
            phi[cell].append([])
            resu = []
            # alc = [round(((1 - 0.005) * np.random.random() + 0.005), 3) for i in range(0, 5)] #random.random()生成[0,1)区间
            # alc = [((1 - 0.005) * np.random.random() + 0.005) for i in range(0, 5)]  # random.random()生成[0,1)区间
            alc = [round(np.random.random(), 4) for i in range(0, 5)]  # random.random()生成[0,1)区间
            alc.append(0)
            alc = sorted(alc)
            alc.append(1)
            for i in range(1, 7):
                # resu.append((user[cell][rb][i-1], round(alc[i] - alc[i-1] + 0.001, 4)))
                resu.append((user[cell][rb][i - 1], round(0.1666, 4)))

            phi[cell][rb] = resu
    return phi


# 交换两个用户的RB，不交换功率
def swap_ue(phi, cell, row1, col1, row2, col2):
    phi[cell][row1][col1], phi[cell][row2][col2] = (phi[cell][row2][col2][0], phi[cell][row1][col1][1]), \
                                                   (phi[cell][row1][col1][0], phi[cell][row2][col2][1])


# 保留六位小数
def integer(num, default=1000000):
    return round(int(num * default) / default, 6)


# 保留实部和虚部的小数位
def my_dot(arrA, arrB, bit=32):
    for i in range(0, arrA.shape[0]):
        arrA[i] = np.round(arrA[i].real, bit) + 1j * np.round(arrA[i].imag, bit)
    for i in range(0, arrB.shape[0]):
        arrB[i] = np.round(arrB[i].real, bit) + 1j * np.round(arrB[i].imag, bit)
    return np.dot(np.array(arrA), np.array(arrB))


def balanceO(phi, SINR_mtx, cell):
    total = SINR_mtx.sum(axis=1)
    row1, row2 = total.argmax(), total.argmin()
    # row1 = total.argmin()  # rb
    # row2 = np.array([total[i] for i in range(0, len(total)) if i != row1]).argmin()  # rb
    col1 = np.argmax(SINR_mtx[row1])
    col2 = np.argmin(SINR_mtx[row2])
    swap_ue(phi, cell, row1, col1, row2, col2)


def balance(phi, SINR_mtx, h, baseline):
    best_SINR = baseline
    best_phi = copy.deepcopy(phi)

    rank_swap_pos_list = []
    for rank in range(3):  # 表示每个RB排序后交换的用户序号
        swap_pos_list = []
        for cell in range(0, N_cell):
            swap_seq = []
            rb_pos = [i for i in range(N_RB)]
            np.random.shuffle(rb_pos)
            ue_pos = np.argsort(SINR_mtx[cell], axis=1)
            for i in range(0, N_RB - 1, 2):
                row1, row2 = rb_pos[i], rb_pos[i + 1]
                col1, col2 = ue_pos[i][rank], ue_pos[i + 1][rank]
                swap_ue(phi, cell, row1, col1, row2, col2)

                swap_seq.append((row1, col1))
                swap_seq.append((row2, col2))
            swap_pos_list.append(swap_seq)
        rank_swap_pos_list.append(swap_pos_list)
    SINR_mtx = np.array(getSINR(phi, h)).reshape((4, 15, 6))
    for swap_pos_list in rank_swap_pos_list:
        for cell in range(0, N_cell):
            for j in range(0, N_RB - 1, 2):
                row1, row2 = swap_pos_list[cell][j][0], swap_pos_list[cell][j + 1][0]
                if np.array(SINR_mtx[cell][row1]).min() < best_SINR or np.array(SINR_mtx[cell][row2]).min() < best_SINR:
                    col1, col2 = swap_pos_list[cell][j][1], swap_pos_list[cell][j + 1][1]
                    swap_ue(phi, cell, row1, col1, row2, col2)

    SINR_mtx = np.array(getSINR(phi, h)).reshape((4, 15, 6))
    min_sinr = np.min(SINR_mtx)
    if min_sinr > best_SINR:
        best_SINR = np.min(SINR_mtx)
        best_phi = copy.deepcopy(phi)
    return best_SINR, best_phi


def balance1(phi, SINR_mtx, h, baseline):
    best_phi = copy.deepcopy(phi)
    best_SINR = baseline
    for cell in range(0, N_cell):
        before = np.array(phi[cell], dtype='i,f').reshape(N_UE)
        np.random.shuffle(before)
        phi[cell] = np.array(before).reshape(N_RB, N_layer)

    SINR_mtx = np.array(getSINR(phi, h)).reshape((4, 15, 6))
    min_sinr = np.min(SINR_mtx)
    if min_sinr > best_SINR:
        best_SINR = min_sinr
        best_phi = copy.deepcopy(phi)
    return best_SINR, best_phi


def start():
    beg = time.time()
    result = []  # 10 * phi
    sample_all = readSample()
    for sample_id in range(1, N_sample + 1):
        sample_beg = time.time()
        h = generateChMtx(sample_all, sample_id)
        phi = init()
        SINR_mtx = np.array(getSINR(phi, h)).reshape((4, 15, 6))
        best_SINR = np.min(SINR_mtx)
        best_phi = copy.deepcopy(phi)
        for t in range(0, 5):
            for p in range(2):
                best_SINR, best_phi = balance1(best_phi, SINR_mtx, h, best_SINR)
                print("-----random choose best allocate： " + str(best_SINR))
            for cycle in range(8):
                best_SINR, best_phi = balance(best_phi, SINR_mtx, h, best_SINR)
                print("-----average choose best allocate： " + str(best_SINR))
        for i in range(25):
            allo_pos_list = []
            for cell in range(0, N_cell):
                allo_pos_list.append([])
                for rb in range(0, N_RB):
                    allo_pos = []
                    max_idx = np.argmax(SINR_mtx[cell][rb])
                    min_idx = np.argmin(SINR_mtx[cell][rb])
                    if max_idx == min_idx:
                        min_idx = np.random.choice([j for j in range(0, N_layer) if j != max_idx])

                    if phi[cell][rb][max_idx][1] - 0.02 > 0:
                        # phi[rb][cell][rb][min_idx] = (phi[rb][cell][rb][min_idx][0], round(phi[rb][cell][rb][min_idx][1] + 0.01, 3))
                        # phi[rb][cell][rb][max_idx] = (phi[rb][cell][rb][max_idx][0], round(phi[rb][cell][rb][max_idx][1] - 0.01, 3))
                        phi[cell][rb][min_idx] = (phi[cell][rb][min_idx][0], integer(phi[cell][rb][min_idx][1] + 0.02))
                        phi[cell][rb][max_idx] = (phi[cell][rb][max_idx][0], integer(phi[cell][rb][max_idx][1] - 0.02))

                        allo_pos.append((min_idx, max_idx))

            SINR_mtx = np.array(getSINR(phi, h)).reshape((4, 15, 6))
            min = np.min(SINR_mtx)
            # for cell in range(0, N_cell):
            #     for rb in range(0, N_RB):
            #         if np.array(SINR_mtx[cell][rb]).min() < best_SINR:
            #             allo_pos_list

            if min > best_SINR:
                best_SINR = min
                best_phi = copy.deepcopy(phi)
            print("-----choose best P： " + str(best_SINR))
        sample_end = time.time()
        print("=" * 16 + " sample_" + str(sample_id) + " end, 耗时： " + str(sample_end - sample_beg) + "  " + "=" * 16)
        # break

        result.append(best_phi)

    for i in range(0, len(result)):
        SINR = getSINR(result[i], generateChMtx(sample_all, i + 1))
        print(str(i + 1) + "   " + str(np.min(SINR)))
    writeResult(result)
    end = time.time()
    print("total time: ", end - beg)


start()
