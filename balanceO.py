"""
python版优化函数
"""

from itertools import count
import numpy as np
import time
from generate_data import generate_int_data, generate_double_data


def balanceO(phi, SINR_mtx, shape, cell, diff_expected=0.001, max_exchange=1000000):
    rows, cols, concav = shape
    total = SINR_mtx.sum(axis=1)

    begin_time = time.time()
    total_exchange = 0
    for epoch in count(1, step=1):
        diff_begin = total.ptp()
        if diff_begin < diff_expected:
            print('=' * 80)
            print('极差已达标，成功优化!')
            break

        row1, row2 = total.argmax(), total.argmin()
        bak = total[[row1, row2]]
        valid_col1 = cols - (row1 >= concav)  # 最大行有效数据个数
        valid_col2 = cols - (row2 >= concav)  # 最小行有效数据个数
        diff_end = diff_begin
        for i in range(max_exchange):
            col1 = np.argmax(SINR_mtx[row1])
            col2 = np.argmin(SINR_mtx[row2])
            phi[row1][cell][row1][col1], phi[row2][cell][row2][col2] = phi[row2][cell][row2][col2], phi[row1][cell][row1][col1]

            diff_end += (SINR_mtx[row1, col1] - SINR_mtx[row2, col2]) * 2
            if abs(diff_end) < diff_begin:
                temp = total[row1] + total[row2]
                total[row1] = (temp + diff_end) / 2
                total[row2] = (temp - diff_end) / 2
                if epoch % 1000 == 0 or i > 100000:
                    print(f'轮次：{epoch:<8}， 交换次数：{i + 1:<10}， 初始差距{diff_begin:<8}， 结束差距：{abs(diff_end):<8}')
                total_exchange += i + 1
                break
        else:
            total[[row1, row2]] = bak
            print('超过最大允许交换数，未达到优化目标!')
            total_exchange += max_exchange
            break

    print('最终极差为：', total.ptp())
    low = SINR_mtx.min()
    high = SINR_mtx.max()
    # low = min(SINR_mtx[:concav].min(), SINR_mtx[concav:, :-1].min())
    # high = max(SINR_mtx[:concav].max(), SINR_mtx[concav:, :-1].max())
    elapsed_time = time.time() - begin_time
    print(
        f'{rows}行{cols}列的矩阵，其元素在{low}和{high}之间,优化{epoch - 1}轮，累计交换{total_exchange}次，耗时{elapsed_time:.2f}秒,平均每毫秒交换{total_exchange / (elapsed_time * 1000):.0f}次')

