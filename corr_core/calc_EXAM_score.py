import numpy as np


def calc_EXAM(rank_data, total_line_data):
    # 计算百分比
    result = np.round(rank_data/total_line_data.reshape(len(total_line_data),-1), decimals=3)*100
    result[np.isnan(result)] = 1
    return result
