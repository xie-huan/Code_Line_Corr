import numpy as np
from data_config import *

# 参数：
# all_df_dict：表示所有数据字典
# fault_line_data：实际错误的代码行，从faultLine.txt中提取
# method_list：具体使用算法列表
def calc_rank(all_df_dict, fault_line_data, method_list):

    real_fault_line_data = list()  # 实际代码行

    # 得到所有代码行
    real_line_data = all_df_dict[method_list[0]]['line_num'].tolist()
    for line in fault_line_data:
        if line in real_line_data:
            real_fault_line_data.append(line)
        else:
            real_fault_line_data.extend(find_closest_num(real_line_data, line))
    real_fault_line_data = list(set(real_fault_line_data))

    # 处理排名并列的情况，以及虽然有排名，但是值却为nan的情况
    for method in method_list:
        concrete_df = all_df_dict[method]
        temp_df = concrete_df[concrete_df["line_num"].isin(real_fault_line_data)]

        # # 处理排名并列情况
        # val_list = temp_df[method].values.tolist()
        # correct_df = concrete_df[concrete_df[method].isin(val_list)][method]
        # rank = correct_df.index.values[0]
        # val = correct_df.loc[rank]

        # 不处理排名并列情况
        rank = temp_df.index.values[0]
        val = temp_df[method].values[0]

        if val != val:
            result_dict[method] = np.nan
        else:
            result_dict[method] = rank + 1
    return result_dict


# faultLine中的行可能对应不了具体的行数，则选择最近的代码行
def find_closest_num(real_line_data, target):
    target = int(target)
    line_data_np = np.array(real_line_data, dtype=int)
    min_diff_val = min(abs(line_data_np-target))

    if int(target + min_diff_val) in real_line_data and int(target - min_diff_val) in real_line_data:
        return list([int(target + min_diff_val), int(target - min_diff_val)])
    if int(target + min_diff_val) in real_line_data:
        return list([int(target + min_diff_val)])
    if int(target - min_diff_val) in real_line_data:
        return list([int(target - min_diff_val)])
