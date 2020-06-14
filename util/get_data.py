import os
import pandas as pd
from util.process_content import *
from data_config import *


# 返回参数：
#   total_line: 该文件语句总行数
#   data: 符合接口标准数据
def get_curr_data(curr_path):
    # columns
    columns_path = os.path.join(curr_path, 'componentinfo.txt')
    total_line, concrete_columns = process_content(columns_path)

    # print(len(concrete_columns))
    # 特征矩阵
    feature_path = os.path.join(curr_path, 'covMatrix.txt')
    feature_data = process_coding(feature_path)
    feature_data = process_feature_data(feature_data)
    # print(len(feature_data[0]))
    # TODO: 需要判断columns的长度是否和特征矩阵匹配
    try:
        feature_df = pd.DataFrame(feature_data, columns=concrete_columns[:-1])    # DataFrame
    except:
        print("创建特征矩阵失败，请检查columns和data的匹配")
        exit()

    # 标签
    label_path = os.path.join(curr_path, 'error.txt')
    label_data = process_coding(label_path)

    label_data = process_label_data(label_data)
    label_df = pd.DataFrame(label_data, columns=['error'])   # DataFrame

    # 合并数据
    data = pd.concat([feature_df, label_df], axis=1)

    return total_line, data


def get_corr(path):
    all_df_dict = dict()

    for method in method_list:
        file_name = method + ".txt"
        corr = process_coding(os.path.join(path, file_name))
        corr = process_corr_data(corr)
        all_df_dict[method] = pd.DataFrame(corr, columns=["line_num", method])

    fault_line_data = process_coding(os.path.join(path, "faultLine.txt"))
    fault_line_data = process_fault_line_data(fault_line_data)

    return all_df_dict, fault_line_data

# 从rank.txt和total_line.txt文件中读取数据
def get_rank_total_line_data(path):

    total_line_raw_data = process_coding(os.path.join(path, "total_line.txt"))
    total_line_data = process_total_line_data(total_line_raw_data)

    rank_raw_data = process_coding(os.path.join(path, "rank.txt"))
    rank_data = process_rank_data(rank_raw_data)

    # data类型为ndarray
    # data = np.hstack((rank_data,total_line_data.reshape(len(total_line_data),-1)))

    return rank_data,total_line_data


# 从rank_percent.txt文件中读取数据
def get_rank_percent_data(path):
    return pd.read_csv(path, header=0, sep="\s+")
