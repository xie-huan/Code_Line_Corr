import os
import pandas as pd
from util.process_content import *


def get_curr_data(curr_path):
    # columns
    columns_path = os.path.join(curr_path, 'componentinfo.txt')
    concrete_columns = process_content(columns_path)

    # print(len(concrete_columns))
    # 特征矩阵
    feature_path = os.path.join(curr_path, 'covMatrix.txt')
    feature_data = process_coding(feature_path)
    feature_data = process_feature_data(feature_data)
    # print(len(feature_data[0]))
    # TODO: 需要判断columns的长度是否和特征矩阵匹配
    try:
        feature_df = pd.DataFrame(feature_data, columns=concrete_columns)    # DataFrame
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

    return data


def get_corr(path):
    all_df_dict = dict()
    method_list = ["pearson",
                   "spearman",
                   "kendall",
                   "chisquare",
                   "mutual_information",
                   "fisher_score",
                   "dstar",
                   "ochiai",
                   "barinel"]
    for method in method_list:
        file_name = method + ".txt"
        corr = process_coding(os.path.join(path, file_name))
        corr = process_corr_data(corr)
        all_df_dict[method] = pd.DataFrame(corr, columns=["line_num", method])

    fault_line_data = process_coding(os.path.join(path, "faultLine.txt"))
    fault_line_data = process_fault_line_data(fault_line_data)

    return all_df_dict, fault_line_data


def get_rank_percent_data(path):
    return pd.read_csv(path, header=0, sep="\s+")
