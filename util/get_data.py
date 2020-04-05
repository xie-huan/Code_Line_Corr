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
        feature_df = pd.DataFrame(feature_data,columns=concrete_columns)    # DataFrame
    except:
        print("创建特征矩阵失败，请检查columns和data的匹配")
        exit()

    # 标签
    label_path = os.path.join(curr_path, 'error.txt')
    label_data = process_coding(label_path)

    label_data = process_label_data(label_data)
    label_df = pd.DataFrame(label_data,columns=['error'])   # DataFrame

    # 合并数据
    data = pd.concat([feature_df, label_df], axis=1)

    return data


def get_corr(path):
    pearson_corr = process_coding(os.path.join(path, "pearson.txt"))
    pearson_corr = process_corr_data(pearson_corr)
    pearson_corr = pd.DataFrame(pearson_corr, columns=["line_num", "pearson_corr"])

    spearman_corr = process_coding(os.path.join(path, "spearman.txt"))
    spearman_corr = process_corr_data(spearman_corr)
    spearman_corr = pd.DataFrame(spearman_corr, columns=["line_num", "pearson_corr"])

    kendall_corr = process_coding(os.path.join(path, "kendall.txt"))
    kendall_corr = process_corr_data(kendall_corr)
    kendall_corr = pd.DataFrame(kendall_corr, columns=["line_num", "pearson_corr"])

    fault_line_data = process_coding(os.path.join(path, "faultLine.txt"))
    fault_line_data = process_fault_line_data(fault_line_data)

    return pearson_corr,spearman_corr,kendall_corr,fault_line_data


def get_rank_percent_data(path):
    return pd.read_csv(path, header=0, sep="\s+")
