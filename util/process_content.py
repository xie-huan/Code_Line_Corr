from util.process_coding import *
import re


# 功能：判断文件中的换行符是windows、Linux还是Mac下的换行符
def choose_newlines(content):
    newlines_list = ["\r\n", "\n", "\r"]
    for token in newlines_list:
        if token in content:
            return token
    return "\r\n"   # 默认返回windows下的换行符


# 针对componentinfo.txt，目前源数据有两种格式
# （1）
#  4
#  1 2 3 4
# （2）
#  4 1 2 3 4
# 去掉总行数，只留下测试对应的编号
def process_content(columns_path):
    columns = process_coding(columns_path)
    token = choose_newlines(columns)
    if token in columns:
        temp_content = columns.split(token)
        total_line = int(temp_content[0])
        concrete_columns = temp_content[1].split()[:total_line]
    else:
        temp_content = columns.split()
        total_line = int(temp_content[0])
        concrete_columns = columns.split()[1:total_line+1]
    return concrete_columns


# 读取特征矩阵
def process_feature_data(feature_data):

    # "\n","\r","\r\n"，不同平台生成的文件，换行符不一样，需判断后用split()
    token = choose_newlines(feature_data)
    feature_data = feature_data.split(token)

    feature_data = [feature_str.strip().split() for feature_str in feature_data]
    feature_data = [list(map(int, arr)) for arr in feature_data]
    feature_data = [[0 if a == 0 else 1 for a in elem] for elem in feature_data]

    return feature_data


# 读取标签矩阵
def process_label_data(label_data):
    token = choose_newlines(label_data)
    label_data = label_data.split(token)

    label_data = [list(map(int, arr)) for arr in label_data]
    return label_data


# 针对faultLine.txt
# 提取出其中的数字，并以list形式返回
def process_fault_line_data(fault_line_data):
    temp_data = re.findall("\"(.*?)\"", fault_line_data)[0]  # type:str
    temp_data = temp_data.strip().split()
    return list(map(int,temp_data))


# 针对pearson.txt、spearman.txt、kendall.txt
def process_corr_data(corr_data):
    token = choose_newlines(corr_data)
    corr_data = [x.strip().split() for x in corr_data.strip().split(token)]
    for elem in corr_data:
        elem[0] = int(elem[0])
        elem[1] = float(elem[1])
    return corr_data


# # 针对rank-percent.txt
# def process_rank_percent(rank_percent_data):
#     pass