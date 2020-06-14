import os
from data_config import *

# def sort_dict(corr_dict):
#     num_dict = dict()
#     nan_dict = dict()
#     for line_num, corr in corr_dict.items():
#         if corr == 'nan':
#             nan_dict[line_num] = corr
#         else:
#             num_dict[line_num] = corr
#
#     num_dict = sorted(num_dict.items(), key=lambda x: x[1], reverse=True)
#
#     return corr_dict


def write_corr_to_txt(method, corr_dict, path):
    print("     将结果写入txt文档中......"+"方法："+method)

    # 在写入txt文档之前，先对相关系数进行排序
    corr_dict.sort_values(ascending=False, inplace=True)

    # 计算存储路径
    path = os.path.dirname(path)
    res_file_name = method + ".txt"
    concrete_path = os.path.join(path, res_file_name)

    # 写入数据
    with open(concrete_path, 'w') as f:
        for d in corr_dict.keys():
            print(d+"  "+str(corr_dict.get(d)), file=f)

    print("     写入完成。")


def write_rank_to_txt(path, rank_dict):
    print("     将faultLine中的排名写入txt文档中......")

    # file_name = "rank.txt"
    # concrete_path = os.path.join(os.path.dirname(path), file_name)

    with open(concrete_path, 'a') as f:
        # for method, rank in rank_dict.items():
        #     print(method+" "+str(rank), file=f)
        print(path.split("\\")[-2], file=f)
        table_head = ""
        for method in rank_dict.keys():
            table_head += method[:3] + "\t"
        value = ""
        for v in rank_dict.values():
            value += str(v) + "\t"
        print(table_head, file=f)
        print(value, file=f)
        print("", file=f)

    print("     写入完成。路径：" + concrete_path)


def write_rank_percent_to_txt(EXAM_score):

    print("将rank_percent写入txt中")
    for row in EXAM_score:
        content = ""
        for percent in row:
            content += str(percent) +" "
        with open(rank_percent_path,'a') as f:
            print(content, file=f)
    print("写入完成\n")
