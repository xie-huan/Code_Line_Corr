from util.get_data import *
from util.write_to_txt import *
from corr_core.calc_corr import *
from corr_core.calc_rank import *
from corr_core.rank_plot import *


# 递归调用搜索目录，计算相关系数
def start_calc_corr(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)  # 获取当前目录下的所有文件或目录

    for elem in curr_file_list:
        curr_path = os.path.join(pre_dir, elem)
        # 判断当前路径是目录还是文件，是目录则递归访问，是文件则计算数据，并输出到txt文档中
        if os.path.isdir(curr_path):
            start_calc_corr(curr_path)
        elif os.path.isfile(curr_path):
            # 如果不包含covMatrix.txt、error.txt、componentinfo.txt三个文件，则不进行读取数据操作
            correct_file_list = ['covMatrix.txt', 'error.txt', 'componentinfo.txt']
            if not set(correct_file_list).issubset(set(curr_file_list)):
                continue

            # 读取数据
            data = get_curr_data(os.path.dirname(curr_path))

            # 计算相关系数并输出到txt文件中
            print("计算" + os.path.dirname(curr_path) + "的相关系数:")
            method_list = ["pearson",
                           "spearman",
                           "kendall",
                           "chisquare",
                           "mutual_information",
                           "fisher_score",
                           "dstar",
                           "ochiai",
                           "barinel"]

            for concrete_method in method_list:
                concrete_corr = calc_corr_bymyself(data, concrete_method)
                write_corr_to_txt(concrete_method, concrete_corr, curr_path)
            print("计算当前文件夹结束\n")

            # 读取数据后，在list中删除这三个文件
            curr_file_list = [curr_file_list.remove(file) for file in correct_file_list]


# 递归调用已经完成计算的目录，给出排名
def start_calc_rank(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)
    for elem in curr_file_list:

        curr_path = os.path.join(pre_dir, elem)
        if os.path.isdir(curr_path):
            start_calc_rank(curr_path)

        elif os.path.isfile(curr_path):
            correct_file_list = ["pearson.txt",
                                 "spearman.txt",
                                 "kendall.txt",
                                 "chisquare.txt",
                                 "mutual_information.txt",
                                 "fisher_score.txt",
                                 "dstar.txt",
                                 "ochiai.txt",
                                 "barinel.txt"]
            if not set(correct_file_list).issubset(set(curr_file_list)):
                continue

            print("计算" + os.path.dirname(curr_path) + "的最高排名:")
            curr_file_list = [curr_file_list.remove(file) for file in correct_file_list]

            all_df_dict, fault_line_data = get_corr(os.path.dirname(curr_path))
            rank_dict = calc_rank(all_df_dict, fault_line_data)
            write_rank_to_txt(curr_path, rank_dict)
            print("计算当前文件夹结束\n")


# 读取排名，计算百分比，存入rank_percent.txt文件中
# 转为用excel实现
def start_rank_percent(dir):
    pass


# 绘图
def start_plot(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)

    if "rank-percent.txt" in curr_file_list:
        curr_file_list.clear()
        rank_percent_df = get_rank_percent_data(os.path.join(pre_dir, "rank-percent.txt"))
        plot_data(rank_percent_df, pre_dir)
    else:
        for elem in curr_file_list:
            curr_path = os.path.join(pre_dir, elem)
            if os.path.isdir(curr_path):
                start_plot(curr_path)
