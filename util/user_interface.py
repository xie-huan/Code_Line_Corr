from util.get_data import *
from util.write_to_txt import *
from corr_core.calc_corr import *
from corr_core.calc_rank import *
from corr_core.rank_plot import *
from corr_core.calc_EXAM_score import *
from data_config import *

# 流程1：计算相关系数
# 递归调用搜索目录，计算相关系数
def start_calc_corr(start_dir):
    # 判断输出的代码行数文件是否存在
    if os.path.exists(total_line_path):
        os.remove(total_line_path)

    calc_corr_(start_dir)


def calc_corr_(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)  # 获取当前目录下的所有文件或目录
    for elem in curr_file_list:
        curr_path = os.path.join(pre_dir, elem)
        # 判断当前路径是目录还是文件，是目录则递归访问，是文件则计算数据，并输出到txt文档中
        if os.path.isdir(curr_path):
            calc_corr_(curr_path)
        elif os.path.isfile(curr_path):
            # 如果不包含covMatrix.txt、error.txt、componentinfo.txt三个文件，则不进行读取数据操作
            if not set(correct_file_list).issubset(set(curr_file_list)):
                continue

            # 如果文件夹里面包含已经计算完的数据，则不进行数据读取
            flag = False
            for concrete_method in method_list:
                # 如果文件已经存在（表明已经计算过），无需再次计算
                if os.path.exists(os.path.dirname(curr_path)+"/"+concrete_method+".txt"):
                    continue
                else:
                    flag = True

            # 读取数据
            total_line, data = get_curr_data(os.path.dirname(curr_path))

            # 记录总语句数
            with open(total_line_path,mode='a') as f:
                print(os.path.dirname(curr_path).split("\\")[-1]+":"+str(total_line),file=f)

            # 计算相关系数并输出到txt文件中
            print("计算" + os.path.dirname(curr_path) + "的相关系数:")
            for concrete_method in method_list:
                # 如果文件已经存在（表明已经计算过），无需再次计算
                if os.path.exists(os.path.dirname(curr_path)+"/"+concrete_method+".txt"):
                    continue
                concrete_corr = calc_corr_bymyself(data, concrete_method)
                write_corr_to_txt(concrete_method, concrete_corr, curr_path)
            print("计算当前文件夹结束\n")

            # 读取数据后，在list中删除这三个文件
            curr_file_list = [curr_file_list.remove(file) for file in correct_file_list]


# 流程2：得到排名
# 递归调用已经完成计算的目录，给出排名
def start_calc_rank(start_dir):
    # 判断输出排名文件是否存在
    if os.path.exists(concrete_path):
        os.remove(concrete_path)
    calc_rank_(start_dir)

def calc_rank_(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)
    for elem in curr_file_list:

        curr_path = os.path.join(pre_dir, elem)
        if os.path.isdir(curr_path):
            calc_rank_(curr_path)

        elif os.path.isfile(curr_path):
            correct_file_list = [method+".txt" for method in method_list]
            # 判断文件是否正确
            if not set(correct_file_list).issubset(set(curr_file_list)):
                continue

            print("计算" + os.path.dirname(curr_path) + "的最高排名:")
            curr_file_list = [curr_file_list.remove(file) for file in correct_file_list]

            all_df_dict, fault_line_data = get_corr(os.path.dirname(curr_path))
            rank_dict = calc_rank(all_df_dict, fault_line_data, method_list)

            write_rank_to_txt(curr_path, rank_dict)
            print("计算当前文件夹结束\n")

# 流程3：计算EXAM所需百分比
# 读取排名，计算百分比，存入rank_percent.txt文件中
# TODO: 自动化
# TODO已完成
def start_rank_percent(dir):
    # 创建文件
    # 判断输出的文件是否存在
    if os.path.exists(rank_percent_path):
        os.remove(rank_percent_path)

    header = ""
    for method in method_list:
        header += method + " "
    with open(rank_percent_path, 'a') as f:
        print(header, file=f)

    # 读取数据
    rank_data, total_line_data = get_rank_total_line_data(dir)
    # 计算
    EXAM_score = calc_EXAM(rank_data, total_line_data)
    # 写入文件
    write_rank_percent_to_txt(EXAM_score)

# 流程4：绘制EXAM图
# 绘图
def start_plot(start_dir):
    pre_dir = start_dir

    curr_file_list = os.listdir(start_dir)

    if "rank_percent.txt" in curr_file_list:
        curr_file_list.clear()
        rank_percent_df = get_rank_percent_data(os.path.join(pre_dir, "rank_percent.txt"))
        plot_data(rank_percent_df, pre_dir)
    else:
        for elem in curr_file_list:
            curr_path = os.path.join(pre_dir, elem)
            if os.path.isdir(curr_path):
                start_plot(curr_path)
