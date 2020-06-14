"""
 下面是需要配置的数据
"""
# 方法列表
method_list = [
               # "pearson",
               "spearman",
               "kendall",
               # "chisquare",
               # "mutual_information",
               # "fisher_score",
               # "dstar",
               "ochiai",
               "barinel"
               ]
# 存储数据文件夹名
DATA_DIR_NAME = "DATA_TEST"


"""
下面是无需配置的数据
"""
# 正确文件列表，无需修改
correct_file_list = ['covMatrix.txt', 'error.txt', 'componentinfo.txt']

# 定义变量
# result_dict = dict({"pearson": float('-inf'),
#                     "spearman": float('-inf'),
#                     "kendall": float('-inf'),
#                     "chisquare": float('-inf'),
#                     "mutual_information": float('-inf'),
#                     "fisher_score": float('-inf'),
#                     "dstar": float('-inf'),
#                     "ochiai": float('-inf'),
#                     "barinel": float('-inf')
#                     })
result_dict = dict()
for method in method_list:
    result_dict[method] = float('-inf')


# 程序语句数记录位置
total_line_path = "D:/Work/Python/Code_Line_Corr/"+DATA_DIR_NAME+"/total_line.txt"

# write_to_txt.py中write_rank_to_txt函数的一个数据设置
# 将faultLine中的排名写入txt文档中
concrete_path = "D:/Work/Python/Code_Line_Corr/"+DATA_DIR_NAME+"/rank.txt"

# rank_percent.txt存储路径
rank_percent_path = "D:/Work/Python/Code_Line_Corr/"+DATA_DIR_NAME+"/rank_percent.txt"

# 绘制EXAM图的文件名
EXAM_file_name = "D:/Work/Python/Code_Line_Corr/"+DATA_DIR_NAME+"/"
for method in method_list:
    EXAM_file_name += method + "_"
EXAM_file_name += ".png"