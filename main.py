from util.user_interface import *
# 单元测试
# test_path = "D:/Work/Python/bysj/Test_Data/gzip-bug-2010-02-19-3eb6091d69-884ef6d16c"
# data = get_curr_data(test_path)

#数据开始查找的路径
dir_name = os.path.abspath(os.path.dirname(__file__))
start_dir = os.path.join(dir_name,"DATA_TEST")
# 用户接口，传入数据所在路径即可
start_calc_corr(start_dir)

