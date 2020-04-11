from util.user_interface import *
# 单元测试
# test_path = "D:/Work/Python/bysj/Test_Data/gzip-bug-2010-02-19-3eb6091d69-884ef6d16c"
# data = get_curr_data(test_path)

# # #数据开始查找的路径
# dir_name = os.path.abspath(os.path.dirname(__file__))
# start_dir = os.path.join(dir_name, "DATA_TEST")
# # 用户接口，传入数据所在路径即可
# start_calc_corr(start_dir)
#
# # 计算排名
# dir_name = os.path.abspath(os.path.dirname(__file__))
# start_dir = os.path.join(dir_name, "DATA_TEST")
# start_calc_rank(start_dir)

# TODO：自动化计算rank/LOC的百分比

# 绘图
# dir_name = os.path.abspath(os.path.dirname(__file__))
# start_dir = os.path.join(dir_name, "DATA_TEST")
# start_plot(start_dir)


from scipy import stats
# x1 = np.arange(0,11)
# y1 = x1
# y2 = x1*2
# y3 = -0.32 * (x1**2) + 3.2*x1
# # tau = stats.kendalltau(x1, y1)
# # print(stats.pearsonr(x1,y1))
# # print(stats.pearsonr(x1,y2))
# # print(stats.pearsonr(x1,y3))
# # stats.spearmanr(x1,x2)
# # print(tau)
#
# d = {"feature":x1,"feature":y1}
# data = pd.DataFrame(d)
#
# print(data)
# corr_bymyself = calc_corr_bymyself(data,"kendall")
# corr = calc_corr(data,"kendall")
# print(corr_bymyself)
# print(corr)
#
# import matplotlib.pyplot as plt
#
# x1 = np.arange(0,11)
# y1 = x1
# x2 = np.arange(0,6)
# y2 = x2*2
# y3 = -0.32 * (x1**2) + 3.2*x1
# plt.figure()
#
# plt.plot(x1, y1, ".-k", markerfacecolor='none', label="Y1 = X", linewidth=0.5, markersize="4")
# plt.annotate("ρ1=1",xy=(8,9),xytext=(8,9),color="black",size=10) #, arrowprops=dict(arrowstyle = "->")
# plt.plot(x2, y2, "v--k", markerfacecolor='none', label="Y2 = 2 * X", linewidth=0.5, markersize="4")
# plt.annotate("ρ2=1",xy=(8.5,8.5),xytext=(3.5,9),color="black",size=10)
# plt.plot(x1, y3, "o:k", markerfacecolor='none', label="Y3 = -0.32 * X^2 + 3.2 * X", linewidth=0.5, markersize="4")
# plt.annotate("ρ3=0",xy=(8,6),xytext=(9,3),color="black",size=10)
# plt.xticks(np.arange(0,11))
# plt.yticks(np.arange(0,11))
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend(loc="lower center")
# plt.title("")
# plt.savefig("X-Y_PCC.png",dpi=600)
# plt.show()

