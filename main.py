from util.user_interface import *

# 配置路径
dir_name = os.path.abspath(os.path.dirname(__file__))
start_dir = os.path.join(dir_name, "DATA_TEST")

# 计算相关系数
start_calc_corr(start_dir)
# 给出故障代码在排名表中的排名
start_calc_rank(start_dir)
# 计算EXAM标准所需的百分比
start_rank_percent(start_dir)
# 绘图
start_plot(start_dir)








# plot_RImp()


# from scipy import stats
# x1 = np.arange(0,11)
# y1 = x1
# y2 = x1*2
# y3 = -0.32 * (x1**2) + 3.2*x1
# x=[12,2,1,12,2]
# y=[1,4,7,1,0]
# tau = stats.kendalltau(x, y)
# print(stats.pearsonr(x1,y1))
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

# 检验Dstar算法的正确性
# covMatrix = pd.DataFrame([
#     [1,1,1,1,1,0,0,0,0],
#     [1,1,1,1,0,1,1,0,0],
#     [1,1,1,1,0,1,0,1,0],
#     [1,1,1,1,0,1,0,1,0],
#     [1,1,0,0,0,0,0,0,1],
#     [1,1,1,1,1,0,0,0,0],
#     [1,1,1,1,1,0,0,0,0],
#     [1,1,1,1,0,1,0,1,0],
#     [1,1,1,1,0,1,1,0,0],
#     [1,1,1,1,1,0,0,0,0],
#     [1,1,1,1,1,0,0,0,0],
#     [1,1,1,1,1,0,0,0,0],
# ])
#
# error = pd.DataFrame([1,1,1,1,0,0,0,0,0,0,0,0])
#
# # a = covMatrix[0]
# # i = 0
# for i in range(0,9):
#     print(dstar(pd.DataFrame(covMatrix[i]),error))



