import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 参数rank_percent_df表示数据DataFrame，pre_dir表示存储图片的路径
def plot_data(rank_percent_df, pre_dir):
    steps = np.arange(0, 101, 5)

    methods = ["pearson","spearman","kendall","Dstar"]
    data_list = list()
    for method in methods:
        concrete_data = rank_percent_df[method]
        concrete_data_rank_percent_list = [len(concrete_data[concrete_data<step])/len(concrete_data)*100 for step in steps]
        data_list.append(concrete_data_rank_percent_list)

    plt.figure()
    plt.title("")
    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[0], ".-k", markerfacecolor='none', label="pearson", linewidth = 0.5,markersize="4")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar", linewidth = 0.5, markersize="4")
    plt.legend(loc="lower right")
    plt.savefig("pearson-Dstar.png", dpi=500)
    plt.show()

    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[1], "v--k", markerfacecolor='none', label="spearman", linewidth = 0.5, markersize="4")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar", linewidth = 0.5, markersize="4")
    plt.legend(loc="lower right")
    plt.savefig("spearman-Dstar.png", dpi=500)
    plt.show()

    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[2], "s-.k", markerfacecolor='none', label="kendall", linewidth = 0.5, markersize="4")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar", linewidth = 0.5, markersize="4")
    plt.legend(loc="lower right")
    plt.savefig("kendall-Dstar.png", dpi=500)
    plt.show()
