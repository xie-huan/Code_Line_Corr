import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 参数rank_percent_df表示数据DataFrame，pre_dir表示存储图片的路径
def plot_data(rank_percent_df, pre_dir):
    steps = np.arange(0, 101, 5)

    methods = ["pearson", "spearman", "kendall", "Dstar"]
    data_list = list()
    for method in methods:
        concrete_data = rank_percent_df[method]
        concrete_data_rank_percent_list = [len(concrete_data[concrete_data < step]) / len(concrete_data) * 100 for step in steps]
        data_list.append(concrete_data_rank_percent_list)

    plt.figure()
    plt.title("")
    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[0], "*-k", markerfacecolor='none', label="pearson")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar")
    plt.legend(loc="lower right")
    plt.savefig("pearson-Dstar.png", dpi=500)
    plt.show()

    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[1], "v--k", markerfacecolor='none', label="spearman")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar")

    plt.legend(loc="lower right")
    plt.savefig("spearman-Dstar.png", dpi=500)
    plt.show()

    plt.xticks(steps)
    plt.xlabel("% of the executable statements examined")
    plt.yticks(steps)
    plt.ylabel("% of faulty versions")
    plt.plot(steps, data_list[2], "s-.k", markerfacecolor='none', label="kendall")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar")

    plt.legend(loc="lower right")
    plt.savefig("kendall-Dstar.png", dpi=500)
    plt.show()


    # plt.title("")
    plt.xticks(steps)
    plt.xlabel("EXAM标准")
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel("失效版本百分比")
    plt.plot(steps, data_list[0], "*-b", markerfacecolor='none', label="pearson")
    plt.plot(steps, data_list[1], "v--r", markerfacecolor='none', label="spearman")
    plt.plot(steps, data_list[2], "s-.g", markerfacecolor='none', label="kendall")
    plt.plot(steps, data_list[3], "o:k", markerfacecolor='none', label="Dstar")
    plt.xlim(-5,80)
    plt.legend(loc="lower right")
    plt.savefig("all-Dstar.png", dpi=500)
    plt.show()

# RImp
def plot_RImp():
    def auto_text(rects):
        for rect in rects:
            plt.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

    x_list = ["gzip","libtiff","python"]

    pearson = [40.98,76.89,2.35]
    spearman = [37.70,76.89,2.35]
    kendall = [48.20,76.89,10.24]

    total_width = 0.6
    n = 3
    width = total_width / n

    plt.figure()
    bar1 = plt.bar(np.arange(len(pearson)), pearson, width, label="pearson", color="salmon")
    bar2 = plt.bar(np.arange(len(pearson)) + width, spearman, width, tick_label=x_list, label="spearman",color='lightblue')
    bar3 = plt.bar(np.arange(len(pearson)) + 2 * width, kendall, width, label="kendall",color='tan')
    plt.ylim(0,100)
    plt.ylabel("RImp")
    plt.legend()

    auto_text(bar1)
    auto_text(bar2)
    auto_text(bar3)
    plt.savefig("RImp.png", dpi=500)

