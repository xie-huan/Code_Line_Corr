import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import random

from data_config import *

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 参数rank_percent_df表示数据DataFrame，pre_dir表示存储图片的路径
def plot_data(rank_percent_df, pre_dir):
    steps = np.arange(0, 101, 5)

    data_list = list()
    for method in method_list:
        concrete_data = rank_percent_df[method]
        concrete_data_rank_percent_list = [len(concrete_data[concrete_data < step]) / len(concrete_data) * 100 for step in steps]
        data_list.append(concrete_data_rank_percent_list)

    # plt.title("")
    plt.xticks(steps)
    plt.xlabel("EXAM标准")
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel("失效版本百分比")
    # 0"pearson",
    # 1"spearman",
    # 2"kendall",
    # 3"chisquare",
    # 4"mutual_information",
    # 5"fisher_score",
    # 6"dstar",
    # 7"ochiai",
    # 8"barinel"
    style=["*-b","v-r","s-g","^--","<--",">--","2-.","3-.","4-."]

    for i in range(len(method_list)):
        index = random.randint(0,len(style)-1)
        plt.plot(steps, data_list[i], style[index],markerfacecolor='none', label=method_list[i])
        style.remove(style[index])

    # 自定义样式
    # plt.plot(steps, data_list[0], "*-b", markerfacecolor='none', label=method_list[0])
    # plt.plot(steps, data_list[1], "v-r", markerfacecolor='none', label=method_list[1])
    # plt.plot(steps, data_list[2], "s-g", markerfacecolor='none', label=method_list[2])
    # plt.plot(steps, data_list[3], "^--", markerfacecolor='none', label=method_list[3])
    # plt.plot(steps, data_list[4], "<--", markerfacecolor='none', label=method_list[4])
    # plt.plot(steps, data_list[5], ">--", markerfacecolor='none', label=method_list[5])
    # plt.plot(steps, data_list[6], "2-.", markerfacecolor='none', label=method_list[6])
    # plt.plot(steps, data_list[7], "3-.", markerfacecolor='none', label=method_list[7])
    # plt.plot(steps, data_list[8], "4-.", markerfacecolor='none', label=method_list[8])

    plt.xlim(-5,100)
    plt.legend(loc="lower right")
    plt.savefig(EXAM_file_name, dpi=500)
    plt.show()

# RImp
def plot_RImp():
    def auto_text(rects):
        for rect in rects:
            plt.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom', fontsize=7)

    x_list = ["gzip","libtiff","python","space"]

    # pearson:
    # dstar = [63.70, 70.90, 77.16, 66.31]
    # ochiai = [92.47, 90.77, 97.39, 68.52]
    # barinel = [92.47, 95.75, 97.39, 60.63]

    # spearman:
    # dstar = [63.89, 71.02, 77.16, 66.31]
    # ochiai = [92.47, 90.92, 97.39, 68.52]
    # barinel = [92.74, 95.92, 97.39, 60.63]

    # kendall：
    dstar = [62.96, 70.96, 79.00, 66.31]
    ochiai = [91.40, 90.84, 99.71, 68.52]
    barinel = [91.40, 95.84, 99.71, 60.63]

    total_width = 0.6
    n = 3
    width = total_width / n

    plt.figure()
    bar1 = plt.bar(np.arange(len(dstar)), dstar, width, label="Dstar", color="salmon")
    bar2 = plt.bar(np.arange(len(dstar)) + width, ochiai, width, label="Ochiai",color='lightblue', tick_label=x_list)
    bar3 = plt.bar(np.arange(len(dstar)) + 2 * width, barinel, width, label="Barinel",color='tan')
    # bar4 = plt.bar(np.arange(len(gzip_pearson)) + 3 * width, space_pearson, width, label="kendall",color='tan')

    plt.ylim(0, 105)
    plt.ylabel("RImp")
    plt.legend()

    auto_text(bar1)
    auto_text(bar2)
    auto_text(bar3)
    plt.savefig("RImp-kendall.png", dpi=500)
    # plt.show()
    print("绘图完成")

