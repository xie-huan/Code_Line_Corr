import numpy as np
import pandas as pd


# Pearson 相关系数的实现，给定特征和标签series，返回相关系数值
def pearson(feature, label):
    cov = feature.cov(label)  # 协方差
    std_x = feature.std()  # 标准差
    std_y = label.std()  # 标准差
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        pearson_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
        pearson_corr = min(1., max(-1., pearson_corr))  # 限制结果范围区间为[-1, 1]
        return round(pearson_corr, 6)  # 浮点数精度不准


# Spearman 相关系数的实现，给定特征和标签series，返回相关系数值
def spearman(feature, label):
    # 排名没有并列的情况
    # feature_rank = data[feature].rank()
    # label_rank = data[label].rank()
    # diff = feature_rank.sub(label_rank)
    # diff_square = diff.mul(diff)
    #
    # N = int(data[label].count())
    # spearman_corr = 1 - 6 * sum(diff_square) / (N * (N * N - 1))
    # print(feature + ":" + str(spearman_corr))

    # 排名有并列的情况
    feature_rank = feature.rank()
    label_rank = label.rank()

    cov = feature_rank.cov(label_rank)  # 协方差
    std_x = feature_rank.std()  # 标准差
    std_y = label_rank.std()  # 标准差
    # 分母为0，相关系数则为nan
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        spearman_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
        spearman_corr = min(1., max(-1., spearman_corr))  # 限制结果范围区间为[-1, 1]
        return round(spearman_corr, 6)


# 计算kendall相关系数接口
# Kendall 相关系数的实现，给定特征和标签series，返回相关系数值
def kendall(feature, label):
    x = np.array(feature)
    y = np.array(label)

    size = x.size
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = calc_dis(y)    # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()
    xtie = count_tie(x)
    ytie = count_tie(y)

    tot = (size * (size - 1)) // 2

    # tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #     = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

    # tau = min(1., max(-1., tau))
    return round(tau, 6)


# 计算kendall相关系数中的xtie，ytie
def count_tie(vector):
    cnt = np.bincount(vector).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return (cnt * (cnt - 1) // 2).sum()


# 求逆序对
def mergeSortInversion(data, aux, low, high):
    if low >= high:
        return 0

    mid = low + (high - low) // 2
    # 递归调用过程
    leftCount = mergeSortInversion(data, aux, low, mid)
    rightCount = mergeSortInversion(data, aux, mid+1, high)

    # merge 过程
    for index in range(low, high+1):
        aux[index] = data[index]
    count = 0
    i = low
    j = mid + 1
    k = i
    while k <= high:
        if i > mid and j <= high:
            data[k] = aux[j]
            j += 1
        elif j > high and i <= mid:
            data[k] = aux[i]
            i += 1
        elif aux[i] <= aux[j]:
            data[k] = aux[i]
            i += 1
        elif aux[i] > aux[j]:
            data[k] = aux[j]
            j += 1
            count += mid - i + 1
        k += 1

    return leftCount + rightCount + count


# 计算kendall相关系数中的不一致对
def calc_dis(y):
    aux = [y[i] for i in range(len(y))]
    nSwap = mergeSortInversion(y, aux, 0, len(y)-1)
    return nSwap


# 时间复杂度较高的kendall相关系数实现，已优化，暂不使用
def k(data):
    # # 计算分母所需参数
    # xtie = count_tie(list(data[feature]))  # n1
    # ytie = count_tie(list(data[label]))  # n2
    # size = len(data[feature])  # n
    # tot = size * (size - 1) // 2  # n0
    #
    # # 计算分子
    # # TODO: 优化计算concordant - discordant的算法，详见维基百科（已优化，见上面归并排序求逆序对）
    # # 时间复杂度O(m * n ^ 2)，数据量大的时候慢得有点让人难以忍受。。。
    # # m 为特征数量，n为数据量
    # feature_rank = data[feature].rank()
    # label_rank = data[label].rank()
    # dis = 0  # 分子所需参数，dis = concordant - discordant
    # for i in range(1, size):
    #     for j in range(0, i):
    #         dis = dis + \
    #               np.sign(feature_rank[i] - feature_rank[j]) * \
    #               np.sign(label_rank[i] - label_rank[j])
    # # 计算并输出结果
    # # 分母为0，相关系数则为nan
    # if abs(np.sqrt(tot - xtie) * np.sqrt(tot - ytie)) < 1e-5:
    #     corr_dict[feature] = 'nan'
    # else:
    #     kendall_corr = dis / (np.sqrt(tot - xtie) * np.sqrt(tot - ytie))
    #     kendall_corr = min(1., max(-1., kendall_corr))  # 限制结果范围区间为[-1, 1]
    #     corr_dict[feature] = kendall_corr
    return 0
