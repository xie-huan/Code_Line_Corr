import numpy as np
import pandas as pd
from scipy.stats._stats import _kendall_dis
from sklearn import metrics
from scipy import stats

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

# 计算kendall秩相关系数
def calc_dis(y):
    aux = [y[i] for i in range(len(y))]
    nSwap = mergeSortInversion(y,aux,0,len(y)-1)
    return nSwap


# 计算kendall相关系数
def tau_b(feature_series, label_series):
    x = np.array(feature_series)
    y = np.array(label_series)

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

    tau = min(1., max(-1., tau))
    return tau


def calc_corr_bymyself(data, method):
    """
        ----------------------------------------------------------
        自己实现的person、kendall、spearman相关系数
        ----------------------------------------------------------
    """
    # 获取数据的特征和标签
    features_list = list(data.columns)[:-1]
    label = list(data.columns)[-1]
    corr_dict = {}  # 字典存储各个特征的协方差

    # 根据method计算相关系数，并存储到字典中
    if method == "pearson":
        for feature in features_list:
            cov = data[feature].cov(data[label])  # 协方差
            std_x = data[feature].std()  # 标准差
            std_y = data[label].std()  # 标准差
            if abs(std_x * std_y) < 1e-5:
                corr_dict[feature] = np.nan
            else:
                pearson_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
                pearson_corr = min(1., max(-1., pearson_corr))  # 限制结果范围区间为[-1, 1]
                corr_dict[feature] = pearson_corr

    if method == "spearman":
        for feature in features_list:
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
            feature_rank = data[feature].rank()
            label_rank = data[label].rank()
            data[feature] = feature_rank
            data[label] = label_rank

            cov = data[feature].cov(data[label])  # 协方差
            std_x = data[feature].std()  # 标准差
            std_y = data[label].std()  # 标准差
            # 分母为0，相关系数则为nan
            if abs(std_x * std_y) < 1e-5:
                corr_dict[feature] = np.nan
            else:
                spearman_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
                spearman_corr = min(1., max(-1., spearman_corr))  # 限制结果范围区间为[-1, 1]
                corr_dict[feature] = spearman_corr

    # 以下实现的是tau-b
    # tau = (c - d) / sqrt((n0 - n1) * (n0 - n2))
    # where c is the number of concordant pairs, d the number of discordant pairs,
    # n1 the number of ties only in `x`, and n2 the number of ties only in `y`.
    if method == "kendall":
        for feature in features_list:
            # 调用自己实现的tau_b计算kendall等级相关系数
            kendall_corr = tau_b(data[feature], data[label])
            corr_dict[feature] = kendall_corr
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

    if method == 'mutual information':
        for feature in features_list:
            labels_true = data[label]  # 标签@wzy
            labels_pred = data[feature]  # 样本@wzy
            # 调用metrics.mutual_info_score()计算互信息@wzy
            mutual_information_coe = metrics.mutual_info_score(labels_pred,
                                                               labels_true)
            corr_dict[feature] = mutual_information_coe

    if method == 'chisquare':
        for feature in features_list:
            exp = data[label]  # 标签@wzy
            obs = data[feature]  # 样本@wzy
            # 调用stats.chisquare，返回是一个列表，第二个值是所需值@wzy
            p = stats.chisquare(obs, exp)
            chisquare_coe = p[1]
            corr_dict[feature] = chisquare_coe
    corr = pd.Series(corr_dict, dtype=float)
    return corr


def calc_corr(data, method):
    # method: {'pearson', 'kendall', 'spearman'}
    # *pearson: standard correlation coefficient
    # *kendall: Kendall Tau correlation coefficient
    # *spearman: Spearman rank correlation

    """
    ----------------------------------------------------------
    调用pandas内置函数实现person、kendall、spearman相关系数
    ----------------------------------------------------------
    """
    # 计算相关系数
    corr = data.corr(method)
    # 处理结果并返回
    features_list = list(corr.index)[:-1]
    label = list(corr.index)[-1]
    corr = corr[label].loc[features_list]
    return corr
