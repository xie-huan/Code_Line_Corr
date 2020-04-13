from .metrics import *

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
            corr_dict[feature] = pearson(data[feature], data[label])

    if method == "spearman":
        for feature in features_list:
            corr_dict[feature] = spearman(data[feature], data[label])


    # 以下实现的是tau-b
    # tau = (c - d) / sqrt((n0 - n1) * (n0 - n2))
    # where c is the number of concordant pairs, d the number of discordant pairs,
    # n1 the number of ties only in `x`, and n2 the number of ties only in `y`.
    if method == "kendall":
        for feature in features_list:
            # 调包stats.kendalltau()，调自己写的函数tau_b()
            # kendall_corr = stats.kendalltau(data[feature].tolist(), data[label].tolist())
            # corr_dict[feature] = round(kendall_corr[0], 6)

            # 调用自己实现的tau_b计算kendall等级相关系数
            corr_dict[feature] = kendall(data[feature], data[label])

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
