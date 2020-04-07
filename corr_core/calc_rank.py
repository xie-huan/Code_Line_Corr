import numpy as np


def calc_rank(pearson_corr, spearman_corr, kendall_corr, fault_line_data):
    # rank_list_pearson = list()
    # rank_list_spearman = list()
    # rank_list_kendall = list()
    # for line_num in fault_line_data:
    #     rank_list_pearson.append(pearson_corr[pearson_corr['line_num'] == line_num].index+1)
    #     rank_list_spearman.append(spearman_corr[spearman_corr['line_num'] == line_num].index+1)

    # fault_line_data = list(set(pearson_corr['line_num'].tolist()).intersection(set(fault_line_data)))
    real_line_data = pearson_corr['line_num'].tolist()
    real_fault_line_data = list()
    for line in fault_line_data:
        if line in real_line_data:
            real_fault_line_data.append(line)
        else:
            prob_val = find_closest_num(real_line_data, line)
            for val in prob_val:
                real_fault_line_data.append(val)
    real_fault_line_data = list(set(real_fault_line_data))

    rank_list_pearson = [pearson_corr[pearson_corr['line_num'] == line_num].index.values[0] + 1
                         for line_num in real_fault_line_data]
    rank_list_spearman = [spearman_corr[spearman_corr['line_num'] == line_num].index.values[0] + 1
                          for line_num in real_fault_line_data]
    rank_list_kendall = [kendall_corr[kendall_corr['line_num'] == line_num].index.values[0] + 1
                         for line_num in real_fault_line_data]

    top_rank_pearson = min(rank_list_pearson)
    top_rank_spearman = min(rank_list_spearman)
    top_rank_kendall = min(rank_list_kendall)
    return dict({"pearson rank":top_rank_pearson,
                 "spearman rank":top_rank_spearman,
                 "kendall rank":top_rank_kendall})

# faultLine中的行可能对应不了具体的行数，则选择最近的代码行
def find_closest_num(real_line_data,target):
    target = int(target)
    line_data_np = np.array(real_line_data, dtype=int)
    min_diff_val = min(abs(line_data_np-target))

    if str(target + min_diff_val) in real_line_data and str(target - min_diff_val) in real_line_data:
        return list([str(target + min_diff_val), str(target - min_diff_val)])
    if str(target + min_diff_val) in real_line_data:
        return list(str(target + min_diff_val))
    if str(target - min_diff_val) in real_line_data:
        return list(str(target - min_diff_val))
