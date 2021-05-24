import numpy as np
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu

# # Max depth
# airlinesEnsemble = np.array([59.70,61.74,61.27,61.19])
# seaEnsemble = np.array([86.13,88.36,88.35,88.30])
# hyperEnsemble = np.array([81.90,85.82,85.23,85.29])
# airlinesIncremental = np.array([60.34,60.5,60.63,60.76])
# seaIncremental = np.array([83.14,83.16,83.14,83.06])
# hyperIncremental = np.array([81.44,81.53,81.55,81.57])

# # learning rate
# airlinesEnsemble2 = np.array([63.30,63.39,62.96,58.87])
# seaEnsemble2 = np.array([88.90,88.94,88.82,83.87])
# hyperEnsemble2 = np.array([84.14,85.42,86.17,80.90])
# airlinesIncremental2 = np.array([58.74,58.78,58.90,59.15])
# seaIncremental2 = np.array([80.25,80.26,80.33,80.29])
# hyperIncremental2 = np.array([80.82,80.84,80.91,81.02])

# z_statistic, p_value = wilcoxon(airlinesEnsemble, airlinesIncremental)
# print("Max depth airlines", p_value, z_statistic)

# z_statistic, p_value = wilcoxon(seaEnsemble, seaIncremental)
# print("Max depth sea", p_value, z_statistic)

# z_statistic, p_value = wilcoxon(hyperEnsemble, hyperIncremental)
# print("Max depth hyper", p_value, z_statistic)

# print("------------")

# z_statistic, p_value = wilcoxon(airlinesEnsemble2, airlinesIncremental2)
# print("learning rate airlines", p_value, z_statistic)

# z_statistic, p_value = wilcoxon(seaEnsemble2, seaIncremental2)
# print("learning rate sea", p_value, z_statistic)

# z_statistic, p_value = wilcoxon(hyperEnsemble2, hyperIncremental2)
# print("learning rate hyper", p_value, z_statistic)


# ######################
# print("Tempo")
# # Max depth
# airlinesEnsemble = np.array([576.44,556.36,553.98,554.54])
# seaEnsemble = np.array([508.75,493.73,493.28,493.45])
# hyperEnsemble = np.array([574.13,555.95,554.01,555.12])
# airlinesIncremental = np.array([360.31,360.08,365.33,366.97])
# seaIncremental = np.array([247.17,246.99,249.08,250.17])
# hyperIncremental = np.array([220.43,220.69,223.36,223.34])

# # learning rate
# airlinesEnsemble2 = np.array([532.15,516.77,515.89,515.79])
# seaEnsemble2 = np.array([510.52,495.47,494.53,494.72])
# hyperEnsemble2 = np.array([526.38,511.09,510.01,510.15])
# airlinesIncremental2 = np.array([213.55,211.59,211.16,211.23])
# seaIncremental2 = np.array([195.20,192.96,192.86,192.84])
# hyperIncremental2 = np.array([129.21,172.10,126.81,125.80])



# z_statistic, p_value = wilcoxon(airlinesEnsemble, airlinesIncremental)
# print("Max depth airlines", p_value)

# z_statistic, p_value = wilcoxon(seaEnsemble, seaIncremental)
# print("Max depth sea", p_value)

# z_statistic, p_value = wilcoxon(hyperEnsemble, hyperIncremental)
# print("Max depth hyper", p_value)

# print("------------")

# z_statistic, p_value = wilcoxon(airlinesEnsemble2, airlinesIncremental2)
# print("learning rate airlines", p_value)

# z_statistic, p_value = wilcoxon(seaEnsemble2, seaIncremental2)
# print("learning rate sea", p_value)

# z_statistic, p_value = wilcoxon(hyperEnsemble2, hyperIncremental2)
# print("learning rate hyper", p_value)

