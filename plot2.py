import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

m = np.array(["o", "s", "D", "*"])
label = ["0.01", "0.05", "0.1", "0.5"]
legend = "LR "

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

## learning rate

# airlines

## ensemble
# axgb = np.array([[63.30,-532.15],
#         [63.39,-516.77],
#         [62.96,-515.89],
#         [58.87,-515.79]])

# x1 = axgb[:, 0]
# y1 = axgb[:, 1]

# plt.scatter(x1, y1, marker="D", label='Ensemble')

# ## sea_a
# axgb = np.array([[88.90,-510.52],
#         [88.94,-495.47],
#         [88.82,-494.53],
#         [83.87,-494.72]])

# x1 = axgb[:, 0]
# y1 = axgb[:, 1]

# plt.scatter(x12, y12, marker="D", label='Ensemble')

## hyper_f
axgb = np.array([[84.14,-526.38],
        [85.42,-511.09],
        [86.17,-510.01],
        [80.90,-510.15]])

x1 = axgb[:, 0]
y1 = axgb[:, 1]

# plt.scatter(x13, y13, marker="D", label='Ensemble')

## incremental

# axgbi = np.array([[58.74,-213.55],
#         [58.78,-211.59],
#         [58.90,-211.16],
#         [59.15,-211.23]])

# x2 = axgbi[:, 0]
# y2 = axgbi[:, 1]
# plt.scatter(x2, y2, label='Incremental')


# ## sea_a
# axgbi = np.array([[80.25,-195.20],
#         [80.26,-192.96],
#         [80.33,-192.86],
#         [80.29,-192.84]])

# x2 = axgbi[:, 0]
# y2 = axgbi[:, 1]

# plt.scatter(x12, y12, label='Incremental')

## hyper_f
axgbi = np.array([[80.82,-129.21],
        [80.84,-127.10],
        [80.91,-126.81],
        [81.02,-125.80]])

x2 = axgbi[:, 0]
y2 = axgbi[:, 1]

# plt.scatter(x13, y13, label='Incremental')


for i in range(len(x1)): 
    plt.scatter(y1[i], x1[i], marker=m[i], label=legend + label[i], color="orange")
    plt.scatter(y2[i], x2[i], marker=m[i], label=legend + label[i], color="blue")

# naming the x axis 
plt.ylabel('Acuracia') 
# naming the y axis 
plt.xlabel('Tempo') 

plt.legend()
plt.title("Hyper_f")


# function to show the plot 
# plt.show()

points = np.concatenate((axgb, axgbi))
print(points)
pareto = identify_pareto(points)
# print ('Pareto front index vales')
# print ('Points on Pareto front: \n',pareto)

pareto_front = points[pareto]
# print ('\nPareto front scores')
# print (pareto_front)

pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values

x_all = points[:, 0]
y_all = points[:, 1]
x_pareto = pareto_front[:, 0]
y_pareto = pareto_front[:, 1]

# plt.scatter(x_all, y_all)
plt.plot(y_pareto, x_pareto, color='r')
ax = plt.gca()
ax.invert_xaxis()


plt.show()
