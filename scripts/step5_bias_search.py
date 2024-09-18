import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import least_squares


def objective_func(b, all_data):
    """
    Objective function to minimize the difference between lambda values,
    considering the first value of b as the reference state.
    """
    diff_sum = 0
    for row in all_data:
        lambs = [row[0]]
        for i in range(1, len(row)):
            lambs.append(row[i] - row[0] - b[i-1])
        diff_sum += np.sum(np.abs(lambs[1:] - np.mean(lambs[1:])))
    return diff_sum
p = argparse.ArgumentParser()
p.add_argument('-i', '--input_folder', help='Input folder')
p.add_argument('-c', '--cut_off', type=float, help='Cut-off value for the bias search', default=0.99)
args = p.parse_args()
input_folder = args.input_folder
cut_off = args.cut_off

# Understand how many analysis* folders we have
folders = os.listdir(input_folder)
folders = [f for f in os.listdir(input_folder) if 'analysis' in f and f.split('analysis')[1].isdigit()]
folders = sorted(folders, key=lambda x: int(x.split('analysis')[1]))
folders = [f for f in folders if 'data' in os.listdir(os.path.join(input_folder, f))]

# Initialize lists to store data
all_data = []
iterations = []

# Iterate through the folders and extract the data
for folder in folders:
    itt = int(folder.split('analysis')[1])
    iterations.append(itt)
    data_path = os.path.join(input_folder, folder, 'data')
    data_files = [f for f in os.listdir(data_path) if 'Lambda' in f]
    data = np.array([])
    for data_file in data_files:
        dat = np.loadtxt(os.path.join(data_path, data_file))
        print(f'Run {itt}, file {data_file}: {np.sum(dat > cut_off, axis=0) / dat.shape[0]}')
        if data.size == 0:
            data = dat
        else:
            data = np.concatenate([data, dat], axis=0)
    # Count the number of values greater than 0.99 for each column (substituent)
    col_counts = np.sum(data > cut_off, axis=0) / data.shape[0]
    all_data.append(col_counts)

# b_data = []
# for folder in folders:
#     b = np.loadtxt(os.path.join(input_folder, folder, 'b_prev.dat'))
#     b_data = np.concatenate([b_data, b], axis=0)

# fig, ax = plt.subplots(figsize=(12, 6))
# for sub in range(1, len(all_data[0])):
#     # estimate the intersection point
#     x = [a - b for a, b in zip(b_data[sub::len(all_data[0])], b_data[0::len(all_data[0])])]
#     y = [row[sub] - row[0] for row in all_data]
#     ax.plot(x, y, label=f'Substituent {sub}', marker='o', linestyle='None')
#     # m, b = np.polyfit(x, y, 1)
#     #plot the estimated line
#     # plt.plot(x, m*np.array(x) + b, label=f'Line {sub}, y={m:.2f}x+{b:.2f}', linestyle='--')


# ax.set_xlabel('b')
# ax.set_ylabel('lambda')
# ax.set_title('Free Energy over Lambda')
# ax.legend()
# fig.savefig(f'{input_folder}/free_energy_over_lambda.png')

    
    

    

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the data
for sub in range(len(all_data[0])):
    ax.plot(iterations, [row[sub] for row in all_data], label=f'Substituent {sub}', marker='o')

# Plot the median value across all substituents as a flat line
mean_value = np.mean(np.array(all_data), axis=1)
mean_value = sum(mean_value) / len(mean_value)
mean_value = np.full(len(all_data), mean_value)
ax.plot(iterations[:len(all_data)], mean_value, label='Mean', linewidth=3, color='k')


# Set the plot properties
ax.set_xlabel('Iteration')
ax.set_ylabel('Fraction of values > {}'.format(cut_off))
ax.set_title('Bias Search')
ax.set_ylim(0, 1/len(all_data[0]))
ax.legend()

# Highlight the best runs
best_runs = []
for i, row in enumerate(all_data):
    if np.all(np.abs(row - row[0]) < 0.005):
        best_runs.append(i)
        ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')
        print(f'Best run found at iteration {iterations[i]}, difference: {np.abs(row - row[0])}')
if best_runs == []:
    for i, row in enumerate(all_data):
        if np.all(np.abs(row - row[0]) < 0.0075):
            best_runs.append(i)
            ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')
            print(f'Best run found at iteration {iterations[i]}, difference: {np.abs(row - row[0])}')
if best_runs == []:
    for i, row in enumerate(all_data):
        if np.all(np.abs(row - row[0]) < 0.01):
            best_runs.append(i)
            ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')
            print(f'Best run found at iteration {iterations[i]}, difference: {np.abs(row - row[0])}')
# Add a legend for the best runs
if best_runs:
    ax.legend(loc='upper left')

# Save the plot
fig.savefig(f'{input_folder}/bias_search.png')
print(f'Plot saved to {input_folder}/bias_search.png')



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse

# p = argparse.ArgumentParser()
# p.add_argument('-i', '--input_folder', help='Input folder')
# p.add_argument('-c', '--cut_off', help='Cut-off value for the bias search', default=0.99)
# args = p.parse_args()
# input_folder = args.input_folder
# cut_off = args.cut_off

# # Understand how many analysis* folders we have
# folders = os.listdir(input_folder)
# folders = [f for f in folders if 'analysis' in f]
# # Sort the folders by the number after 'analysis', like analysis1, analysis2, ...
# folders = sorted(folders, key=lambda x: int(x.split('analysis')[1]))
# # Check if the folders contain the 'data' folder, otherwise skip
# folders = [f for f in folders if 'data' in os.listdir(os.path.join(input_folder, f))]

# fig, ax = plt.subplots(figsize=(12, 6))
# all_data = []
# iterations = []
# for folder in folders:
#     itt = int(folder.split('analysis')[1])
#     iterations.append(itt)
#     data_path = os.path.join(input_folder, folder, 'data')
#     data_files = os.listdir(data_path)
#     data_files = [f for f in data_files if 'Lambda' in f]
#     data = np.array([])
#     for data_file in data_files:
#         dat = np.loadtxt(os.path.join(data_path, data_file))
#         if data.size == 0:
#             data = dat
#         else:
#             data = np.concatenate([data, dat], axis=0)
    
#     # Count the number of values greater than 0.99 for each column (substituent)
#     col_counts = np.sum(data > 0.99, axis=0) / (data.shape[0])
    
#     # Append the column-wise counts to the all_data list
#     all_data.append(col_counts)

# # Plot the column-wise counts over the iterations
# for sub in range(len(all_data[0])):
#     # no line
#     ax.plot(iterations, [row[sub] for row in all_data], label=f'substituent {sub}', marker='o', linestyle='None')

# ax.set_xlabel('Iteration')
# ax.set_ylabel('Fraction of values > 0.99')
# ax.set_title('Bias Search')
# # set maximum y-axis value to 1/number of substituents
# ax.set_ylim(0, 1/len(all_data[0]))
# ax.legend()
# fig.savefig(f'{input_folder}/bias_search.png')