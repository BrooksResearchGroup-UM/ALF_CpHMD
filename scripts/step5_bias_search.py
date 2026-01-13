import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import least_squares
from pathos.multiprocessing import ProcessingPool as Pool  # Make sure to install pathos: pip install pathos
import shutil
import re
import sys


p = argparse.ArgumentParser()
p.add_argument('-i', '--input_folder', help='Input folder')
p.add_argument('-c', '--cut_off', type=float, help='Cut-off value for the bias search', default=0.985)
p.add_argument('-v', '--verbose', action='store_true', help='Verbose mode', default=False)
# p.add_argument('-a', '--adjustment', options=['+', '0', '-'], help='Adjustment type: +, 0, -', default='0')
p.add_argument('-a', '--adjustment', choices=['+', '0', '-'], help='Adjustment type: +, 0, -', default='0')
args = p.parse_args()
input_folder = args.input_folder
cut_off = args.cut_off
verbose = args.verbose

# Identify analysis folders with data
folders = [f for f in os.listdir(input_folder) if 'analysis' in f and f.split('analysis')[1].isdigit()]
folders = sorted(folders, key=lambda x: int(x.split('analysis')[1]))
folders = [f for f in folders if 'data' in os.listdir(os.path.join(input_folder, f))]

def process_folder(folder):
    """Process a single analysis folder, return iteration number, column counts, and a log string.
    Reports fractions, means and stds in percentage.
    """
    itt = int(folder.split('analysis')[1])
    data_path = os.path.join(input_folder, folder, 'data')
    data_files = [f for f in os.listdir(data_path) if 'Lambda' in f]
    data = np.array([])
    for data_file in data_files:
        dat = np.loadtxt(os.path.join(data_path, data_file))
        if data.size == 0:
            data = dat
        else:
            data = np.concatenate([data, dat], axis=0)
    num_rows = data.shape[0]
    col_counts = np.sum(data > cut_off, axis=0) / num_rows
    col_means = np.mean(data, axis=0)
    
    # Convert to percentage
    pct_counts = np.round(col_counts * 100, 2)
    pct_means = np.round(col_means * 100, 2)
    log_str = (f"Run {itt}:\n"
               f"  Files processed: {len(data_files)}\t Rows: {num_rows}\n"
               f"  Fraction > {cut_off}: {pct_counts} %\n"
               f"  Column means: {pct_means} %\n")
    return (itt, col_counts, log_str)

# Use pathos for multiprocessing
pool = Pool()
results = pool.map(process_folder, folders)
results = [r for r in results if r is not None]
results = sorted(results, key=lambda x: x[0])  # sort by iteration

# Check if we have any valid results
if not results:
    print(f"Error: No valid analysis folders with data found in {input_folder}")
    print("Make sure analysis folders contain 'data' subdirectories with Lambda files.")
    sys.exit(1)

iterations, all_data, logs = zip(*results)
logs = list(logs)
if verbose:
    verbose_logs = logs
else:
    verbose_logs = []

# Now print the logs sequentially

# Continue with the plotting as before
fig, ax = plt.subplots(figsize=(12, 6))
for sub in range(len(all_data[0])):
    ax.plot(iterations, [row[sub] for row in all_data], label=f'Substituent {sub}', marker='o')

mean_value = np.mean(np.array(all_data), axis=1)
mean_value = sum(mean_value) / len(mean_value)
mean_value = np.full(len(all_data), mean_value)
ax.plot(iterations[:len(all_data)], mean_value, label='Mean', linewidth=3, color='k')

ax.set_xlabel('Iteration')
ax.set_ylabel('Fraction of values > {}'.format(cut_off))
ax.set_title('Bias Search')
ax.set_ylim(0, 1/len(all_data[0]))
ax.legend()

# Improved best run selection using a combined metric
alpha = 10.0  # Penalty factor for imbalance; adjust as needed
scores = []
for i, row in enumerate(all_data):
    avg = np.mean(row)
    diff = np.max(row) - np.min(row)
    score = avg - alpha * diff
    scores.append(score)

scores = np.array(scores)
# Get indices of runs sorted in descending order of score
sorted_indices = np.argsort(scores)[::-1]
top5_indices = sorted_indices[:5]

# Log and highlight the top 5 runs
verbose_logs.append("Top 5 runs based on combined score:")
for rank, idx in enumerate(top5_indices):
    iter_value = iterations[idx]
    run_data = all_data[idx]
    score = scores[idx]
    diff_percent = np.round((np.max(run_data) - np.min(run_data)) * 100, 2)
    avg_percent = np.round(np.mean(run_data) * 100, 2)
    verbose_logs.append(
        f"Rank {rank+1}: Iteration {iter_value} with average {avg_percent}% and diff {diff_percent}%. Score: {score:.4f}"
    )
    # Highlight top 5 runs on the plot
    if rank == 0:
        ax.axvline(x=iter_value, color='red', linestyle='--', label=f'Top Run (Rank {rank+1})')
    else:
        ax.axvline(x=iter_value, color='red', linestyle='--')

ax.legend(loc='upper left')

# Write logs to a file and print if verbose
log_file = os.path.join(input_folder, "bias_search.log")
with open(log_file, "w") as f:
    for log in logs:
        f.write(log + "\n")
        if verbose:
            print(log)
    for log in verbose_logs:
        f.write(log + "\n")
        print(log)

fig.savefig(f'{input_folder}/bias_search.png', dpi=300)
print(f'Plot saved to {input_folder}/bias_search.png')

# For top run, copy variables{iter_value}.inp to input_folder and name it variables_final.inp
var_file = os.path.join(input_folder, f'variables{iterations[top5_indices[0]]}.inp')
# Define file paths
temp_file = f"{input_folder}/prep/temp"
os.makedirs('variables', exist_ok=True)  # Ensure the variables directory exists
new_var_file = os.path.join('variables', f'var-{input_folder}.inp')
# new_var_file = f"{input_folder}/variables_final.inp"

# Read temperature from prep/temp, default to 298.15 if missing or empty
temperature = 298.15  # Default temperature
if os.path.isfile(temp_file):
    try:
        with open(temp_file, 'r') as f:
            temp_content = f.read().strip()
            if temp_content:
                temperature = float(temp_content)
                print(f"Temperature read from {temp_file}: {temperature} K")
            else:
                print(f"Warning: {temp_file} is empty, using default temperature: {temperature} K")
    except ValueError:
        print(f"Warning: Invalid temperature value in {temp_file}, using default temperature: {temperature} K")
else:
    print(f"Warning: {temp_file} not found, using default temperature: {temperature} K")
    
    
# copy variables{iter_value}.inp to variables_final.inp
shutil.copy(var_file, new_var_file)
print(f"Copied {var_file} to {new_var_file}")

R = 0.0019872041  # kcal/(mol*K)
RT = R * temperature

# Calculate adjustments
# use cutoff values from the best run
data = all_data[top5_indices[0]]

# Handle non-positive values by adding a small epsilon or skipping adjustment
if np.any(data <= 0):
    print("Warning: Data contains non-positive values, cannot compute log-based adjustments.")
    print("Setting adjustments to zero (no adjustment will be applied).")
    adjustments = np.zeros_like(data)
else:
    # mean data for cutoff
    mean_data = np.mean(data)
    if mean_data == 0:
        print("Warning: Mean of data is zero, setting adjustments to zero.")
        adjustments = np.zeros_like(data)
    else:
        adjustments = np.log(data / mean_data)
        adjustments -= adjustments[0]  # Normalize to the first value
        if args.adjustment == '-':
            adjustments = -adjustments  # Invert the sign for negative adjustment
        elif args.adjustment == '0':
            adjustments = np.zeros_like(adjustments)
        else: # No adjustment
            adjustments = adjustments

adjustments *= RT  # Scale by RT for free energy in kcal/mol
print(f"Adjustment factor for free energy (kcal/mol): {adjustments}")

# Modify file
new_lines = []
with open(var_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'lams1' in line:
            match = re.match(r'^set\s+(lams1s\d+)\s*=\s*([-+]?\d*\.?\d+)', line.strip())
            if match:
                var_name, number = match.groups()
                try:
                    index = int(var_name[-1]) - 1
                    if 0 <= index < len(adjustments):
                        number = float(number)
                        new_number = np.round(number + adjustments[index], 3)
                        print(f"Adjusting {var_name} from {number} to {new_number}")
                        new_lines.append(f"set {var_name} = {new_number:>8.3f}\n")
                    else:
                        print(f"Warning: Index {index} out of range for {line.strip()}")
                        new_lines.append(line)
                except ValueError:
                    print(f"Warning: Invalid number in {line.strip()}")
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
# Write back to file
with open(new_var_file, 'w') as f:
    f.writelines(new_lines)
print(f"Updated {new_var_file} with new free energy adjustments.")


# OLD ALGORITHM
# # Highlight best runs (using the same criteria)
# best_runs = []
# for i, row in enumerate(all_data):
#     if np.all(np.abs(row - row[0]) < 0.005):
#         best_runs.append(i)
#         ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')
#         diff = np.abs(row - row[0])
#         diff = np.round(diff * 100, 2)
#         run_details = logs[i]
#         verbose_logs.append(f'Best run found at iteration {iterations[i]}, difference: {diff} %')
#         verbose_logs.append(run_details)
# if not best_runs:
#     for i, row in enumerate(all_data):
#         if np.all(np.abs(row - row[0]) < 0.0075):
#             best_runs.append(i)
#             ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')      
#             diff = np.abs(row - row[0])
#             diff = np.round(diff * 100, 2)
#             run_details = logs[i]
#             verbose_logs.append(f'Best run found at iteration {iterations[i]}, difference: {diff} %')
#             verbose_logs.append(run_details)
# if not best_runs:   
#     for i, row in enumerate(all_data):
#         if np.all(np.abs(row - row[0]) < 0.01):
#             best_runs.append(i)
#             ax.axvline(x=iterations[i], color='g', linestyle='--', label='Best Run' if i == 0 else '')
#             diff = np.abs(row - row[0])
#             diff = np.round(diff * 100, 2)
#             run_details = logs[i]
#             verbose_logs.append(f'Best run found at iteration {iterations[i]}, difference: {diff} %')
#             verbose_logs.append(run_details)
# if best_runs:
#     ax.legend(loc='upper left')

# log_file = os.path.join(input_folder, "bias_search.log")
# with open(log_file, "w") as f:
#     for log in logs:
#         f.write(log + "\n")
#         if verbose:
#             print(log)
#     for log in verbose_logs:
#         f.write(log + "\n")
#         print(log)

# fig.savefig(f'{input_folder}/bias_search.png', dpi=300)
# print(f'Plot saved to {input_folder}/bias_search.png')
