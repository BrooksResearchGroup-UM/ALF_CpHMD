import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool, cpu_count
import argparse
import imageio_ffmpeg as ffmpeg

def generate_simplex_grid(N, num_points):
    grids = [np.linspace(0, 1, num_points) for _ in range(N)]
    mesh = np.array(list(product(*grids)))
    simplex_points = mesh[np.isclose(np.sum(mesh, axis=1), 1, atol=1e-6)]
    return simplex_points

def energy_E_b(lambda_vec, b):
    N = len(lambda_vec)
    E_b = 0
    for i in range(N):
        E_b += -(lambda_vec[i] * (b[i]))
    return E_b

def energy_E_c(lambda_vec, c):
    N = len(lambda_vec)
    E_c = 0
    for i in range(N):
        for j in range(N):
            E_c += (c[i, j] + c[j, i]) * lambda_vec[i] * lambda_vec[j]
    return E_c

def energy_E_s(lambda_vec, s):
    N = len(lambda_vec)
    E_s = 0
    for i in range(N):
        for j in range(N):
            E_s += (s[i, j] / (lambda_vec[i] + 0.017) + s[j, i] / (lambda_vec[j] + 0.017)) * (lambda_vec[i] * lambda_vec[j])
    return E_s

def energy_E_x(lambda_vec, x):
    N = len(lambda_vec)
    E_x = 0
    for i in range(N):
        for j in range(N):
            E_x += x[i, j] * (lambda_vec[j] * (1 - np.exp(-5.56 * lambda_vec[i]))) + x[j, i] * (lambda_vec[i] * (1 - np.exp(-5.56 * lambda_vec[j])))
    return E_x

def total_energy(lambda_vec, b, c, s, x):
    E_b_val = energy_E_b(lambda_vec, b)
    E_c_val = energy_E_c(lambda_vec, c)
    E_s_val = energy_E_s(lambda_vec, s)
    E_x_val = energy_E_x(lambda_vec, x)
    return E_b_val + E_c_val + E_s_val + E_x_val

def calculate_energy_profile(args):
    folder, input_folder, simplex_grid = args
    itt = int(folder.split('analysis')[1])
    b = np.loadtxt(os.path.join(input_folder, folder, 'b_prev.dat'))
    c = np.loadtxt(os.path.join(input_folder, folder, 'c_prev.dat'))
    x = np.loadtxt(os.path.join(input_folder, folder, 'x_prev.dat'))
    s = np.loadtxt(os.path.join(input_folder, folder, 's_prev.dat'))
    
    # Calculate the energy profile for the current iteration
    energy_profile = [total_energy(lambda_vec, b, c, s, x) for lambda_vec in simplex_grid]
    
    return itt, np.array(energy_profile)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate energy profiles.')
parser.add_argument('-i', '--input', type=str, required=True, help='Input folder')
args = parser.parse_args()
input_folder = args.input

# Initialize lists to store data
RMSD_energy = []
iterations = []

# Folder containing the data
folders = [f for f in os.listdir(input_folder) if 'analysis' in f and f.split('analysis')[1].isdigit()]
folders.sort(key=lambda x: int(x.split('analysis')[1]))

# Generate a simplex grid
b_sample = np.loadtxt(os.path.join(input_folder, folders[0], 'b_prev.dat'))
N = len(b_sample)
num_points = 200  # Number of points per dimension
simplex_grid = generate_simplex_grid(N, num_points)

# Create a pool of workers
pool = Pool(cpu_count())

# Prepare arguments for parallel processing
args = [(folder, input_folder, simplex_grid) for folder in folders]

# Calculate energy profiles in parallel
results = pool.map(calculate_energy_profile, args)
pool.close()
pool.join()

# Process results
energy_profiles_dict = {itt: energy_profile for itt, energy_profile in results}
iterations = list(energy_profiles_dict.keys())
iterations.sort()

# Calculate min and max energy values
all_energies = np.concatenate(list(energy_profiles_dict.values()))
q1, q3 = np.percentile(all_energies, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_energies = all_energies[(all_energies >= lower_bound) & (all_energies <= upper_bound)]
min_energy = np.min(filtered_energies)
max_energy = np.max(filtered_energies)



# Debugging prints
print(f"Number of iterations: {len(iterations)}")
print(f"Number of energy profiles: {len(energy_profiles_dict)}")

# Calculate RMSD between consecutive iterations
rmsd_values = []
for i in range(1, len(iterations)):
    prev_profile = energy_profiles_dict[iterations[i - 1]]
    curr_profile = energy_profiles_dict[iterations[i]]
    rmsd = np.sqrt(np.mean((curr_profile - prev_profile) ** 2))
    rmsd_values.append(rmsd)

# Plot RMSD values over iterations
plt.figure()
plt.plot(iterations[1:], rmsd_values, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('RMSD')
plt.yscale('log')
plt.title(f'RMSD of Energy Profiles Between Consecutive Iterations for {input_folder}')
plt.grid(visible=True)
plt.savefig('RMSD_Energy_Profiles.png')



# Create Interactive 2D Plot with Slider (N = 2)
if N == 2:
    fig_2d = go.Figure()

    for itt, energy_profile in energy_profiles_dict.items():
        fig_2d.add_trace(go.Scatter(x=simplex_grid[:, 0], y=energy_profile, mode='markers',
                                    marker=dict(color=energy_profile, colorscale='Viridis', colorbar=dict(title='Energy')),
                                    name=f'Iteration {itt}', visible=False))

    # Make the first iteration visible
    fig_2d.data[0].visible = True

    # Create slider steps
    steps = []
    for i in range(len(fig_2d.data)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig_2d.data)},
                  {'title': f"2D Energy Profile - Iteration {iterations[i]}"}],
        )
        step['args'][0]['visible'][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_2d.update_layout(
        sliders=sliders,
        title='2D Energy Profile',
        xaxis_title='λ1',
        yaxis_title='Energy',
        showlegend=False
    )

    fig_2d.show()

# Create Interactive 3D Plot with Slider (N = 3)
if N == 3:
    fig_3d = go.Figure()

    for itt, energy_profile in energy_profiles_dict.items():
        fig_3d.add_trace(go.Scatter3d(
            x=simplex_grid[:, 0] - simplex_grid[:, 1],
            y=simplex_grid[:, 0] - (1 - simplex_grid[:, 0] - simplex_grid[:, 1]),
            z=energy_profile,
            mode='markers',
            marker=dict(size=5, color=energy_profile, colorscale='Viridis', colorbar=dict(title='Energy')),
            name=f'Iteration {itt}', visible=False))

    # Make the first iteration visible
    fig_3d.data[0].visible = True

    # Create slider steps
    steps = []
    for i in range(len(fig_3d.data)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig_3d.data)},
                  {'title': f"3D Energy Profile - Iteration {iterations[i]}"}],
        )
        step['args'][0]['visible'][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_3d.update_layout(
        sliders=sliders,
        title='3D Energy Profile',
        scene=dict(
            xaxis=dict(title='λ1 - λ2', range=[-1, 1]),
            yaxis=dict(title='λ1 - λ3', range=[-1, 1]),
            zaxis=dict(title='Energy', range=[-2, max_energy])
        ),
        showlegend=False
    )

    # fig_3d.show()
    # Save the figure as an HTML file
    fig_3d.write_html('3D_Energy_Profile.html')

    # Create a directory to save the frames
    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Generate frames and save as images
    for itt, energy_profile in energy_profiles_dict.items():
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=simplex_grid[:, 0] - simplex_grid[:, 1],
                    y=simplex_grid[:, 0] - (1 - simplex_grid[:, 0] - simplex_grid[:, 1]),
                    z=energy_profile,
                    mode='markers',
                    marker=dict(size=5, color=energy_profile, colorscale='Viridis', colorbar=dict(title='Energy'))
                )
            ]
        )
        fig.update_layout(
            title=f'3D Energy Profile - Iteration {itt}',
            scene=dict(
                xaxis=dict(title='λ1 - λ2', range=[-1, 1]),
                yaxis=dict(title='λ1 - λ3', range=[-1, 1]),
                zaxis=dict(title='Energy', range=[-2, max_energy])
            )
        )
        
        frame_filename = os.path.join(frames_dir, f'frame_{itt:04d}.png')
        fig.write_image(frame_filename)

    # Compile images into a video using ffmpeg
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    input_pattern = os.path.join(frames_dir, 'frame_%04d.png')
    output_video = '3D_Energy_Profile.mp4'

    # Use ffmpeg to create video
    os.system(f'{ffmpeg_path} -r 10 -i {input_pattern} -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_video}')

    print(f"3D animation saved as {output_video}")





# # Initialize previous energy profile as None
# previous_energy_profile = None

# # Initialize lists to store data
# RMSD_energy = []
# iterations = []

# # Iterate through the folders and extract the data
# for folder in folders:
#     itt = int(folder.split('analysis')[1])
#     iterations.append(itt)
#     b = np.loadtxt(os.path.join(input_folder, folder, 'b_prev.dat'))
#     c = np.loadtxt(os.path.join(input_folder, folder, 'c_prev.dat'))
#     x = np.loadtxt(os.path.join(input_folder, folder, 'x_prev.dat'))
#     s = np.loadtxt(os.path.join(input_folder, folder, 's_prev.dat'))
    
#     simplex_grid = generate_simplex_grid(len(b), num_points)
    
#     # Calculate the energy profile
#     energy_profile = []
#     for lambda_vec in simplex_grid:
#         E_b_val = energy_E_b(lambda_vec)
#         E_c_val = energy_E_c(lambda_vec)
#         E_s_val = energy_E_s(lambda_vec)
#         E_x_val = energy_E_x(lambda_vec)
#         total_energy = E_b_val + E_c_val + E_s_val + E_x_val
#         energy_profile.append(total_energy)

#     energy_profile = np.array(energy_profile)
#     # Calculate RMSD with the previous iteration
#     if previous_energy_profile is not None:
#         rmsd = np.sqrt(np.mean((energy_profile - previous_energy_profile) ** 2))
#         RMSD_energy.append(rmsd)
        
#     # Update the previous energy profile
#     previous_energy_profile = energy_profile
    
# # Plot the RMSD over iterations
# plt.plot(iterations[1:], RMSD_energy, marker='o', linestyle='-', color='b')
# plt.xlabel('Iteration')
# plt.ylabel('RMSD of Energy Profile')
# plt.title('RMSD of Energy Profile Over Iterations')
# plt.grid(True)
# plt.show()