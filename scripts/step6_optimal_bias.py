#!/usr/bin/env python

# step6_optimal_bias.py
# Copyright (c) 2023-2025 Stanislav Cherepanov
#
# This script analyzes bias search results for multi-state lambda simulations.
# It processes all analysis folders, extracts physical state populations (where lambda > cutoff),
# and evaluates both population equality and transition efficiency between states.
# The script ranks all runs, predicts optimal bias parameters using machine learning, and
# provides detailed statistics on both population balance and transition metrics.
#
# Usage example:
#   python step6_optimal_bias.py -i arg -c 0.985 -p 8
#
# Requirements: numpy, scipy, scikit-learn, matplotlib


import os
import numpy as np
import argparse
from scipy.optimize import least_squares, minimize, Bounds
from multiprocessing import Pool
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_nsubs_info(input_folder):
    """
    Load nsubs file to understand how lambda columns are grouped into sites.
    
    Returns:
        sites_info: dict with 'n_sites', 'sites' (list of site sizes), 'groups' (list of column ranges for each site)
    """
    nsubs_file = os.path.join(input_folder, 'prep', 'nsubs')
    
    if not os.path.exists(nsubs_file):
        print(f"Warning: {nsubs_file} not found, assuming single site with all lambdas")
        return None
    
    try:
        with open(nsubs_file, 'r') as f:
            content = f.read().strip()
        
        # Parse nsubs format
        if content == '':
            return None
        
        # Handle both single number (one site) and space-separated numbers (multiple sites)
        parts = content.split()
        
        if len(parts) == 1:
            # Single site with n substituents
            n_substituents = int(parts[0])
            return {
                'n_sites': 1,
                'sites': [n_substituents],
                'groups': [list(range(n_substituents))]  # All lambdas belong to site 0
            }
        else:
            # Multiple sites
            sites = [int(x) for x in parts]
            n_sites = len(sites)
            
            # Create column groupings
            groups = []
            col_start = 0
            for site_size in sites:
                groups.append(list(range(col_start, col_start + site_size)))
                col_start += site_size
            
            return {
                'n_sites': n_sites,
                'sites': sites,
                'groups': groups
            }
            
    except Exception as e:
        print(f"Warning: Could not parse {nsubs_file}: {e}")
        return None

def aggregate_lambda_by_sites(lambda_data, sites_info, cut_off):
    """
    Aggregate lambda columns by sites as defined in nsubs.
    
    For each site, calculate the fraction of time where ANY lambda in that site > cut_off.
    
    Args:
        lambda_data: numpy array of shape (n_steps, n_lambdas)
        sites_info: dict from load_nsubs_info()
        cut_off: threshold for physical state
    
    Returns:
        site_fractions: array of shape (n_sites,) with fraction for each site
    """
    if sites_info is None:
        # No nsubs file, treat each lambda as separate site
        return np.sum(lambda_data > cut_off, axis=0) / lambda_data.shape[0]
    
    site_fractions = []
    
    for site_group in sites_info['groups']:
        # For this site, check if ANY lambda in the group is > cut_off
        site_lambdas = lambda_data[:, site_group]
        is_physical = np.any(site_lambdas > cut_off, axis=1)
        fraction = np.sum(is_physical) / lambda_data.shape[0]
        site_fractions.append(fraction)
    
    return np.array(site_fractions)

def process_folder_bias(args):
    """
    Process a single analysis folder:
      - Loads all files with "Lambda" in the name from the 'data' subfolder.
      - Concatenates the data into one numpy array.
      - Computes the fraction (per site) of λ values > cut_off using nsubs information.
      - Loads the bias matrix data from b_sum.dat, c_sum.dat, s_sum.dat, x_sum.dat files.
    
    Returns:
      iteration (int), col_counts: a 1D array of fractions (one per site),
      and bias_matrices: dict with 'b', 'c', 's', 'x' matrices.
    """
    folder, input_folder, cut_off, sites_info = args
    itt = int(folder.split('analysis')[1])
    folder_path = os.path.join(input_folder, folder)
    
    # Load Lambda data
    data_path = os.path.join(folder_path, 'data')
    if not os.path.exists(data_path):
        return None
        
    data_files = [f for f in os.listdir(data_path) if 'Lambda' in f and f.endswith('.dat')]
    if not data_files:
        return None
        
    data = None
    total_rows = 0
    
    print(f"Processing {folder}: found {len(data_files)} Lambda files")
    
    for data_file in data_files:
        file_path = os.path.join(data_path, data_file)
        try:
            dat = np.loadtxt(file_path)
            if dat.ndim == 1:
                dat = dat.reshape(1, -1)  # Handle single row case
            
            if data is None:
                data = dat
            else:
                data = np.concatenate([data, dat], axis=0)
            total_rows += dat.shape[0]
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if data is None or data.size == 0:
        print(f"Warning: No valid data found in {folder}")
        return None
        
    num_rows = data.shape[0]
    n_cols = data.shape[1]
    
    # Use site-based aggregation if nsubs information is available
    if sites_info is not None:
        col_counts = aggregate_lambda_by_sites(data, sites_info, cut_off)
    else:
        # Fallback to per-column analysis
        col_counts = np.sum(data > cut_off, axis=0) / num_rows
    
    # Analyze transition behavior
    transition_metrics = analyze_transitions(data, cut_off)
    
    # Load bias matrices
    bias_matrices = {}
    for matrix_name in ['b', 'c', 's', 'x']:
        matrix_file = os.path.join(folder_path, f'{matrix_name}_sum.dat')
        if os.path.exists(matrix_file):
            try:
                matrix = np.loadtxt(matrix_file)
                if matrix_name == 'b':
                    # b should be 1D array
                    if matrix.ndim == 2 and matrix.shape[0] == 1:
                        matrix = matrix.flatten()
                    elif matrix.ndim == 2 and matrix.shape[1] == 1:
                        matrix = matrix.flatten()
                bias_matrices[matrix_name] = matrix
            except Exception as e:
                print(f"Warning: Could not load {matrix_file}: {e}")
                bias_matrices[matrix_name] = None
        else:
            print(f"Warning: {matrix_file} not found in {folder}")
            bias_matrices[matrix_name] = None
    
    # Verify matrix dimensions
    if bias_matrices['b'] is not None and len(bias_matrices['b']) != n_cols:
        print(f"Warning: b matrix dimension mismatch in {folder}: {len(bias_matrices['b'])} vs {n_cols}")
    
    for matrix_name in ['c', 's', 'x']:
        matrix = bias_matrices[matrix_name]
        if matrix is not None and matrix.shape != (n_cols, n_cols):
            print(f"Warning: {matrix_name} matrix dimension mismatch in {folder}: {matrix.shape} vs ({n_cols}, {n_cols})")
    
    return itt, col_counts, bias_matrices, transition_metrics

def gather_runs(input_folder, cut_off, n_processes=None, sites_info=None):
    """
    Find all analysis folders (names like "analysisX") that contain a "data" subfolder,
    process them in parallel, and return a list of tuples (iteration, col_counts, bias_matrices, transition_metrics).
    """
    folders = [f for f in os.listdir(input_folder)
               if 'analysis' in f and f.split('analysis')[1].isdigit()]
    folders = sorted(folders, key=lambda x: int(x.split('analysis')[1]))
    folders = [f for f in folders if os.path.exists(os.path.join(input_folder, f, 'data'))]
    
    if not folders:
        print("No valid analysis folders found.")
        return []
    
    print(f"Found {len(folders)} analysis folders to process")
    
    # Prepare arguments for parallel processing
    args_list = [(f, input_folder, cut_off, sites_info) for f in folders]
    
    # Use multiprocessing to process folders in parallel
    if n_processes is None:
        n_processes = min(len(folders), os.cpu_count())
    
    print(f"Using {n_processes} processes for parallel processing")
    
    start_time = time.time()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_folder_bias, args_list)
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} out of {len(folders)} folders")
    
    return valid_results

def compute_effective_bias(lambda_vec, bias_matrices):
    """
    Compute the total energy/bias for a given lambda vector using the bias matrices.
    This follows the energy calculation from step7_energy_profiles.py:
    
    Total Energy = E_b + E_c + E_s + E_x
    
    where:
    - E_b = -sum_i(lambda_i * b[i])
    - E_c = sum_i,j((c[i,j] + c[j,i]) * lambda_i * lambda_j)
    - E_s = sum_i,j((s[i,j]/(lambda_i + 0.017) + s[j,i]/(lambda_j + 0.017)) * lambda_i * lambda_j)
    - E_x = sum_i,j(x[i,j] * lambda_j * (1 - exp(-5.56 * lambda_i)) + x[j,i] * lambda_i * (1 - exp(-5.56 * lambda_j)))
    """
    b = bias_matrices['b']
    c = bias_matrices['c']
    s = bias_matrices['s']
    x = bias_matrices['x']
    
    N = len(lambda_vec)
    
    # E_b contribution
    E_b = 0
    for i in range(N):
        E_b += -(lambda_vec[i] * b[i])
    
    # E_c contribution
    E_c = 0
    for i in range(N):
        for j in range(N):
            E_c += (c[i, j] + c[j, i]) * lambda_vec[i] * lambda_vec[j]
    
    # E_s contribution
    E_s = 0
    for i in range(N):
        for j in range(N):
            E_s += (s[i, j] / (lambda_vec[i] + 0.017) + s[j, i] / (lambda_vec[j] + 0.017)) * (lambda_vec[i] * lambda_vec[j])
    
    # E_x contribution
    E_x = 0
    for i in range(N):
        for j in range(N):
            E_x += x[i, j] * (lambda_vec[j] * (1 - np.exp(-5.56 * lambda_vec[i]))) + x[j, i] * (lambda_vec[i] * (1 - np.exp(-5.56 * lambda_vec[j])))
    
    return E_b + E_c + E_s + E_x

def compute_energy_profile_for_run(run_data, bias_matrices, cut_off):
    """
    Compute energy profile using actual lambda trajectories from the run.
    Returns energy statistics for physical states.
    """
    if any(matrix is None for matrix in bias_matrices.values()):
        return None
    
    # Use actual lambda data from the run
    itt, f, bias_matrices_run, trans_metrics = run_data
    
    # Load the actual lambda data for this run
    folder_path = os.path.join(input_folder, f'analysis{itt}')
    data_path = os.path.join(folder_path, 'data')
    
    if not os.path.exists(data_path):
        return None
    
    data_files = [f for f in os.listdir(data_path) if 'Lambda' in f and f.endswith('.dat')]
    if not data_files:
        return None
    
    # Load all lambda data
    all_lambda_data = []
    for data_file in data_files:
        file_path = os.path.join(data_path, data_file)
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            all_lambda_data.append(data)
        except:
            continue
    
    if not all_lambda_data:
        return None
    
    # Concatenate all lambda data
    lambda_trajectories = np.vstack(all_lambda_data)
    
    # Identify physical states (any lambda > cut_off)
    is_physical = np.any(lambda_trajectories > cut_off, axis=1)
    physical_indices = np.where(is_physical)[0]
    
    if len(physical_indices) == 0:
        return None
    
    # Sample a subset of physical states for energy calculation (to speed up)
    # max_samples = min(1000, len(physical_indices))
    max_samples = physical_indices
    sampled_indices = np.random.choice(physical_indices, max_samples, replace=False)
    
    # Compute energies for sampled physical states
    energies = []
    for idx in sampled_indices:
        lambda_vec = lambda_trajectories[idx]
        try:
            energy = compute_effective_bias(lambda_vec, bias_matrices)
            energies.append(energy)
        except:
            continue
    
    if len(energies) == 0:
        return None
    
    energies = np.array(energies)
    
    return {
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'min_energy': np.min(energies),
        'max_energy': np.max(energies),
        'num_samples': len(energies)
    }

def analyze_energy_landscape(runs, input_folder, cut_off):
    """
    Analyze the energy landscape using actual lambda trajectories.
    """
    np.random.seed(42)  # For reproducible sampling
    energy_profiles = []
    
    for run_data in runs:
        itt, f, bias_matrices, trans_metrics = run_data
        
        # Load actual lambda data for this run
        folder_path = os.path.join(input_folder, f'analysis{itt}')
        data_path = os.path.join(folder_path, 'data')
        
        if not os.path.exists(data_path):
            continue
        
        data_files = [f for f in os.listdir(data_path) if 'Lambda' in f and f.endswith('.dat')]
        if not data_files:
            continue
        
        # Load all lambda data
        all_lambda_data = []
        for data_file in data_files:
            file_path = os.path.join(data_path, data_file)
            try:
                data = np.loadtxt(file_path)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                all_lambda_data.append(data)
            except:
                continue
        
        if not all_lambda_data:
            continue
        
        # Concatenate all lambda data
        lambda_trajectories = np.vstack(all_lambda_data)
        
        # Identify physical states (any lambda > cut_off)
        is_physical = np.any(lambda_trajectories > cut_off, axis=1)
        physical_indices = np.where(is_physical)[0]
        
        if len(physical_indices) == 0:
            continue
        
        # Sample physical states for energy calculation (to speed up)
        max_samples = min(500, len(physical_indices))
        sampled_indices = np.random.choice(physical_indices, max_samples, replace=False)
        
        # Compute energies for sampled physical states
        energies = []
        for idx in sampled_indices:
            lambda_vec = lambda_trajectories[idx]
            try:
                energy = compute_effective_bias(lambda_vec, bias_matrices)
                energies.append(energy)
            except:
                continue
        
        if len(energies) == 0:
            continue
        
        energies = np.array(energies)
        
        energy_profile = {
            'iteration': itt,
            'fractions': f,
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'energy_range': np.max(energies) - np.min(energies),
            'num_samples': len(energies),
            'bias_matrices': bias_matrices
        }
        
        energy_profiles.append(energy_profile)
    
    return energy_profiles

def extract_bias_features(bias_matrices):
    """
    Extract features from bias matrices for analysis.
    Returns a flattened array of all bias parameters.
    """
    b = bias_matrices['b']
    c = bias_matrices['c']
    s = bias_matrices['s']
    x = bias_matrices['x']
    
    features = []
    
    # Add b parameters (flatten the matrix)
    features.extend(b.flatten())
    
    # Add c parameters (upper triangular only)
    n = c.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            features.append(c[i, j])
    
    # Add s parameters (off-diagonal)
    for i in range(n):
        for j in range(n):
            if i != j:
                features.append(s[i, j])
    
    # Add x parameters (off-diagonal)
    for i in range(n):
        for j in range(n):
            if i != j:
                features.append(x[i, j])
    
    return np.array(features)

def analyze_bias_relationships(runs):
    """
    Analyze the relationship between bias parameters and population distributions.
    """
    X = []  # Bias features
    y_fractions = []  # Fraction distributions
    y_scores = []  # Quality scores
    transition_data = []  # Transition metrics
    
    for itt, f, bias_matrices, trans_metrics in runs:
        if any(matrix is None for matrix in bias_matrices.values()):
            continue
            
        # Extract bias features
        bias_features = extract_bias_features(bias_matrices)
        X.append(bias_features)
        y_fractions.append(f)
        transition_data.append(trans_metrics)
        
        # Calculate quality score (80% equal populations, 20% good transitions)
        f_std = np.std(f)
        f_max_diff = np.max(f) - np.min(f)
        f_min = np.min(f)
        
        # Population equality score (lower is better)
        population_score = f_std + 10 * f_max_diff
        
        # Transition quality score based on actual transition metrics
        transition_score = 0
        
        # Penalize very low physical state sampling
        if trans_metrics['physical_fraction'] < 0.5:
            transition_score += 100 * (0.5 - trans_metrics['physical_fraction'])
        
        # Penalize very low transition rates (poor mixing)
        if trans_metrics['transition_rate'] < 0.001:  # Less than 0.1% transitions per step
            transition_score += 50 * (0.001 - trans_metrics['transition_rate'])
        
        # Penalize very long alchemical states (inefficient transitions)
        if trans_metrics['avg_alchemical_length'] > 100:
            transition_score += 10 * (trans_metrics['avg_alchemical_length'] - 100)
        
        # Penalize low fractions in physical states
        if f_min < 0.05:
            transition_score += 100 * (0.05 - f_min)
        
        # Combined score (80% population equality, 20% transition quality)
        combined_score = 0.8 * population_score + 0.2 * transition_score
        y_scores.append(combined_score)
    
    X = np.array(X)
    y_fractions = np.array(y_fractions)
    y_scores = np.array(y_scores)
    
    print(f"Analyzing {len(X)} runs with {X.shape[1]} bias parameters")
    
    return X, y_fractions, y_scores, transition_data

def predict_optimal_bias(X, y_fractions, y_scores, target_fractions=None, run_scores=None):
    """
    Use machine learning to predict optimal bias parameters.
    
    The goal is to achieve EQUAL PHYSICAL STATE FRACTIONS for each substituent.
    Physical state fraction = fraction of time when λᵢ > cutoff for substituent i.
    
    This is NOT about equal overall averages, but about equal time spent in physical states.
    """
    n_substituents = y_fractions.shape[1]
    
    print(f"OPTIMIZATION GOAL: Equal physical state fractions (λ > cutoff) for all substituents")
    print(f"This means equal fractions like step5's 'Fraction > 0.985' values")
    
    # Default target: equal fractions based on successful runs' scale
    if target_fractions is None:
        # Use the scale from successful runs rather than theoretical 1/3
        if run_scores is not None and len(run_scores) > 0:
            best_fractions = run_scores[0]['fractions']
            mean_fraction = np.mean(best_fractions)
            target_fractions = np.full(n_substituents, mean_fraction)
            print(f"Best run fractions: {best_fractions}")
            print(f"Target: equal fractions at {mean_fraction:.5f} ({mean_fraction*100:.3f}%) for each substituent")
        else:
            # Fallback: assume equal at 10% level (typical for physical states)
            target_fractions = np.full(n_substituents, 0.1)
            print(f"Fallback target: 10% physical state fraction for each substituent")
    
    print(f"Target fractions: {target_fractions}")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train separate models for each substituent's fraction
    models = []
    for i in range(n_substituents):
        model = LinearRegression()
        model.fit(X_scaled, y_fractions[:, i])
        models.append(model)
        print(f"Model for substituent {i}: R² = {model.score(X_scaled, y_fractions[:, i]):.3f}")
    
    # Analyze successful runs to understand the free energy landscape
    successful_runs = []
    if run_scores is not None:
        successful_runs = [run for run in run_scores[:10] if run['f_std'] < 0.01 and run['f_max_diff'] < 0.05]
    
    if successful_runs:
        print(f"\nAnalyzing {len(successful_runs)} successful runs:")
        successful_b_matrices = []
        for run in successful_runs:
            b_matrix = np.array(run['bias_matrices']['b'])
            # Normalize so first element is 0 (reference state)
            b_normalized = b_matrix - b_matrix[0]
            successful_b_matrices.append(b_normalized)
            print(f"  Run {run['iteration']}: B = {b_normalized}, score = {run['score']:.3f}")
        
        # Find patterns in successful B matrices
        successful_b_matrices = np.array(successful_b_matrices)
        mean_b = np.mean(successful_b_matrices, axis=0)
        std_b = np.std(successful_b_matrices, axis=0)
        
        print(f"  Mean B pattern: {mean_b}")
        print(f"  Std B pattern: {std_b}")
        
        # Use successful runs to define physical constraints for ML optimization
        if len(successful_runs) > 0:
            print(f"\nUsing ML optimization with physical constraints from successful runs")
            
            # Define bounds based on successful runs
            successful_features = []
            for run in successful_runs:
                features = extract_bias_features(run['bias_matrices'])
                successful_features.append(features)
            successful_features = np.array(successful_features)
            
            # Calculate bounds: mean ± 2*std for each parameter
            feature_means = np.mean(successful_features, axis=0)
            feature_stds = np.std(successful_features, axis=0)
            
            # For parameters with very small std (< 0.1), use a minimum range
            feature_stds = np.maximum(feature_stds, 0.1 * np.abs(feature_means))
            
            lower_bounds = feature_means - 2 * feature_stds
            upper_bounds = feature_means + 2 * feature_stds
            
            print(f"  B matrix bounds: [{lower_bounds[0]:.1f}, {upper_bounds[0]:.1f}] to [{lower_bounds[n_substituents-1]:.1f}, {upper_bounds[n_substituents-1]:.1f}]")
            
            # Convert bounds to scaled space
            dummy_data = np.vstack([lower_bounds, upper_bounds]).reshape(2, -1)
            scaled_bounds = scaler.transform(dummy_data)
            lower_bounds_scaled = scaled_bounds[0]
            upper_bounds_scaled = scaled_bounds[1]
            
            # Enhanced objective function with physical constraints
            def constrained_objective(x_scaled):
                # Ensure we're within bounds
                x_scaled = np.clip(x_scaled, lower_bounds_scaled, upper_bounds_scaled)
                
                predicted_fractions = np.array([model.predict(x_scaled.reshape(1, -1))[0] for model in models])
                
                # Primary objective: minimize deviation from equal populations
                pop_error = np.sum((predicted_fractions - target_fractions)**2)
                
                # Reconstruct bias parameters to check physical constraints
                x_original = scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
                b_values = x_original[:n_substituents]
                
                # Normalize B matrix so first element is 0 (reference state)
                b_normalized = b_values - b_values[0]
                
                # Strong penalty for deviation from successful B patterns
                pattern_penalty = 0
                for i in range(1, len(b_normalized)):
                    if std_b[i] > 0:
                        deviation = abs(b_normalized[i] - mean_b[i]) / std_b[i]
                        pattern_penalty += deviation**2
                    else:
                        # If std is 0, penalize any deviation from mean
                        pattern_penalty += (b_normalized[i] - mean_b[i])**2
                
                # Penalty for very low fractions (physical constraint)
                transition_penalty = 0
                for frac in predicted_fractions:
                    if frac < 0.05:
                        transition_penalty += 1000 * (0.05 - frac)**2
                    elif frac < 0.15:
                        transition_penalty += 100 * (0.15 - frac)**2
                
                # Penalty for being outside successful parameter bounds
                bounds_penalty = 0
                x_original_full = scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
                for i, (val, lower, upper) in enumerate(zip(x_original_full, lower_bounds, upper_bounds)):
                    if val < lower:
                        bounds_penalty += 10 * (lower - val)**2
                    elif val > upper:
                        bounds_penalty += 10 * (val - upper)**2
                
                return 0.5 * pop_error + 0.3 * pattern_penalty + 0.1 * transition_penalty + 0.1 * bounds_penalty
            
            # Use best successful run as starting point
            best_successful_run = successful_runs[0]
            best_features = extract_bias_features(best_successful_run['bias_matrices'])
            initial_guess_scaled = scaler.transform(best_features.reshape(1, -1))[0]
            
            # Constrained optimization
            bounds = Bounds(lower_bounds_scaled, upper_bounds_scaled)
            
            try:
                result = minimize(constrained_objective, initial_guess_scaled, 
                                method='L-BFGS-B', bounds=bounds)
                optimal_bias_scaled = result.x
                print(f"  Optimization converged: {result.success}")
                print(f"  Final objective value: {result.fun:.6f}")
            except:
                print("  Optimization failed, using best successful run")
                optimal_bias_scaled = initial_guess_scaled
            
            # Convert back to original scale
            optimal_bias = scaler.inverse_transform(optimal_bias_scaled.reshape(1, -1))[0]
            
            # Predict fractions with optimal bias
            predicted_fractions = np.array([model.predict(optimal_bias_scaled.reshape(1, -1))[0] for model in models])
            
            return optimal_bias, predicted_fractions, scaler, models
    
    # Optimize bias parameters to achieve target fractions
    def objective(x_scaled):
        predicted_fractions = np.array([model.predict(x_scaled.reshape(1, -1))[0] for model in models])
        
        # Primary objective: minimize deviation from equal populations (80% weight)
        pop_error = np.sum((predicted_fractions - target_fractions)**2)
        
        # Reconstruct bias parameters to check physical constraints
        x_original = scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
        b_values = x_original[:n_substituents]  # First n elements are b parameters
        
        # Physics-based constraint: B matrix represents relative free energies
        # Normalize B matrix so first element is 0 (reference state)
        b_normalized = b_values - b_values[0]
        
        # Penalize deviation from successful patterns if available
        pattern_penalty = 0
        if successful_runs:
            for i in range(1, len(b_normalized)):  # Skip first element (always 0)
                if std_b[i] > 0:
                    deviation = abs(b_normalized[i] - mean_b[i]) / std_b[i]
                    pattern_penalty += deviation**2
        
        # Transition quality objective - penalize very low fractions
        transition_penalty = 0
        for frac in predicted_fractions:
            if frac < 0.05:
                transition_penalty += 100 * (0.05 - frac)**2
            elif frac < 0.10:
                transition_penalty += 10 * (0.10 - frac)**2
        
        return 0.7 * pop_error + 0.2 * transition_penalty + 0.1 * pattern_penalty
    
    # Better initial guess: use the best performing run as starting point
    best_run_idx = np.argmin(y_scores)
    initial_guess = X_scaled[best_run_idx]
    
    # Also try mean of top 5 runs
    best_indices = np.argsort(y_scores)[:5]
    mean_guess = np.mean(X_scaled[best_indices], axis=0)
    
    # Try both initial guesses and pick the better result
    results = []
    for guess, name in [(initial_guess, "best_run"), (mean_guess, "top5_mean")]:
        try:
            result = minimize(objective, guess, method='L-BFGS-B')
            results.append((result, name))
        except:
            continue
    
    # Pick the result with lowest objective value
    if results:
        best_result, best_name = min(results, key=lambda x: x[0].fun)
        print(f"Best optimization result from: {best_name}")
        optimal_bias_scaled = best_result.x
    else:
        # Fallback to simple mean
        optimal_bias_scaled = mean_guess
        print("Using fallback: mean of top 5 runs")
    
    # Convert back to original scale
    optimal_bias = scaler.inverse_transform(optimal_bias_scaled.reshape(1, -1))[0]
    
    # Predict fractions with optimal bias
    predicted_fractions = np.array([model.predict(optimal_bias_scaled.reshape(1, -1))[0] for model in models])
    
    return optimal_bias, predicted_fractions, scaler, models

def reconstruct_bias_matrices(optimal_bias, n_sites, example_bias_matrices=None):
    """
    Reconstruct bias matrices from flattened optimal bias parameters.
    
    Args:
        optimal_bias: flattened array of bias parameters
        n_sites: number of sites (for site-based aggregation)
        example_bias_matrices: example bias matrices to determine actual lambda dimensions
    """
    # Determine the actual number of lambda columns from example matrices
    if example_bias_matrices is not None:
        n_lambdas = len(example_bias_matrices['b'])
    else:
        # Fallback: try to infer from the total number of parameters
        # For n lambdas: b has n elements, c has n(n-1)/2, s has n(n-1), x has n(n-1)
        # Total = n + n(n-1)/2 + n(n-1) + n(n-1) = n + n(n-1)(1/2 + 1 + 1) = n + n(n-1)*2.5
        # Solve: total = n + 2.5*n*(n-1) = n(1 + 2.5*(n-1)) = n(2.5*n - 1.5) for n
        total_params = len(optimal_bias)
        # Solve quadratic equation: 2.5*n^2 - 1.5*n - total = 0
        a, b, c = 2.5, -1.5, -total_params
        discriminant = b**2 - 4*a*c
        n_lambdas = int((-b + np.sqrt(discriminant)) / (2*a))
        print(f"Inferred {n_lambdas} lambda columns from {total_params} parameters")
    
    idx = 0
    
    # Reconstruct b (1D array)
    b = optimal_bias[idx:idx+n_lambdas]
    idx += n_lambdas
    
    # Reconstruct c (upper triangular)
    c = np.zeros((n_lambdas, n_lambdas))
    for i in range(n_lambdas):
        for j in range(i+1, n_lambdas):
            c[i, j] = optimal_bias[idx]
            idx += 1
    
    # Reconstruct s (symmetric, off-diagonal)
    s = np.zeros((n_lambdas, n_lambdas))
    for i in range(n_lambdas):
        for j in range(n_lambdas):
            if i != j:
                s[i, j] = optimal_bias[idx]
                idx += 1
    
    # Reconstruct x (symmetric, off-diagonal)
    x = np.zeros((n_lambdas, n_lambdas))
    for i in range(n_lambdas):
        for j in range(n_lambdas):
            if i != j:
                x[i, j] = optimal_bias[idx]
                idx += 1
    
    return {'b': b, 'c': c, 's': s, 'x': x}

def analyze_transitions(data, cut_off):
    """
    Analyze transition behavior in the lambda data.
    Returns metrics about transitions between physical states.
    """
    # Identify physical states (any lambda > cut_off)
    is_physical = np.any(data > cut_off, axis=1)
    
    # Count transitions: physical -> alchemical -> physical
    transitions = 0
    transition_lengths = []
    alchemical_lengths = []
    
    in_alchemical = False
    alchemical_start = 0
    
    for i, physical in enumerate(is_physical):
        if physical and in_alchemical:
            # End of alchemical state
            alchemical_length = i - alchemical_start
            alchemical_lengths.append(alchemical_length)
            transitions += 1
            in_alchemical = False
        elif not physical and not in_alchemical:
            # Start of alchemical state
            alchemical_start = i
            in_alchemical = True
    
    # Calculate metrics
    total_steps = len(data)
    physical_steps = np.sum(is_physical)
    alchemical_steps = total_steps - physical_steps
    
    transition_rate = transitions / total_steps if total_steps > 0 else 0
    avg_alchemical_length = np.mean(alchemical_lengths) if alchemical_lengths else 0
    
    return {
        'total_steps': total_steps,
        'physical_steps': physical_steps,
        'alchemical_steps': alchemical_steps,
        'transitions': transitions,
        'transition_rate': transition_rate,
        'avg_alchemical_length': avg_alchemical_length,
        'physical_fraction': physical_steps / total_steps if total_steps > 0 else 0
    }

def plot_2d_energy_landscape(optimal_matrices, n_substituents, output_dir=".", filename_prefix="energy_landscape"):
    """
    Create 2D energy landscape plots for all pairs of substituents.
    """
    print(f"\nCreating 2D energy landscape plots...")
    
    # Create grid of lambda values
    lambda_values = np.linspace(0, 1, 50)
    
    # For 3 substituents, create plots for pairs (0,1), (0,2), (1,2)
    pairs = [(i, j) for i in range(n_substituents) for j in range(i+1, n_substituents)]
    
    # Skip plotting if there are no pairs (single substituent system)
    if len(pairs) == 0:
        print("Single substituent system - no pairs to plot for 2D energy landscape")
        return
    
    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    
    for idx, (i, j) in enumerate(pairs):
        # Create meshgrid for this pair
        lambda_i, lambda_j = np.meshgrid(lambda_values, lambda_values)
        energies = np.zeros_like(lambda_i)
        
        # Calculate energy for each point in the grid
        for row in range(lambda_i.shape[0]):
            for col in range(lambda_i.shape[1]):
                lambda_vec = np.zeros(n_substituents)
                lambda_vec[i] = lambda_i[row, col]
                lambda_vec[j] = lambda_j[row, col]
                # Set other lambdas to small values
                for k in range(n_substituents):
                    if k != i and k != j:
                        lambda_vec[k] = 0.01
                
                try:
                    energy = compute_effective_bias(lambda_vec, optimal_matrices)
                    energies[row, col] = energy
                except:
                    energies[row, col] = np.nan
        
        # Create contour plot
        ax = axes[idx]
        im = ax.contourf(lambda_i, lambda_j, energies, levels=20, cmap='viridis')
        ax.contour(lambda_i, lambda_j, energies, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        
        # Mark physical states (high lambda values)
        physical_threshold = 0.985
        ax.axhline(y=physical_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Physical threshold λ={physical_threshold}')
        ax.axvline(x=physical_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Mark equal population point (if applicable)
        if n_substituents == 3:
            equal_lambda = 1.0 / 3.0
            ax.plot(equal_lambda, equal_lambda, 'ro', markersize=8, label=f'Equal population (λ={equal_lambda:.3f})')
        
        ax.set_xlabel(f'λ{i}')
        ax.set_ylabel(f'λ{j}')
        ax.set_title(f'Energy Landscape: λ{i} vs λ{j}')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Energy (kT)')
    
    plt.tight_layout()
    
    # Save plot
    landscape_filename = f"{output_dir}/{filename_prefix}_2d_landscape.png"
    plt.savefig(landscape_filename, dpi=300, bbox_inches='tight')
    print(f"2D energy landscape saved to: {landscape_filename}")
    plt.close()

def plot_predicted_lambda_distribution(optimal_matrices, run_scores, n_substituents, output_dir=".", filename_prefix="lambda_distribution"):
    """
    Create plots showing predicted vs observed lambda distributions.
    """
    print(f"\nCreating lambda distribution plots...")
    
    # Sample lambda values for monte carlo estimation
    # Use uniform sampling in [0,1] for each lambda (NOT constrained to sum=1)
    n_samples = 500
    np.random.seed(42)
    
    # Generate random lambda vectors - each lambda independently uniform in [0,1]
    random_lambdas = np.random.uniform(0, 1, size=(n_samples, n_substituents))
    
    # Calculate energies for all samples
    energies = []
    valid_lambdas = []
    
    for lambda_vec in random_lambdas:
        try:
            energy = compute_effective_bias(lambda_vec, optimal_matrices)
            energies.append(energy)
            valid_lambdas.append(lambda_vec)
        except:
            continue
    
    energies = np.array(energies)
    valid_lambdas = np.array(valid_lambdas)
    
    if len(energies) == 0:
        print("Warning: Could not calculate energies for lambda distribution")
        return None, None
    
    # Calculate Boltzmann weights (assuming kT = 1)
    kT = 1.0
    boltzmann_weights = np.exp(-energies / kT)
    boltzmann_weights /= np.sum(boltzmann_weights)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Energy distribution
    ax1 = axes[0, 0]
    ax1.hist(energies, bins=50, alpha=0.7, density=True, edgecolor='black')
    ax1.set_xlabel('Energy (kT)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Energy Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Boltzmann-weighted lambda distributions
    ax2 = axes[0, 1]
    for i in range(n_substituents):
        # Calculate weighted histogram
        hist, bins = np.histogram(valid_lambdas[:, i], bins=30, weights=boltzmann_weights, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(bin_centers, hist, label=f'λ{i}', linewidth=2)
    
    ax2.set_xlabel('λ value')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Predicted λ Distributions (Boltzmann weighted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison with best observed run
    ax3 = axes[1, 0]
    best_run = run_scores[0]
    observed_fractions = best_run['fractions']
    
    # Calculate predicted fractions from Boltzmann weighting
    # Match the original analysis: fraction = P(λᵢ > threshold) for each substituent
    threshold = 0.985
    predicted_fractions = []
    
    for i in range(n_substituents):
        # For each substituent, calculate weighted probability that λᵢ > threshold
        mask = valid_lambdas[:, i] > threshold
        if np.sum(mask) > 0:
            weighted_prob = np.sum(boltzmann_weights[mask])
            predicted_fractions.append(weighted_prob)
        else:
            predicted_fractions.append(0.0)
    
    predicted_fractions = np.array(predicted_fractions)
    
    x = np.arange(n_substituents)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, observed_fractions, width, label='Observed (best run)', alpha=0.8)
    bars2 = ax3.bar(x + width/2, predicted_fractions, width, label='Predicted (Boltzmann)', alpha=0.8)
    
    ax3.set_xlabel('Substituent')
    ax3.set_ylabel('Fraction in Physical State')
    ax3.set_title('Observed vs Predicted Fractions')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Sub {i}' for i in range(n_substituents)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Lambda space projection (for 3D case)
    ax4 = axes[1, 1]
    if n_substituents == 3:
        # Project onto 2D triangle (ternary plot approximation)
        # Use first two lambdas as coordinates
        scatter = ax4.scatter(valid_lambdas[:, 0], valid_lambdas[:, 1], 
                            c=boltzmann_weights, cmap='plasma', alpha=0.6, s=1)
        ax4.set_xlabel('λ0')
        ax4.set_ylabel('λ1')
        ax4.set_title('Lambda Space Distribution\n(colored by Boltzmann weight)')
        
        # Mark physical region
        physical_threshold = 0.985
        ax4.axhline(y=physical_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax4.axvline(x=physical_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Mark observed best run point
        ax4.plot(observed_fractions[0], observed_fractions[1], 'ro', markersize=8, 
                label=f'Best observed run')
        ax4.legend()
        
        plt.colorbar(scatter, ax=ax4, label='Boltzmann Weight')
    else:
        # For other cases, show lambda correlations
        for i in range(min(2, n_substituents-1)):
            ax4.scatter(valid_lambdas[:, i], valid_lambdas[:, i+1], 
                       c=boltzmann_weights, cmap='plasma', alpha=0.6, s=1)
        ax4.set_xlabel(f'λ0')
        ax4.set_ylabel(f'λ1')
        ax4.set_title('Lambda Correlations')

    plt.tight_layout()
    
    # Save plot
    distribution_filename = f"{output_dir}/{filename_prefix}_distribution.png"
    plt.savefig(distribution_filename, dpi=300, bbox_inches='tight')
    print(f"Lambda distribution plots saved to: {distribution_filename}")
    plt.close()
    
    # Print summary statistics
    print(f"\nPredicted distribution summary:")
    print(f"Observed fractions: {observed_fractions}")
    print(f"Predicted fractions: {predicted_fractions}")
    print(f"Fraction difference: {np.abs(observed_fractions - predicted_fractions)}")
    print(f"Mean energy: {np.mean(energies):.3f} ± {np.std(energies):.3f} kT")
    
    return predicted_fractions, energies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder',
                        help='Input folder containing analysis folders', required=True)
    parser.add_argument('-c', '--cut_off',
                        type=float,
                        help='Cut-off value for physical state (default 0.985)', default=0.985)
    parser.add_argument('-p', '--processes',
                        type=int,
                        help='Number of processes for parallel processing (default: auto)', default=None)
    args = parser.parse_args()
    input_folder = args.input_folder
    cut_off = args.cut_off
    n_processes = args.processes

    # Load nsubs information for site-based aggregation
    sites_info = load_nsubs_info(input_folder)
    if sites_info is None:
        print("Warning: Could not load nsubs information, using per-column analysis")
    else:
        print(f"Loaded nsubs information: {sites_info['n_sites']} sites")
        for i, site_cols in enumerate(sites_info['groups']):
            print(f"Site {i+1}: columns {site_cols}")
    
    # For consistency with step5, always use per-column analysis regardless of nsubs
    print("Note: Using per-column analysis for step5 compatibility")
    sites_info = None
    
    # Gather bias data from all runs (each run returns fractions, bias matrices).
    runs = gather_runs(input_folder, cut_off, n_processes, sites_info)
    if len(runs) == 0:
        print("No valid analysis folders found.")
        return

    print(f"Found {len(runs)} analysis folders.")
    print(f"Using cut-off value: {cut_off}")
    print(f"Run scoring: using step5_bias_search.py method (score = avg - 10*diff)")
    
    # Analyze each run to find the one with most equal distributions (focus on populations)
    # NOTE: This uses the same scoring approach as step5_bias_search.py for consistency:
    # score = avg - alpha * diff, where alpha = 10.0 (penalty factor for imbalance)
    run_scores = []
    
    for itt, f, bias_matrices, trans_metrics in runs:
        if any(matrix is None for matrix in bias_matrices.values()):
            continue
        
        # Calculate metrics for this run
        f_mean = np.mean(f)
        f_std = np.std(f)
        f_max_diff = np.max(f) - np.min(f)
        f_min = np.min(f)
        
        # Check if any substituent is dominating (> 80% or < 5%)
        f_max = np.max(f)
        is_dominating = f_max > 0.8 or f_min < 0.05
        
        # Use the same scoring approach as step5_bias_search.py
        # Score = avg - alpha * diff (higher is better, lower imbalance is better)
        alpha = 10.0  # Penalty factor for imbalance; same as step5
        score = f_mean - alpha * f_max_diff
        
        # Optional: Apply small penalty for very poor transitions (but keep step5 compatibility)
        transition_penalty = 0
        if is_dominating:
            transition_penalty += 0.1  # Small penalty for dominating states
        if trans_metrics['physical_fraction'] < 0.2:
            transition_penalty += 0.05 * (0.2 - trans_metrics['physical_fraction'])
        
        # Final score (matches step5 approach with minimal transition consideration)
        score = score - transition_penalty
        
        # Keep coefficient of variation for reporting compatibility
        cv = f_std / f_mean if f_mean > 0 else 100
        
        run_scores.append({
            'iteration': itt,
            'fractions': f,
            'transition_metrics': trans_metrics,
            'f_mean': f_mean,
            'f_std': f_std,
            'f_max_diff': f_max_diff,
            'f_min': f_min,
            'f_max': f_max,
            'is_dominating': is_dominating,
            'cv': cv,
            'score': score,
            'bias_matrices': bias_matrices
        })
    
    # Sort by score (best first - higher score is better, same as step5)
    run_scores.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 10 runs with best population balance (using step5 scoring: avg - 10*diff):")
    print("Rank | Iter | F_mean | F_std  | F_diff | F_min  | CV    | Phys% | Trans/k | Score")
    print("-" * 85)
    
    for rank, run_info in enumerate(run_scores[:10]):
        trans = run_info['transition_metrics']
        print(f"{rank+1:4d} | {run_info['iteration']:4d} | "
              f"{run_info['f_mean']:6.4f} | {run_info['f_std']:6.4f} | "
              f"{run_info['f_max_diff']:6.4f} | {run_info['f_min']:6.4f} | "
              f"{run_info['cv']:5.2f} | "
              f"{trans['physical_fraction']*100:5.1f} | "
              f"{trans['transition_rate']*1000:7.2f} | "
              f"{run_info['score']:8.2f}")
    
    # Log top 5 runs in step5 format for consistency
    print(f"\nTop 5 runs based on combined score (step5 compatible):")
    for rank, run_info in enumerate(run_scores[:5]):
        f = run_info['fractions']
        diff_percent = np.round(run_info['f_max_diff'] * 100, 2)
        avg_percent = np.round(run_info['f_mean'] * 100, 2)
        print(f"Rank {rank+1}: Iteration {run_info['iteration']} with average {avg_percent}% and diff {diff_percent}%. Score: {run_info['score']:.4f}")
    
    # ADVANCED ANALYSIS: Find optimal bias parameters
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS: Finding optimal bias parameters using actual lambda trajectories")
    print("="*80)
    
    # Analyze energy landscape using actual lambda data
    print("Analyzing energy landscape from actual lambda trajectories...")
    energy_profiles = analyze_energy_landscape(runs, input_folder, cut_off)
    
    if len(energy_profiles) > 0:
        print(f"Successfully analyzed energy profiles for {len(energy_profiles)} runs")
        
        # Find runs with flat energy landscapes (good for equal populations)
        energy_profiles.sort(key=lambda x: x['std_energy'])
        
        print("\nTop 5 runs with most stable energy landscapes:")
        print("Rank | Iter | Energy_mean | Energy_std | Energy_range | Samples | F_std")
        print("-" * 75)
        
        for rank, profile in enumerate(energy_profiles[:5]):
            f_std = np.std(profile['fractions'])
            print(f"{rank+1:4d} | {profile['iteration']:4d} | "
                  f"{profile['mean_energy']:11.2f} | {profile['std_energy']:10.2f} | "
                  f"{profile['energy_range']:12.2f} | {profile['num_samples']:7d} | "
                  f"{f_std:.6f}")
    
    # Analyze bias-population relationships
    X, y_fractions, y_scores, transition_data = analyze_bias_relationships(runs)
    
    if len(X) > 0:
        # Predict optimal bias parameters
        n_substituents = y_fractions.shape[1]
        optimal_bias, predicted_fractions, scaler, models = predict_optimal_bias(X, y_fractions, y_scores, run_scores=run_scores)
        
        # Reconstruct bias matrices using example from successful runs to get correct dimensions
        example_bias_matrices = run_scores[0]['bias_matrices'] if run_scores else None
        optimal_matrices = reconstruct_bias_matrices(optimal_bias, n_substituents, example_bias_matrices)
        
        print(f"\nOPTIMAL BIAS PARAMETERS (from ML with physical constraints):")
        print(f"GOAL: Equal physical state fractions (λ > {cut_off}) for all substituents")
        print(f"This matches step5's goal: equal 'Fraction > {cut_off}' values")
        print(f"NOT targeting equal overall averages (column means)")
        print(f"")
        print(f"Predicted fractions: {predicted_fractions}")
        print(f"Target fractions: {np.full(n_substituents, 1.0/n_substituents)}")
        print(f"Fraction std: {np.std(predicted_fractions):.6f}")
        print(f"Fraction max diff: {np.max(predicted_fractions) - np.min(predicted_fractions):.6f}")
        
        print(f"\nOptimal bias matrices (ML prediction with physical constraints):")
        for matrix_name in ['b', 'c', 's', 'x']:
            matrix = optimal_matrices[matrix_name]
            print(f"{matrix_name.upper()} matrix:")
            print(matrix)
            print()
        
        # Explain the physics of the B matrix
        print(f"PHYSICS INTERPRETATION:")
        print(f"B matrix represents bias potentials (kT units) applied to counteract natural free energy differences:")
        b_matrix = optimal_matrices['b']
        print(f"  Substituent 0 (reference): bias = {b_matrix[0]:.2f} kT")
        for i in range(1, len(b_matrix)):
            print(f"  Substituent {i}: bias = {b_matrix[i]:.2f} kT (relative to sub 0)")
        
        # Show that this bias is designed to achieve equal populations
        print(f"\nPURPOSE OF BIAS:")
        if sites_info is not None:
            print(f"This bias potential is designed to achieve equal populations between sites by counteracting")
            print(f"the natural free energy differences between sites.")
            print(f"With this bias applied, the expected result is:")
            equal_site_fraction = 1.0 / sites_info['n_sites']
            for i in range(sites_info['n_sites']):
                print(f"  Site {i+1}: ~{equal_site_fraction*100:.1f}% population (equal distribution)")
        else:
            print(f"This bias potential is designed to achieve equal populations by counteracting")
            print(f"the natural free energy differences between substituents.")
            print(f"With this bias applied, the expected result is:")
            equal_fraction = 1.0 / len(b_matrix)
            for i in range(len(b_matrix)):
                print(f"  Substituent {i}: ~{equal_fraction*100:.1f}% population (equal distribution)")
        
        # Create visualization plots
        print(f"\n" + "="*80)
        print("VISUALIZATION: Creating energy landscape and distribution plots")
        print("="*80)
        
        # Create 2D energy landscape plots
        plot_2d_energy_landscape(optimal_matrices, n_substituents, 
                               output_dir=input_folder, filename_prefix="optimal_bias")
        
        # Create predicted lambda distribution plots
        predicted_dist, energies = plot_predicted_lambda_distribution(optimal_matrices, run_scores, n_substituents,
                                                                    output_dir=input_folder, filename_prefix="optimal_bias")
        
        # Compare with best observed run
        best_run = run_scores[0]
        print(f"\nCOMPARISON WITH BEST OBSERVED RUN (iteration {best_run['iteration']}):")
        print(f"Observed fractions: {best_run['fractions']}")
        print(f"Observed B matrix: {best_run['bias_matrices']['b']}")
        print(f"Observed std: {best_run['f_std']:.6f}")
        print(f"Predicted improvement in std: {best_run['f_std'] - np.std(predicted_fractions):.6f}")
        
        # Show transition analysis summary
        print(f"\nTRANSITION ANALYSIS SUMMARY:")
        avg_phys_frac = np.mean([t['physical_fraction'] for t in transition_data])
        avg_trans_rate = np.mean([t['transition_rate'] for t in transition_data])
        avg_alch_length = np.mean([t['avg_alchemical_length'] for t in transition_data])
        
        print(f"Average physical state fraction: {avg_phys_frac:.3f}")
        print(f"Average transition rate: {avg_trans_rate*1000:.2f} per 1000 steps")
        print(f"Average alchemical state length: {avg_alch_length:.1f} steps")
        
        best_trans = run_scores[0]['transition_metrics']
        print(f"\nBest run transition metrics:")
        print(f"Physical state fraction: {best_trans['physical_fraction']:.3f}")
        print(f"Transition rate: {best_trans['transition_rate']*1000:.2f} per 1000 steps")
        print(f"Average alchemical length: {best_trans['avg_alchemical_length']:.1f} steps")
        print(f"Total transitions: {best_trans['transitions']}")
        
        # Show top contributing bias parameters
        print(f"\nMOST IMPORTANT BIAS PARAMETERS:")
        feature_names = []
        
        # B parameters
        for i in range(n_substituents):
            feature_names.append(f"b[{i}]")
        
        # C parameters (upper triangular)
        for i in range(n_substituents):
            for j in range(i+1, n_substituents):
                feature_names.append(f"c[{i},{j}]")
        
        # S parameters (off-diagonal)
        for i in range(n_substituents):
            for j in range(n_substituents):
                if i != j:
                    feature_names.append(f"s[{i},{j}]")
        
        # X parameters (off-diagonal)
        for i in range(n_substituents):
            for j in range(n_substituents):
                if i != j:
                    feature_names.append(f"x[{i},{j}]")
        
        # Calculate feature importance based on model coefficients
        total_importance = np.zeros(len(feature_names))
        for i, model in enumerate(models):
            if hasattr(model, 'coef_') and model.coef_ is not None:
                coef = np.atleast_1d(model.coef_)
                if len(coef) == len(feature_names):
                    total_importance += np.abs(coef)
                else:
                    print(f"Warning: Model {i} coefficient shape {coef.shape} doesn't match feature names {len(feature_names)}")
        
        # Sort by importance
        importance_indices = np.argsort(total_importance)[::-1]
        print("Top 10 most important parameters:")
        for i in range(min(10, len(importance_indices))):
            idx = importance_indices[i]
            if idx < len(optimal_bias):
                print(f"{feature_names[idx]:>8s}: {total_importance[idx]:8.3f} (optimal value: {optimal_bias[idx]:8.3f})")
            else:
                print(f"{feature_names[idx]:>8s}: {total_importance[idx]:8.3f} (optimal value: N/A)")
    
    else:
        print("No valid runs found for analysis.")
    
    # Print detailed info for the best observed run
    best_run = run_scores[0]
    print(f"\n" + "="*80)
    print(f"BEST OBSERVED RUN (iteration {best_run['iteration']}):")
    print("="*80)
    print(f"Fractions: {best_run['fractions']}")
    print(f"  - Mean: {best_run['f_mean']:.6f}")
    print(f"  - Std: {best_run['f_std']:.6f}")
    print(f"  - Min: {best_run['f_min']:.6f}")
    print(f"  - Max: {best_run['f_max']:.6f}")
    print(f"  - Difference: {best_run['f_max_diff']:.6f}")
    print(f"  - Dominating states: {'Yes' if best_run['is_dominating'] else 'No'}")
    
    # Print transition metrics for best run
    best_trans = best_run['transition_metrics']
    print(f"\nTransition metrics:")
    print(f"  - Physical state fraction: {best_trans['physical_fraction']:.3f}")
    print(f"  - Transition rate: {best_trans['transition_rate']*1000:.2f} per 1000 steps")
    print(f"  - Average alchemical length: {best_trans['avg_alchemical_length']:.1f} steps")
    print(f"  - Total transitions: {best_trans['transitions']}")
    print(f"  - Physical steps: {best_trans['physical_steps']}/{best_trans['total_steps']}")
    
    print(f"\nScoring breakdown:")
    print(f"  - Base score (avg - 10*diff): {best_run['f_mean'] - 10*best_run['f_max_diff']:.3f}")
    print(f"  - Final score (with transition penalty): {best_run['score']:.3f}")
    print(f"  - Scoring method: same as step5_bias_search.py")
    
    # Also show some statistics about transition quality
    print(f"\nTransition quality assessment:")
    good_transition_runs = [r for r in run_scores if not r['is_dominating'] and r['f_min'] > 0.1]
    print(f"Runs with good transitions (no dominating states, all fractions > 10%): {len(good_transition_runs)}")
    
    excellent_runs = [r for r in run_scores if not r['is_dominating'] and r['f_min'] > 0.15 and r['f_std'] < 0.01]
    print(f"Excellent runs (balanced + good transitions): {len(excellent_runs)}")
    
    efficient_transition_runs = [r for r in run_scores 
                               if r['transition_metrics']['physical_fraction'] > 0.7 
                               and r['transition_metrics']['transition_rate'] > 0.005]
    print(f"Runs with efficient transitions (>70% physical, >0.5% transition rate): {len(efficient_transition_runs)}")
    
    if excellent_runs:
        print(f"Top excellent run: iteration {excellent_runs[0]['iteration']} (score: {excellent_runs[0]['score']:.2f})")
    
    # Print the bias matrices for the best run
    print(f"\nBias matrices for best observed run (iteration {best_run['iteration']}):")
    for matrix_name in ['b', 'c', 's', 'x']:
        matrix = best_run['bias_matrices'][matrix_name]
        print(f"{matrix_name.upper()} matrix:")
        print(matrix)
        print()

    # Generate plots using the functions defined outside main()
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    if len(X) > 0 and 'optimal_matrices' in locals():
        # Create 2D energy landscape plots
        plot_2d_energy_landscape(optimal_matrices, n_substituents, 
                               output_dir=input_folder, filename_prefix="optimal_bias")
        
        # Create predicted lambda distribution plots
        predicted_dist, energies = plot_predicted_lambda_distribution(optimal_matrices, run_scores, n_substituents,
                                                                    output_dir=input_folder, filename_prefix="optimal_bias")
        
        print("Visualization plots completed successfully!")
    else:
        print("Cannot create plots: insufficient data or missing optimal matrices")
    

if __name__ == "__main__":
    main()