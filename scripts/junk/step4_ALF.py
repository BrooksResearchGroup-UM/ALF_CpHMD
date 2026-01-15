#!/usr/bin/env python
"""
Advanced MPI-enabled ALF (Alchemical Lambda Free energy) simulation script with 
comprehensive CpHMD integration and Henderson-Hasselbalch curve analysis.

This script implements sophisticated constant-pH molecular dynamics (CpHMD) simulations
using the ALF method with full Henderson-Hasselbalch curve generation and analysis.

Key Features:
============

MPI Parallelization:
- Uses MPI to run multiple replicas in parallel with automatic GPU assignment
- Each MPI rank handles one replica, with rank 0 dedicated to ALF analysis
- Supports multi-node setups with multiple GPUs per node
- Respects CUDA_VISIBLE_DEVICES environment variable

CpHMD Integration:
- Comprehensive micro-pKa parameter computation from TAG specifications
- Multi-site pH coupling with automatic pH₀ determination
- Bias shift calculations for replica exchange between pH conditions
- Support for UPOS (basic), UNEG (acidic), and NONE (reference) states

Henderson-Hasselbalch Analysis:
- Advanced logistic fitting with UPOS/UNEG/NONE state handling
- Adaptive sigmoid direction detection (rising vs falling curves)
- Global weight normalization across all micro-states
- Individual population curve preservation with unique identifiers
- Site-specific plotting with subspecies grouping
- Physical value filtering and robust error handling

Mathematical Framework:
- Proper Henderson-Hasselbalch equation implementation:
  * UPOS (s=+1): P / (1 + 10^(pH - pKa)) - falling with pH
  * UNEG (s=-1): P / (1 + 10^(pKa - pH)) - rising with pH
  * NONE (s=0): Constant baseline or complementary populations
- Global normalization: W = Σ(w_raw) ensures Σpopulations ≤ 1.0
- Element-wise NONE scaling for population conservation

Simulation Phases:
- Phase 1: Initial equilibration with adaptive bias adjustment
- Phase 2: Refined sampling with reduced bias cutoffs
- Phase 3: Production runs with minimal bias constraints
- Automatic phase progression based on convergence criteria

Output Generation:
- Henderson-Hasselbalch curves (PNG/PDF) with comprehensive fitting
- Individual microstate population analysis
- Site-specific titration curves with direction indicators
- Data export in tabular format for further analysis
- Automatic plot generation for nreps > 3 when pH is specified

GPU Assignment:
- Intelligent GPU distribution based on local rank
- Automatic CUDA device selection for optimal performance
- Support for heterogeneous GPU configurations

Usage:
======
    mpirun -np <num_processes> python step4_ALF_ph_noclass.py [options]

The number of replicas equals num_processes. Henderson-Hasselbalch curves are
automatically generated when nreps > 3 and pH is specified, saved to the
plots/ directory with comprehensive fitting analysis.

Dependencies:
=============
- mpi4py: MPI parallelization
- pycharmm: CHARMM molecular dynamics engine
- matplotlib: Plotting and visualization
- scipy: Advanced curve fitting (optional, graceful fallback)
- numpy, pandas: Numerical computation and data handling
- alf: Alchemical Lambda Free energy analysis library

Authors: Enhanced with comprehensive Henderson-Hasselbalch analysis
"""

# In[1]: Importing modules

from mpi4py import MPI
import os
import subprocess

import numpy as np
import pandas as pd
import shlex
import re
import itertools
import nglview as nv
import ipywidgets
import random
import argparse
import shutil
import time
import random
import sys
from pathlib import Path
sys.path.insert(0, '/home/stanislc/software/ALF')
import alf
import alf.GetLambda 
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
############################################
# Initialize MPI safely
# Check if MPI is already initialized to avoid double initialization

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Import pyCHARMM after MPI initialization is handled
import pycharmm
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.settings as settings
import pycharmm.write as write
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.crystal as crystal
import pycharmm.minimize as minimize
import pycharmm.shake as shake
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.psf as psf
import pycharmm.charmm_file as charmm_file
import pycharmm.param as param
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.scalar as scalar
import pycharmm.select as select
import pycharmm.image as image
size = comm.Get_size()


############################################
# Set up global parameters
input_folder = 'his'
toppar = 'toppar'
topology_files = [
    'top_all36_prot.rtf',
    'par_all36m_prot.prm',
    'top_all36_na.rtf',
    'par_all36_na.prm',
    'toppar_water_ions.str',
    'top_all36_cgenff.rtf',
    'par_all36_cgenff.prm',
    'my_files/titratable_residues.str',
    'my_files/nucleic_c36.str'
]

# topology_files = [
#     'top_all22_prot.rtf',
#     'par_all22_prot.prm',
#     'toppar_water_ions.str',
#     'my_files/titratable_residues_c22.str'
# ]

# non-bonded conditions
nb_fswitch = False
nb_pme = False
nb_pme_vswitch = True
cutnb = 14
cutim = cutnb
ctofnb = 12
ctonnb = 10

# dynamics conditions and paramaters
cpt_on = True # run with CPT for NVT?
temperature = 298.15
pH = None
hmr = False
cent_ncres = False
hydrogen = False

# ALF run parameters
start = 1
end = 20
phase = 1
nreps = size  # Default to MPI size, can be overridden by --nrep argument
phase_runs = size  # Use MPI size as number of replicas
no_x_bias = False  # Default to False, can be overridden by --no_x_bias argument
no_s_bias = False  # Default to False, can be overridden by --no_s_bias argument

alphabet = 'abcdefghijklmnopqrstuvwxyz'
fix_bias = False

box_size = [0.0, 0.0, 0.0]
type_ = None
angles = [90, 90, 90]

log_unit = 30
dcd_unit = 40
lmd_unit = 50
rpr_unit = 60
rst_unit = 70
restart_run = None
clog = None

patch_info = None
alf_info = None
site_pH0 = {}
site_pKa_shifts = {}


# Set up non-bonded parameters
nb_param = {'elec': True, 'atom': True,
            'cdie': True,'eps': 1,
            'cutnb': cutnb,'cutim': cutim,
            'ctofnb': ctofnb,'ctonnb': ctonnb,
            'inbfrq': -1,'imgfrq': -1,
            'nbxmod': 5
            }
if nb_pme:
    nb_param.update({'switch': True, 'vfswitch': True,
                     'ewald': True, 'pmewald': True,
                    'kappa': 0.320, 'order': 6
                    }) 
elif nb_fswitch:
    nb_param.update({'fswitch': True, 'vfswitch': True,
                     'ewald': False, 'pmewald': False
                    })
elif nb_pme_vswitch:
    nb_param.update({'switch': True, 'vswitch': True,
                     'ewald': True, 'pmewald': True,
                    'kappa': 0.320, 'order': 6
                    })
else:
    print('No non-bonded conditions set')


def check_charmm():
    # Check if CHARMM_LIB_DIR environment variable is set and points to a directory
    charmm_lib_dir = os.environ.get('CHARMM_LIB_DIR')
    if charmm_lib_dir is None:
        print("Error: CHARMM_LIB_DIR environment variable is not set.")
        sys.exit(1)
    if not os.path.isdir(charmm_lib_dir):
        print(f"Error: CHARMM_LIB_DIR '{charmm_lib_dir}' does not exist or is not a directory.")
        sys.exit(1)
    print(f"CHARMM_LIB_DIR found: {charmm_lib_dir}")
    # Check toppar directory exists
    global toppar
    if not os.path.isdir(toppar):
        print(f"Topology folder '{toppar}' not found.")
        sys.exit(1)
    return None

def read_topology_files(verbose=True):
    if not verbose:
        lingo.charmm_script('prnlev -1')
    
    prm_files = [f for f in topology_files if f.endswith('.prm')]
    rtf_files = [f for f in topology_files if f.endswith('.rtf')]
    str_files = [f for f in topology_files if f.endswith('.str')]
    lingo.charmm_script('bomblevel -2')
    settings.set_warn_level(-1)
    if rtf_files:
        read.rtf(os.path.join(toppar, rtf_files[0]))
        for file in rtf_files[1:]:
            read.rtf(os.path.join(toppar, file), append=True)
            
    if prm_files:
        read.prm(os.path.join(toppar, prm_files[0]), flex=True)
        for file in prm_files[1:]:
            read.prm(os.path.join(toppar, file), flex=True, append=True)
            
    for file in str_files:
        lingo.charmm_script(f'stream {os.path.join(toppar, file)}')
    
    settings.set_warn_level(5)
    lingo.charmm_script('bomblevel 0')
    if not verbose:
        lingo.charmm_script('prnlev 5')
    lingo.charmm_script('IOFOrmat EXTEnded')
    return None

def get_replica_assignments(nreps, size):
    """
    Calculate replica assignments for asynchronous execution.
    
    Args:
        nreps: Total number of replicas to run
        size: Number of MPI processes
        
    Returns:
        Dict mapping rank to list of replica indices
    """
    assignments = {rank: [] for rank in range(size)}
    
    # Distribute replicas
    for replica in range(nreps):
        rank = replica % size  # Round-robin assignment
        assignments[rank].append(replica)
    
    return assignments

# Update phase_runs after parsing arguments
def update_phase_runs():
    global phase_runs, nreps
    phase_runs = nreps
    
    if nreps > size:
        print(f"Running {nreps} replicas asynchronously across {size} MPI processes")
        print(f"Each process will handle {(nreps + size - 1) // size} replicas on average")
    else:
        print(f"Running {nreps} replicas synchronously with {size} MPI processes")

def parser():
    global input_folder, temperature, pH, hmr, cent_ncres, hydrogen, start, end, phase, restrains, nreps, no_x_bias, no_s_bias
    p = argparse.ArgumentParser(description='Run ALF simulations')
    p.add_argument('-i', '--input',
                        dest='input_folder',
                        default=input_folder,
                        help='Input folder')
    p.add_argument('-t', '--temperature',
                        dest='temperature',
                        type=float,
                        default=temperature,
                        help='Temperature (in K)')
    p.add_argument('-pH', '--pH',
                        dest='pH',
                        type=float,
                        default=pH,
                        help='pH')
    p.add_argument('-hmr', '--hmr',
                        dest='hmr',
                        type=bool,
                        default=hmr,
                        help='Hydrogen mass repartitioning (True or False)',
                        action=argparse.BooleanOptionalAction)
    p.add_argument('-c', '--cent_ncres',
                        dest='cent_ncres',
                        type=int,
                        default=cent_ncres,
                        help='reCENTering command allows to recenter the system at the geometric center of the first NCRES residues in the psf file')
    p.add_argument('-r', '--restrains', default='SCAT', 
                        choices=['SCAT', 'NOE'],
                        help='Restrain atoms in BLOCK (default: SCAT)')
    p.add_argument('-s', '--start',
                        dest='start',
                        type=int,
                        default=start,
                        help='Start run')
    p.add_argument('-e', '--end',
                        dest='end',
                        type=int,
                        default=end,
                        help='End run')
    p.add_argument('-p', '--phase',
                        dest='phase',
                        type=int,
                        default=phase,
                        help='Phase (1, 2 or 3)')
    p.add_argument('-H', '--hydrogen',
                        dest='hydrogen',
                        #if present, make hydrogen True
                        action='store_true',
                        help='Restrain hydrogen atoms in BLOCK (default: False)')
    p.add_argument('-nr', '--nrep',
                        dest='nreps',
                        type=int,
                        default=size,
                        help='Number of replicas to run (default: MPI size)')
    p.add_argument('-nx', '--no_x_bias',
                        dest='no_x_bias',
                        type=bool,
                        default=False,
                        help='Fix x bias (True or False)',
                        action=argparse.BooleanOptionalAction)
    p.add_argument('-ns', '--no_s_bias',
                        dest='no_s_bias',
                        type=bool,
                        default=False,
                        help='Fix s bias (True or False)',
                        action=argparse.BooleanOptionalAction)

    args, unknown = p.parse_known_args()
    # Reassign arguments
    input_folder = args.input_folder
    temperature = args.temperature
    pH = args.pH
    hmr = args.hmr
    cent_ncres = args.cent_ncres
    hydrogen = args.hydrogen
    start = args.start
    end = args.end
    phase = args.phase
    restrains = args.restrains
    nreps = args.nreps
    no_x_bias = args.no_x_bias
    no_s_bias = args.no_s_bias

    if hydrogen:
        print('Hydrogen atoms will be restrained')
        
    required_files = ['system.crd', 'system.psf', 'patches.dat', 'box.dat', 'fft.dat']

    # check if all required files are present and not empty
    for file in required_files:
        if not os.path.isfile(os.path.join(input_folder, 'prep', file)):
            raise FileNotFoundError(f'File {file} not found in {input_folder}')
        if os.path.getsize(os.path.join(input_folder, 'prep', file)) == 0:
            raise FileNotFoundError(f'File {file} is empty')
    return None

# name is folder name of input_folder

# In[2]: arguments

# Get arguments as flags from command line

parser()
update_phase_runs()

name = input_folder.split('/')[-1]

# Redirect stdout to include MPI rank
sys.stdout = open(f'{input_folder}/python_log_rank{rank}.out', 'w')


# In[3]: CHARMM topology and parameter files
# All ranks need to read topology files
check_charmm()
read_topology_files()




def get_gpu_id():
    """
    Determine GPU ID based on local rank and CUDA_VISIBLE_DEVICES.
    This function handles multi-node setups where each node has multiple GPUs
    and multiple MPI tasks per node.
    """
    import os
    
    # Try to get local rank from different MPI implementations
    local_rank = None
    
    # OpenMPI
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    # SLURM
    elif 'SLURM_LOCALID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
    # Intel MPI
    elif 'MPI_LOCALRANKID' in os.environ:
        local_rank = int(os.environ['MPI_LOCALRANKID'])
    # MPICH
    elif 'PMI_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['PMI_LOCAL_RANK'])
    else:
        # Fallback: use global rank modulo assumed GPUs per node
        print(f"Warning: Could not determine local rank. Using global rank {rank} as fallback.")
        local_rank = rank % 4  # Assume 4 GPUs per node as fallback
    
    # Get available GPUs from CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if cuda_visible_devices:
        # Parse comma-separated GPU list
        available_gpus = [gpu.strip() for gpu in cuda_visible_devices.split(',') if gpu.strip()]
        
        if local_rank < len(available_gpus):
            gpu_id = int(available_gpus[local_rank])
            print(f"Rank {rank} (local rank {local_rank}): Assigned GPU {gpu_id}")
            return gpu_id
        else:
            print(f"Error: Local rank {local_rank} exceeds available GPUs {available_gpus}")
            return 0  # Fallback to GPU 0
    else:
        # No CUDA_VISIBLE_DEVICES set, use local rank directly
        print(f"Warning: CUDA_VISIBLE_DEVICES not set. Using local rank {local_rank} as GPU ID.")
        return local_rank

def run():
    global start, phase_runs, fix_bias, phase, box_size, restart_run, clog, site_pH0, site_pKa_shifts
    # Find last run
    comm.barrier()
    if rank == 0:
        start = find_last_run()
    # Broadcast the start and phase values to all ranks
    start = comm.bcast(start, root=0)
    
    comm.barrier()
    
    # Debug print to verify all ranks have the same values
    print(f'Rank {rank}: start={start}, phase={phase}, nreps={nreps}')
    
    # Prevent re-running the same simulation
    if os.path.isfile(f'{input_folder}/variables{end}.inp'):
        return print('Simulation already executed up to run {}'.format(end))
    
    # Get replica assignments for asynchronous execution
    if nreps > size:
        replica_assignments = get_replica_assignments(nreps, size)
        my_replicas = replica_assignments[rank]
        print(f'Rank {rank}: Assigned replicas {my_replicas}')
    else:
        my_replicas = [rank] if rank < nreps else []
    
    # Run simulation
    for i in range(start, end+1):
        phase = comm.bcast(phase, root=0)
        repeats = 1
        if phase == 1:
            repeats = 1
        elif phase == 2 or phase == 3:
            repeats = 2
        for k in range(repeats):
            #just a fallback, read from previous analysis/phase.dat if exists
            if os.path.isfile(f'analysis{i-1}/phase.dat'):
                phase = np.loadtxt(f'analysis{i-1}/phase.dat', dtype=int)
            start_time = time.time()
            
            # Use nreps for phase_runs
            phase_runs = nreps
            
            # Barrier to ensure all processes are synchronized
            comm.Barrier()
            
            # Run simulations - each rank handles its assigned replicas
            for replica_idx in my_replicas:
                j = replica_idx  # Use actual replica index
                print(f'Rank {rank}: Run {i}, Replica {j}')
                
                # Always use letter suffix when nreps > 1
                if nreps > 1:
                    letter = '_' + alphabet[j]
                else:
                    letter = ''
                run_mkdir(i)
                comm.barrier()
                redirect_output(i, letter, k, j)
                setup_crystal(i, letter, k, j)
                print(f'Rank {rank}: Creating block file... for run {i}, replica {j}')
                block(i, letter, k, j)
                change_bias(delta_pKa)
                
                if restrains == 'NOE':
                    if i == 1:
                        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
                        lingo.charmm_script('mini sd nstep 50 nprint 10 step 0.005')
                    noe(hydrogen=hydrogen)
                else:
                    scat(hydrogen)
                
                minimization(i)
                dynamics(i, letter, k, j)
                
                lingo.charmm_script('! Returning to initial output unit. Bye!')
                print(f'Rank {rank}: Run {i}, Replica {j}{k} completed')
                return_output()
            
            # Barrier to ensure all simulations are complete before analysis
            comm.Barrier()
            end_time = time.time() - start_time
            print(f'Run {i}.{k} dynamics completed in {round(end_time, 1)} seconds')
            # Only rank 0 performs ALF analysis
        start_time = time.time()
        if rank == 0:
            print(f'Rank {rank}: Moving to ALF analysis for run {i}...')
            alf_analysis(i,repeats)

        # Barrier to ensure analysis is complete before next iteration
        comm.Barrier()
        
        if rank == 0:
            end_time = time.time() - start_time
            print(f'Analysis completed in {round(end_time, 1)} seconds')
            

def redirect_output(run, letter='', k=0, j=0):
    global clog, log_unit
    clog = charmm_file.CharmmFile(file_name=f'{input_folder}/run{run}/log.{k}.{j}.out', file_unit=log_unit , read_only=False, formatted=True)
    pycharmm.charmm_script(f'OUTUnit {str(log_unit)}')

def return_output():
    global clog
    if clog:
        pycharmm.charmm_script('OUTUnit 6')
        clog.close()

def find_last_run():
    global phase  # Make sure we can modify the global phase variable
    for i in range(end+1,start-1, -1):
        # if files has word 'set', it is good, otherwise, go to lower, if it has nan, then remove the run
        if os.path.isfile(f'{input_folder}/variables{i}.inp') and open(f'{input_folder}/variables{i}.inp').read().find('set') != -1:
            # If it has 'nan', then remove the run
            if 'nan' in open(f'{input_folder}/variables{i}.inp').read():
                print(f'Found NaN in {input_folder}/variables{i}.inp, removing run {i}')
                if os.path.isdir(f'{input_folder}/run{i}'):
                    shutil.rmtree(f'{input_folder}/run{i}')
                if os.path.isdir(f'{input_folder}/analysis{i}'):
                    shutil.rmtree(f'{input_folder}/analysis{i}')
                continue
            # delete analysis and run folders for runs for this run
            if os.path.isdir(f'{input_folder}/run{i}'):
                shutil.rmtree(f'{input_folder}/run{i}')
            if os.path.isdir(f'{input_folder}/analysis{i}'):
                shutil.rmtree(f'{input_folder}/analysis{i}')
            # Determine phase based on NSTEP value in log files
            phase_determined = False
            run_dir = f'run{i-1}' if i > 1 else f'run{i}'
            
            # Look for log files in the run directory
            import glob
            log_files = glob.glob(f'{input_folder}/{run_dir}/log*')
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        # Look for NSTEP pattern in the content
                        import re
                        nstep_matches = re.findall(r'NSTEP\s*=\s*(\d+)', content)
                        
                        if nstep_matches:
                            # Get the last NSTEP value (most recent)
                            nstep = int(nstep_matches[-1])
                            
                            # Determine phase based on NSTEP value
                            if nstep < 50000:
                                phase = 1
                                print(f'Phase determined from NSTEP={nstep} in {log_file}: Phase 1')
                            elif nstep < 1000000:
                                phase = 2
                                print(f'Phase determined from NSTEP={nstep} in {log_file}: Phase 2')
                            else:
                                phase = 3
                                print(f'Phase determined from NSTEP={nstep} in {log_file}: Phase 3')
                            
                            phase_determined = True
                            break
                            
                except (FileNotFoundError, ValueError, IndexError, OSError):
                    # Continue to next log file if this one fails
                    continue
                
                if phase_determined:
                    break
            
            if not phase_determined:
                print(f'Could not determine phase from log files in {run_dir}, using provided phase {phase}')
            
            if i == 1:
                print('No previous runs found, starting from run 1')
                return 1  
            else:
                print(f'Found parameters from run {i-1}, starting from run {i}')
                return i
    else:
        raise FileNotFoundError(f'File {input_folder}/variables1.inp not found')

def setup_crystal(run, letter='', k=0, j=0):
    global nb_param, box_size, type_, angles, restart_run, patch_info, delta_pKa
    settings.set_verbosity(level=5)
    if run == start and (letter == '' or letter == '_a' or size > 1):
        if hmr:
            read.psf_card(f'{input_folder}/prep/system_hmr.psf')
        else:
            read.psf_card(f'{input_folder}/prep/system.psf')
        alf_initialize()
        read_selections(f'{input_folder}/prep/patches.dat')
    else:
        pycharmm.charmm_script('BLOCK\n CLEAR\n END')
        if restrains == 'NOE':
            pycharmm.charmm_script('NOE\n RESET\n END')
        # pycharmm.charmm_script('image clear al')
        pycharmm.charmm_script('CRYSTAL FREE')
        # pycharmm.charmm_script('blade off')
        # pycharmm.charmm_script('faster off')
        # shake.off()


    if run > 5:
        restart_run = random.randint(run-5, run-1)
    else: 
        restart_run = 1
    
    # check that the previous run folder exists
    file = open(f'{input_folder}/prep/box.dat').readlines()
    type_ = file[0].strip()
    box_size = list(map(float, file[1].strip().split()))
    angles = list(map(float, file[2].strip().split()))
    if os.path.isfile(f'{input_folder}/prep/system_min.crd') == True:
            read.coor_card(f'{input_folder}/prep/system_min.crd')
    elif hmr:
        read.coor_card(f'{input_folder}/prep/system_hmr.crd')
    else:
        read.coor_card(f'{input_folder}/prep/system.crd')
    if restart_run != 1:
        # Try to read the coordinate file from possible locations
        crd_candidates = [
            f'{input_folder}/run{restart_run}/prod.{k}.{j}.crd',
            f'{input_folder}/run{restart_run}/prod.crd{letter}',
            f'{input_folder}/run{restart_run}/prod.crd',
            f'{input_folder}/run{restart_run}/prod.crd_a',
            f'{input_folder}/run{restart_run}/prod.0.0.crd',
        ]
        for crd_file in crd_candidates:
            if os.path.isfile(crd_file):
                read.coor_card(crd_file)
            break

        # Try to read the box file from possible locations
        box_candidates = [
            f'{input_folder}/run{restart_run}/box.{k}.{j}.dat',
            f'{input_folder}/run{restart_run}/box.dat{letter}',
            f'{input_folder}/run{restart_run}/box.dat',
            f'{input_folder}/run{restart_run}/box.dat_a'
            f'{input_folder}/run{restart_run}/box.0.0.dat',
        ]
        for box_file in box_candidates:
            if os.path.isfile(box_file):
                file = open(box_file).readlines()
            break
            
        box_size = list(map(float, file[1].strip().split()))

        # give a small addition of 0.1 to the box size
        # box_size = [x + 0.1 for x in box_size]
    

    pycharmm.charmm_script(f'CRYSTAL DEFINE {type_} {" ".join(map(str, box_size))} {" ".join(map(str, angles))}')
        
    pycharmm.crystal.build(nb_param['cutim'])
    # pycharmm.charmm_script(script=f'CRYSTAL BUILD NOPEr 0 CUToff {nb_param['cutim']}')
    if not cent_ncres:
        pycharmm.charmm_script('IMAGE BYRESid SELE        segid SOLV .or. segid IONS  END')
        pycharmm.charmm_script('IMAGE BYSEGid SELE .not. (segid SOLV .or. segid IONS) END')
    fft = open(f'{input_folder}/prep/fft.dat').read().strip().split()
    nb_param.update({'fftx': fft[0], 'ffty': fft[1], 'fftz': fft[2]})
    pycharmm.NonBondedScript(**nb_param).run()

def minimization(run):
    if int(run) < 6 and os.path.isfile(f'{input_folder}/prep/system_min.crd') != True:
        print('Minimization requested, but no system_min.crd file found')
        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
        minimize.run_sd(nstep = 100)
        pycharmm.charmm_script('faster on')
        pycharmm.charmm_script(f'blade on gpuid {gpuid}')
        minimize.run_abnr(nstep = 1000, tolenr = 1e-3, tolgrd=1e-3)
        write.coor_card(f'{input_folder}/prep/system_min.crd')     
        energy.show()
        #pycharmm.charmm_script('energy domdec gpu only dlb off ndir 1 1 1 ')
        pycharmm.charmm_script('energy blade')

def dyn_init():
    shake.on(fast=True, bonh=True, param=True, tol=1e-7)
    pycharmm.charmm_script('faster on')
    pycharmm.charmm_script(f'blade on gpuid {gpuid}')
    # pycharmm.charmm_script('shake fast bonh param sele .not. resname H* end')
    # n = psf.get_natom()
    # scalar.set_fbetas([1.0] * n)
    gscale = 10
    dyn.set_fbetas(np.full(psf.get_natom(), gscale, dtype=float))
    

def run_mkdir(run):
    # Only rank 0 creates directories to avoid race conditions
    if rank == 0:
        os.makedirs(f'{input_folder}/run{run}', exist_ok=True)
        os.makedirs(f'{input_folder}/run{run}/dcd', exist_ok=True)
        os.makedirs(f'{input_folder}/run{run}/res', exist_ok=True)
    # Wait for directories to be created
    comm.Barrier()


def read_selections(selections_file=None):
    global patch_info
    print(patch_info)
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called and patch_info is set.")
    for index, row in patch_info.iterrows():
        name = row['SELECT']
        segid = row['SEGID']
        resid = row['RESID']
        resname = row['PATCH']
        atoms = '-\n(type ' + " -\n .or. type ".join(row["ATOMS"].split(' ')) + ')'
        pycharmm.charmm_script(f'DEFine {name} SELEction SEGID {segid} .AND. RESId {resid} .AND. RESName {resname} .AND. {atoms} END')


def alf_initialize():
    global patch_info, alf_info
    comm.barrier()
     # Read patches.info
    patch_info = pd.read_csv(os.path.join(input_folder, 'prep', 'patches.dat'), sep=',')
    patch_info[['site', 'sub']] = patch_info['SELECT'].str.extract(r's(\d+)s(\d+)')

    required_files = ['name', 'nsubs', 'nblocks', 'nreps', 'ncentral', 'engine']
    # if one of the required files is missing in f'{input_folder}/prep', initialize alf_info
    if not all([file in required_files for file in os.listdir(os.path.join(input_folder, 'prep'))]):


        alf_info = {}
        alf_info['name'] = input_folder.split('/')[-1]
        alf_info['nsubs'] = np.array([],dtype=int)
        alf_info['nblocks'] = 0
        alf_info['nreps'] = nreps
        alf_info['ncentral'] = nreps // 2  # central value of self.size
        alf_info['nnodes'] = 1
        alf_info['temp'] = temperature
        alf_info['engine'] = 'charmm'
        for site in patch_info['site'].unique():
            alf_info['nblocks'] += len(patch_info[patch_info['site'] == site]['sub'].unique()) 
            # how many subsites in this block
            alf_info['nsubs'] = np.append(alf_info['nsubs'], len(patch_info[patch_info['site'] == site]['sub'].unique()))
            
        for key in alf_info.keys():

            f = open(f'{input_folder}/prep/{key}', 'w') 
            if key == 'nsubs':
                f.write(' '.join(map(str, alf_info[key])))
            else:
                f.write(str(alf_info[key]))
            f.close()
        # copy a folder 'G_imp' from alf module to current directory
        if rank == 0:
            # Try all G_imp_candidates in both package and cwd locations
            G_imp_candidates = ['G_imp_20', 'G_imp']
            src_candidates = []
            pkg_dir = Path(alf.__file__).resolve().parent
            dst = Path(input_folder).expanduser().resolve() / "G_imp"
            for name in G_imp_candidates:
                src_candidates.append(pkg_dir / name)

            for src in src_candidates:
                if src.is_dir():
                    try:
                        shutil.copytree(src, dst, dirs_exist_ok=True)   # Python ≥ 3.8
                        print(f"Copied G_imp from {src} → {dst}")
                        break
                    except (PermissionError, OSError) as err:
                        raise RuntimeError(f"Failed to copy {src} → {dst}: {err}") from err
                else:
                    print("No G_imp directory found in the expected locations.")
        home_dir = os.getcwd()
        try:
            os.chdir(f'{input_folder}')
            if rank == 0:
                alf.InitVars(alf_info)
                alf.SetVars(alf_info,1)
        finally:
            os.chdir(home_dir)    
    comm.barrier()
            
def change_bias(delta_pKa):
    global patch_info
    
    # If pH is None, create zero bias files and skip CpHMD setup
    if pH is None:
        if patch_info is not None:
            n_sites = len(patch_info)
            zero_bias = [0.0] * n_sites
        else:
            zero_bias = [0.0]  # Default single zero
        
        # Create zero bias files
        os.makedirs(f'{input_folder}/nbshift', exist_ok=True)
        np.savetxt(f'{input_folder}/nbshift/b_shift.dat', np.reshape(np.array(zero_bias), (1, -1)), fmt="%.18e")
        np.savetxt(f'{input_folder}/nbshift/b_fix_shift.dat', np.reshape(np.array(zero_bias), (1, -1)), fmt="%.18e")
        
        print(f"No pH specified: Created zero bias files with {len(zero_bias)} sites")
        print(f"  b_shift.dat: {[f'{x:.6f}' for x in zero_bias]}")
        print(f"  b_fix_shift.dat: {[f'{x:.6f}' for x in zero_bias]}")
        
        return zero_bias
    
    # # CpHMD mode: proceed with original bias computation
    # try:
    #     analysis_folders = [f for f in os.listdir(os.path.join('../09_bias_vswitch', input_folder)) if f.startswith('analysis')]
    #     # if input folder starts with digit, and patches_info 'PATCH' contains only single PATCH which ends with O, 
    #     # get resname from this, and check analysis folders for that resname
    #     if (input_folder[0].isdigit() and patch_info is not None and 
    #         patch_info['PATCH'].str.endswith('O').sum() == 1):
    #         resname = patch_info['PATCH'].str.extract(r'(\w+)O')[0].values[0][:3].lower()
    #         analysis_folders = [f for f in os.listdir(os.path.join('../09_bias_vswitch', resname)) if f.startswith('analysis')]

        # if analysis_folders:
        #     analysis_folders.sort(key=lambda x: int(match.group(1)) if (match := re.search(r'analysis(\d+)', x)) else 0, reverse=True)
        #     last_analysis_folder = analysis_folders[0]
        #     match = re.search(r'analysis(\d+)', last_analysis_folder)
        #     last_analysis_number = int(match.group(1)) if match else 0
        #     source_folder = os.path.join('../09_bias_vswitch', input_folder, last_analysis_folder)
        #     dest_folder = os.path.join(input_folder, 'analysis0')
        #     os.makedirs(dest_folder, exist_ok=True)
        #     for file in os.listdir(source_folder):
        #         if file.endswith('.dat') or file.endswith('_sum.dat'):
        #             shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))
        #     if os.path.isfile(os.path.join('../09_bias_vswitch', input_folder, f'variables{last_analysis_number+1}.inp')):
        #         shutil.copy(os.path.join('../09_bias_vswitch', input_folder, f'variables{last_analysis_number+1}.inp'), os.path.join(input_folder, 'variables1.inp'))
        #         print(f'Copied variables{last_analysis_number+1}.inp to {input_folder}/variables1.inp')
    # except Exception as e:
    #     print(f'Bias files not found, skipping bias initialization. Error: {e}')
    # return None
    

def manual_bias_adjustment(lam: list):
    # change the fixed bias in variables1.inp
    with open(f'{input_folder}/variables1.inp', 'r') as f:
        lines = f.readlines() 
    np.savetxt(f'{input_folder}/analysis0/b.dat', np.reshape(np.array(lam),(1,-1)))
    # np.savetxt(f'{pdb_name}/analysis0/b_prev.dat', np.reshape(np.array(lam),(1,-1)))
    # np.savetxt(f'{pdb_name}/nbshift/b_shift.dat', np.array(lam) ,fmt="%.18e", newline=" ")
    np.savetxt(f'{input_folder}/analysis0/b_sum.dat',np.reshape(np.array(lam),(1,-1)),fmt=' %7.2f')
    

    updated_lines = []    
    for line in lines:
        if line.startswith('set lams'):
            line_parts = line.split('=')
            line_prefix = line_parts[0].strip()
            current_value = float(line_parts[1].strip())
            new_value = '{:.2f}'.format(lam.pop(0))
            line = f"{line_prefix}= {new_value}\n"
        updated_lines.append(line)
        
    #Write the updated content back to variables1.inp
    with open(f'{input_folder}/variables1.inp', 'w') as file:
        file.writelines(updated_lines)

def block(i, letter='', k=0, j=0):
    global patch_info, alf_info, site_pH0, site_pKa_shifts, phase ,delta_pKa
    
    # Guard clauses to ensure variables are initialized
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called.")
    if alf_info is None:
        raise ValueError("alf_info is not initialized. Please ensure alf_initialize() has been called.")
    
    run = i
    def read_variable_file(i):
        global delta_pKa
        """
        This function reads variables{i}.inp file and returns a dictionary of variable names and values.
        i is the variable in the filename.
        """
        filename = f'{input_folder}/variables{i}.inp'
        variables = {}
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("set"):
                    # Remove the "set" keyword and any whitespace characters
                    line = line.replace("set", "").strip()
                    # Split the line into variable name and value
                    var_name, var_value = line.split("=")
                    # Remove any whitespace characters from variable name and value
                    var_name = var_name.strip()
                    var_value = var_value.strip()
                    # Convert variable value to float if possible
                    try:
                        var_value = float(var_value)
                    except ValueError:
                        pass
                    # Add variable to the dictionary
                    variables[var_name] = var_value
        return variables
    
    # Compute CpHMD parameters if pH is set
    effective_pH = pH  # Default value
    replica_effective_pH = pH  # Initialize for all cases
    if pH is not None:
        kTln10 = compute_cphmd_parameters()
        
        # Determine effective pH based on site pH₀ values
        if len(set(site_pH0.values())) == 1:
            # Single site: use its pH₀ as effective pH, keep original pKa shifts
            site_pH0_val = list(site_pH0.values())[0]
            effective_pH = site_pH0_val 
            pH0_ref = site_pH0_val
            delta_pH = pH - pH0_ref
            
            # Keep original pKa shifts (already computed relative to site pH₀)
            # No need to recalculate site_pKa_shifts - they are correct as computed
            
            print(f"Single site: effective_pH={effective_pH:.3f} (site pH₀), user_pH={pH:.3f}, delta_pH={delta_pH:.3f}")
            print(f"Using original pKa shifts relative to site pH₀={effective_pH:.3f}")
        elif len(set(site_pH0.values())) > 1:
            # Multiple sites with different pH₀: use pH=0 as neutral reference
            effective_pH = 0.0
            pH0_ref = 0.0  # Neutral reference
            delta_pH = pH - pH0_ref
            print(f"Multiple sites: effective_pH={effective_pH:.3f} (neutral), user_pH={pH:.3f}, delta_pH={delta_pH:.3f}")
            print(f"Site pH₀ values: {site_pH0}")
        else:
            # No sites computed: fallback to user pH
            effective_pH = pH
            pH0_ref = 7.0  # Default reference
            delta_pH = pH - pH0_ref
            print(f"No computed pH₀: effective_pH={effective_pH:.3f}, delta_pH={delta_pH:.3f}")
        
        
        # Update TAG values based on effective pH conditions
        if patch_info is not None:
            for site, sub in patch_info[['site', 'sub']].itertuples(index=False):
                tag_val = patch_info.loc[(patch_info['site'] == site) & (patch_info['sub'] == sub), 'TAG']
                if hasattr(tag_val, 'values'):
                    tag = str(tag_val.values[0])
                else:
                    tag = str(tag_val)
                if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                    original_pKa = tag.split()[1]
                    new_pKa = float(original_pKa) + (effective_pH - site_pH0.get(site, 7.0))
                    new_tag = f"{tag.split(' ')[0]} {new_pKa:.2f}"
                    patch_info.loc[(patch_info['site'] == site) & (patch_info['sub'] == sub), 'TAG'] = new_tag
                    print(f"Site {site} Subsite {sub}: Updated TAG from {tag} to {new_tag} based on effective_pH={effective_pH:.3f}")
        
        
        # delta_pKa remains constant regardless of nreps (contribution averages to zero)
        if phase == 1:
            delta_pKa = 1.0
        elif phase == 2:
            delta_pKa = 0.5
        else:
            delta_pKa = 0.25  

        # Calculate replica-specific effective pH
        replica_idx = 0
        if nreps > 1 and letter and letter.startswith('_'):
            replica_idx = ord(letter[1]) - ord('a')
        
        ncentral = alf_info['ncentral']
        replica_shift = delta_pKa * (replica_idx - ncentral)
        
        # Apply replica shift to effective pH for PHMD command
        replica_effective_pH = effective_pH + replica_shift
        
        # For multiple replicas, add replica-specific shifts
        if nreps > 1:
            total_delta_pH = delta_pH + replica_shift
            # For single site: only use replica shift (delta_pH ignored for b_shift)
            if len(set(site_pH0.values())) == 1:
                print(f"Replica {replica_idx}: effective_pH={replica_effective_pH:.3f}, replica_shift={replica_shift:.3f}, total_delta_pH={total_delta_pH:.3f} (single site)")
            else:
                print(f"Replica {replica_idx}: effective_pH={replica_effective_pH:.3f}, base_delta_pH={delta_pH:.3f}, replica_shift={replica_shift:.3f}, total_delta_pH={total_delta_pH:.3f}")
        else:
            # For single site: no pH shift for b_shift (only pKa shifts matter)
            if len(set(site_pH0.values())) == 1:
                total_delta_pH = delta_pH
                print(f"Single replica, single site: effective_pH={replica_effective_pH:.3f}, total_delta_pH={total_delta_pH:.3f}")
            else:
                total_delta_pH = delta_pH
                print(f"Single replica: effective_pH={replica_effective_pH:.3f}, total_delta_pH={total_delta_pH:.3f}")
        
        # Create bias shift files
        create_bias_shift_file(delta_pKa, kTln10)
        print(f"Shift b_shift with delta_pKa={delta_pKa:.3f}, kTln10={kTln10:.3f}")
    else:
        # When pH is None, set delta_pKa to 0 for consistency
        delta_pKa = 0.0

    
    block_command = ''

    variables = read_variable_file(i)
    
    
    block_command += f'BLOCK {len(patch_info)+1} '
    # if variables['nreps'] > 1:
    #     block_command += f'NREP {variables["nreps"]}\n\n'
    # else:
    #     block_command += '\n'
    block_command += '\n'
    
    # Define blocks
    block_command += '!----------------------------------------\n'
    block_command += '! Set up l-dynamics by setting BLOCK parameters\n'
    block_command += '!----------------------------------------\n\n'
    
    # for index, row in patch_info.iterrows():
    #     block_command += f'CALL {index+2} SELEct {row["SELECT"]} END\n'
    block_command += ''.join(
        f'CALL {index+2} SELECT {row["SELECT"]} END\n'
        for index, row in patch_info.iterrows()
    )
    
    
    # Exclude blocks from each other
    block_command += '!----------------------------------------\n'
    block_command += '! Exclude blocks from each other\n'
    block_command += '!----------------------------------------\n\n'
    
    # exclude blocks with same RESID from each other
    # for index, row in patch_info.iterrows():
    #     for index2, row2 in patch_info.iterrows():
    #         if index2 > index and row['RESID'] == row2['RESID']:
    #             block_command += 'adexcl {:<3} {:<3}\n'.format(index+2, index2+2)
    
    block_command += ''.join(
        'adexcl {:<3} {:<3}\n'.format(index+2, index2+2)
        for index, row in patch_info.iterrows()
        for index2, row2 in patch_info.iterrows()
        if index2 > index and row['site'] == row2['site']
    )

    
    block_command += '\n'            
    block_command += '!----------------------------------------\n'
    block_command += '!QLDM turns on lambda-dynamics option\n'
    block_command += '!----------------------------------------\n\n'
    block_command += 'QLDM THETa\n\n'
    
    block_command += '!----------------------------------------\n'
    block_command += '!LANGEVIN turns on the langevin heatbath\n'
    block_command += '!----------------------------------------\n\n'
    block_command += f'LANG TEMP {temperature:.2f}\n\n'
    
    if pH != None and delta_pKa != 0:
        block_command += '!----------------------------------------\n'
        block_command += '!Setup CpHMD with replica-specific pH\n'
        block_command += '!----------------------------------------\n\n'
        block_command += f'PHMD pH {replica_effective_pH:.3f}\n'
    elif pH != None and delta_pKa == 0:
        block_command += '!----------------------------------------\n'
        block_command += '!CpHMD disabled (delta_pKa=0)\n'
        block_command += '!----------------------------------------\n\n'
        print(f"Running simulation without PHMD pH command (delta_pKa={delta_pKa})")
    
        
    block_command += '!----------------------------------------\n'
    block_command += '!Soft-core potentials\n'
    block_command += '!----------------------------------------\n\n'
    block_command += 'SOFT ON\n'
    
    block_command += '!----------------------------------------\n'
    block_command += '!lambda-dynamics energy constrains (from ALF) AKA fixed bias\n'
    block_command += '!----------------------------------------\n\n'
    
    # Explicitly initialize 'l0' column to 0.0 for all rows
    patch_info['l0'] = 0.0

    # for each site, randomly select one of the subsites to have l0=1, and the others l0=0
    for site in patch_info['site'].unique():
        site_iter = patch_info.loc[patch_info['site'] == site]
        subsites = site_iter['sub'].tolist()
        # Generate random l0 values with 3 decimal digits, sum to 1
        l0_values = np.random.dirichlet(np.ones(len(subsites)))
        l0_values = np.round(l0_values, 3)
        # Adjust last value to ensure exact sum to 1.0 (fix rounding error)
        l0_values[-1] = np.round(1.0 - np.sum(l0_values[:-1]), 3)
        for sub, l0 in zip(subsites, l0_values):
            patch_info.loc[(patch_info['site'] == site) & (patch_info['sub'] == sub), 'l0'] = l0
    
    if pH != None and delta_pKa != 0:
        block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4} {:<5}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0, 'NONE')
        for index, row in patch_info.iterrows():
            l0 = row['l0']
            var = variables['lam'+str(row['SELECT'])]
            # Keep original TAG values (UPOS/UNEG with pKa values, NONE)
            block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4} {:<5}\n'.format(
                index+2, l0, 0.0, 12.0, var, 5.0, row['TAG']
            )
            
    else:
        # When pH is None or delta_pKa is 0, run without TAG values
        block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0)
        for index, row in patch_info.iterrows():
            l0 = row['l0']
            var = variables['lam'+str(row['SELECT'])]
            block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4}\n'.format(index+2, l0, 0.0, 12.0, var, 5.0)

        if pH != None and delta_pKa == 0:
            print(f"Running simulation without TAG values (delta_pKa={delta_pKa})")
    
        
    block_command += '!------------------------------------------\n'
    block_command += '! All bond/angle/dihe terms treated at full str (no scaling),\n'
    block_command += '! prevent unphysical results'
    block_command += '!------------------------------------------\n\n'
    block_command += 'rmla bond theta impr\n\n'
    
    
    block_command += '!------------------------------------------\n'
    block_command += '! Selects MSLD, the numbers assign each block to the specified site on the core\n'
    block_command += '!------------------------------------------\n\n'
    block_command += 'MSLD 0 -\n'
    
    index = 1
    for select in patch_info['SELECT']:
        block_command += f'{select.split("s")[1]} '
        if select.split("s")[1] != patch_info['SELECT'].iloc[-1].split("s")[1]:
            block_command += ' -\n'
    block_command += '-\nfnex 5.5 \n\n'
        
    block_command += '!------------------------------------------\n'
    block_command += '! Constructs the interaction matrix and assigns lambda & theta values for each block\n'
    block_command += '!------------------------------------------\n\n'
    block_command += 'MSMA\n\n'
    
    block_command += '!------------------------------------------\n'
    block_command += '! PME for electrostatics\n'
    block_command += '!------------------------------------------\n\n'
    block_command += 'PMEL EX\n\n'
    
    block_command += '!------------------------------------------\n'
    block_command += '! Variable lambda potentials\n'
    block_command += '!------------------------------------------\n\n'
    
    # iterate over f with same resid
    ldbi = '!------------------------------------------\n'
    ldbi += '! Quadratic Barriers\n'
    ldbi += '!------------------------------------------\n\n'
    i = 0
    # for resid in patch_info['RESID'].unique():
    #     iter = patch_info.loc[patch_info['RESID'] == resid]

    for site in patch_info['site'].unique():
        iter = patch_info.loc[patch_info['site'] == site]
        for index, row in iter.iterrows():
            for index2, row2 in iter.iterrows():
                if index < index2:
                    index_1 = index + 2
                    index_2 = index2 + 2
                    ldbv = str(row['SELECT'])+str(row2['SELECT'])
                    if index2 > index:
                        try:
                            i += 1
                            ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 6, 0.0, variables["c"+ldbv], 0)
                        except: pass
    
    ldbi += '!------------------------------------------\n'
    ldbi += '!End point Potentials\n'
    ldbi += '!------------------------------------------\n\n'
    for site in patch_info['site'].unique():
        iter = patch_info.loc[patch_info['site'] == site]
        for index, row in iter.iterrows():
            for index2, row2 in iter.iterrows():
                index_1 = index + 2
                index_2 = index2 + 2
                ldbv = str(row['SELECT'])+str(row2['SELECT'])
                
                if index2 != index:
                    try:
                        i += 1
                        ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 8, 0.017, variables["s"+ldbv], 0)
                    except: pass
    ldbi += '!------------------------------------------\n'
    ldbi += '! Skew Potentials\n'
    ldbi += '!------------------------------------------\n\n'                  
    for site in patch_info['site'].unique():
        iter = patch_info.loc[patch_info['site'] == site]
        for index, row in iter.iterrows():
            for index2, row2 in iter.iterrows():
                index_1 = index + 2
                index_2 = index2 + 2
                ldbv = str(row['SELECT'])+str(row2['SELECT'])
                if index2 != index:
                    try:
                        i += 1
                        ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 10, -5.56, variables["x"+ldbv], 0)
                    except: pass
    block_command += f'LDBI {i}\n' +ldbi + '\n'
    block_command += 'END'
    # print(f'{input_folder}/run{run}/block.{k}.{j}.str')
    with open(f'{input_folder}/run{run}/block.{k}.{j}.str', 'w') as f:
        f.write(block_command)
    # print(block_command)
    # settings.set_bomb_level(-5)
    pycharmm.charmm_script(block_command)
    # settings.set_bomb_level(0)
    
def scat(hydrogen):
    global patch_info
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called.")
    
    scat_command = 'BLOCK\n scat on\nscat k 300\n'
    for site in patch_info['site'].unique():
        atoms = patch_info.loc[patch_info['site'] == site]['ATOMS']
        atoms = set([atom for atom in atoms.str.split().sum()])
        h_atoms = [atom for atom in atoms if atom.startswith('H')]
        atoms = [atom for atom in atoms if not atom.startswith('H')]
        for atom in atoms:
            scat_command += f'cats SELE type {atom} .and. ({" .or. ".join(map(str, patch_info.loc[patch_info["site"] == site]["SELECT"]))}) END\n'
        if hydrogen:
            for atom in h_atoms:
                scat_command += f'cats SELE type {atom} .and. ({" .or. ".join(map(str, patch_info.loc[patch_info["site"] == site]["SELECT"]))}) END\n'
    scat_command += 'END\n'
    with open(f'{input_folder}/prep/restrains.str', 'w') as f:
        f.write(scat_command)
    pycharmm.charmm_script(scat_command)

def noe(hydrogen):
    global patch_info
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called.")
    
    noe_command = 'NOE\n'
    
    index = 1
    # Group by site from patch_info and process each site
    for site, group in patch_info.groupby('site', sort=False):
        segid = group['SEGID'].iloc[0]
        resid = group['RESID'].iloc[0]
        resname = group['PATCH'].iloc[0][0:3]  # Assuming first 3 chars of PATCH is resname
        
        # Get atom names from the ATOMS column for this site
        atoms = group['ATOMS'].str.split().explode().dropna().unique()
        
        # Filter atoms based on hydrogen parameter
        if not hydrogen:
            atoms = [atom for atom in atoms if not atom.startswith('H')]
        
        # Count occurrences of each atom name to find repeats
        atom_counts = group['ATOMS'].str.split().explode().value_counts()
        repeats = atom_counts[atom_counts > 1].index.tolist()
        
        if repeats:  # Only proceed if there are repeated atom names
            noe_command += f'!---------------------------------------------------------------\n! Restrains for {segid} {resname} {resid}, SITE {site}, GROUP {index}\n!---------------------------------------------------------------\n'
            index += 1
            for repeat_atom in repeats:
                # Filter patches containing the repeat_atom
                atom_patches = group[group['ATOMS'].str.contains(repeat_atom, na=False)]
                # If hydrogen is False, exclude patches where repeat_atom starts with 'H'
                if not hydrogen and repeat_atom.startswith('H'):
                    atom_patches = atom_patches.iloc[0:0]  # Empty DataFrame to skip hydrogen atoms
                
                if len(atom_patches) > 1:  # Ensure more than one occurrence for restraint
                    for i1, i2 in itertools.combinations(atom_patches.index, 2):
                        patch1 = atom_patches.loc[i1, 'PATCH']
                        patch2 = atom_patches.loc[i2, 'PATCH']
                        noe_command += (
                            f'assign sele segid {segid} .and. resid {resid} .and. resn {patch1} .and. type {repeat_atom} end '
                            f'sele segid {segid} .and. resid {resid} .and. resn {patch2} .and. type {repeat_atom} end -\n'
                            'kmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0\n'
                        )
    
    noe_command += 'END\n'
    
    # Write to file (assuming input_folder is accessible in the class scope)
    with open(f'{input_folder}/prep/restrains.str', 'w') as f:
        f.write(noe_command)
    
    # Execute via PyCHARMM
    pycharmm.charmm_script(noe_command)


def dynamics(run, letter='', k=0, j=0):
    global cpt_on, temperature, phase, box_size, type_, angles, restart_run, dcd_unit, rst_unit, lmd_unit, rpr_unit
    lingo.charmm_script(f'blade on gpuid {gpuid}')
    if run == start:
        # settings.set_verbosity(5)
        dyn_init()
    # else: settings.set_verbosity(3)
    if phase == 1:
        nsteps_eq = 10000 # 20 ps
        nsteps_prod = 40000 # 80 ps
        nsavc = 1000 # 2 ps
        nsavl = 1 # 2 fs

    elif phase == 2:
        nsteps_eq = 50000 # 200 ps
        nsteps_prod = 450000 # 800 ps
        nsavc = 10000 # 2 ps
        nsavl = 1 # 2 fs

    elif phase == 3:
        nsteps_eq = 0 # 0 ps
        nsteps_prod = 500000 # 1000 ps
        nsavc = 10000 # 2 ps
        nsavl = 1 # 2 fs
    
    else:
        raise ValueError('Phase must be 1, 2 or 3')
    
    
    dyn_param = {
        'start': True, 
        'restart': False,
        'blade': True,
        'prmc': True,
        'iprs': 100,
        'prdv': 100,
        'cpt': cpt_on, 
        'timestep': 0.002,
        'firstt': temperature, 
        'finalt': temperature,
        'tstruc': temperature, 
        'tbath': temperature,
        'ichecw': 0, #do not scale velocities to final temp (i.e. equilibrate)
        'ihtfrq': 0, #frequency of heating
        'ieqfrq': 0, #frequence of scaling/assigning velocities
        'iasors': 1, #assign velocities during heating (0 will be scale velocities)
        'iasvel': 1, #using gaussian distribution to assign velocities
        'iscvel': 0,
        'inbfrq': 0, # BLaDE does it's own neighbor searching
        'ilbfrq': 0,
        'imgfrq': 0, # BLADE does it's own neighbor searching
        'ntrfrq': 0,
        'echeck': -1, # energy tolerance before crash
        'iunldm': lmd_unit,
        'iunwri': rst_unit,
        'iuncrd': dcd_unit
                }
    # 'leap': True, # some code doesn't have this flag
    # 'langevin': True, # some code doesn't have this flag
    # 'iscale': 0, # some code doesn't have this flag
    
    if cpt_on:
        dyn_param.update({
            'pconstant': True, 'pmass': psf.get_natom() * 0.12,
            'pref': 1.0, 'pgamma': 20.0,
            'hoover': True, 'reft': temperature, 'tmass': 1000
        })
    

    # add item to dyn_param
    if cent_ncres:
        molecule = (pycharmm.SelectAtoms().by_seg_id('SOLV') | 
                    pycharmm.SelectAtoms().by_seg_id('IONS'))
        molecule = molecule.__invert__()
        n_molecule = len(list(set(molecule.get_res_ids())))
        molecule.unstore()
        dyn_param.update({'cent ncres': n_molecule})
    
    
    if hmr:
        nsteps_eq = nsteps_eq // 2
        nsteps_prod = nsteps_prod // 2
        nsavc = nsavc // 2
        nsavl = nsavl // 2
        nsavl = max(nsavl, 1)
        dyn_param.update({'timestep': 0.004})
    
    if phase == 3:
        dyn_param.update({
            'start': False,
            'restart': True,
            'iunrea': rpr_unit})
        
    dyn_param.update({'nsavc': nsavc, 'nsavl': nsavl,
                     'nprint': nsavc,'iprfrq': nsavc,
                     'isvfrq': nsavc
                     })
    
    # Equalibration Run
    if nsteps_eq > 0:
        dcd_fn = f'{input_folder}/run{run}/dcd/{name}_eq.{k}.{j}.dcd'
        rst_fn = f'{input_folder}/run{run}/res/{name}_eq.{k}.{j}.rst'
        lmd_fn = f'{input_folder}/run{run}/res/{name}_eq.{k}.{j}.lmd'

        dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = dcd_unit, read_only = False, formatted = False)
        rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = rst_unit, read_only = False, formatted = True)
        # lingo.charmm_script(f'OPEN WRITE UNIT {rst_unit} CARD NAME {rst_fn}')
        lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = lmd_unit, read_only = False, formatted = False)

        dyn_param.update({'nstep': nsteps_eq})
        pycharmm.DynamicsScript(**dyn_param).run()
        dcd.close()
        rst.close()
        lmd.close()
        verbosity = 1  # Default verbosity value
        if not cent_ncres:
            verbosity = settings.set_verbosity(1)
            dcd = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_eq.{k}.{j}.dcd', 
                                      file_unit=dcd_unit,read_only=True, formatted=False)
            dcd_new = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_eq_1.{k}.{j}.dcd',
                                          dcd_unit+5, False, False)
            lingo.charmm_script(f'merge first {dcd_unit} nunit 1 output {dcd_unit+5} sele all end  -\n sele .not. (segid SOLV .or. segid IONS) .and. .not. hydrogen end')
            os.remove(f'{input_folder}/run{run}/dcd/{name}_eq.{k}.{j}.dcd')
            os.rename(f'{input_folder}/run{run}/dcd/{name}_eq_1.{k}.{j}.dcd', f'{input_folder}/run{run}/dcd/{name}_eq.{k}.{j}.dcd')
            dcd.close()
            dcd_new.close()
            settings.set_verbosity(verbosity)
    lingo.charmm_script('energy blade')
    
    # Production Run
    sim_type = None  # Ensure sim_type is always defined
    
    if nsteps_prod > 0:
        if phase in (1, 2):
            sim_type = 'flat'
        elif phase == 3:
            sim_type = 'prod'
        dcd_fn = f'{input_folder}/run{run}/dcd/{name}_{sim_type}.{k}.{j}.dcd'
        lmd_fn = f'{input_folder}/run{run}/res/{name}_{sim_type}.{k}.{j}.lmd'
        rst_fn = f'{input_folder}/run{run}/res/{name}_{sim_type}.{k}.{j}.rst'
        dyn_param.update({'start': False, 'restart': True, 'iunrea': rpr_unit})
        rpr_fn = None

        # Build a list of candidate restart filenames
        if sim_type == 'flat':
            candidates = [
                f'{input_folder}/run{run}/res/{name}_eq.rst{letter}',
                f'{input_folder}/run{run}/res/{name}_eq.rst_a',
                f'{input_folder}/run{run}/res/{name}_eq.rst',
                f'{input_folder}/run{run}/res/{name}_eq.{k}.{j}.rst',
                f'{input_folder}/run{run}/res/{name}_eq.rst{k}',
                f'{input_folder}/run{run}/res/{name}_eq_0.0.rst'
            ]
        else:
            candidates = [
                f'{input_folder}/run{restart_run}/res/{name}_prod.rst{letter}',
                f'{input_folder}/run{restart_run}/res/{name}_prod.rst_a',
                f'{input_folder}/run{restart_run}/res/{name}_flat.rst{letter}',
                f'{input_folder}/run{restart_run}/res/{name}_flat.rst_a',
                f'{input_folder}/run{restart_run}/res/{name}_prod.rst',
                f'{input_folder}/run{restart_run}/res/{name}_flat.rst',
                f'{input_folder}/run{restart_run}/res/{name}_prod.{k}.{j}.rst',
                f'{input_folder}/run{restart_run}/res/{name}_flat.{k}.{j}.rst',
                f'{input_folder}/run{restart_run}/res/{name}_prod.rst{k}',
                f'{input_folder}/run{restart_run}/res/{name}_flat.rst{k}',
                f'{input_folder}/run{restart_run}/{name}_prod_0.0.rst',
                f'{input_folder}/run{restart_run}/{name}_flat_0.0.rst',
            ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                rpr_fn = candidate
                break

        if rpr_fn is None:
            print(f'No restart file found for {name} from {restart_run}. Starting from scratch.')
            dyn_param.update({'start': True, 'restart': False})
            dyn_param.pop('iunrea', None)

        if rpr_fn is not None:
            rpr = pycharmm.CharmmFile(file_name=rpr_fn, file_unit=rpr_unit, read_only=True, formatted=True)
        else:
            rpr = None
        dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = dcd_unit, read_only = False, formatted = False)
        rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = rst_unit, read_only = False, formatted = True)
        lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = lmd_unit, read_only = False, formatted = False)

        dyn_param.update({'nstep': nsteps_prod})

        pycharmm.DynamicsScript(**dyn_param).run()
        dcd.close()
        lmd.close()
        rst.close()
        if rpr is not None:
            rpr.close()
        if not cent_ncres:
            settings.set_verbosity(1)
            dcd = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_{sim_type}.{k}.{j}.dcd', 
                                      file_unit=dcd_unit,read_only=True, formatted=False)
            dcd_new = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_{sim_type}_1.{k}.{j}.dcd',
                                          dcd_unit+5, False, False)
            lingo.charmm_script(f'merge first {dcd_unit} nunit 1 output {dcd_unit+5} sele all end -\n sele .not. (segid SOLV .or. segid IONS) .and. .not. hydrogen end')
            os.remove(f'{input_folder}/run{run}/dcd/{name}_{sim_type}.{k}.{j}.dcd')
            os.rename(f'{input_folder}/run{run}/dcd/{name}_{sim_type}_1.{k}.{j}.dcd', f'{input_folder}/run{run}/dcd/{name}_{sim_type}.{k}.{j}.dcd')
            dcd.close()
            dcd_new.close()
            settings.set_verbosity(verbosity)
        pycharmm.charmm_script('! Simulation Complete')
        settings.set_verbosity(5)
        lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit = lmd_unit, read_only = True, formatted = False)
        # pycharmm.charmm_script(f'traj lamb print ctlo 0.95 cthi 0.99 first {lmd_unit} nunit 1')
        lmd.close()
    #BOLD RED checkmark: calculating the box size
    xtla = pycharmm.get_energy_value('XTLA')
    xtlb = pycharmm.get_energy_value('XTLB')
    xtlc = pycharmm.get_energy_value('XTLC')
    
    # Safely assign box size values with fallback
    box_size[0] = float(xtla) if xtla is not None else box_size[0]
    box_size[1] = float(xtlb) if xtlb is not None else box_size[1]
    box_size[2] = float(xtlc) if xtlc is not None else box_size[2]
    box_size = [max(box_size), max(box_size), max(box_size)]
    write.coor_card(f'{input_folder}/run{run}/prod.{k}.{j}.crd')
    f = open(f'{input_folder}/run{run}/box.{k}.{j}.dat', 'w')
    f.write(f'{type_}\n')
    f.write(f'{box_size[0]} {box_size[1]} {box_size[2]}\n')
    f.write(f'{angles[0]} {angles[1]} {angles[2]}')
    f.close()

    # check that log{letter}.{k}.out exists and has line DYNA>    {nsteps} steps completed
    log_file = f'{input_folder}/run{run}/log.{k}.{j}.out'
    if os.path.isfile(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            if f'DYNA>    {nsteps_prod}' in log_content:
                print(f'\u001b[32;1m\u2714 Simulation completed successfully for run {run}, phase {phase}, replica k={k}, j={j}\u001b[0m')
            else:
                print(f'\u001b[31;1m\u2718 Simulation doesn\'t seem to have completed successfully for run {run}, phase {phase}, replica k={k}, j={j}. Please check the log file. Try to re-run or adjust fnex values (requires regeneration of G_profiles).\u001b[0m')
    lingo.charmm_script('blade off')

def compute_cphmd_parameters():
    global delta_pKa
    """
    Compute CpHMD parameters from micro-pKa values and prepare bias shifts.
    """
    global alf_info, patch_info, site_pH0, site_pKa_shifts
    
    # Guard clauses to ensure variables are initialized
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called.")
    if alf_info is None:
        raise ValueError("alf_info is not initialized. Please ensure alf_initialize() has been called.")
    
    site_pH0 = {}
    site_pKa_shifts = {}
    
    # Constants
    kB = 0.0019872041  # kcal·mol⁻¹·K⁻¹
    T = alf_info['temp']  # K
    kBT = kB * T
    kTln10 = kBT * np.log(10.0)
    
    # Initialize storage for site-wise parameters
    
    # Process each site
    for site in patch_info['site'].unique():
        site_patches = patch_info[patch_info['site'] == site]
        
        # Sort micro-pKₐ values by flag
        pKa_upos = []
        pKa_uneg = []
        pKa_none = []
        
        for _, row in site_patches.iterrows():
            tag = row['TAG']
            if tag.startswith('UPOS'):
                pKa_value = float(tag.split()[1])
                pKa_upos.append(pKa_value)
            elif tag.startswith('UNEG'):
                pKa_value = float(tag.split()[1])
                pKa_uneg.append(pKa_value)
            else:  # NONE or other
                pKa_none.append(0.0)  # Keep for bookkeeping
        
        # Compute pH₀ and pKa shifts for this site
        if pKa_upos and pKa_uneg:
            # Case 1: Both UPOS and UNEG states present (e.g., histidine)
            pKa_pos = -np.log10(np.sum(10**(-np.array(pKa_upos))))  # pKa(+↔0)
            pKa_neg = np.log10(np.sum(10**(np.array(pKa_uneg))))  # pKa(0↔–)
            pH0 = 0.5 * (pKa_pos + pKa_neg)
            
            site_pH0[site] = pH0
            
            # Compute pKa shifts for each patch in this site
            site_shifts = {}
            for _, row in site_patches.iterrows():
                tag = row['TAG']
                if tag.startswith('UPOS') or tag.startswith('UNEG'):
                    pKa_i = float(tag.split()[1])
                    shift = pH0 - pKa_i
                    site_shifts[row['SELECT']] = shift
                else:
                    site_shifts[row['SELECT']] = 0.0
            
            site_pKa_shifts[site] = site_shifts
            
            print(f"Site {site}: pH0={pH0:.3f}, pKa_pos={pKa_pos:.3f}, pKa_neg={pKa_neg:.3f}")
            
        elif pKa_uneg:
            # Case 2: Only UNEG states present (e.g., aspartate, glutamate)
            pKa_neg = np.log10(np.sum(10**(np.array(pKa_uneg))))  # pKa(0↔–)
            pH0 = pKa_neg  # For acids, pH0 = pKa
            
            site_pH0[site] = pH0
            
            # Compute pKa shifts for each patch in this site
            site_shifts = {}
            for _, row in site_patches.iterrows():
                tag = row['TAG']
                if tag.startswith('UNEG'):
                    pKa_i = float(tag.split()[1])
                    shift = pH0 - pKa_i
                    site_shifts[row['SELECT']] = shift
                else:
                    site_shifts[row['SELECT']] = 0.0
            
            site_pKa_shifts[site] = site_shifts
            
            print(f"Site {site} (acid): pH0={pH0:.3f}, pKa_neg={pKa_neg:.3f}")
            
        elif pKa_upos:
            # Case 3: Only UPOS states present (e.g., lysine, arginine)
            pKa_pos = -np.log10(np.sum(10**(-np.array(pKa_upos))))  # pKa(+↔0)
            pH0 = pKa_pos  # For bases, pH0 = pKa
            
            site_pH0[site] = pH0
            
            # Compute pKa shifts for each patch in this site
            site_shifts = {}
            for _, row in site_patches.iterrows():
                tag = row['TAG']
                if tag.startswith('UPOS'):
                    pKa_i = float(tag.split()[1])
                    shift = pH0 - pKa_i
                    site_shifts[row['SELECT']] = shift
                else:
                    site_shifts[row['SELECT']] = 0.0
            
            site_pKa_shifts[site] = site_shifts
            
            print(f"Site {site} (base): pH0={pH0:.3f}, pKa_pos={pKa_pos:.3f}")
    
    return kTln10

def create_bias_shift_file(total_delta_pH, kTln10):
    """
    Create nbshift/b_shift.dat file with pH-dependent bias shifts.
    
    For single sites: total_delta_pH contains only replica shifts (delta_pKa effects)
    For multiple sites: total_delta_pH contains user pH difference + replica shifts
    """
    global patch_info, site_pH0, site_pKa_shifts
    
    # Guard clause to ensure variables are initialized
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called.")
    
    os.makedirs(f'{input_folder}/nbshift', exist_ok=True)
    
    # Prepare bias shifts for all patches
    bias_shifts = []
    
    if patch_info is None:
        raise ValueError("patch_info is not initialized. Please ensure alf_initialize() has been called and patch_info is set.")
    for _, row in patch_info.iterrows():
        tag = row['TAG']
        site = row['site']
        
        if tag.startswith('UPOS'):
            sign_i = +1 
        elif tag.startswith('UNEG'):
            sign_i = -1 
        else:  # NONE
            sign_i = 0
        
        # For b_shift.dat: bias shift for replica differences and pH effects
        b_shift_i = sign_i * kTln10 * total_delta_pH
        bias_shifts.append(b_shift_i)
    
    # Save bias shifts for dynamics (b_shift.dat)
    np.savetxt(f'{input_folder}/nbshift/b_shift.dat', np.reshape(np.array(bias_shifts), (1, -1)), fmt="%.18e")
    
    # Create b_fix_shift.dat with original pKa shifts (used in GetEnergy)
    # These are always relative to the original site pH₀, regardless of effective pH
    bias_fix = []
    for _, row in patch_info.iterrows():
        site = row['site']
        if site in site_pKa_shifts:
            pKa_shift = site_pKa_shifts[site].get(row['SELECT'], 0.0)
            tag = row['TAG']
            if tag.startswith('UPOS'):
                sign_i = +1 
            elif tag.startswith('UNEG'):
                sign_i = -1
            else:  # NONE
                sign_i = 0
            
            # b_fix_shift uses original pKa shifts (site pH₀ - pKa_i)
            b_fix_i = sign_i * kTln10 * pKa_shift
            bias_fix.append(b_fix_i)
        else:
            bias_fix.append(0.0)
    
    # Save fixed bias shifts (used by GetEnergy for ALF analysis)
    np.savetxt(f'{input_folder}/nbshift/b_fix_shift.dat', np.reshape(np.array(bias_fix), (1, -1)), fmt="%.18e")
    
    print(f"Created bias files:")
    print(f"  b_shift.dat (total_delta_pH={total_delta_pH:.3f}): {[f'{x:.6f}' for x in bias_shifts]}")
    print(f"  b_fix_shift.dat (original pKa shifts): {[f'{x:.6f}' for x in bias_fix]}")

    return bias_shifts

def fit_lambda_data_to_hh_formulas(pH_data, lambda_data, microstates):
    """
    Fit simulation lambda data to the correct Henderson-Hasselbalch formulas
    based on the microstate composition.
    
    Returns fitted parameters including effective pKa and Hill coefficient.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        print("scipy not available for fitting")
        return None
    
    # Categorize microstates
    upos = [m for m in microstates if m["tag"].upper().startswith("UPOS")]
    uneg = [m for m in microstates if m["tag"].upper().startswith("UNEG")]
    none = [m for m in microstates if m["tag"].upper().startswith("NONE")]
    
    fitted_params = {}
    
    try:
        # Case 1: UPOS + UNEG + NONE (e.g., histidine) - 3-state system
        if upos and uneg and none:
            # Calculate theoretical macroscopic pKa values
            pKa_pos_theory = -np.log10(np.sum(10**(-np.array([m["pKa"] for m in upos]))))
            pKa_neg_theory = np.log10(np.sum(10**(np.array([m["pKa"] for m in uneg]))))
            
            # Define 3-state fitting function
            def three_state_hh(pH, pKa_pos, pKa_neg):
                # NONE population: 1 / (1 + 10^(pKa_neg - pH) + 10^(pH - pKa_pos))
                return 1.0 / (1.0 + 10**(pKa_neg - pH) + 10**(pH - pKa_pos))
            
            # Initial guess based on theoretical values
            p0 = [pKa_pos_theory, pKa_neg_theory]
            bounds = ([pH_data.min()-2, pH_data.min()-2], [pH_data.max()+2, pH_data.max()+2])
            
            popt, pcov = curve_fit(three_state_hh, pH_data, lambda_data, p0=p0, bounds=bounds)
            fitted_params = {
                "type": "three_state",
                "pKa_pos": popt[0],
                "pKa_neg": popt[1], 
                "pKa_eff": 0.5 * (popt[0] + popt[1]),  # Isoelectric point
                "formula": "1 / (1 + 10^(pKa_neg - pH) + 10^(pH - pKa_pos))"
            }
            print(f"Fitted 3-state HH: pKa_pos={popt[0]:.2f}, pKa_neg={popt[1]:.2f}, pI={fitted_params['pKa_eff']:.2f}")
        
        # Case 2: UPOS + NONE only (e.g., lysine) - 2-state basic system  
        elif upos and none and not uneg:
            # Define 2-state basic fitting function
            def two_state_basic_hh(pH, pKa_eff, n=1):
                # NONE population: 1 / (1 + 10^(pKa_eff - pH))
                return 1.0 / (1.0 + 10**(n * (pKa_eff - pH)))
            
            # Initial guess
            pKa_guess = np.mean(pH_data)
            p0 = [pKa_guess, 1.0]
            bounds = ([pH_data.min()-2, 0.1], [pH_data.max()+2, 5.0])
            
            popt, pcov = curve_fit(two_state_basic_hh, pH_data, lambda_data, p0=p0, bounds=bounds)
            fitted_params = {
                "type": "two_state_basic",
                "pKa_eff": popt[0],
                "n": popt[1],
                "formula": "1 / (1 + 10^(n * (pKa_eff - pH)))"
            }
            print(f"Fitted 2-state basic HH: pKa_eff={popt[0]:.2f}, n={popt[1]:.2f}")
        
        # Case 3: UNEG + NONE only (e.g., aspartate) - 2-state acidic system
        elif uneg and none and not upos:
            # Define 2-state acidic fitting function  
            def two_state_acidic_hh(pH, pKa_eff, n=1):
                # NONE population: 1 / (1 + 10^(pKa_eff - pH))
                return 1.0 / (1.0 + 10**(n * (pKa_eff - pH)))
            
            # Initial guess
            pKa_guess = np.mean(pH_data)
            p0 = [pKa_guess, 1.0]
            bounds = ([pH_data.min()-2, 0.1], [pH_data.max()+2, 5.0])
            
            popt, pcov = curve_fit(two_state_acidic_hh, pH_data, lambda_data, p0=p0, bounds=bounds)
            fitted_params = {
                "type": "two_state_acidic",
                "pKa_eff": popt[0],
                "n": popt[1],
                "formula": "1 / (1 + 10^(n * (pKa_eff - pH)))"
            }
            print(f"Fitted 2-state acidic HH: pKa_eff={popt[0]:.2f}, n={popt[1]:.2f}")
        
        # Case 4: Only UPOS states - fit total UPOS population
        elif upos and not uneg and not none:
            # Define UPOS-only fitting function
            def upos_only_hh(pH, pKa_eff, n=1):
                # Total UPOS population: 1 / (1 + 10^(n * (pH - pKa_eff)))
                return 1.0 / (1.0 + 10**(n * (pH - pKa_eff)))
            
            pKa_guess = np.mean(pH_data)
            p0 = [pKa_guess, 1.0]
            bounds = ([pH_data.min()-2, 0.1], [pH_data.max()+2, 5.0])
            
            popt, pcov = curve_fit(upos_only_hh, pH_data, lambda_data, p0=p0, bounds=bounds)
            fitted_params = {
                "type": "upos_only",
                "pKa_eff": popt[0],
                "n": popt[1],
                "formula": "1 / (1 + 10^(n * (pH - pKa_eff)))"
            }
            print(f"Fitted UPOS-only HH: pKa_eff={popt[0]:.2f}, n={popt[1]:.2f}")
        
        # Case 5: Only UNEG states - fit total UNEG population
        elif uneg and not upos and not none:
            # Define UNEG-only fitting function
            def uneg_only_hh(pH, pKa_eff, n=1):
                # Total UNEG population: 1 / (1 + 10^(n * (pKa_eff - pH)))
                return 1.0 / (1.0 + 10**(n * (pKa_eff - pH)))
            
            pKa_guess = np.mean(pH_data)
            p0 = [pKa_guess, 1.0]
            bounds = ([pH_data.min()-2, 0.1], [pH_data.max()+2, 5.0])
            
            popt, pcov = curve_fit(uneg_only_hh, pH_data, lambda_data, p0=p0, bounds=bounds)
            fitted_params = {
                "type": "uneg_only", 
                "pKa_eff": popt[0],
                "n": popt[1],
                "formula": "1 / (1 + 10^(n * (pKa_eff - pH)))"
            }
            print(f"Fitted UNEG-only HH: pKa_eff={popt[0]:.2f}, n={popt[1]:.2f}")
        
        return fitted_params
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None

def generate_fitted_hh_curve(pH_grid, fitted_params, microstates):
    """
    Generate Henderson-Hasselbalch curve using fitted parameters.
    """
    if not fitted_params:
        return np.zeros_like(pH_grid)
    
    param_type = fitted_params["type"]
    
    if param_type == "three_state":
        pKa_pos = fitted_params["pKa_pos"]
        pKa_neg = fitted_params["pKa_neg"]
        return 1.0 / (1.0 + 10**(pKa_neg - pH_grid) + 10**(pH_grid - pKa_pos))
    
    elif param_type == "two_state_basic":
        pKa_eff = fitted_params["pKa_eff"]
        n = fitted_params["n"]
        return 1.0 / (1.0 + 10**(n * (pKa_eff - pH_grid)))
    
    elif param_type == "two_state_acidic":
        pKa_eff = fitted_params["pKa_eff"]
        n = fitted_params["n"]
        return 1.0 / (1.0 + 10**(n * (pKa_eff - pH_grid)))
    
    elif param_type == "upos_only":
        pKa_eff = fitted_params["pKa_eff"]
        n = fitted_params["n"]
        return 1.0 / (1.0 + 10**(n * (pH_grid - pKa_eff)))
    
    elif param_type == "uneg_only":
        pKa_eff = fitted_params["pKa_eff"]
        n = fitted_params["n"]
        return 1.0 / (1.0 + 10**(n * (pKa_eff - pH_grid)))
    
    else:
        return np.zeros_like(pH_grid)

def generate_hh_curve(run, lambda_data, l_files):
    global delta_pKa
    """
    Generate Henderson-Hasselbalch curve from lambda data and save to plots folder.
    Uses advanced logistic fitting with UPOS/UNEG/NONE state handling.
    
    Args:
        run: Current simulation run number
        lambda_data: Combined lambda data from all replicas
        l_files: List of lambda files
    """
    # --- user‑defined colour palette -------------------------------------
    COLORS = ['tomato', 'dodgerblue', 'yellowgreen', 'orange', 'plum']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)
    # Check if matplotlib is available
    if plt is None:
        print("matplotlib not available - skipping HH curve generation")
        return
    
    def logistic(pH, pKa, s, P):
        """
        Henderson–Hasselbalch term.
        s = +1  → population falls with pH  (UPOS or 'downward')
        s = –1  → population rises with pH  (UNEG or 'upward')
        s =  0  → constant (used only if both UPOS and UNEG are present)
        """
        if s == 0:
            return P * np.ones_like(pH)
        return P / (1.0 + 10.0**(s * (pH - pKa)))
    
    def block_weights(block):
        """
        Assign raw weights (w_raw) and logistic‐curve signs (m['sign'])
        for each micro-state in the *block*.

        * Weight rules
            NONE  -> w_raw = 1
            UPOS  -> w_raw ∝ 10^(−pKa)
            UNEG  -> w_raw ∝ 10^(−pKa)          # keeps ‘lower pKa ⇒ higher weight’

        * Curve-direction (sign) rules based on user requirements:
            
            Scenario 1 - ASP (NONE, UNEG, UNEG): 
                NONE : sign = -1  (positive slope - fails with pH)
                UNEG : sign = +1  (negative slope - rises with pH)

            Scenario 2 - HSP (NONE, UPOS, UPOS):
                NONE : sign = +1  (negative slope - rises with pH)  
                UPOS : sign = -1  (positive slope - fails with pH)

            Scenario 3 - Histidine (UPOS, NONE, UNEG):
                UPOS : sign = -1  (positive slope - fails with pH)
                UNEG : sign = +1  (negative slope - rises with pH)
                NONE : sign = 0   (bell function - handled separately)
        """
        has_upos = any(m["tag"].upper().startswith("UPOS") for m in block)
        has_uneg = any(m["tag"].upper().startswith("UNEG") for m in block)
        has_none = any(m["tag"].upper().startswith("NONE") for m in block)

        for m in block:
            tag = m["tag"].upper()

            # ---- weights ----------------------------------------------------
            if tag.startswith("NONE"):
                m["w_raw"] = 1.0
            else:                               # UPOS or UNEG
                m["w_raw"] = 10.0 ** (-m["pKa"])

            # ---- logistic signs based on scenarios ------------------------
            if tag.startswith("NONE"):
                if has_upos and has_uneg:
                    # Scenario 3: Histidine-like (UPOS + NONE + UNEG)
                    m["sign"] = 0               # Bell function
                elif has_upos and not has_uneg:
                    # Scenario 2: HSP-like (NONE + UPOS)
                    m["sign"] = -1              # Rises with pH (logistic s=-1)
                elif has_uneg and not has_upos:
                    # Scenario 1: ASP-like (NONE + UNEG)  
                    m["sign"] = +1              # Falls with pH (logistic s=+1)
                else:
                    # Only NONE states
                    m["sign"] = 0               # Flat
            elif tag.startswith("UPOS"):
                m["sign"] = +1                  # Always falls with pH (logistic s=+1)
            elif tag.startswith("UNEG"):
                m["sign"] = -1                  # Always rises with pH (logistic s=-1)
            else:
                m["sign"] = 0                   # safeguard

        return block
    
    def fit_comprehensive_hh(pH_data, lambda_data, microstates):
        """
        Fit comprehensive Henderson-Hasselbalch model with UPOS/UNEG/NONE states.
        """
        # 1. Group micro-states by flag (case-insensitive)
        upos = [m for m in microstates if m["tag"].upper().replace(" ", "").startswith("UPOS")]
        uneg = [m for m in microstates if m["tag"].upper().replace(" ", "").startswith("UNEG")]
        none = [m for m in microstates if m["tag"].upper().replace(" ", "").startswith("NONE")]


        # 2. Give every UPOS/UNEG micro-state a raw weight
        upos = block_weights(upos)
        uneg = block_weights(uneg)
        
        
        # 3. Global normalization across all directional micro-states
        W = sum(m["w_raw"] for m in upos + uneg)
        if W > 0:
            for m in upos + uneg:
                m["Pmax"] = m["w_raw"] / W
        else:
            # No directional states - all weights are zero
            for m in upos + uneg:
                m["Pmax"] = 0.0
        
        none_sign = None  # Default NONE sign (will be set later)
        # 4. Decide what shape the NONE block must take
        if none:
            if upos and not uneg:           # Case A: UPOS + NONE
                none_sign = -1
            elif uneg and not upos:         # Case B: UNEG + NONE
                none_sign = +1
            else:                           # Case C: UPOS + UNEG + NONE
                none_sign = None            # baseline will be flat
        
        # 5-a Weight and shape in Cases A & B
        if none_sign in (+1, -1):
            none = block_weights(none)
            # Global normalization including NONE states
            W_total = sum(m["w_raw"] for m in upos + uneg + none)
            if W_total > 0:
                for m in upos + uneg + none:
                    m["Pmax"] = m["w_raw"] / W_total
        # 5-b Weight and shape in Case C
        elif none_sign is None and none:
            # Take an arbitrary reference (the lowest pKa for stability)
            none.sort(key=lambda m: m["pKa"])
            pKa_ref = none[0]["pKa"]
            # Raw weights: higher pKa ⇒ larger share
            w_raw_none = [10.0**(m["pKa"] - pKa_ref) for m in none]
            S_none = sum(w_raw_none)
            for m, w_i in zip(none, w_raw_none):
                m["Pshare"] = w_i / S_none if S_none > 0 else 1.0/len(none)  # relative share
                m["sign"] = 0              # flat, as it's a remainder of UPOS and UNEG
        
        # 6. Compute theoretical populations using correct formulas
        def compute_populations(pH_grid):
            pop_dict = {}  # Store individual population curves with unique IDs
            
            # Calculate macroscopic pKa values
            if upos:
                # pKa_pos: macroscopic pKa for UPOS/NONE transition
                pKa_pos = -np.log10(np.sum(10**(-np.array([m["pKa"] for m in upos]))))
            else:
                pKa_pos = None
                
            if uneg:
                # pKa_neg: macroscopic pKa for NONE/UNEG transition  
                pKa_neg = np.log10(np.sum(10**(np.array([m["pKa"] for m in uneg]))))
            else:
                pKa_neg = None
            
            # Case 1: UPOS + UNEG + NONE (e.g., histidine)
            if upos and uneg and none:
                # NONE population: 1 / (1 + 10^(pKa_neg - pH) + 10^(pH - pKa_pos))
                none_pop = 1.0 / (1.0 + 10**(pKa_neg - pH_grid) + 10**(pH_grid - pKa_pos))
                
                # Total UPOS population: NONE_pop * 10^(pKa_pos - pH)
                upos_total = none_pop * 10**(pKa_pos - pH_grid)
                
                # Total UNEG population: NONE_pop * 10^(pH - pKa_neg)
                uneg_total = none_pop * 10**(pH_grid - pKa_neg)
                
                # Distribute NONE population among individual NONE states
                if len(none) == 1:
                    uid = f'{none[0]["tag"]}_{none[0].get("site", 0)}_{id(none[0])}'
                    pop_dict[uid] = none_pop
                else:
                    # Multiple NONE states - distribute based on relative weights
                    none_weights = [10**(m["pKa"] - none[0]["pKa"]) for m in none]  # relative to first
                    none_weight_sum = sum(none_weights)
                    for i, m in enumerate(none):
                        weight_frac = none_weights[i] / none_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = none_pop * weight_frac
                
                # Distribute UPOS population among individual UPOS states
                if len(upos) == 1:
                    uid = f'{upos[0]["tag"]}_{upos[0].get("site", 0)}_{id(upos[0])}'
                    pop_dict[uid] = upos_total
                else:
                    # Multiple UPOS states - distribute based on relative Boltzmann weights
                    upos_weights = [10**(-m["pKa"]) for m in upos]
                    upos_weight_sum = sum(upos_weights)
                    for i, m in enumerate(upos):
                        weight_frac = upos_weights[i] / upos_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = upos_total * weight_frac
                
                # Distribute UNEG population among individual UNEG states  
                if len(uneg) == 1:
                    uid = f'{uneg[0]["tag"]}_{uneg[0].get("site", 0)}_{id(uneg[0])}'
                    pop_dict[uid] = uneg_total
                else:
                    # Multiple UNEG states - distribute based on relative Boltzmann weights
                    uneg_weights = [10**(-m["pKa"]) for m in uneg]
                    uneg_weight_sum = sum(uneg_weights)
                    for i, m in enumerate(uneg):
                        weight_frac = uneg_weights[i] / uneg_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = uneg_total * weight_frac
            
            # Case 2: UPOS + NONE only (e.g., lysine)
            elif upos and none and not uneg:
                # NONE population: 1 / (1 + 10^(pKa_pos - pH))
                none_pop = 1.0 / (1.0 + 10**(pKa_pos - pH_grid))
                
                # Total UPOS population: 1 / (1 + 10^(pH - pKa_pos))
                upos_total = 1.0 / (1.0 + 10**(pH_grid - pKa_pos))
                
                # Distribute NONE population
                if len(none) == 1:
                    uid = f'{none[0]["tag"]}_{none[0].get("site", 0)}_{id(none[0])}'
                    pop_dict[uid] = none_pop
                else:
                    none_weights = [10**(m["pKa"] - none[0]["pKa"]) for m in none]
                    none_weight_sum = sum(none_weights)
                    for i, m in enumerate(none):
                        weight_frac = none_weights[i] / none_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = none_pop * weight_frac
                
                # Distribute UPOS population
                if len(upos) == 1:
                    uid = f'{upos[0]["tag"]}_{upos[0].get("site", 0)}_{id(upos[0])}'
                    pop_dict[uid] = upos_total
                else:
                    upos_weights = [10**(-m["pKa"]) for m in upos]
                    upos_weight_sum = sum(upos_weights)
                    for i, m in enumerate(upos):
                        weight_frac = upos_weights[i] / upos_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = upos_total * weight_frac
            
            # Case 3: UNEG + NONE only (e.g., aspartate)
            elif uneg and none and not upos:
                # NONE population: 1 / (1 + 10^(pKa_neg - pH))
                none_pop = 1.0 / (1.0 + 10**(pKa_neg - pH_grid))
                
                # Total UNEG population: 1 / (1 + 10^(pH - pKa_neg))
                uneg_total = 1.0 / (1.0 + 10**(pH_grid - pKa_neg))
                
                # Distribute NONE population
                if len(none) == 1:
                    uid = f'{none[0]["tag"]}_{none[0].get("site", 0)}_{id(none[0])}'
                    pop_dict[uid] = none_pop
                else:
                    none_weights = [10**(m["pKa"] - none[0]["pKa"]) for m in none]
                    none_weight_sum = sum(none_weights)
                    for i, m in enumerate(none):
                        weight_frac = none_weights[i] / none_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = none_pop * weight_frac
                
                # Distribute UNEG population
                if len(uneg) == 1:
                    uid = f'{uneg[0]["tag"]}_{uneg[0].get("site", 0)}_{id(uneg[0])}'
                    pop_dict[uid] = uneg_total
                else:
                    uneg_weights = [10**(-m["pKa"]) for m in uneg]
                    uneg_weight_sum = sum(uneg_weights)
                    for i, m in enumerate(uneg):
                        weight_frac = uneg_weights[i] / uneg_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = uneg_total * weight_frac
            
            # Case 4: Only UPOS states
            elif upos and not uneg and not none:
                # All population goes to UPOS states, distributed by weights
                upos_weights = [10**(-m["pKa"]) for m in upos]
                upos_weight_sum = sum(upos_weights)
                for i, m in enumerate(upos):
                    weight_frac = upos_weights[i] / upos_weight_sum
                    uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                    pop_dict[uid] = np.ones_like(pH_grid) * weight_frac
            
            # Case 5: Only UNEG states
            elif uneg and not upos and not none:
                # All population goes to UNEG states, distributed by weights
                uneg_weights = [10**(-m["pKa"]) for m in uneg]
                uneg_weight_sum = sum(uneg_weights)
                for i, m in enumerate(uneg):
                    weight_frac = uneg_weights[i] / uneg_weight_sum
                    uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                    pop_dict[uid] = np.ones_like(pH_grid) * weight_frac
            
            # Case 6: Only NONE states
            elif none and not upos and not uneg:
                # All population goes to NONE states, distributed equally or by weight
                if len(none) == 1:
                    uid = f'{none[0]["tag"]}_{none[0].get("site", 0)}_{id(none[0])}'
                    pop_dict[uid] = np.ones_like(pH_grid)
                else:
                    none_weights = [10**(m["pKa"] - none[0]["pKa"]) for m in none]
                    none_weight_sum = sum(none_weights)
                    for i, m in enumerate(none):
                        weight_frac = none_weights[i] / none_weight_sum
                        uid = f'{m["tag"]}_{m.get("site", 0)}_{id(m)}'
                        pop_dict[uid] = np.ones_like(pH_grid) * weight_frac
            
            # Return total population (should sum to 1.0 at all pH values)
            total_pop = np.sum(list(pop_dict.values()), axis=0)
            return total_pop
        
        return compute_populations, upos + uneg + none
    
    # Create plots directory
    plots_dir = f'../plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract replica information and pH values
    replica_phs = []
    replica_lambdas = []
    
    
    # Guard against None alf_info
    if alf_info is None or 'ncentral' not in alf_info:
        print("Warning: alf_info not properly initialized - using default ncentral=0")
        ncentral = 0
    else:
        ncentral = alf_info['ncentral']
    
    # Get effective_pH from the global scope or compute it
    if pH is not None:
        # Use the same logic as in change_bias function
        if len(set(site_pH0.values())) == 1:
            effective_pH = list(site_pH0.values())[0]
        elif len(set(site_pH0.values())) > 1:
            effective_pH = 0.0
        else:
            effective_pH = pH
    else:
        effective_pH = 7.0  # Default if pH not set
    
    print(f"Using effective_pH = {effective_pH:.2f} for theoretical pKa calculations")
    
    # Process each replica's lambda data to get replica-specific data
    cutoff_threshold = 0.985  # Threshold for counting "active" states
    print(f"Using cutoff threshold {cutoff_threshold} for relative fraction calculation")
    
    # Parse replica information from l_files (which now contains representative files for each replica)
    replica_data_dict = {}
    
    # First, organize the combined lambda_data by replica
    data_start_idx = 0
    for file in sorted(l_files):
        # Extract replica number from filename (Lambda.{repeat}.{replica}.dat)
        try:
            replica_idx = int(file.split('.')[2])
        except:
            continue
        
        # Find how many rows belong to this replica in the combined lambda_data
        # This is tricky because we combined all repeats for each replica
        # We need to reload the combined data for this specific replica
        
        # Reload and combine all repeat files for this replica
        replica_files = []
        for fname in sorted(f for f in os.listdir("data") if f.startswith("Lambda")):
            try:
                parts = fname.split('.')
                if len(parts) >= 4 and int(parts[2]) == replica_idx:
                    replica_files.append(fname)
            except (ValueError, IndexError):
                continue
        
        # Combine all repeats for this replica
        replica_combined_data = None
        for fname in sorted(replica_files):
            l = np.loadtxt(f'data/{fname}')
            if replica_combined_data is None:
                replica_combined_data = l
            else:
                replica_combined_data = np.vstack([replica_combined_data, l])
        
        if replica_combined_data is not None:
            replica_data_dict[replica_idx] = replica_combined_data
            print(f"HH Curve: Replica {replica_idx} combined data shape: {replica_combined_data.shape} from {len(replica_files)} repeat files")
    
    print(f"HH Curve: Processing {len(replica_data_dict)} replicas for curve generation")
    
    # Now process each replica's combined data
    for replica_idx in sorted(replica_data_dict.keys()):
        replica_lambda = replica_data_dict[replica_idx]
        
        # Calculate replica-specific pH
        replica_shift = delta_pKa * (replica_idx - ncentral)
        replica_ph = effective_pH + replica_shift
        replica_phs.append(replica_ph)
        
        # Calculate relative fraction of values above cutoff for each titratable site
        if replica_lambda.ndim == 1:
            # Single column - count values above cutoff and normalize
            above_cutoff_count = np.sum(replica_lambda > cutoff_threshold)
            total_above_cutoff = above_cutoff_count  # Only one column
            
            if total_above_cutoff > 0:
                fraction_lambda = above_cutoff_count / total_above_cutoff
            else:
                fraction_lambda = 0.0  # No values above cutoff
        else:
            # Multiple columns (multiple titratable sites)
            above_cutoff_counts = np.sum(replica_lambda > cutoff_threshold, axis=0)
            total_above_cutoff = np.sum(above_cutoff_counts)
            
            if total_above_cutoff > 0:
                fraction_lambda = above_cutoff_counts / total_above_cutoff
            else:
                # No values above cutoff for any site
                fraction_lambda = np.zeros(replica_lambda.shape[1])
        
        replica_lambdas.append(fraction_lambda)
    
    # Convert to numpy arrays for easier handling
    replica_phs = np.array(replica_phs)
    replica_lambdas = np.array(replica_lambdas)

    # Filter out non-physical values - for fractions, we expect values between 0 and 1
    # No need for strict cutoff filtering since fractions are already normalized
    # Instead, filter out any NaN or infinite values
    if replica_lambdas.ndim == 1:
        # Single site: keep finite values
        physical_mask = np.isfinite(replica_lambdas)
    else:
        # Multiple sites: keep points where ALL fraction values are finite
        physical_mask = np.all(np.isfinite(replica_lambdas), axis=1)
    
    replica_phs = replica_phs[physical_mask]
    replica_lambdas = replica_lambdas[physical_mask]
    
    # Check if we have enough physical data points
    if len(replica_phs) < 3:
        print(f"Warning: Only {len(replica_phs)} finite data points - skipping HH curve generation")
        return
    
    # Sort by pH for plotting
    sort_indices = np.argsort(replica_phs)
    replica_phs = replica_phs[sort_indices]
    replica_lambdas = replica_lambdas[sort_indices]
    
    # Check for dominance patterns - skip plotting if one substituent dominates across all pH
    def check_dominance(data, dominance_threshold=0.99, min_variance=0.001):
        global phase
        """
        Check if one substituent dominates across all pH values with NO variation.
        Returns True if data should be skipped (truly flat/uninformative pattern found).
        
        Only skips data that shows complete lack of pH-dependent behavior:
        - Perfect dominance: one column = 1.0, others = 0.0 for ALL pH points
        - Extremely low variance: essentially flat lines with no meaningful change
        """
        if data.ndim == 1:
            # Single site: only skip if completely flat (no pH dependence)
            variance = np.var(data)
            if variance < min_variance:
                print(f"Skipping plot: Single site shows no pH dependence (variance={variance:.6f})")
                return True
            return False
        else:
            # Multiple sites: only skip if there's perfect dominance with zero variation
            
            # Check for perfect 1.0/0.0 patterns across ALL pH points
            # BUT ensure it's the SAME column dominating everywhere (no pH dependence)
            dominant_columns = []  # Track which column dominates at each pH
            perfect_pattern_count = 0
            total_rows = data.shape[0]
            
            for i in range(total_rows):
                row = data[i, :]
                max_val = np.max(row)
                max_indices = np.where(row == max_val)[0]
                
                # Perfect dominance: one value = 1.0, all others = 0.0
                if max_val > dominance_threshold and len(max_indices) == 1:
                    others = row[row != max_val]
                    if len(others) > 0 and np.max(others) < (1.0 - dominance_threshold):
                        perfect_pattern_count += 1
                        dominant_columns.append(max_indices[0])  # Store which column dominates
                    else:
                        dominant_columns.append(None)  # Not a perfect pattern
                else:
                    dominant_columns.append(None)  # Not a perfect pattern
            
            # Only skip if 95% or more rows show perfect dominance AND it's always the same column
            if perfect_pattern_count >= 0.95 * total_rows:
                # Check if the same column dominates throughout (no pH dependence)
                valid_dominant_cols = [col for col in dominant_columns if col is not None]
                if len(set(valid_dominant_cols)) == 1:
                    # Same column dominates everywhere - no pH dependence
                    print(f"Skipping plot: Same column ({valid_dominant_cols[0]}) dominates in {perfect_pattern_count}/{total_rows} rows")
                    print("  No pH dependence detected - same state dominates at all pH values")
                    return True
                elif len(set(valid_dominant_cols)) < total_rows:
                    # Different columns dominate at different pH - this is GOOD data!
                    print(f"Good pH-dependent data detected: {len(set(valid_dominant_cols))} different columns dominate across pH range")
                    print(f"  Dominant columns by pH: {valid_dominant_cols}")
                    return False  # Do NOT skip this data
                else: 
                    print("Skipping plot: No clear dominance pattern found - all columns vary")
                    if phase == 2: phase = 3
                    return False

            
            # Additionally check: if any column has near-zero variance AND high mean
            # (flat line at high values)
            for col in range(data.shape[1]):
                col_data = data[:, col]
                col_variance = np.var(col_data)
                col_mean = np.mean(col_data)
                
                # Very flat line at high values (no pH dependence)
                if col_variance < min_variance and col_mean > 0.98:
                    print(f"Skipping plot: Column {col} shows flat dominance")
                    print(f"  Mean={col_mean:.4f}, Variance={col_variance:.6f} - no pH dependence")
                    return True
            return False
    
    if check_dominance(replica_lambdas):
        print("Skipping Henderson-Hasselbalch curve generation - one substituent dominates across all pH values")
        # Still save the data file for inspection
        data_file = f'{plots_dir}/hh_data_run{run}_skipped.txt'
        with open(data_file, 'w') as f:
            f.write("# Henderson-Hasselbalch curve data (SKIPPED - dominance pattern detected)\n")
            f.write(f"# Generated from run {run}\n")
            f.write("# Method: Relative fraction of values above cutoff (0.985)\n")
            f.write("# pH\tRelative_Fraction(s)\n")
            
            for i, ph in enumerate(replica_phs):
                if replica_lambdas.ndim == 1:
                    f.write(f"{ph:.6f}\t{replica_lambdas[i]:.6f}\n")
                else:
                    lambda_str = '\t'.join([f"{replica_lambdas[i, j]:.6f}" for j in range(replica_lambdas.shape[1])])
                    f.write(f"{ph:.6f}\t{lambda_str}\n")
        
        print(f"Data saved to {data_file} for inspection")
        return
    
    # Print summary of data ranges for debugging
    if replica_lambdas.ndim > 1:
        print(f"Data summary for {replica_lambdas.shape[1]} substituents:")
        for col in range(replica_lambdas.shape[1]):
            col_data = replica_lambdas[:, col]
            print(f"  Column {col}: min={np.min(col_data):.3f}, max={np.max(col_data):.3f}, var={np.var(col_data):.4f}")
    
    # Create the Henderson-Hasselbalch curve plot
    if replica_lambdas.ndim == 1:
        # Single titratable site - create microstate info from patch_info
        microstates = []
        if patch_info is not None and len(patch_info) > 0:
            for _, row in patch_info.iterrows():
                # Extract pKa from TAG if available, otherwise use effective_pH
                pKa_val = effective_pH  # Use effective_pH instead of 7.0
                tag = row['TAG']
                if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                    try:
                        # Try to extract pKa from tag like "UPOS4.2" or "upos4.2"
                        pKa_val = float(tag[4:]) if len(tag) > 4 else effective_pH
                    except:
                        pKa_val = effective_pH
                
                microstates.append({
                    "pKa": pKa_val,
                    "tag": tag,
                    "site": row['site']
                })
            
            site_title = f"{patch_info.iloc[0]['SEGID']}{patch_info.iloc[0]['RESID']}"
        else:
            site_title = name.upper()
            microstates = [{"pKa": effective_pH, "tag": "NONE", "site": "1"}]
        
        plt.figure(figsize=(12, 8))
        plt.plot(replica_phs, replica_lambdas, 'bo', markersize=8, label='Simulation data')
        
        # Generate theoretical curve based on pKa and weights from patch_info
        if patch_info is not None and len(patch_info) > 0:
            # Create microstates from patch_info
            microstates = []
            for _, row in patch_info.iterrows():
                pKa_val = effective_pH  # Use effective_pH instead of 7.0
                tag = row['TAG']
                if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                    try:
                        pKa_val = float(tag[4:]) if len(tag) > 4 else effective_pH
                    except:
                        pKa_val = effective_pH
                if tag.startswith('NONE'):
                    try:
                        pKa_val = site_pH0[row['site']]
                    except KeyError:
                        pKa_val = effective_pH
                microstates.append({
                    "pKa": pKa_val,
                    "tag": tag,
                    "site": row['site']
                })
            # Use fit_comprehensive_hh logic to get populations
            ph_theory = np.linspace(replica_phs.min()-1, replica_phs.max()+1, 200)
            compute_pops, _ = fit_comprehensive_hh(replica_phs, replica_lambdas, microstates)
            lambda_theory = compute_pops(ph_theory)
            plt.plot(ph_theory, lambda_theory, 'g--', linewidth=2, alpha=0.8, label='Theoretical (block-weighted)')
        
        # Fit comprehensive HH model if we have microstates
        if microstates and len(replica_phs) >= 4:
            try:
                # First, fit the simulation data to extract parameters
                fitted_params = fit_lambda_data_to_hh_formulas(replica_phs, replica_lambdas, microstates)
                
                # Generate theoretical curve using fitted parameters
                compute_pops, fitted_states = fit_comprehensive_hh(replica_phs, replica_lambdas, microstates)
                
                # Generate smooth curve for plotting
                ph_smooth = np.linspace(replica_phs.min()-1, replica_phs.max()+1, 200)
                lambda_smooth = compute_pops(ph_smooth)
                
                plt.plot(ph_smooth, lambda_smooth, 'r-', linewidth=2, 
                        label='Comprehensive HH fit')
                
                # Plot fitted curve using extracted parameters
                if fitted_params:
                    lambda_fitted = generate_fitted_hh_curve(ph_smooth, fitted_params, microstates)
                    plt.plot(ph_smooth, lambda_fitted, 'm:', linewidth=3, alpha=0.8,
                            label=f'Fitted HH (pKa={fitted_params.get("pKa_eff", "N/A"):.2f})')
                
                # Show individual components
                colors = COLORS
                for i, state in enumerate(fitted_states[:4]):  # Limit to 4 for visibility
                    if 'Pmax' in state:
                        individual = logistic(ph_smooth, state["pKa"], state["sign"], state["Pmax"])
                        plt.plot(ph_smooth, individual, '--', color=colors[i % len(colors)], 
                                alpha=0.7, linewidth=1, 
                                label=f'{state["tag"]} (pKa={state["pKa"]:.1f})')
                
            except Exception as e:
                print(f"Comprehensive HH fitting failed: {e}")
                # Fallback to simple fitting with proper direction detection
                try:
                    from scipy.optimize import curve_fit
                    
                    # Detect sigmoid direction from data trend
                    correlation = np.corrcoef(replica_phs, replica_lambdas)[0, 1]
                    is_rising = correlation > 0  # Positive correlation = rising with pH
                    print(f"Direction detection: correlation={correlation:.3f}, is_rising={is_rising}")
                    
                    def adaptive_henderson_hasselbalch(pH, pKa, n=1):
                        """Henderson-Hasselbalch with adaptive direction"""
                        if is_rising:
                            # Rising sigmoid (UNEG-like): s=-1 form
                            return 1.0 / (1.0 + 10**(n * (pKa - pH)))
                        else:
                            # Falling sigmoid (UPOS-like): s=+1 form
                            return 1.0 / (1.0 + 10**(n * (pH - pKa)))
                    
                    pKa_guess = np.mean(replica_phs)
                    
                    # Constrain n to be positive
                    bounds = ([replica_phs.min()-2, 0.1], [replica_phs.max()+2, 5])
                    
                    popt, pcov = curve_fit(adaptive_henderson_hasselbalch, replica_phs, replica_lambdas, 
                                         p0=[pKa_guess, 1], bounds=bounds)
                    
                    fitted_pKa, fitted_n = popt
                    ph_smooth = np.linspace(replica_phs.min()-1, replica_phs.max()+1, 200)
                    lambda_smooth = adaptive_henderson_hasselbalch(ph_smooth, fitted_pKa, fitted_n)
                    
                    direction_label = "rising" if is_rising else "falling"
                    plt.plot(ph_smooth, lambda_smooth, 'r-', linewidth=2, 
                            label=f'Adaptive HH fit ({direction_label}): pKa={fitted_pKa:.2f}, n={fitted_n:.2f}')
                    
                except ImportError:
                    print("scipy not available for fallback fitting")
                except Exception as e2:
                    print(f"Fallback fitting also failed: {e2}")
        
        plt.xlabel('pH', fontsize=14)
        plt.ylabel('Relative Fraction (above cutoff)', fontsize=14)
        plt.title(f'{site_title} - Henderson-Hasselbalch Curve (Run {run})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.ylim(-0.05, 1.05)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/hh_curve_run{run}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Multiple titratable sites - group by site and plot all subs on same plot
        n_sites = replica_lambdas.shape[1]
        
        # Group columns by site (SEGID+RESID)
        site_groups = {}
        if patch_info is not None:
            for site_idx in range(min(n_sites, len(patch_info))):
                site_info = patch_info.iloc[site_idx]
                site_key = f"{site_info['SEGID']}{site_info['RESID']}"
                sub_label = str(site_info['sub'])
                
                if site_key not in site_groups:
                    site_groups[site_key] = []
                site_groups[site_key].append({
                    'idx': site_idx,
                    'sub': sub_label,
                    'patch': site_info['PATCH'],
                    'tag': site_info['TAG']
                })
        else:
            # Fallback: treat each column as a separate site
            for site_idx in range(n_sites):
                site_key = f"Site_{site_idx+1}"
                site_groups[site_key] = [{'idx': site_idx, 'sub': '1', 'patch': 'UNK', 'tag': 'NONE'}]
        
        # Create one plot per site with all subs
        for site_key, subs in site_groups.items():
            plt.figure(figsize=(12, 8))
            
            # Use different colors for each sub
            colors = COLORS[:len(subs)]
            
            # Plot each subspecies
            for sub_idx, sub_info in enumerate(subs):
                site_lambdas = replica_lambdas[:, sub_info['idx']]
                sub_label = f"Sub {sub_info['sub']} ({sub_info['tag']})"
                
                plt.plot(replica_phs, site_lambdas, 'o-', color=colors[sub_idx], 
                        linewidth=2, markersize=6, label=sub_label)
                
                # Add theoretical curve for this subspecies using block weights
                if len(replica_phs) >= 3:
                    ph_theory = np.linspace(replica_phs.min()-1, replica_phs.max()+1, 200)
                    tag = sub_info['tag']
                    pKa_val = effective_pH  # Use effective_pH instead of 7.0
                    
                    # Extract pKa from tag
                    if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                        try:
                            pKa_val = float(tag[4:]) if len(tag) > 4 else effective_pH
                        except:
                            pKa_val = effective_pH
                    
                    
                    # For theoretical curves, we need to consider this subspecies in context
                    # of all subspecies for the same site to get proper block weighting
                    site_microstates = []
                    for other_sub in subs:
                        other_tag = other_sub['tag']
                        other_pKa = effective_pH  # Use effective_pH instead of 7.0
                        if other_tag.upper().startswith('UPOS') or other_tag.upper().startswith('UNEG'):
                            try:
                                other_pKa = float(other_tag[4:]) if len(other_tag) > 4 else effective_pH
                            except:
                                other_pKa = effective_pH
                        site_microstates.append({
                            "pKa": other_pKa, 
                            "tag": other_tag, 
                            "site": other_sub.get('site', '1'),
                            "is_current": other_sub['idx'] == sub_info['idx']
                        })
                    
                    # Generate proper theoretical curve using individual microstate populations
                    # Each subspecies gets its own weighted Henderson-Hasselbalch curve
                    has_upos = any(m["tag"].upper().startswith("UPOS") for m in site_microstates)
                    has_uneg = any(m["tag"].upper().startswith("UNEG") for m in site_microstates)
                    has_none = any(m["tag"].upper().startswith("NONE") for m in site_microstates)
                    
                    current_tag = sub_info['tag'].upper()
                    current_pKa = pKa_val
                    lambda_theory = None
                    
                    if has_upos and has_none and not has_uneg:
                        # Case: UPOS + NONE system (e.g., lysine)
                        # Get all UPOS pKa values and calculate weights
                        upos_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UPOS")]
                        upos_weights = [10**(-pKa) for pKa in upos_pkas]
                        total_upos_weight = sum(upos_weights)
                        
                        # Calculate macroscopic pKa for the transition
                        pKa_macro = -np.log10(total_upos_weight)
                        
                        if current_tag.startswith("UPOS"):
                            # Individual UPOS state weight relative to all UPOS states
                            individual_weight = 10**(-current_pKa) / total_upos_weight
                            # Total UPOS population decreases with pH
                            total_upos_pop = 1.0 / (1.0 + 10**(pKa_macro - ph_theory))
                            # This UPOS state gets its weighted share
                            lambda_theory = total_upos_pop * individual_weight
                            
                        elif current_tag.startswith("NONE"):
                            # NONE population increases with pH (complementary to UPOS)
                            lambda_theory = 1.0 / (1.0 + 10**(ph_theory - pKa_macro))

                    elif has_uneg and has_none and not has_upos:
                        # Case: UNEG + NONE system (e.g., aspartate)
                        # Get all UNEG pKa values and calculate weights
                        uneg_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UNEG")]
                        uneg_weights = [10**(-pKa) for pKa in uneg_pkas]  # Using same weight scheme
                        total_uneg_weight = sum(uneg_weights)
                        
                        # Calculate macroscopic pKa for the transition  
                        pKa_macro = np.log10(sum(10**pKa for pKa in uneg_pkas))
                        
                        if current_tag.startswith("UNEG"):
                            # Individual UNEG state weight relative to all UNEG states
                            individual_weight = 10**(-current_pKa) / total_uneg_weight
                            # Total UNEG population increases with pH
                            total_uneg_pop = 1.0 / (1.0 + 10**(ph_theory - pKa_macro))
                            # This UNEG state gets its weighted share
                            lambda_theory = total_uneg_pop * individual_weight
                            
                        elif current_tag.startswith("NONE"):
                            # NONE population decreases with pH (complementary to UNEG)
                            lambda_theory = 1.0 / (1.0 + 10**( pKa_macro - ph_theory))
                    
                    elif has_upos and has_uneg and has_none:
                        # Case: 3-state system (e.g., histidine)
                        upos_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UPOS")]
                        uneg_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UNEG")]
                        pKa_pos = -np.log10(np.sum(10**(-np.array(upos_pkas))))
                        pKa_neg = np.log10(np.sum(10**(np.array(uneg_pkas))))
                        
                        # Calculate the three macroscopic populations
                        denominator = 1.0 + 10**(ph_theory - pKa_neg) + 10**(ph_theory - pKa_pos)
                        none_pop = 1.0 / denominator
                        upos_total = none_pop * 10**(pKa_pos - ph_theory)
                        uneg_total = none_pop * 10**(ph_theory - pKa_neg)
                        
                        if current_tag.startswith("UPOS"):
                            # Individual UPOS state weight relative to all UPOS states
                            upos_weights = [10**(-pKa) for pKa in upos_pkas]
                            total_upos_weight = sum(upos_weights)
                            individual_weight = 10**(-current_pKa) / total_upos_weight
                            lambda_theory = upos_total * individual_weight
                            
                        elif current_tag.startswith("UNEG"):
                            # Individual UNEG state weight relative to all UNEG states  
                            uneg_weights = [10**(-pKa) for pKa in uneg_pkas]
                            total_uneg_weight = sum(uneg_weights)
                            individual_weight = 10**(-current_pKa) / total_uneg_weight
                            lambda_theory = uneg_total * individual_weight
                            
                        elif current_tag.startswith("NONE"):
                            # For NONE states in 3-state system, could distribute based on pKa
                            # For now, use equal sharing among NONE states
                            none_count = sum(1 for m in site_microstates if m["tag"].upper().startswith("NONE"))
                            lambda_theory = none_pop / none_count
                    
                    elif has_upos and not has_uneg and not has_none:
                        # Case: Only UPOS states - distribute by weight, flat with pH
                        upos_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UPOS")]
                        upos_weights = [10**(-pKa) for pKa in upos_pkas]
                        total_upos_weight = sum(upos_weights)
                        individual_weight = 10**(-current_pKa) / total_upos_weight
                        lambda_theory = individual_weight * np.ones_like(ph_theory)
                        
                    elif has_uneg and not has_upos and not has_none:
                        # Case: Only UNEG states - distribute by weight, flat with pH
                        uneg_pkas = [m["pKa"] for m in site_microstates if m["tag"].upper().startswith("UNEG")]
                        uneg_weights = [10**(-pKa) for pKa in uneg_pkas]
                        total_uneg_weight = sum(uneg_weights)
                        individual_weight = 10**(-current_pKa) / total_uneg_weight
                        lambda_theory = individual_weight * np.ones_like(ph_theory)
                        
                    else:
                        # Fallback: single state or unknown case
                        lambda_theory = 0.5 * np.ones_like(ph_theory)
                    
                    if lambda_theory is not None:
                        plt.plot(ph_theory, lambda_theory, ':', color=colors[sub_idx], 
                                linewidth=2, alpha=0.8, 
                                label=f'Sub {sub_info["sub"]} theory (pKa={pKa_val:.1f})')
                
                # Adaptive individual fitting for each subspecies
                if len(replica_phs) >= 3:
                    try:
                        from scipy.optimize import curve_fit
                        
                        # Detect sigmoid direction from data trend for this subspecies
                        correlation = np.corrcoef(replica_phs, site_lambdas)[0, 1]
                        is_rising = correlation > 0  # Positive correlation = rising with pH
                        print(f"Sub {sub_info['sub']} direction: correlation={correlation:.3f}, is_rising={is_rising}")
                        
                        def adaptive_henderson_hasselbalch(pH, pKa, n=1):
                            """Henderson-Hasselbalch with adaptive direction"""
                            if is_rising:
                                # Rising sigmoid (UNEG-like): s=-1 form
                                return 1.0 / (1.0 + 10**(n * (pKa - pH)))
                            else:
                                # Falling sigmoid (UPOS-like): s=+1 form
                                return 1.0 / (1.0 + 10**(n * (pH - pKa)))
                        
                        pKa_guess = np.mean(replica_phs)
                        
                        # Constrain n to be positive
                        bounds = ([replica_phs.min()-2, 0.1], [replica_phs.max()+2, 5])
                        
                        popt, pcov = curve_fit(adaptive_henderson_hasselbalch, replica_phs, site_lambdas, 
                                             p0=[pKa_guess, 1], bounds=bounds)
                        
                        fitted_pKa, fitted_n = popt
                        ph_smooth = np.linspace(replica_phs.min()-1, replica_phs.max()+1, 200)
                        lambda_smooth = adaptive_henderson_hasselbalch(ph_smooth, fitted_pKa, fitted_n)
                        
                        direction_label = "↗" if is_rising else "↘"
                        plt.plot(ph_smooth, lambda_smooth, '--', color=colors[sub_idx], linewidth=1.5, 
                                alpha=0.8, label=f'Sub {sub_info["sub"]} fit {direction_label}: pKa={fitted_pKa:.2f}')
                        
                    except:
                        pass  # Skip fitting if it fails
            
            plt.xlabel('pH', fontsize=14)
            plt.ylabel('Relative Fraction (above cutoff)', fontsize=14)
            plt.title(f'{site_key} - Henderson-Hasselbalch Curves (Run {run})', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.ylim(-0.05, 1.05)
            
            # Save plot
            plt.tight_layout()
            safe_filename = site_key.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{plots_dir}/hh_curve_run{run}_{safe_filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save the data to text files
    data_file = f'{plots_dir}/hh_data_run{run}.txt'
    with open(data_file, 'w') as f:
        f.write("# Henderson-Hasselbalch curve data\n")
        f.write(f"# Generated from run {run}\n")
        f.write("# Method: Relative fraction of values above cutoff (0.985)\n")
        f.write("# pH\tRelative_Fraction(s)\n")
        
        for i, ph in enumerate(replica_phs):
            if replica_lambdas.ndim == 1:
                f.write(f"{ph:.6f}\t{replica_lambdas[i]:.6f}\n")
            else:
                lambda_str = '\t'.join([f"{replica_lambdas[i, j]:.6f}" for j in range(replica_lambdas.shape[1])])
                f.write(f"{ph:.6f}\t{lambda_str}\n")
    
    print(f"Henderson-Hasselbalch curve saved to {plots_dir}/hh_curve_run{run}.png")
    print(f"Data saved to {data_file}")

# --- helper --------------------------------------------------------------
def good_overlap(fracs, spread_tol):
    """
    Return True if every pair of columns differs by < spread_tol.
    Much faster than outer subtraction on big arrays.
    """
    return np.max(fracs) - np.min(fracs) < spread_tol

def enough_samples(lambda_data, min_hits=200):
    """True if every column has at least min_hits samples above threshold."""
    hits = np.sum(lambda_data, axis=0)   # boolean mask already applied
    return np.all(hits >= min_hits)

def check_pka_convergence(lambda_data, l_files, pka_tolerance=1.5):
    """
    Check if all pKa values are within ±pka_tolerance of the fitted pKa.
    Returns True if convergence is achieved (should move to phase 3).
    """
    global patch_info, alf_info, pH, site_pH0, delta_pKa
    
    # Guard clauses
    if patch_info is None or len(patch_info) == 0:
        print("No patch_info available for pKa convergence check")
        return False
    
    if lambda_data is None or len(l_files) < 3:
        print("Insufficient lambda data for pKa convergence check")
        return False
    
    if pH is None:
        print("pH not set - cannot perform pKa convergence check")
        return False
    
    try:
        # Extract replica information and pH values (same logic as in generate_hh_curve)
        replica_phs = []
        replica_lambdas = []
        
        # Guard against None alf_info
        if alf_info is None or 'ncentral' not in alf_info:
            ncentral = 0
        else:
            ncentral = alf_info['ncentral']
        
        # Get effective_pH
        if len(set(site_pH0.values())) == 1:
            effective_pH = list(site_pH0.values())[0]
        elif len(set(site_pH0.values())) > 1:
            effective_pH = 0.0
        else:
            effective_pH = pH
        
        # Process each replica's combined lambda data
        cutoff_threshold = 0.8
        
        # Parse replica information from l_files and organize data by replica
        replica_data_dict = {}
        
        # Organize combined lambda_data by replica (same logic as in generate_hh_curve)
        for file in sorted(l_files):
            try:
                replica_idx = int(file.split('.')[2])
            except:
                continue
            
            # Reload and combine all repeat files for this replica
            replica_files = []
            for fname in sorted(f for f in os.listdir("data") if f.startswith("Lambda")):
                try:
                    parts = fname.split('.')
                    if len(parts) >= 4 and int(parts[2]) == replica_idx:
                        replica_files.append(fname)
                except (ValueError, IndexError):
                    continue
            
            # Combine all repeats for this replica
            replica_combined_data = None
            for fname in sorted(replica_files):
                l = np.loadtxt(f'data/{fname}')
                if replica_combined_data is None:
                    replica_combined_data = l
                else:
                    replica_combined_data = np.vstack([replica_combined_data, l])
            
            if replica_combined_data is not None:
                replica_data_dict[replica_idx] = replica_combined_data
        
        print(f"pKa Convergence: Processing {len(replica_data_dict)} replicas")
        
        # Process each replica's combined data
        for replica_idx in sorted(replica_data_dict.keys()):
            replica_lambda = replica_data_dict[replica_idx]
            
            # Calculate replica-specific pH
            replica_shift = delta_pKa * (replica_idx - ncentral)
            replica_ph = effective_pH + replica_shift
            replica_phs.append(replica_ph)
            
            # Load lambda data for this replica
            # replica_lambda = np.loadtxt(f'data/{file}')  # Already loaded above
            
            # Calculate relative fraction (same logic as generate_hh_curve)
            if replica_lambda.ndim == 1:
                above_cutoff_count = np.sum(replica_lambda > cutoff_threshold)
                total_above_cutoff = above_cutoff_count
                if total_above_cutoff > 0:
                    fraction_lambda = above_cutoff_count / total_above_cutoff
                else:
                    fraction_lambda = 0.0
            else:
                above_cutoff_counts = np.sum(replica_lambda > cutoff_threshold, axis=0)
                total_above_cutoff = np.sum(above_cutoff_counts)
                if total_above_cutoff > 0:
                    fraction_lambda = above_cutoff_counts / total_above_cutoff
                else:
                    fraction_lambda = np.zeros(replica_lambda.shape[1])
            
            replica_lambdas.append(fraction_lambda)
        
        # Convert to numpy arrays
        replica_phs = np.array(replica_phs)
        replica_lambdas = np.array(replica_lambdas)
        
        # Filter out non-physical values
        if replica_lambdas.ndim == 1:
            physical_mask = np.isfinite(replica_lambdas)
        else:
            physical_mask = np.all(np.isfinite(replica_lambdas), axis=1)
        
        replica_phs = replica_phs[physical_mask]
        replica_lambdas = replica_lambdas[physical_mask]
        
        if len(replica_phs) < 3:
            print("Insufficient physical data points for pKa convergence check")
            return False
        
        # Create microstates from patch_info for fitting
        microstates = []
        for _, row in patch_info.iterrows():
            pKa_val = effective_pH  # Use effective_pH instead of 7.0
            tag = row['TAG']
            if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                try:
                    pKa_val = float(tag.split()[1])
                except:
                    pKa_val = effective_pH
            
            microstates.append({
                "pKa": pKa_val,
                "tag": tag,
                "site": row['site']
            })
        
        # For single site analysis, fit Henderson-Hasselbalch curve
        if replica_lambdas.ndim == 1:
            fitted_params = fit_lambda_data_to_hh_formulas(replica_phs, replica_lambdas, microstates)
            
            if fitted_params and 'pKa_eff' in fitted_params:
                fitted_pKa = fitted_params['pKa_eff']
                print(f"Fitted pKa: {fitted_pKa:.2f}")
                
                # Extract all pKa values from patch_info
                theoretical_pkas = []
                for _, row in patch_info.iterrows():
                    tag = row['TAG']
                    if tag.upper().startswith('UPOS') or tag.upper().startswith('UNEG'):
                        try:
                            pKa_val = float(tag.split()[1])
                            theoretical_pkas.append(pKa_val)
                        except:
                            pass
                
                if not theoretical_pkas:
                    print("No theoretical pKa values found in patch_info")
                    return False
                
                # Check convergence: all pKa values within ±pka_tolerance of fitted pKa
                pka_diffs = [abs(pKa - fitted_pKa) for pKa in theoretical_pkas]
                max_diff = max(pka_diffs)
                all_converged = all(diff <= pka_tolerance for diff in pka_diffs)
                
                print(f"pKa convergence check:")
                print(f"  Theoretical pKas: {theoretical_pkas}")
                print(f"  Fitted pKa: {fitted_pKa:.2f}")
                print(f"  Max difference: {max_diff:.2f}")
                print(f"  Tolerance: ±{pka_tolerance}")
                print(f"  Converged: {all_converged}")
                
                return all_converged
            else:
                print("Failed to fit Henderson-Hasselbalch curve")
                return False
        else:
            print("Multi-site pKa convergence check not yet implemented")
            return False
    
    except Exception as e:
        print(f"Error in pKa convergence check: {e}")
        return False

def alf_analysis(run,repeats=0):
    global phase_runs, alf_info, phase, fix_bias, alphabet, homedir
    
    # Guard clause to ensure variables are initialized
    if alf_info is None:
        raise ValueError("alf_info is not initialized. Please ensure alf_initialize() has been called.")
    
    home_path = os.getcwd()
    try:
        os.chdir(f'{input_folder}')
        print(os.getcwd())
        im5 = max(run-5,1)
        # im5 = max(run-2,1)
        N = run - im5 + 1
        print(f'Analysis{run} started')
        if not os.path.exists(f'analysis{run}'):
            os.mkdir(f'analysis{run}')
        shutil.copy(f'analysis{run-1}/b_sum.dat',f'analysis{run}/b_prev.dat')
        shutil.copy(f'analysis{run-1}/c_sum.dat',f'analysis{run}/c_prev.dat')
        shutil.copy(f'analysis{run-1}/x_sum.dat',f'analysis{run}/x_prev.dat')
        shutil.copy(f'analysis{run-1}/s_sum.dat',f'analysis{run}/s_prev.dat')
        # copy nbshift folder into analysis{run} folder
        shutil.copytree(f'nbshift',f'analysis{run}/nbshift')
        # Weighted addition of the bias
        # for site in patch_info['site'].unique():
        #     tags = patch_info.loc[patch_info['site'] == site]['TAG']
            # tags can be 'NONE', 'UPOS float' or 'UNEG float'
            # if it it is UPOS, it's positive value of pKa, if it is UNEG, it's negative value of pKa
            # if it is NONE, it's reference state
            

        if not os.path.exists(f'analysis{run}/G_imp'):
            G_imp_dir = os.path.join(os.getcwd(),'G_imp')
            target_path = os.path.join(os.getcwd(),f'analysis{run}/G_imp')
            try:
                os.symlink(G_imp_dir,target_path)
            except FileExistsError:
                pass

        os.chdir(path=f'analysis{run}')
        alf_info['nreps'] = phase_runs
        os.makedirs('data', exist_ok=True)
        for j in range(0,phase_runs):
            for k in range(repeats):
                if phase_runs == 1:
                    letter = ''
                else:
                    letter = '_' + alphabet[j]
                if phase in [1,2]:
                    fnmsin = [f'../run{run}/res/{name}_flat.{k}.{j}.lmd']
                else:
                    fnmsin = [f'../run{run}/res/{name}_prod.{k}.{j}.lmd']
                fnmout = f'data/Lambda.{k}.{j}.dat'

                # Check if input file exists before processing
                if not os.path.isfile(fnmsin[0]):
                    print(f"Warning: Missing input file for GetLambda: {fnmsin[0]}")
                    print(f"Skipping replica {j} for run {run}")
                    continue  # Skip this replica
            
                alf.GetLambda.GetLambda(alf_info, fnmout, fnmsin)
        # check how many previous folders have same structure
        alf.GetEnergy(alf_info, im5, run)


        # load all Lambda values - combine repeats for each replica
        threshold  = 0.8               # adaptive?  maybe 0.9*lambda_data.max()
        spread12   = 0.3               # tolerance Phase 1 → 2
        spread23   = 0.1               # tighter  Phase 2 → 3
        min_hits   = 1000                # safeguard on tiny trajectories

        lambda_data = None
        l_files = []
        
        # First, identify all unique replicas and collect all repeat files for each replica
        replica_files = {}
        for fname in sorted(f for f in os.listdir("data") if f.startswith("Lambda")):
            try:
                # Parse filename: Lambda.{repeat}.{replica}.dat
                parts = fname.split('.')
                if len(parts) >= 4:
                    repeat_idx = int(parts[1])
                    replica_idx = int(parts[2])
                    
                    if replica_idx not in replica_files:
                        replica_files[replica_idx] = []
                    replica_files[replica_idx].append(fname)
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed lambda filename: {fname}")
                continue
        
        # For each replica, combine all repeats into a single dataset
        for replica_idx in sorted(replica_files.keys()):
            repeat_files = sorted(replica_files[replica_idx])
            
            replica_combined_data = None
            for fname in repeat_files:
                l = np.loadtxt(f"data/{fname}")
                if replica_combined_data is None:
                    replica_combined_data = l
                else:
                    # Concatenate along rows (combine time series from different repeats)
                    replica_combined_data = np.vstack([replica_combined_data, l])
            
            # Store the combined data for this replica
            if lambda_data is None:
                lambda_data = replica_combined_data
            else:
                # Concatenate all replica data
                lambda_data = np.vstack([lambda_data, replica_combined_data])
            
            # Keep track of which files represent each replica (use first repeat file as representative)
            l_files.append(repeat_files[0])
        
        print(f"Loaded lambda data from {len(replica_files)} replicas with {sum(len(files) for files in replica_files.values())} total repeat files")
        print(f"Expected: {phase_runs} replicas × {max(1, repeats)} repeats = {phase_runs * max(1, repeats)} files")
        for replica_idx, files in replica_files.items():
            print(f"  Replica {replica_idx}: {len(files)} files - {files}")
        
        if lambda_data is not None:
            print(f"Final combined lambda_data shape: {lambda_data.shape}")
        else:
            print("Warning: No lambda data was loaded!")

        if lambda_data is None:
            print(f"Run {run}: WARNING – no Lambda files.  Staying in phase {phase}.")
        else:
            mask       = lambda_data > threshold
            col_fracs  = mask.mean(axis=0)

            # gate 1 → 2
            if phase == 1 and good_overlap(col_fracs, spread12) \
                        and enough_samples(mask, min_hits) \
                        and check_pka_convergence(lambda_data, l_files, pka_tolerance=1.5):
                phase = 2
                print(f"{time.asctime()}: Run {run} -> phase 2 "
                    f"(spread={col_fracs.ptp():.3f}, min_hits={mask.sum(axis=0).min()})")

            # gate 2 → 3
            elif phase == 2 and good_overlap(col_fracs, spread23) \
                            and enough_samples(mask, min_hits) \
                            and check_pka_convergence(lambda_data, l_files, pka_tolerance=0.3):
                phase = 3
                print(f"{time.asctime()}: Run {run} -> phase 3 "
                    f"(spread={col_fracs.ptp():.3f}, min_hits={mask.sum(axis=0).min()})")

            else:
                # optional: tell yourself why you *didn’t* promote
                print(f"{time.asctime()}: Run {run}: still in phase {phase}; "
                    f"spread={col_fracs.ptp():.3f}, min_hits={mask.sum(axis=0).min()})")
        np.savetxt('phase.dat', np.array([phase]), fmt='%d')
        # fpout = open('output.dat','w')
        # fperr = open('error.dat','w')
        # check how many variables alf.RunWham takes, it should be 3 or 4
        N_sims = len(os.listdir('Energy'))
        # if alf.RunWham.__code__.co_argcount == 3 and fix_bias == False:
        #     alf.RunWham(nf=N*alf_info['nreps'], 0, 0)
        # elif alf.RunWham.__code__.co_argcount == 3 and fix_bias == True:
        #     alf.RunWham(N*alf_info['nreps'], 1, 0)
        # if alf.RunWham.__code__.co_argcount == 4 and fix_bias == False:
        #     alf.RunWham(N_sims, alf_info['temp'], 0, 0)
        # elif alf.RunWham.__code__.co_argcount == 4 and fix_bias == True:
        #     alf.RunWham(N*alf_info['nreps'], alf_info['temp'], 1, 0)
        # else:
            # raise ValueError('RunWham function takes wrong number of arguments')
        # Apply retry loop for ALF analysis
        cut_params = {}       
        for attempt in range(3):
            
            try:
                fpout = open(f'output_{attempt}.dat','w')
                fperr = open(f'error_{attempt}.dat','w')
                cmd = (
                    "GPU_ID=$((SLURM_LOCALID % SLURM_GPUS_ON_NODE)); "
                    "export CUDA_VISIBLE_DEVICES=$GPU_ID; "
                    f"python -c 'import sys; sys.path.insert(0, \"/home/stanislc/software/ALF\"); import alf; alf.RunWham2({N_sims}, {alf_info['temp']}, 0, 0)'"
                )
                subprocess.call(cmd, shell=True, stdout=fpout, stderr=fperr)
                fpout.close()
                fperr.close()
                # alf.RunWham({N_sims}, {alf_info['temp']}, 0, 0)
                cutb = 2.0
                if phase == 1:
                    if run < 5: 
                        cutb = 5.0
                        cut_params = dict(cutc=3*cutb)
                    elif run < 30: 
                        cutb = 2.5
                        cut_params = dict(cutc=4*cutb)
                    else: 
                        cutb = 1.0
                    cut_params = dict(cutc=4*cutb)
                elif phase == 2: 
                    cutb = 0.5
                    cut_params = dict(cutc=2*cutb)
                elif phase == 3: 
                    cutb = 0.1
                    cut_params = dict(cutc=1*cutb)
                    
                if no_x_bias:
                    cut_params.update(dict(cutx=0.0))
                if no_s_bias:
                    cut_params.update(dict(cuts=0.0))
                if run < 10:
                    micropka_protection = 0.0
                else:
                    micropka_protection = 0.3
                try:
                    alf.GetFreeEnergy5_micropka(alf_info, ms=0, msprof=0, cutb=cutb, **cut_params, micropka_protection=micropka_protection)
                except Exception as e:
                    print(f"GetFreeEnergy5_micropka failed: {e}, trying GetFreeEnergy5_xs")
                    try:
                        alf.GetFreeEnergy5_xs(alf_info, ms=0, msprof=0, cutb=cutb, **cut_params)
                    except Exception as e2:
                        print(f'GetFreeEnergy5_xs failed: {e2}')
                        continue  # Optionally continue to next attempt in the loop

                alf.SetVars(alf_info, run+1)
                # Load summary files and check for all-zero, NaN, or Inf values
                def is_invalid(arr):
                    return np.all(arr == 0) or np.any(np.isnan(arr)) or np.any(np.isinf(arr))

                try:
                    b = np.loadtxt('b.dat')
                    c = np.loadtxt('c.dat')
                    if any(is_invalid(arr) for arr in [b, c]):
                        print(f'Invalid values detected in b/c files, rerunning WHAM... (attempt {attempt + 1}/3)')
                        try:
                            os.remove(f'../variables{run+1}.inp')
                        except OSError:
                            pass
                        continue

                    b_sum = np.loadtxt('b_sum.dat')
                    c_sum = np.loadtxt('c_sum.dat')
                    vars_file = f'../variables{run+1}.inp'
                    if any(is_invalid(arr) for arr in [b_sum, c_sum]):
                        print(f'Invalid values detected in summary files, rerunning WHAM... (attempt {attempt + 1}/3)')
                        try:
                            os.remove(vars_file)
                        except OSError:
                            pass
                        continue
                    break
                except Exception as e:
                    print(f'Error during WHAM execution: {e}, rerunning... (attempt {attempt + 1}/3)')
                    continue
            except Exception as e:
                print(f'Error during WHAM execution: {e}, rerunning... (attempt {attempt + 1}/3)')
                continue
        else:
            # If all 3 attempts failed, raise an exception
            raise RuntimeError("WHAM execution failed after 3 attempts")

        # Generate Henderson-Hasselbalch curve if we have enough replicas and pH is specified
        # Calculate delta_pKa based on phase to determine if HH curve should be generated
        if phase == 1:
            delta_pKa_analysis = 1.0
        elif phase == 2:
            delta_pKa_analysis = 0.5
        else:
            delta_pKa_analysis = 0.25
        
        if lambda_data is not None and nreps > 3 and pH is not None and delta_pKa_analysis != 0:
            generate_hh_curve(run, lambda_data, l_files)
        elif lambda_data is not None and nreps > 3 and pH is not None and delta_pKa_analysis == 0:
            print(f"Skipping HH curve generation (delta_pKa={delta_pKa_analysis})")
        
        
        alf.SetVars(alf_info, run+1)
        try:
            shutil.rmtree('Energy')
            shutil.rmtree('Lambda')
        except FileNotFoundError:
            pass  # Ignore if the directory doesn't exist

    except:
        print(f'Analysis{run} failed')
        raise Exception(f'Analysis{run} failed')
    finally:
        os.chdir(home_path)
        sys.stdout.flush()
        # if run > 5:
        #     shutil.rmtree(f'{input_folder}/analysis{run-6}')
        

gpuid = get_gpu_id()
homedir = os.getcwd()
comm.barrier()
alf_initialize()
comm.barrier()
run()
sys.stdout.close()