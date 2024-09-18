import os
import re
from itertools import product
from scipy.io import FortranFile
import numpy as np
import pandas as pd
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
import argparse
import pyarrow.parquet as pq


def find_subfolders(folder):
    sub_folders = []
    for subfolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subfolder)):
            if os.path.isdir(os.path.join(folder, subfolder, 'lambdas')) \
                    and os.path.isdir(os.path.join(folder, subfolder, 'logs')):
                sub_folders.append(subfolder)
    sub_folders.sort(key=lambda x: int(x.split('_')[1]))
    return sub_folders


def read_log_metadata(log_dir):
    resid_resn_headers = []
    pH_map = {}
    log_files = [os.path.join(log_dir, f"prestart_{i:03d}.out") for i in [1, 2]]
    content = ""
    for log_file in log_files:
        with open(log_file, "r") as f:
            content = f.read()
            if re.search(r'CALL\s+(.*)end', content, re.IGNORECASE):
                break

    calls = re.findall(r'CALL\s+(.*)end', content, re.IGNORECASE)
    for call in calls:
        try:
            resid = re.search(r'resi\D+\s+(\w+)', call, re.IGNORECASE).group(1)
            resn = re.search(r'resn\D+\s+(\w+)', call, re.IGNORECASE).group(1)
            if f'{resid} {resn}' not in resid_resn_headers:
                resid_resn_headers.append(f"{resid} {resn}")
        except:
            pass
    # If it's not in CALL, it's in DEFINE
    if not resid_resn_headers:
        calls = re.findall(r'DEFINE\s+(.*)', content, re.IGNORECASE)
        for call in calls:
            try:
                resid = re.search(r'resi\D+\s+(\w+)+\s', call, re.IGNORECASE).group(1)
                resn = re.search(r'resn\D+\s+(\w+)+\s', call, re.IGNORECASE).group(1)
                if f'{resid} {resn}' not in resid_resn_headers:
                    resid_resn_headers.append(f"{resid} {resn}")
            except:
                pass
            
    ph_values = re.findall(r'PH value assigned =\s+([0-9.-]+)\s+NREP\s+(\d+)',
                           content, re.IGNORECASE)
    ph_values = list(dict.fromkeys(ph_values))
    for ph, nrep in ph_values:
        pH_map[str(int(nrep)-1).zfill(2)] = str(float(ph))
    
    # print(f"Residues: {resid_resn_headers}"
    #       f"\nPH values: {pH_map}")
    return resid_resn_headers, pH_map


def lambda_reader(lmd_fn):
    fp = FortranFile(lmd_fn, 'r')
    header = (fp.read_record(
        [('hdr', np.string_, 4), ('icntrl', np.int32, 20)]))
    hdr = header['hdr'][0]
    icntrl = header['icntrl'][0][:]
    nfile = icntrl[0]     # Total number of dynamics steps in lambda file
    npriv = icntrl[1]     # Number of steps preceding this run
    nsavl = icntrl[2]     # Save frequency for lambda in file
    nblocks = icntrl[6]   # Total number of blocks = env + subsite blocks
    # Total number of substitution sites (R-groups) in MSLD
    nsitemld = icntrl[10]
    Lambdas = np.zeros((0, nblocks-1))

    # print('Reading lambda file...', lmd_fn)
    # print('Total number of dynamics steps in lambda file: ', nfile)
    # print('Number of steps preceding this run: ', npriv)
    # print('Save frequency for lambda in file: ', nsavl)
    # print('Total number of blocks: ', nblocks)
    # print('Total number of substitution sites (R-groups) in MSLD: ', nsitemld)

    # Time step for dynamics in AKMA units
    delta4 = fp.read_record(dtype=np.float32) * 4.888821477E-2
    # print('Time step for dynamics: %.3f ps' % delta4)

    # Title in trajectoory file
    title = fp.read_record(dtype=[('h', np.int32),
                                  ('title', 'S80')])['title'][0].decode()
    # print('Title in trajectoory file: ', title)

    # ? Unused in current processing
    nbiasv = fp.read_record(dtype=np.int32)
    junk = fp.read_record(dtype=np.float32)

    # Array (length nblocks) indicating which subsites
    # below to which R-substitiution site
    isitemld = fp.read_record(dtype=np.int32)

    # print('Array (length nblocks) indicating which subsites
    # below to which R-substitiution site: ', isitemld)

    # Temeprature used in lambda dynamics thermostat
    temp = fp.read_record(dtype=np.float32)
    # print('Temeprature used in lambda dynamics thermostat: ', temp)

    # Unsed data for this processing
    junk3 = fp.read_record(dtype=np.float32)
    # print('Unused data: ', junk3)

    Lambda = np.zeros((nfile, nblocks-1))
    # print('Assigning lambda values to array...', Lambda)

    timestart = round((npriv*delta4).item(), 1)
    # print('timestart: ', timestart) # in ps
    timestep = round((nsavl*delta4).item(), 1)
    # print('timestep: ', timestep) # in ps

    for i in range(nfile):
        # Read a line of lambda values
        try:
            lambdav = fp.read_record(dtype=np.float32)
            theta = fp.read_record(dtype=np.float32)
            Lambda[i, :] = lambdav[1:]
        except:
            print('Lambda file ', lmd_fn,
                  ' ended prematurely at step ', i+1, 'of ', nfile)
            # delete rows after after i row in Lambda
            Lambda = np.delete(Lambda, np.s_[i:], axis=0)
            # Lambda = np.round(Lambda, 2)
            break
        # print('Lambda values for reading ', i+1,':\n' , Lambda[i,:])
        # print('Lambda size: ', Lambda.shape, 'Step:', i,'/',nfile)
    Lambdas = np.concatenate((Lambdas, Lambda), axis=0, dtype=object)
    # Rounding Lambda values to 2 decimal places
    Lambdas = np.array(Lambdas, dtype=float).round(3)

    # timestamps = np.arange(0, Lambdas.shape[0]*timestep, timestep, dtype=int)
    #! This have rounding errors
    # timestamps = np.arange(timestart, timestart+Lambdas.shape[0] * timestep, timestep)

    timestamps = np.array([round(timestart + i * timestep, 2)
                          for i in range(Lambdas.shape[0])])
    Lambdas = np.concatenate((timestamps[:, None], Lambdas), axis=1)

    # Format Lambdas: first column integer, others float with 2 digits after decimal point
    # print('Lambdas file: ', Lambdas)
    fp.close()

    return Lambdas

def lambda_reader_prq(lmd_fn):
    if lmd_fn.endswith('.lmd'):
        Lambdas = lambda_reader(lmd_fn)
    elif lmd_fn.endswith('.parquet'):
        Lambdas = pq.read_table(lmd_fn).to_pandas()
        Lambdas = Lambdas.values
    else:
        raise ValueError(f"Unknown file extension: {lmd_fn}")
        Lambdas = None
    return Lambdas

# Argument parser
parser = argparse.ArgumentParser(description='Process directory arguments.')
parser.add_argument('-i', '--input', nargs='+',
                    help='List of input directories', required=True)
args = parser.parse_args()

directories = args.input
# delete duplicates
directories = list(dict.fromkeys(directories))

# Converter to parquet
# Reading 8.4 sec vs 0.5 sec (hdf vs parquet). Size of parquet is 1/8 of hdf
all_pairs = [(folder, sub)
             for folder in directories for sub in find_subfolders(folder)]

for folder, sub_folder in all_pairs:
    resid_resn_headers, pH_map = read_log_metadata(
        os.path.join(folder, sub_folder, 'logs'))
    for pH_index, pH_value in pH_map.items():
        files = os.listdir(os.path.join(folder, sub_folder, 'lambdas'))
        files.sort(key=lambda x: int(x.split('_')[0].split('/')[-1]))
        base_filenames = set(f.rsplit('_', 1)[0] for f in files)
        selected_files = []
        for base_filename in base_filenames:
            parquet_file = os.path.join(folder, sub_folder, 'lambdas', f"{base_filename}_{pH_index}.parquet")
            lmd_file = os.path.join(folder, sub_folder, 'lambdas', f"{base_filename}_{pH_index}.lmd")
            if os.path.exists(parquet_file):
                selected_files.append(parquet_file)
            elif os.path.exists(lmd_file):
                selected_files.append(lmd_file)
                
        files = selected_files

        
        # read previously proccessed dataframe
        analysis_dir = os.path.join(os.path.dirname(folder),
                                    'analysis/data', os.path.basename(folder))
        output_file = os.path.join(
            analysis_dir, f"{sub_folder}_[{pH_value}].parquet")
        
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        if os.path.exists(output_file):
            dataframe = pd.read_parquet(output_file)
            # get highest timestamp
            last_timestamp = dataframe['Time (ps)'].max()
            last_full_file, remainder = divmod(last_timestamp, 1000)
            files = [f for f in files if
                     int(os.path.basename(f).split('_')[0]) > last_full_file]
            # each lmd file is 1000 ps
            if remainder != 0:
                
                # delete entries after last full file
                dataframe = dataframe[dataframe['Time (ps)']
                                      <= last_full_file * 1000]
        else:
            dataframe = pd.DataFrame(columns=['Time (ps)'] + resid_resn_headers)
        
        if len(files) == 0:
            continue
        
        # read files using lmd_reader in parallel
        print(f"Processing {len(files)} files for pH {pH_value} in {folder}/{sub_folder}")
        with ThreadPoolExecutor() as executor:
            # results = list(executor.map(lambda_reader, files))
            results = list(executor.map(lambda_reader_prq, files))
        
        # Concatenate results
        dataframe_new = pd.DataFrame(np.concatenate(results, axis=0), columns=[
                                     'Time (ps)'] + resid_resn_headers)
        # dataframe_new = pd.DataFrame(pd.concat(results, axis=0))
        
        
        # Merge with previous dataframe
        dataframe = pd.concat([dataframe, dataframe_new]
                              ).reset_index(drop=True)
        # dataframe = pd.concat([dataframe, results]).reset_index(drop=True)
        output_file = os.path.join(analysis_dir, f"{sub_folder}_[{pH_value}]")
        dataframe.to_parquet(output_file + '.parquet')
