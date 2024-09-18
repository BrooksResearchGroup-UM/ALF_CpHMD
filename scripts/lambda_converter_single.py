import os
import re
from itertools import product
from scipy.io import FortranFile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

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
    return resid_resn_headers, pH_map

def lambda_reader(lmd_fn, verbose=False):
    fp = FortranFile(lmd_fn, 'r')
    header = (fp.read_record([('hdr', np.string_, 4), ('icntrl', np.int32, 20)]))
    hdr = header['hdr'][0]
    icntrl = header['icntrl'][0][:]
    nfile = icntrl[0]     # Total number of dynamics steps in lambda file
    npriv = icntrl[1]     # Number of steps preceding this run
    nsavl = icntrl[2]     # Save frequency for lambda in file
    nblocks = icntrl[6]   # Total number of blocks = env + subsite blocks
    nsitemld = icntrl[10] # Total number of substitution sites (R-groups) in MSLD
    Lambdas = np.zeros((0, nblocks-1))

    if verbose:
        print('Header: ', hdr)
        print('Control parameters: ', icntrl)
        print('Reading lambda file...', lmd_fn)
        print('Total number of dynamics steps in lambda file: ', nfile)
        print('Number of steps preceding this run: ', npriv)
        print('Save frequency for lambda in file: ', nsavl)
        print('Total number of blocks: ', nblocks)
        print('Total number of substitution sites (R-groups) in MSLD: ', nsitemld)

    # Time step for dynamics in AKMA units
    delta4 = fp.read_record(dtype=np.float32) * 4.888821477E-2
    if verbose:
        print('Time step for dynamics: %.3f ps' % delta4)

    # Title in trajectory file
    title = fp.read_record(dtype=[('h', np.int32), ('title', 'S80')])['title'][0].decode()
    if verbose:
        print('Title in trajectory file: ', title)

    # Unused in current processing
    nbiasv = fp.read_record(dtype=np.int32)
    junk = fp.read_record(dtype=np.float32)
    if verbose:
        print('Unused data: ', nbiasv, junk)

    # Array (length nblocks) indicating which subsites belong to which R-substitution site
    isitemld = fp.read_record(dtype=np.int32)
    if verbose:
        print('Array indicating which subsites belong to which R-substitution site: ', isitemld)

    # Temperature used in lambda dynamics thermostat
    temp = fp.read_record(dtype=np.float32)
    temp = temp[0]
    if verbose:
        print('Temperature used in lambda dynamics thermostat: ', temp)

    # Unused data for this processing
    junk3 = fp.read_record(dtype=np.float32)
    if verbose:
        print('Unused data: ', junk3)

    Lambda = np.zeros((nfile, nblocks-1))
    if verbose:
        print('Assigning lambda values to array...')

    timestart = round((npriv*delta4).item(), 4)
    timestep = round(delta4.item() * nsavl, 4)
    if verbose:
        print('timestart: ', timestart, 'ps')
        print('timestep: ', timestep, 'ps')

    for i in range(nfile):
        # Read a line of lambda values
        try:
            lambdav = fp.read_record(dtype=np.float32)
            theta = fp.read_record(dtype=np.float32)
            Lambda[i, :] = lambdav[1:]
            # if verbose and i % 1000 == 0:
                # print(f'Processing step {i+1}/{nfile}')
        except:
            print('Lambda file ', lmd_fn, ' ended prematurely at step ', i+1, 'of ', nfile)
            # delete rows after i row in Lambda
            Lambda = np.delete(Lambda, np.s_[i:], axis=0)
            break

    Lambdas = np.concatenate((Lambdas, Lambda), axis=0, dtype=object)
    # Rounding Lambda values to 3 decimal places
    Lambdas = np.array(Lambdas, dtype=float).round(3)
    

    timestamps = np.array([round(timestart + i * timestep, 2) for i in range(Lambdas.shape[0])])
    Lambdas = np.concatenate((timestamps[:, None], Lambdas), axis=1)

    if verbose:
        print('Lambdas shape:', Lambdas.shape)
        print('First few rows of Lambdas:')
        print(Lambdas[:5])

    fp.close()
    metadata = {
        'Title': title,
        'Temperature': temp,
        'Time Step': timestep,
        'Time Start': timestart,
        'Time End': round(timestart + (nfile-1) * timestep, 2),
        'Save Frequency': nsavl,
        'nblocks': nblocks,
        'nsites': nsitemld,
        'nsubsites': isitemld,
        'Start Step': npriv,
        'Total Steps': nfile,
        'End Step': npriv + nfile -nsavl}
    metadata = {key: str(value) for key, value in metadata.items()}
        
        
    return Lambdas, metadata

def convert_single_lambda_file(log_dir, lambda_file, output_file,verbose):
    resid_resn_headers, pH_map = read_log_metadata(log_dir)
    lambdas, metadata = lambda_reader(lambda_file, verbose)
    time_column = lambdas[:, 0].astype(np.float32)
    lambda_columns = lambdas[:, 1:].astype(np.float16)
    print(pH_map)
    ph_index = os.path.basename(lambda_file).split("_")[-1].split(".")[0]
    metadata['pH'] = pH_map[ph_index]
    sim_folder = os.path.dirname(os.path.dirname(lambda_file)).split("/")[-1]
    name = os.path.dirname(os.path.dirname(os.path.dirname(lambda_file))).split("/")[-1]
    metadata['Simulation'] = sim_folder
    metadata['Name'] = name
    dataframe = pd.DataFrame(lambdas, columns=['time'] + resid_resn_headers)
    # dataframe.to_parquet(output_file)
    # Create a ParquetWriter
    table = pa.Table.from_pandas(dataframe)
    schema = table.schema.with_metadata(metadata)
    table = table.cast(schema)
    pq.write_table(table, output_file)
    if verbose:
        print(f"Parquet file saved to {output_file}")
        parquet_file = pq.ParquetFile(output_file)
        metadata = parquet_file.schema_arrow.metadata
        metadata = {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
        print("Metadata:")
        print(metadata)
    
    
def parser():
    parser = argparse.ArgumentParser(description="Convert lambda file to parquet")
    parser.add_argument("-i", "--lambda_file", type=str, required=True, help="Path to lambda file")
    parser.add_argument("-o", "--output_file", type=str, required=False, help="Path to output parquet file")
    parser.add_argument("-d", "--log_dir", type=str, required=False, help="Path to log directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    lambda_file = args.lambda_file
    output_file = args.output_file
    log_dir = args.log_dir
    if not os.path.exists(lambda_file):
        raise FileNotFoundError(f"Lambda file {lambda_file} not found")
    if not output_file:
        lambda_filename = os.path.basename(lambda_file).split(".")[0]
        output_file = os.path.join(os.path.dirname(lambda_file), lambda_filename+".parquet")
    if not log_dir:
        # log directory is the parent directory of directory containing lambda file
        log_dir = os.path.dirname(os.path.dirname(lambda_file))
        log_dir = os.path.join(log_dir, "logs")
    verbose = args.verbose
    if verbose:    
        print(f"Log directory: {log_dir}")
        print(f"Lambda file: {lambda_file}")
        print(f"Output file: {output_file}")
    return log_dir, lambda_file, output_file, verbose


log_dir, lambda_file, output_file, verbose = parser()
convert_single_lambda_file(log_dir, lambda_file, output_file, verbose)