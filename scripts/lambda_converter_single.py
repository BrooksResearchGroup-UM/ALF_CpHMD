# Copyright Stanislav Cherepanov

import os
import re
from scipy.io import FortranFile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

def read_log_metadata(log_dir):
    """
    Extract metadata from CALL commands in log files.
    """
    log_files = [os.path.join(log_dir, f"prestart_{i:03d}.out") for i in [1, 999]]
    content = ""

    # Read the log files
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Combine lines into a single string for easier processing
            content = "\n".join(lines)
            if "CALL" in content:
                break

    # Step 1: Extract CALL blocks
    resid_resn_headers = []
    for call in re.findall(r'CALL\s+\d+\s+SELE(?:ct)?\s+(.*?)\s+END', content, re.IGNORECASE | re.DOTALL):
        # Check if it's an alias
        alias_match = re.match(r'(\w+)$', call.strip(), re.IGNORECASE)
        if alias_match:
            alias = alias_match.group(1)
            if alias not in resid_resn_headers:
                resid_resn_headers.append(alias)
        else:
            # Otherwise, parse segid, resid, and resname
            segid = re.search(r'segid\s+(\w+)', call, re.IGNORECASE)
            resid = re.search(r'resid\s+(\d+)', call, re.IGNORECASE)
            resname = re.search(r'resname\s+(\w+)', call, re.IGNORECASE)

            segid = segid.group(1) if segid else ""
            resid = resid.group(1) if resid else ""
            resname = resname.group(1) if resname else ""

            # Construct the header
            header = f"{segid} {resid} {resname}".strip()
            if header and header not in resid_resn_headers:
                resid_resn_headers.append(header)
     
    # Step 2: Extract pH values
    ph_map = {}           
    ph_values = re.findall(r'PH value assigned =\s+([0-9.-]+)\s+NREP\s+(\d+)',
                           content, re.IGNORECASE)
    ph_values = list(dict.fromkeys(ph_values))
    for ph, nrep in ph_values:
        ph_map[str(int(nrep)-1).zfill(2)] = str(float(ph))

    return resid_resn_headers, ph_map

def lambda_reader(lmd_fn, verbose=False):
    """
    Read a lambda dynamics file and extract lambda values with metadata.
    """
    with FortranFile(lmd_fn, 'r') as fp:
        # Read header and control parameters
        header = fp.read_record([('hdr', np.string_, 4), ('icntrl', np.int32, 20)])
        hdr = header['hdr'][0]
        icntrl = header['icntrl'][0]
        nfile, npriv, nsavl, nblocks, nsitemld = icntrl[0], icntrl[1], icntrl[2], icntrl[6], icntrl[10]

        # Verbose output for header and control parameters
        if verbose:
            print(f"Header: {hdr.decode().strip()}")
            print(f"Control parameters: {icntrl}")
            print(f"Total steps: {nfile}, Preceding steps: {npriv}, Save frequency: {nsavl}")
            print(f"Blocks: {nblocks}, Substitution sites: {nsitemld}")

        # Read timestep and convert to picoseconds
        delta4 = fp.read_record(dtype=np.float32)[0] * 4.888821477E-2
        if verbose:
            print(f"Timestep: {delta4:.3f} ps")

        # Read title and clean whitespace
        title = fp.read_record(dtype=[('h', np.int32), ('title', 'S80')])['title'][0].decode().strip()
        title = re.sub(r'\s+', ' ', title)
        if verbose:
            print(f"Title: {title}")

        # Skip unused data
        fp.read_record(dtype=np.int32)  # nbiasv
        fp.read_record(dtype=np.float32)  # junk
        isitemld = fp.read_record(dtype=np.int32)
        temp = fp.read_record(dtype=np.float32)[0]  # Temperature
        fp.read_record(dtype=np.float32)  # junk3

        # Verbose output for unused data
        if verbose:
            print(f"Substitution site mapping: {isitemld}")
            print(f"Temperature: {temp:.2f} K")

        # Preallocate lambda array
        Lambda = np.empty((nfile, nblocks - 1), dtype=np.float32)

        # Timestamps
        timestart = npriv * delta4
        timestep = delta4 * nsavl
        timestamps = np.arange(nfile, dtype=np.float32) * timestep + timestart

        # Read lambda values
        if verbose:
            print("Reading lambda values...")

        try:
            for i in range(nfile):
                lambdav = fp.read_record(dtype=np.float32)
                theta = fp.read_record(dtype=np.float32)
                Lambda[i] = lambdav[1:]
        except EOFError:
            Lambda = Lambda[:i]  # Trim incomplete rows
            timestamps = timestamps[:i]
            if verbose:
                print(f"File ended prematurely at step {i}/{nfile}")

        # Combine timestamps and Lambda values
        Lambdas = np.column_stack((timestamps, Lambda.round(3)))

        # Verbose output for Lambda
        if verbose:
            print(f"Lambdas shape: {Lambdas.shape}")

    # Metadata collection
    metadata = {
        "Title": title,
        "Temperature": f"{temp:.2f}",
        "Time Step": f"{timestep:.3f}",
        "Time Start": f"{timestart:.3f}",
        "Time End": f"{timestart + (nfile - 1) * timestep:.3f}",
        "Save Frequency": str(nsavl),
        "nblocks": str(nblocks),
        "nsites": str(nsitemld),
        "nsubsites": str(isitemld),
        "Start Step": str(npriv),
        "Total Steps": str(nfile),
        "End Step": str(npriv + nfile - nsavl),
    }

    return Lambdas, metadata
           
def convert_single_lambda_file(log_dir, lambda_file, output_file,verbose):
    # Attempt to read log metadata
    try:
        resid_resn_headers, pH_map = read_log_metadata(log_dir)
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to read log metadata. Using default headers. Error: {e}")
        resid_resn_headers, pH_map = [], {}
        
    # Read lambda file and metadata
    lambdas, metadata = lambda_reader(lambda_file, verbose)
    
    # Generate default headers if none were provided
    if not resid_resn_headers:
        nblocks = lambdas.shape[1] - 1  # Subtract 1 for the time column
        resid_resn_headers = [str(i) for i in range(2, nblocks + 2)]

    if pH_map:    
        # Assign pH value, default to 'Unknown' if not found
        ph_index = os.path.basename(lambda_file).split("_")[-1].split(".")[0]
        metadata["pH"] = pH_map.get(ph_index, "Unknown")
    
    # Add simulation and name metadata based on the lambda file path    
    abs_path = os.path.abspath(lambda_file)
    sim_folder = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
    name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(abs_path))))
    metadata['Simulation'] = sim_folder
    metadata['Name'] = name
    
    
    # Create a DataFrame and write to Parquet
    dataframe = pd.DataFrame(data=lambdas, columns=['time'] + resid_resn_headers)
    table = pa.Table.from_pandas(dataframe)
    schema = table.schema.with_metadata(metadata)
    table = table.cast(schema)
    pq.write_table(table, output_file)
    if verbose:
        print(f"Parquet file saved to {output_file}")
        parquet_file = pq.ParquetFile(output_file)
        metadata = parquet_file.schema_arrow.metadata
        metadata = {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
        print("Metadata in parquet file:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
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