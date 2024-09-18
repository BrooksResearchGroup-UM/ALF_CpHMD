import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import os

def reduce_parquet_with_compression(input_file, compression='SNAPPY', verbose=False):
    # Read the Parquet file
    table = pq.read_table(input_file)
    
    # Get the original schema and metadata
    original_schema = table.schema
    metadata = original_schema.metadata
    
    # Convert to pandas DataFrame
    df = table.to_pandas()
    
    # Keep 'Time (ps)' as float64, convert other columns to float32
    new_dtypes = {'Time (ps)': np.float64}
    new_dtypes.update({col: np.float32 for col in df.columns if col != 'Time (ps)'})
    
    for col, dtype in new_dtypes.items():
        df[col] = df[col].astype(dtype)
    
    # Create new schema
    new_fields = [
        pa.field('Time (ps)', pa.float64()),
        *[pa.field(name, pa.float32()) for name in df.columns if name != 'Time (ps)']
    ]
    new_schema = pa.schema(new_fields, metadata=metadata)
    
    # Convert to PyArrow Table
    arrays = [pa.array(df[col], type=new_schema.field(col).type) for col in df.columns]
    new_table = pa.Table.from_arrays(arrays, schema=new_schema)
    
    # Construct the output filename
    dir_name, file_name = os.path.split(input_file)
    base_name, ext = os.path.splitext(file_name)
    output_file = os.path.join(dir_name, f"{base_name}_reduced{ext}")
    
    # Write the new Parquet file with compression
    pq.write_table(new_table, output_file, compression=compression)
    
    if verbose:
        print(f"Reduced and compressed Parquet file saved to {output_file}")
        print("\nOriginal schema:")
        print(original_schema)
        print("\nNew schema:")
        print(new_schema)
        
        original_size = os.path.getsize(input_file)
        reduced_size = os.path.getsize(output_file)
        print(f"\nOriginal file size: {original_size:,} bytes")
        print(f"Reduced file size: {reduced_size:,} bytes")
        print(f"Size reduction: {(1 - reduced_size/original_size)*100:.2f}%")
    
    return output_file

# Example usage
input_file = "/home/stanislc/projects/1_cphmd/benchmark_proteins/analysis/data/1bvi/sim_1_[0.0].parquet"
reduced_file = reduce_parquet_with_compression(input_file, compression='SNAPPY', verbose=True)