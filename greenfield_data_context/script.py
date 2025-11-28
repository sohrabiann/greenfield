import os

def split_file(file_path, chunk_size_mb=24, output_dir=None):
    # Convert MB to bytes
    chunk_size = chunk_size_mb * 1024 * 1024
    
    # Default output directory: same as input file
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # "dcsrawdataextracts_sample"
    ext = os.path.splitext(file_path)[1]  # ".csv"
    
    # Open the input file
    with open(file_path, 'rb') as f:
        i = 1  # start numbering at 1
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            # Create output filename like dcsrawdataextracts_sample1.csv, dcsrawdataextracts_sample2.csv...
            out_file = os.path.join(output_dir, f"{base_name}{i}{ext}")
            with open(out_file, 'wb') as out:
                out.write(data)
            print(f"Created {out_file} ({len(data)} bytes)")
            i += 1
    
    print(f"\n✅ Splitting complete. Created {i-1} parts in {output_dir}")

# Example usage:
split_file(
    r"C:\Users\sohra\Downloads\Greenfield_Data_chat_project\greenfield_data_context\volumes\dcsrawdataextracts\dcsrawdataextracts_sample.csv",
    chunk_size_mb=24,
    output_dir=r"C:\Users\sohra\Downloads\SplitFiles"
)