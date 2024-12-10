#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_folder> [-o <output_folder>]"
    echo "  -i: Input file (required, can be a path)"
    echo "  -o: Output folder (optional, defaults to last part of input folder path)"
    exit 1
}

# Parse command-line options 
while getopts "i:o:" opt; do
    case $opt in
        i) input_folder="$OPTARG" ;;
        o) output_folder="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if input folder was provided
if [ -z "$input_folder" ]; then
    echo "Error: Input folder is required."
    usage
fi

# If output folder is not provided, set it to the last part of the input folder path
if [ -z "$output_folder" ]; then
    output_folder=$(basename "$input_folder")
fi


# Create the input folder if it does not exist
if [ ! -d "$output_folder" ]; then
    echo "Creating input folder: $output_folder"
    mkdir -p "$output_folder"
else
    echo "Input folder already exists: $output_folder"
fi  # Closing the if-else statement

# Create a temporary job submission script
cat << EOF > temp_job_script.sh
#! /bin/bash
#SBATCH --job-name=$output_folder
#SBATCH --output=$output_folder/sbatch_log.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA5500
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate charmm

nvidia-smi

# Check if the 'prep' subdirectory exists in the input folder
if [ ! -d "${output_folder}/prep" ]; then
    echo "'prep' directory does not exist. Running solvation and patching scripts..."

    # Running the solvation script
    python scripts/step2_solvate.py -i "${input_folder}" -o "${output_folder}" --pad 15 --temperature 298.15 --salt_concentration 0.1 --pos_ion POT

    # Running the patching script
    python scripts/step3_patching.py -i "${output_folder}" 

    echo "Solvation and patching completed."
else
    echo "'prep' directory exists. Skipping solvation and patching."
fi

# Execute the Python scripts
python scripts/step4_ALF.py -p 1 -s 1 -e 120  -i ${output_folder} 
#python scripts/step4_ALF.py -p 2 -s 40 -e 120 -hmr -i ${output_folder}
# python scripts/step4_ALF.py -p 3 -s 71 -e 80 -hmr -i ${output_folder}
python scripts/step5_bias_search.py -i ${output_folder}

# prepare system
python scripts/step8_block.py ${output_folder}
EOF

# Submit the job
sbatch temp_job_script.sh

# Optionally, remove the temporary script after submission
rm temp_job_script.sh
