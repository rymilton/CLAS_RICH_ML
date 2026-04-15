#!/bin/bash

data_directory=/home/rmilton/work_dir/RICH_data_generation_12_2025/reco_rgaspring2018_root/
# Get total number of .root files in data_directory
total_files=$(ls ${data_directory}/*.root | wc -l)
batch_size=500
start_index=0
end_index=$total_files
# Loop through the files in batches and run data_processor.py on each batch
first_run=true
while [ $start_index -lt $total_files ]; do
    end_index=$((start_index + batch_size))
    # Print start and end indices for debugging
    if [ $end_index -gt $total_files ]; then
        end_index=$total_files
    fi

    # Make a string representation of the current batch
    batch_string=$(printf "%d-%d" $start_index $end_index)
    echo "Processing files from index $start_index to $end_index"
    python3 data_processor.py --start_file $start_index --end_file $end_index --name_suffix "${batch_string}"
    start_index=$((start_index + batch_size))
done
