#!/bin/bash

data_directory="/home/rmilton/work_dir/RICH_data_generation_12_2025/reco_rgaspring2018_hipo/"
output_directory="/home/rmilton/work_dir/RICH_data_generation_12_2025/reco_rgaspring2018_root/"
mkdir -p "${output_directory}"

count=0
for file in "${data_directory}"/*; do
    echo "Converting ${file}"
    filename=$(basename "$file")
    output_file="${output_directory}/${filename}.root"
    if [[ -f "$output_file" ]]; then
        echo "Skipping ${filename}: ${output_file} already exists."
        continue
    fi
    ./RICH_hipo2root "${data_directory}" "${filename}" "${output_directory}"

    ((count++))
done