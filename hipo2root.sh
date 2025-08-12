#!/bin/bash

data_directory="/work/clas12/pecar/RICH/gemcSim/clas12-rich-sim/clas12Tags-5.10/recoFiles/DIS/default/allOpticsNominal/"
output_directory="/volatile/clas12/rmilton/RICH_data/"

mkdir -p ${output_directory}

for file in "${data_directory}"/*; do
    echo "Converting ${file}"
    filename=$(basename "$file") 
    ./RICH_hipo2root ${data_directory} ${filename} ${output_directory}
done
