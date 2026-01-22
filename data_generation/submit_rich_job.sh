#!/bin/bash
#SBATCH --job-name=RICH_generation_0_499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-499
#SBATCH --output=/volatile/clas12/rmilton/RICH_data_generation_12_2025/logs_3cmshift/slurm_%A_%a.out
#SBATCH --error=/volatile/clas12/rmilton/RICH_data_generation_12_2025/logs_3cmshift/slurm_%A_%a.err

i=${SLURM_ARRAY_TASK_ID}

source /etc/profile.d/modules.sh
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.3
module unload .scons_bm
module load gemc/5.12

LUND_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/lund_3cmshift/
HIPO_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/hipo_3cmshift/
RECO_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/reco_3cmshift/

printf -v ii "%04d" "$i"
lund_file=${LUND_DIRECTORY}/file_${ii}.dat
sim_file=${HIPO_DIRECTORY}/RICH_12_2025_gemc_${ii}.hipo
reco_file=${RECO_DIRECTORY}/RICH_12_2025_reco_${ii}.hipo
err_file=${LOG_DIRECTORY}/RICH_12_2025_${ii}.err

mkdir -p $HIPO_DIRECTORY
mkdir -p $RECO_DIRECTORY

$GEMC/bin/gemc \
-USE_GUI=0 \
-N=1000 \
/volatile/clas12/rmilton/RICH_data_generation_12_2025/rgc_summer2022.gcard \
-INPUT_GEN_FILE="LUND, ${lund_file}" \
-OUTPUT="hipo, ${sim_file}"

recon-util -y \
/volatile/clas12/rmilton/RICH_data_generation_12_2025/rgc_summer2022.yaml \
-i ${sim_file} \
-o ${reco_file}