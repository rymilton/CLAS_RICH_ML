#!/bin/bash
#SBATCH --job-name=RICH_generation_0_9999
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=production
#SBATCH --account=clas12
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-9999
#SBATCH --output=/volatile/clas12/rmilton/RICH_data_generation_12_2025/logs_rgaspring2018/slurm_%A_%a.out
#SBATCH --error=/volatile/clas12/rmilton/RICH_data_generation_12_2025/logs_rgaspring2018/slurm_%A_%a.err

i=${SLURM_ARRAY_TASK_ID}

source /etc/profile.d/modules.sh
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.3
module unload gemc
module load gemc/5.12
module unload coatjava
module load coatjava/11.1.0

LUND_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/lund_rgaspring2018/
HIPO_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/hipo_rgaspring2018/
RECO_DIRECTORY=/volatile/clas12/rmilton/RICH_data_generation_12_2025/reco_rgaspring2018/

printf -v ii "%04d" "$i"
lund_file=${LUND_DIRECTORY}/file_${ii}.dat
sim_file=${HIPO_DIRECTORY}/RICH_12_2025_gemc_${ii}.hipo
reco_file=${RECO_DIRECTORY}/RICH_12_2025_reco_${ii}.hipo
err_file=${LOG_DIRECTORY}/RICH_12_2025_${ii}.err

mkdir -p $HIPO_DIRECTORY
mkdir -p $RECO_DIRECTORY

$GEMC/bin/gemc \
-USE_GUI=0 \
-N=10000 \
/volatile/clas12/rmilton/RICH_data_generation_12_2025/rga_spring2018.gcard \
-INPUT_GEN_FILE="LUND, ${lund_file}" \
-OUTPUT="hipo, ${sim_file}"

recon-util -y \
/volatile/clas12/rmilton/RICH_data_generation_12_2025/rga_spring2018.yaml \
-i ${sim_file} \
-o ${reco_file}