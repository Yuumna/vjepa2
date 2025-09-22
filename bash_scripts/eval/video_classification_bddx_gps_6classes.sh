#!/bin/bash

#SBATCH --partition tflmb_gpu-rtx4090    # short: -p <partition_name>
#SBATCH --job-name bddx_gps_vjepa_w/o_aug     # short: -J <job name>

#SBATCH --output ./logs/%x-%A-%j.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error ./logs/%x-%A-%j.err    # STDERR  short: -e logs/%x-%A-job_name.out

#GET two nodes
# Define the amount of memory required per node
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59


if [[ $(hostname) == "dlc2gpu03" || $(hostname) == "dlc2gpu05" ]] ; then
  export NCCL_P2P_DISABLE=1
fi

cd /work/dlclarge2/aliy-vjepa/vjepa2

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
. ~/.bashrc # Adjust to your path of Miniconda installation
conda activate vjepa2

#move dataset to tmp using bash file in the current directory called dataset_to_tmp.sh
# Running the job
start=`date +%s`

python -m evals.main --fname configs/eval/vitl/bddx_gps.yaml  --devices cuda:0
end=`date +%s`
runtime=$((end-start))


echo Job execution complete
echo Runtime: $runtime