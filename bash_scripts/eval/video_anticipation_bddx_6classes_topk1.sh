#!/bin/bash

#SBATCH --partition lmbdlc2_gpu-l40s  # short: -p <partition_name>
#SBATCH --job-name probes_vid_anti_k1     # short: -J <job name>

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
#SBATCH --signal=TERM@300

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

torchrun --nproc_per_node=1 -m evals.main --fname configs/eval/vitl/bddx_action_anticipation_topk1.yaml  --devices cuda:0
end=`date +%s`
runtime=$((end-start))


echo Job execution complete
echo Runtime: $runtime