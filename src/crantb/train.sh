#! /bin/bash  -l
#SBATCH --job-name="train_caches"
#SBATCH --output=/dev/null 
#SBATCH --partition=hpc_l40s_b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu 4G              
#SBATCH --time=6:00:00
#SBATCH --gpus=2

### program starts below                

# Construct log filename with current date and SLURM job ID
logfile="$(date '+%Y%m%d')_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

# Redirect stdout and stderr to that log file
exec > "$logfile" 2>&1

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running job ID: $SLURM_JOB_ID"

pixi run train --cfg config.yaml --epochs 10