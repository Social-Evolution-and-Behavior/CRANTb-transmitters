#! /bin/bash  -l
#SBATCH --job-name="cloud_data"
#SBATCH --output=/dev/null 
#SBATCH --array=0-9999:1000
#SBATCH --cpus-per-task 4                
#SBATCH --time 12:00:00                  
#SBATCH --mem-per-cpu 4G                
### program starts below                

# Construct log filename with current date and SLURM array job ID and task ID
logfile="$(date '+%Y%m%d')_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

# Redirect stdout and stderr to that log file
exec > "$logfile" 2>&1

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running array job ID: $SLURM_ARRAY_JOB_ID, task ID: $SLURM_ARRAY_TASK_ID"

# Calculate start and end indices for this array task
STRIDE=1000
START_IDX=$SLURM_ARRAY_TASK_ID
END_IDX=$((SLURM_ARRAY_TASK_ID + STRIDE))

echo "Processing synapses from index $START_IDX to $END_IDX"

python -u /ru-auth/local/home/jlee11/scratch/nt_predictions/CRANTb-transmitters_repo/getting_data.py \
    --start-index $START_IDX \
    --end-index $END_IDX