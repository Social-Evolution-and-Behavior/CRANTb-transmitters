#! /bin/bash  -l
#SBATCH --job-name="cloud_data"
#SBATCH --output=/dev/null 
#SBATCH --ntasks 10                 
#SBATCH --cpus-per-task 4                
#SBATCH --time 3:00:00                  
#SBATCH --mem-per-cpu 4G                
### program starts below                

# Construct log filename with current date and SLURM job ID
logfile="$(date '+%Y%m%d')_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

# Redirect stdout and stderr to that log file
exec > "$logfile" 2>&1

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running job ID: $SLURM_JOB_ID"

python -u /ru-auth/local/home/jlee11/scratch/nt_predictions/CRANTb-transmitters_repo/getting_data.py