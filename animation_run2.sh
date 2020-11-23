#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-381
#SBATCH --cpus-per-task=14
#SBATCH -J FLARES-pysphv #Give it something meaningful.
#SBATCH -o logs/output_flythrough.%J.out
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A cosma8
#SBATCH --exclusive
#SBATCH -t 00:30:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma/home/dp004/dc-rope1/cosma7/SWIFT/flares-swift-vis

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1 + 1000))

# Run the program
#./swift_dm_animate.py $i
#./swift_gas_animate.py $i
#./swift_stars_animate.py $i
#./swift_GasStars_animate.py $i
#./swift_DMGas_animate.py $i
./swift_GasStars_flythrough.py $i
./swift_GasTemp_flythrough.py $i

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit


