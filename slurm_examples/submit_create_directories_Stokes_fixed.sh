#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=50:30:00
#SBATCH --job-name=submit_master

# ==========================================================
# Load modules
module load anaconda/anaconda-2023.09

# ==========================================================
# Set start and end indices explicitly
#START=11
#END=20
#START=21
#END=50
START=1
END=100

# Run the Python script with dynamic start and end
python3 create_directories_Stokes_fixed.py $START $END

# Clear loaded modules
module purge