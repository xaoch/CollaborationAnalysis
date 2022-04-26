#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=deWarpingVideo
#SBATCH --mail-type=END
#SBATCH --mail-user=xavier.ochoa@nyu.edu
#SBATCH --output=slurm_videoDeWarp%j.out

module purge
module load python/intel/3.8.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -U numpy
pip install opencv-python

OUTFILE=$SCRATCH/$2
DATAFILE=$SCRATCH/$1
DIR=$USER/Code/CollaborationAnalysis
cd $DIR
python ./dewarpVideo.py -i $DATAFILE -o $OUTFILE