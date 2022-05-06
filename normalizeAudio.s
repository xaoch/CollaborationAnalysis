#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB
#SBATCH --job-name=normalizeAudio
#SBATCH --mail-type=END
#SBATCH --mail-user=xavier.ochoa@nyu.edu
#SBATCH --output=slurm_normalizeAudio%j.out

module purge
module load python/intel/3.8.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install pydub

OUTFILE=$SCRATCH/$2
DATAFILE=$SCRATCH/$1
DIR=$USER/Code/CollaborationAnalysis
cd $DIR
python ./normalizeAudio.py -i $DATAFILE -o $OUTFILE