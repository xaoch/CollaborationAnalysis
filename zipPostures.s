#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=2GB
#SBATCH --job-name=deWarpingVideo
#SBATCH --mail-type=END
#SBATCH --mail-user=xavier.ochoa@nyu.edu
#SBATCH --output=slurm_zipPostures%j.out

DIR=$1

cd $DIR
tar -zcvf postures.tar.gz postures
rm -rf postures