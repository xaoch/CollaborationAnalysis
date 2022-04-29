#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=postureVideoCorrection
#SBATCH --mail-type=END
#SBATCH --mail-user=xavier.ochoa@nyu.edu
#SBATCH --output=slurm_postureVideoCorrection%j.out


VIDEOFILE=$1
OUTPUTFILE=$2

singularity exec \
      --overlay /scratch/work/public/singularity/openpose1.7.0-cuda11.1.1-cudnn8-devel-ubuntu20.04-dep.sqf:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    ffmpeg -y -loglevel info -i $VIDEOFILE $OUTPUTFILE


