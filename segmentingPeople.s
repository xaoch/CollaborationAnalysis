#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=poseExtraction
#SBATCH --mail-type=END
#SBATCH --mail-user=xavier.ochoa@nyu.edu
#SBATCH --output=slurm_poseExtraction%j.out


VIDEOFILE=$1
OUTDIR=$2

SINGULARITY_IMAGE=/scratch/work/public/singularity/ubuntu-20.04.1.sif
OVERLAY_FILE=/scratch/work/public/examples/greene-getting-started/overlay-15GB-500K-pytorch.ext3
singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash /home/xao1/Code/CollaborationAnalysis/segmentingPeople.sh $VIDEOFILE $OUTDIR

cd $OUTJSON
cd ..
tar -zcvf postures.tar.gz postures
rm -rf postures