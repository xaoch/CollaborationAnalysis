#!/bin/bash

VIDFILE=$1
DIR=$2
SPEAKERS=$3

source /ext3/miniconda3/bin/activate

conda activate /scratch/xao1/asr/nemo

python /home/xao1/Code/CollaborationAnalysis/createTranscript.py -i $VIDFILE -o $DIR -s $SPEAKERS