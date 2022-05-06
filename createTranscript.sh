#!/bin/bash

source /ext3/miniconda3/bin/activate

conda activate /scratch/xao1/asr/nemo

python /home/xao1/Code/CollaboratinAnalysis/createTranscript.py $1 $2 $3