#!/bin/bash

VIDFILE=$1
OUTDIR=$2

source /ext3/miniconda3/bin/activate

conda activate /scratch/xao1/yolo/yoloenv

python /home/xao1/Code/yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --save-txt --save-conf --augment --classes 0 --project $OUTDIR --source $VIDFILE