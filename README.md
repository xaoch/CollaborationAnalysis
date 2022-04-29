# Instructions

## Dewarp Video

sbatch dewarp.s \ 
    BiochemS1/Session_1_1230_Sensor_5/Five.mp4 \
    BiochemS1/Session_1_1230_Sensor_5/FiveDewarped.mp4

## Extract Poses

sbatch postureOpenPose.s \ 
    /scratch/xao1/BiochemS1/Session_1_1400_Sensor_7/SevenDewarped.mp4 \
    /scratch/xao1/BiochemS1/Session_1_1400_Sensor_7/SevenPostures.avi \
    /scratch/xao1/BiochemS1/Session_1_1400_Sensor_7/postures/

## Correct Posture Video


