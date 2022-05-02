# Instructions

## Dewarp Video

sbatch dewarp.s \ 
    BiochemS1/Session_1_1230_Sensor_5/Five.mp4 \
    BiochemS1/Session_1_1230_Sensor_5/FiveDewarped.mp4

## Extract Poses

sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreeDewarped.mp4 /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.avi /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/postures/

## Correct Posture Video

sbatch correctPostureVideo.s \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.avi \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.mp4

## Extract Audio

sbatch extractAudio.s \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_4/Four.mp4 \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_4/fourAudio.wav