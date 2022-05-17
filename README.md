# Instructions

## Dewarp Video

sbatch dewarp.s BiochemS1/Session_2_1400_Sensor_8/Eight.mp4 BiochemS1/Session_2_1400_Sensor_8/EightDewarped.mp4

## Extract Poses

sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_1/OneDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_0930_Sensor_1/OnePostures.avi /scratch/xao1/BiochemS1/Session_2_0930_Sensor_1/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_2/TwoDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_0930_Sensor_2/TwoPostures.avi /scratch/xao1/BiochemS1/Session_2_0930_Sensor_2/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_3/ThreeDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_0930_Sensor_3/ThreePostures.avi /scratch/xao1/BiochemS1/Session_2_0930_Sensor_3/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_5/FiveDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1100_Sensor_5/FivePostures.avi /scratch/xao1/BiochemS1/Session_2_1100_Sensor_5/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_6/SixDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1100_Sensor_6/SixPostures.avi /scratch/xao1/BiochemS1/Session_2_1100_Sensor_6/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_7/SevenDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1100_Sensor_7/SevenPostures.avi /scratch/xao1/BiochemS1/Session_2_1100_Sensor_7/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_8/EightDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1100_Sensor_8/EightPostures.avi /scratch/xao1/BiochemS1/Session_2_1100_Sensor_8/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_2/TwoDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1230_Sensor_2/TwoPostures.avi /scratch/xao1/BiochemS1/Session_2_1230_Sensor_2/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_3/ThreeDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1230_Sensor_3/ThreePostures.avi /scratch/xao1/BiochemS1/Session_2_1230_Sensor_3/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_4/FourDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1230_Sensor_4/FourPostures.avi /scratch/xao1/BiochemS1/Session_2_1230_Sensor_4/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_9/NineDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1230_Sensor_9/NinePostures.avi /scratch/xao1/BiochemS1/Session_2_1230_Sensor_9/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_1/OneDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_1/OnePostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_1/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_2/TwoDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_2/TwoPostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_2/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_5/FiveDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_5/FivePostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_5/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_6/SixDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_6/SixPostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_6/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_7/SevenDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_7/SevenPostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_7/postures/
sbatch postureOpenPose.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8/EightDewarped.mp4 /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8/EightPostures.avi /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8/postures/



## Correct Posture Video

sbatch correctPostureVideo.s \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.avi \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.mp4

## Extract Audio

sbatch extractAudio.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8/Eight.mp4 

## Normalize Audio

sbatch normalizeAudio.s BiochemS1/Session_1_0930_Sensor_3/Three.wav BiochemS1/Session_1_0930_Sensor_3/ThreeNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_0930_Sensor_4/Four.wav BiochemS1/Session_1_0930_Sensor_4/FourNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1100_Sensor_2/Two.wav BiochemS1/Session_1_1100_Sensor_2/TwoNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1100_Sensor_5/Five.wav BiochemS1/Session_1_1100_Sensor_5/FiveNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1100_Sensor_6/Six.wav BiochemS1/Session_1_1100_Sensor_6/SixNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1100_Sensor_7/Seven.wav BiochemS1/Session_1_1100_Sensor_7/SevenNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1100_Sensor_9/Nine.wav BiochemS1/Session_1_1100_Sensor_9/NineNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1230_Sensor_3/Three.wav BiochemS1/Session_1_1230_Sensor_3/ThreeNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1230_Sensor_4/Four.wav BiochemS1/Session_1_1230_Sensor_4/FourNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1230_Sensor_5/Five.wav BiochemS1/Session_1_1230_Sensor_5/FiveNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1230_Sensor_8/Eight.wav BiochemS1/Session_1_1230_Sensor_8/EightNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_1/One.wav BiochemS1/Session_1_1400_Sensor_1/OneNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_3/Three.wav BiochemS1/Session_1_1400_Sensor_3/ThreeNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_5/Five.wav BiochemS1/Session_1_1400_Sensor_5/FiveNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_6/Six.wav BiochemS1/Session_1_1400_Sensor_6/SixNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_7/Seven.wav BiochemS1/Session_1_1400_Sensor_7/SevenNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_1_1400_Sensor_9/Nine.wav BiochemS1/Session_1_1400_Sensor_9/NineNormalized.wav

sbatch normalizeAudio.s BiochemS1/Session_2_0930_Sensor_1/One.wav BiochemS1/Session_2_0930_Sensor_1/OneNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_0930_Sensor_2/Two.wav BiochemS1/Session_2_0930_Sensor_2/TwoNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_0930_Sensor_3/Three.wav BiochemS1/Session_2_0930_Sensor_3/ThreeNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1100_Sensor_5/Five.wav BiochemS1/Session_2_1100_Sensor_5/FiveNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1100_Sensor_6/Six.wav BiochemS1/Session_2_1100_Sensor_6/SixNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1100_Sensor_7/Seven.wav BiochemS1/Session_2_1100_Sensor_7/SevenNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1100_Sensor_8/Eight.wav BiochemS1/Session_2_1100_Sensor_8/EightNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1230_Sensor_2/Two.wav BiochemS1/Session_2_1230_Sensor_2/TwoNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1230_Sensor_3/Three.wav BiochemS1/Session_2_1230_Sensor_3/ThreeNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1230_Sensor_4/Four.wav BiochemS1/Session_2_1230_Sensor_4/FourNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1230_Sensor_9/Nine.wav BiochemS1/Session_2_1230_Sensor_9/NineNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_1/One.wav BiochemS1/Session_2_1400_Sensor_1/OneNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_2/Two.wav BiochemS1/Session_2_1400_Sensor_2/TwoNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_5/Five.wav BiochemS1/Session_2_1400_Sensor_5/FiveNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_6/Six.wav BiochemS1/Session_2_1400_Sensor_6/SixNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_7/Seven.wav BiochemS1/Session_2_1400_Sensor_7/SevenNormalized.wav
sbatch normalizeAudio.s BiochemS1/Session_2_1400_Sensor_8/Eight.wav BiochemS1/Session_2_1400_Sensor_8/EightNormalized.wav


## Create Transcript

sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_1/OneNormalized.wav /scratch/xao1/BiochemS1/Session_2_0930_Sensor_1 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_2/TwoNormalized.wav /scratch/xao1/BiochemS1/Session_2_0930_Sensor_2 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_0930_Sensor_3/ThreeNormalized.wav /scratch/xao1/BiochemS1/Session_2_0930_Sensor_3 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_5/FiveNormalized.wav /scratch/xao1/BiochemS1/Session_2_1100_Sensor_5 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_6/SixNormalized.wav /scratch/xao1/BiochemS1/Session_2_1100_Sensor_6 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_7/SevenNormalized.wav /scratch/xao1/BiochemS1/Session_2_1100_Sensor_7 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1100_Sensor_8/EightNormalized.wav /scratch/xao1/BiochemS1/Session_2_1100_Sensor_8 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_2/TwoNormalized.wav /scratch/xao1/BiochemS1/Session_2_1230_Sensor_2 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_3/ThreeNormalized.wav /scratch/xao1/BiochemS1/Session_2_1230_Sensor_3 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_4/FourNormalized.wav /scratch/xao1/BiochemS1/Session_2_1230_Sensor_4 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1230_Sensor_9/NineNormalized.wav /scratch/xao1/BiochemS1/Session_2_1230_Sensor_9 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_1/OneNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_1 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_2/TwoNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_2 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_5/FiveNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_5 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_6/SixNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_6 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_7/SevenNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_7 5
sbatch createTranscript.s /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8/EightNormalized.wav /scratch/xao1/BiochemS1/Session_2_1400_Sensor_8 5


## Segmenting People

sbatch segmentingPeople.s /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreeDewarped.mp4 /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/people


srun --gres=gpu:1 --time=00:20:00 --pty /bin/bash
srun --time=00:20:00 --pty /bin/bash

singularity exec --nv \
      --overlay /scratch/work/public/singularity/openpose1.7.0-cuda11.1.1-cudnn8-devel-ubuntu20.04-dep.sqf:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash /home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON

singularity exec --nv \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash /home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON

srun --gres=gpu:1 --mem=8GB --time=01:00:00 --pty /bin/bash

SINGULARITY_IMAGE=/scratch/work/public/singularity/ubuntu-20.04.1.sif
OVERLAY_FILE=/scratch/work/public/examples/greene-getting-started/overlay-15GB-500K-pytorch.ext3
singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash

source /ext3/miniconda3/bin/activate

conda activate /scratch/xao1/asr/nemo

python examples/asr/transcribe_speech.py \
 pretrained_name="stt_en_conformer_transducer_large" \
 audio_dir="/scratch/xao1/NeMo/audio"


singularity exec \
      --overlay /scratch/work/public/singularity/openpose1.7.0-cuda11.1.1-cudnn8-devel-ubuntu20.04-dep.sqf:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash 

ffmpeg -y -i FourDewarped.mp4 -vf normalize=blackpt=black:whitept=white:smoothing=0 FourNormalized.mp4

VIDEOFILE=/scratch/xao1/
/home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON


Session_1_0930_Sensor_3
Session_1_0930_Sensor_4
Session_1_1100_Sensor_2
Session_1_1100_Sensor_5
Session_1_1100_Sensor_6
Session_1_1100_Sensor_7
Session_1_1100_Sensor_9
Session_1_1230_Sensor_3
Session_1_1230_Sensor_4
Session_1_1230_Sensor_5
Session_1_1230_Sensor_8
Session_1_1400_Sensor_1
Session_1_1400_Sensor_3
Session_1_1400_Sensor_5
Session_1_1400_Sensor_6
Session_1_1400_Sensor_7
Session_1_1400_Sensor_9

Session_2_0930_Sensor_1
Session_2_0930_Sensor_2
Session_2_0930_Sensor_3
Session_2_1100_Sensor_5
Session_2_1100_Sensor_6
Session_2_1100_Sensor_7
Session_2_1100_Sensor_8
Session_2_1230_Sensor_2
Session_2_1230_Sensor_3
Session_2_1230_Sensor_4
Session_2_1230_Sensor_9
Session_2_1400_Sensor_1
Session_2_1400_Sensor_2
Session_2_1400_Sensor_5
Session_2_1400_Sensor_6
Session_2_1400_Sensor_7
Session_2_1400_Sensor_8