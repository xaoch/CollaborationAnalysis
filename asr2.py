import nemo.collections.asr as nemo_asr
import numpy as np
import librosa
import os
import wget
import nemo
import glob
import os
from omegaconf import OmegaConf
import shutil
import json
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
import gzip



import pprint
pp = pprint.PrettyPrinter(indent=4)
data_dir="/scratch/xao1/asr"
ROOT = os.getcwd()
#data_dir = os.path.join(ROOT,'data')
#os.makedirs(data_dir, exist_ok=True)


AUDIO_FILENAME = "/scratch/xao1/asr/Three16Norm.wav"

signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)

CONFIG = os.path.join(data_dir,'offline_diarization_with_asr.yaml')

cfg = OmegaConf.load(CONFIG)
print(OmegaConf.to_yaml(cfg))

# Create a manifest file for input with below format.
# {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-",
# "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}

meta = {
    'audio_filepath': AUDIO_FILENAME,
    'offset': 0,
    'duration':None,
    'label': 'infer',
    'text': '-',
    'num_speakers': 2,
    'rttm_filepath': None,
    'uem_filepath' : None
}
with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')

pretrained_speaker_model='titanet_large'
cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
cfg.diarizer.out_dir = data_dir #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
cfg.diarizer.clustering.parameters.oracle_num_speakers=True

# Using VAD generated from ASR timestamps
cfg.diarizer.asr.model_path = 'stt_en_conformer_ctc_large'
cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD
cfg.diarizer.asr.parameters.asr_based_vad = True
cfg.diarizer.asr.parameters.threshold=100 # ASR based VAD threshold: If 100, all silences under 1 sec are ignored.
cfg.diarizer.asr.parameters.decoder_delay_in_sec=0.2 # Decoder delay is compensated for 0.2 sec


asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
asr_model = asr_ts_decoder.set_asr_model()
word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)

print("Decoded word output dictionary: \n", word_hyp['Three16Norm'])
print("Word-level timestamps dictionary: \n", word_ts_hyp['Three16Norm'])

asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer)
asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset
diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
print("Diarization hypothesis output: \n", diar_hyp['Three16Norm'])

#asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)




def gunzip(file_path, output_path):
    with gzip.open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()

ARPA_FILE = os.path.join(data_dir,'4gram_big.arpa.gz')
f = ARPA_FILE
gunzip(f, f.replace(".gz", ""))

arpa_model_path = os.path.join(data_dir, '4gram_big.arpa')
cfg.diarizer.asr.ctc_decoder_parameters.pretrained_language_model = arpa_model_path

import importlib
import nemo.collections.asr.parts.utils.decoder_timestamps_utils as decoder_timestamps_utils
importlib.reload(decoder_timestamps_utils) # This module should be reloaded after you install pyctcdecode.

asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
asr_model = asr_ts_decoder.set_asr_model()
word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)

print("Decoded word output dictionary: \n", word_hyp['Three16Norm'])
print("Word-level timestamps dictionary: \n", word_ts_hyp['Three16Norm'])

arpa_model_path = os.path.join(data_dir, '4gram_big.arpa')
cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model = arpa_model_path
cfg.diarizer.asr.realigning_lm_parameters.logprob_diff_threshold = 1.2

import importlib
import nemo.collections.asr.parts.utils.diarization_utils as diarization_utils
importlib.reload(diarization_utils) # This module should be reloaded after you install arpa.

# Create a new instance with realigning language model
asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer)
asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset

asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)