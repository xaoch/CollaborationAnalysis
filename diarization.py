from pyannote.core import Segment, notebook
from pyannote.audio.features import RawAudio
from IPython.display import Audio
import torch
import sys
import getopt



def extract(audio,output):
    AUDIO_FILE = {'uri': "audio", 'audio': '/scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/Three.wav'}
    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
    diarization = pipeline(AUDIO_FILE)
    print(diarization)

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   extract(inputfile,outputfile)

if __name__ == '__main__':
    main(sys.argv[1:])