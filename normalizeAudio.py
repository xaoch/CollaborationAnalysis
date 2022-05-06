from pydub import AudioSegment, effects
import sys
import getopt



def normalize(inputFile,outputFile):
    rawsound = AudioSegment.from_file(inputFile, "wav")
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(outputFile, format="wav")
    print("Finished")

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
   normalize(inputfile,outputfile)

if __name__ == '__main__':
    main(sys.argv[1:])

