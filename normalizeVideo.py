import sys
import getopt
import cv2




def normalize(inputFile,outputFile):
    cap = cv2.VideoCapture(inputFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    frame = 0
    while cap.isOpened():
        success, image = cap.read()
        frame = frame + 1
        print(frame)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            # Flip the image horizontally for a selfie-view display.
        cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        out.write(image)
        if frame == 30:
            break
    cap.release()

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