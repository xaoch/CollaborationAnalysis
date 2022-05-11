import cv2
from retinaface import RetinaFace

print("Starting")

# For webcam input:
cap = cv2.VideoCapture("/scratch/xao1/BiochemS1/Session_1_0930_Sensor_4/FourDewarped.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
fps =  int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (width,height))
frame=0
print(width,height,fps)
while cap.isOpened():
  success, image = cap.read()
  frame=frame+1
  print(frame)
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # Flip the image horizontally for a selfie-view display.
  faces = RetinaFace.detect_faces(image)

  for key in faces.keys():
      identity=faces[key]
      print(identity)
      facial_area = identity["facial_area"]
      print(facial_area)
      color = (255, 0, 0)
      cv2.rectangle(image, (facial_area[2], facial_area[3]),(facial_area[0], facial_area[1]),color)

  out.write(image)
  if frame==30:
    break
cap.release()