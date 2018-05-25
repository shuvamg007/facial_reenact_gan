import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream
import time
import os

video_link = "rus_short.mp4"
op_directory = "output3/"
ip_directory = "input3/"
orig_directory = "input_orig3/"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"


init_frame = 0
frame = 10000
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=300)

print("new src")

counter = 0
vs = FileVideoStream(video_link).start()
cap = cv2.VideoCapture(video_link)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
time.sleep(2.0)

print(video_length)

while counter < video_length:

    print(frame)

    flag = 0
    count = 0
    clone = np.zeros([300,300,3], dtype=np.uint8)
    image = vs.read()
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) != 0 :

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            faceAligned, M = fa.align(image, gray, rect)
            faceAligned = imutils.resize(faceAligned, width=256)
            
            cv2.imwrite(ip_directory + str(frame) + ".png", faceAligned)
            cv2.imwrite(orig_directory + str(frame) + ".png", image)
            rectan = dlib.rectangle(0, 0, 300, 300)
            grays = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

            shape = predictor(grays, rectan)
            shape = face_utils.shape_to_np(shape)

            # loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name == "mouth":
                    centre = np.sum(shape[i:j], axis=0)
                    centre = centre/[20, 20]
                    centre = [ int(x) for x in centre ]
                    for (x, y) in shape[i:j]:

                        if flag == 0:
                            (init_a, init_b) = (a, b) = (x, y)
                            flag = 1

                        elif count == 11:
                            cv2.line(clone, (a, b), (x, y), (0, 0, 255), 1)
                            cv2.line(clone, (init_a, init_b), (x, y), (0, 0, 255), 1)
                            flag = 0

                        elif count == 19:
                            cv2.line(clone, (a, b), (x, y), (0, 0, 255), 1)
                            cv2.line(clone, (init_a, init_b), (x, y), (0, 0, 255), 1)

                        else:
                            cv2.line(clone, (a, b), (x, y), (0, 0, 255), 1)
                            (a, b) = (x, y)

                        count = count + 1

                    faceAligned[centre[1]-50:centre[1]+50, centre[0]-50:centre[0]+50] = clone[centre[1]-50:centre[1]+50, centre[0]-50:centre[0]+50]
        
        cv2.imwrite(op_directory + str(frame) + ".png", faceAligned)
        frame = frame + 1
    
    init_frame = init_frame + 1
    counter = counter + 1