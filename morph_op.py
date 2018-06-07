import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream
import time
import os
import argparse

def create_dir(pth):
    cwd = os.getcwd()
    dir = cwd + "/" + pth
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"

def pre_process(video, align_dir, frame_dir=None):
    init_frame = 0
    frame = 10000
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)
    print("New video")

    counter = 0
    vs = FileVideoStream(video).start()
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time.sleep(2.0)
    print(video_length)

    while counter < video_length:
        print(counter)

        flag = 0
        count = 0
        clone = np.zeros([256,256,3], dtype=np.uint8)
        image = vs.read()
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if len(rects) != 0 :

            # Loop over the face detections
            for (i, rect) in enumerate(rects):
                # faceAligned, M = fa.align(image, gray, rect)
                faceAligned = fa.align(image, gray, rect)
                faceAligned = imutils.resize(faceAligned, width=256)

                cv2.imwrite(align_dir + str(frame) + ".png", faceAligned)
                if frame_dir != None:
                    cv2.imwrite(frame_dir + str(frame) + ".png", image)
            frame = frame + 1
        counter = counter + 1

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='Path to source video', required=True)
parser.add_argument('--target', type=str, help='Path to target video', required=True)
parser.add_argument('--predictor', type=str, help='Path to shape_predictor_68_face_landmarks.dat', required=True)
args = parser.parse_args()

src_video = args.source
tgt_video = args.target

src_ip_directory = create_dir("align_src")
tgt_ip_directory = create_dir("align_tgt")
orig_directory = create_dir("frames")
shape_predictor_path = args.predictor

pre_process(src_video, src_ip_directory)
pre_process(tgt_video, tgt_ip_directory, orig_directory)
