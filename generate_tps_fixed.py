import cv2
from imutils import face_utils
import numpy as np
import imutils
import dlib
from imutils.video import FileVideoStream
import tensorflow as tf
import time
from PIL import Image
import os
from tps import from_control_points
import argparse

def create_dir(pth):
    cwd = os.getcwd()
    dir = cwd + "/" + pth
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"

def get_center(img_src):
    gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # changed implementation
        shapes = shape[48:68]
        center = np.sum(shapes, axis=0)
        center = (center/[20, 20]).astype(int)
    center = (center[0], center[1])
    return center

def get_shape(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    shape_d = predictor(gray, rects[0])
    shape_d = face_utils.shape_to_np(shape_d)
    return shape_d

def calc_tps(shape_src_outline, shape_dst_outline):
    t = 0
    for i in range(len(shape_src_outline)):
        if t == 0:
            t = 1
            cp = from_control_points([(shape_src_outline[i][0], shape_src_outline[i][1], shape_dst_outline[i][0], shape_dst_outline[i][1])])
        else:
            cp.add(shape_src_outline[i][0], shape_src_outline[i][1], shape_dst_outline[i][0], shape_dst_outline[i][1])
    return cp

def load_graph(graph_filename):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

parser = argparse.ArgumentParser()
parser.add_argument('--predictor', type=str, help='Path to shape_predictor_68_face_landmarks.dat', required=True)
parser.add_argument('--model', type=str, help='Path to frozen model', required=True)
args = parser.parse_args()

op_directory = create_dir("tgt_oriented")
directory_src = create_dir("align_tgt")
directory_dst = create_dir("align_src")
shape_predictor_path = args.predictor
model_file = args.model
tot = max(len(os.listdir(directory_src)), len(os.listdir(directory_dst))) + 10000

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

graph = load_graph(model_file)
image_tensor = graph.get_tensor_by_name('image_tensor:0')
output_tensor = graph.get_tensor_by_name('generate_output/output:0')
sess = tf.Session(graph=graph)

print("new src")
counter = 10000

while counter < tot:
    flag = 0
    count = 0
    img_src1 = cv2.imread(directory_src + str(counter) + ".png")
    # img_src1 = img_src1[:,:300]
    img_src1 = imutils.resize(img_src1, width=256)

    # print(np.shape(img_src1))
    clone = np.zeros([256,256,3], dtype=np.uint8)
    img_dst1 = cv2.imread(directory_dst + str(counter) + ".png")
    img_dst1 = imutils.resize(img_dst1, width=256)

    img_src = img_dst1
    img_dst = img_src1

    shape_src = get_shape(img_src)
    shape_dst = get_shape(img_dst)

    shape_src_outline = np.concatenate([shape_src[17:49], shape_src[54:55]])
    shape_dst_outline = np.concatenate([shape_dst[17:49], shape_dst[54:55]])

    shape_src_outline = np.concatenate([shape_src_outline, (shape_src[61:62]+shape_src[67:68])/(2,2)])
    shape_dst_outline = np.concatenate([shape_dst_outline, (shape_dst[61:62]+shape_dst[67:68])/(2,2)])

    shape_src_outline = np.concatenate([shape_src_outline, (shape_src[62:63]+shape_src[66:67])/(2,2)])
    shape_dst_outline = np.concatenate([shape_dst_outline, (shape_dst[62:63]+shape_dst[66:67])/(2,2)])

    shape_src_outline = np.concatenate([shape_src_outline, (shape_src[63:64]+shape_src[65:66])/(2,2)])
    shape_dst_outline = np.concatenate([shape_dst_outline, (shape_dst[63:64]+shape_dst[65:66])/(2,2)])

    cp = calc_tps(shape_src_outline, shape_dst_outline)

    for i in range(20):
        shape_dst[i+48] = cp.transform(shape_src[i+48][0], shape_src[i+48][1])

    # shape = face_utils.shape_to_np(shape_dst)
    shape = shape_dst
    src_centre = get_center(img_dst)

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
                    print(counter)
                    cv2.line(clone, (a, b), (x, y), (0, 0, 255), 1)
                    cv2.line(clone, (init_a, init_b), (x, y), (0, 0, 255), 1)
                else:
                    cv2.line(clone, (a, b), (x, y), (0, 0, 255), 1)
                    (a, b) = (x, y)
                count = count + 1

            img_dst[src_centre[1]-45:src_centre[1]+45, src_centre[0]-45:src_centre[0]+45] = clone[centre[1]-45:centre[1]+45, centre[0]-45:centre[0]+45]
    img_dst = np.concatenate((img_dst, img_dst1), axis=1)
    img_fin = sess.run(output_tensor, feed_dict={image_tensor: img_dst})

    img_fin = cv2.cvtColor(np.squeeze(img_fin), cv2.COLOR_RGB2BGR)
    cv2.imwrite(op_directory + str(counter) + ".png", img_fin)
    counter = counter + 1
