import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils
from PIL import Image

crop_img_dir = "photos/val2/"
frame_img_dir = "input_orig4/"
crop_mask_img_dir = "nptel_test/images/"
final_op_dir = "/home/shuvamg007/pix2pix-tensorflow/final_opn/"
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

MIN_MATCH_COUNT = 10

def get_center(img_src):
    gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shapes = shape[48:68]
        center = np.sum(shapes, axis=0)
        center = (center/[20, 20]).astype(int)
    center = (center[0], center[1])
    return center
    
for img_val in range(10020,16350):
    crop_img = crop_img_dir + str(img_val) + ".png"
    frame_img = frame_img_dir + str(img_val) + ".png"
    crop_mask_img = crop_mask_img_dir + str(img_val) + "-outputs.png"

    img_src = cv2.imread(crop_img)
    
    img_src = img_src[:,:300]
    img_src = imutils.resize(img_src, width=256)

    img_dst = cv2.imread(frame_img)
    img_diff = cv2.imread(crop_mask_img)
    
    img1 = cv2.imread(crop_img,0)
    
    img1 = img1[:,:300]
    img1 = imutils.resize(img1, width=256)
    
    img2 = cv2.imread(frame_img,0)
    center = get_center(img_src)
    
    obj = img_diff[center[1]-41:center[1]+41, center[0]-41:center[0]+41]
    mask = 255 * np.ones(obj.shape, obj.dtype)
    
    # The location of the center of the src in the dst
    width, height, channels = img_src.shape

    normal_clone = cv2.seamlessClone(obj, img_src, mask, center, cv2.NORMAL_CLONE)
    img_diff = normal_clone
    
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print(img_val)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        mask = 255 * np.ones(img_src.shape, img_src.dtype)
        
        im_mask_temp  = cv2.warpPerspective(img_diff, M, (img_dst.shape[1],img_dst.shape[0]))
        center = get_center(img_dst)
        img_dst[center[1]-40:center[1]+40, center[0]-40:center[0]+40] = im_mask_temp[center[1]-40:center[1]+40, center[0]-40:center[0]+40]
        cv2.imwrite(final_op_dir + str(img_val) + ".png", img_dst)
        
    else:
        print("Not enough matches are found")
        matchesMask = None
