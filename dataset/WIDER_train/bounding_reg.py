import sys
import os
import cv2
from tqdm import tqdm

def bounding_box_scaler(bbox, image):
    img = cv2.imread("./images/"+image)
    h, w, _ = img.shape
    bbox[0] = int(float(bbox[0])*w)
    bbox[1] = int(float(bbox[1])*h)
    bbox[2] = int(float(bbox[2])*w)
    bbox[3] = int(float(bbox[3])*h)
    return (bbox, h, w)
classname = "hands"
with open("wider_face_train.txt", "r+") as f:
    for line in tqdm(f.readlines()):
        words = line.strip().split("    ")
        image_name = words[0]
        bbox, h, w = bounding_box_scaler([words[1], words[2], words[3], words[4]], image_name)
        xmin = bbox[0]
        xmax = bbox[2]
        ymin = bbox[1]
        ymax = bbox[3]
        x_tip = int(float(words[5])*w)
        y_tip = int(float(words[6])*h)
        x_bottom = int(float(words[7])*w)
        y_bottom = int(float(words[8])*h)
        img = cv2.imread("./images/"+image_name)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.circle(img, (x_tip, y_tip), 3, (0,255,0), 3)
        cv2.circle(img, (x_bottom, y_bottom), 3, (0,0,255), 3)
        cv2.imshow("imge", img)
        cv2.waitKey(1000)
