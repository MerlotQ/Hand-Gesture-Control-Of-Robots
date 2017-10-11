import sys
import os
import signal
import collections as col
import cv2 as cv
import cv2
from tqdm import tqdm

from shutil import copyfile
from lxml import etree


def createXML(image_name,classname, xmin, xmax, ymin, ymax, width, height):
    annotation = etree.Element('annotation')

    fo = etree.Element('folder')
    fo.text ='images'

    annotation.append(fo)

    f = etree.Element('filename')
    f.text = image_name

    annotation.append(f)

    size = etree.Element('size')
    w = etree.Element('width')
    w.text = str(width)
    h = etree.Element('height')
    h.text = str(height)
    d = etree.Element('depth')
    d.text = str(1)

    size.append(w)
    size.append(h)
    size.append(d)

    annotation.append(size)

    seg = etree.Element('segmented')
    seg.text = str(0)

    annotation.append(seg)

    object = etree.Element('object')
    n = etree.Element('name')
    p = etree.Element('pose')
    t = etree.Element('truncated')
    d_1 = etree.Element('difficult')
    bb = etree.Element('bndbox')

    n.text = classname
    p.text = 'center'
    t.text = str(1)
    d_1.text = str(0)

    xmi = etree.Element('xmin')
    ymi = etree.Element('ymin')
    xma = etree.Element('xmax')
    yma = etree.Element('ymax')

    xmi.text = str(xmin)
    yma.text = str(ymax)
    ymi.text = str(ymin)
    xma.text = str(xmax)

    bb.append(xmi)
    bb.append(ymi)
    bb.append(xma)
    bb.append(yma)

    object.append(n)
    object.append(p)
    object.append(t)
    object.append(d_1)
    object.append(bb)

    annotation.append(object)
    return annotation

def saveXML(xml, filename, classname):
    path = './dataset/WIDER_train/annotations/' + filename
    if(False):
    	print ('Creating file ' + path + ':')

    with open(path, "w") as file:
        file.write((etree.tostring(xml,encoding='unicode',method="xml",  pretty_print=True)))

def bounding_box_scaler(bbox, image):
    img = cv2.imread("./dataset/WIDER_train/images/"+image)
    h, w, _ = img.shape
    bbox[0] = int(float(bbox[0])*w)
    bbox[1] = int(float(bbox[1])*h)
    bbox[2] = int(float(bbox[2])*w)
    bbox[3] = int(float(bbox[3])*h)
    return (bbox, h, w)
classname = "hands"
with open("./dataset/wider_face_train.txt", "r+") as f:
    for line in tqdm(f.readlines()):
        words = line.strip().split("    ")
        image_name = words[0]
        bbox, h, w = bounding_box_scaler([words[1], words[2], words[3], words[4]], image_name)
        xmin = bbox[0]
        xmax = bbox[2]
        ymin = bbox[1]
        ymax = bbox[3]
        """print ("./dataset/WIDER_face/images/"+image_name)
        img = cv2.imread("./dataset/WIDER_train/images/"+image_name)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.imshow("imge", img)
        cv2.waitKey(1000)"""
        xml = createXML(image_name, classname, xmin, xmax, ymin, ymax, w, h)
        filename_xml = image_name.replace(".png", ".xml")
        saveXML(xml, filename_xml, classname)
