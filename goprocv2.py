import cv2
from goprocam import GoProCamera
from goprocam import constants
gpCam = GoProCamera.GoPro()
cap = cv2.VideoCapture("udp://127.0.0.1:10000")
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(100)