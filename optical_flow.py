import cv2
import numpy as np
def optical_smooth(prev,cur,x_l,y_l,x_r,y_r,dt):
    prvs=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3.0, 15.0, 3.0, 5.0, 1.2, 0.0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag2=mag[y_l:y_r,x_l:x_r]
    ang2=ang[y_l:y_r,x_l:x_r]
    mmag=np.mean(mag2)
    mang=np.mean(ang2)
    shift_x=mmag*np.cos(mang)*dt
    shift_y=mmag*np.sin(mang)*dt
    x_ln=shift_x+x_l
    x_rn=shift_x+x_r
    y_ln=shift_y+y_l
    y_rn=shift_y+y_r
    bbox_next=[x_ln,y_ln,x_rn,y_rn]
    bbox_next=np.array(bbox_next).astype(int)
    return bbox_next