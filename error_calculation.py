import cv2
# import imutils
import time
import numpy as np
import tkinter
import math

def contours_proc(frame, h, w):

    blur_frame = cv2.GaussianBlur(frame[:,:,1], (11,11), 0)
    ret, bframe = cv2.threshold(blur_frame, 42, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(bframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for c in contours:
            c_size = cv2.contourArea(c)
            if 200 < c_size < 3000 and c_size != 666:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    mu11p = int(M['mu11'] / M['m00'])
                    mu20p = int(M['mu20'] / M['m00'])
                    mu02p = int(M['mu02'] / M['m00'])
                else:
                    cX, cY = 0, 0
                    mu11p = mu20p = mu02p = 0
                dx = cX - (w/2)
                dy = -(cY - (h/2))
                d = math.sqrt((dx ** 2) + (dy ** 2))
                return d
    else:
        d = 0
        return d

def main(name):
    f = open('./' + name + '/error_values', 'w')
    total = 0
    count = 0
    for i in range(500):
        f_name = './' + name + '/vid_frames/frame-' + str(1000 + i) + '.jpg'
        frame = cv2.imread(f_name)
        (h,w) = frame.shape[:2]
        d = contours_proc(frame, h, w)
        if d:
            d_mm = d * 26.5 / 525.4
            total += d_mm
            count += 1
            #f.write(str(d_mm) + ' ')
    print(total / count)
if __name__ == '__main__':
    main('speed_control_test2')
