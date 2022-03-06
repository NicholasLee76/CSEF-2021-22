from picamera.array import PiRGBArray
from picamera import PiCamera
from collections import deque
from imutils.video import VideoStream
from Raspi_MotorHAT import Raspi_MotorHAT, Raspi_DCMotor
import numpy as np
import argparse
import cv2
import imutils
import cv2 as cv
import math
import time
import atexit
import threading
import asyncio
from multiprocessing import Queue
import queue
import struct
import tkinter


# start queues
t_queue = queue.Queue()
mouse_queue = queue.Queue()
m1_queue = queue.Queue()
m2_queue = queue.Queue()
m3_queue = queue.Queue()


# initialize_motors
mh = Raspi_MotorHAT(addr=0x6f)
m1 = mh.getMotor(1)
m2 = mh.getMotor(2)
m3 = mh.getMotor(3)

velocity_constant = 48
x_y_constant = 1
z_constant = 1
Boundary = 15

orientation = False
write = True
show = False


def turnOffMotors():
    mh.getMotor(1).setSpeed(0)
    mh.getMotor(2).setSpeed(0)
    mh.getMotor(3).setSpeed(0)
    mh.getMotor(4).setSpeed(0)
atexit.register(turnOffMotors)


# initialize video frame in main
def init_video():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())

    if not args.get("video", False):
        vs = VideoStream(src=0).start()
    else:
        vs = cv2.VideoCapture(args["video"])

    return vs

def init_frame_and_count(vs, number_of_frames):
    count = 0

    frames = [0] * number_of_frames
    start_time = 0.0
    end_time = 0.0

    processing_end = 0.0
    processing_times = [0] * number_of_frames

    frame = vs.read()
    frame = cv2.flip(frame, -1)

    prev_id = id(frame)
    (h, w) = frame.shape[:2]
    tx = w/2
    ty = h/2

    return count, frames, start_time, end_time, processing_end, processing_times, tx, ty, frame, prev_id, (h,w)


def start_count(frame, frames, i, start_time):
    prev_id = id(frame)
    end_time = time.time()
    time_diff = end_time - start_time
    start_time = end_time
    frames[i] = time_diff

    return prev_id, time_diff, start_time, frames


# jacobian calculations
def radtodeg(rad):
    deg = (rad * 180) / math.pi
    return deg


def angles(x, y, z):
    # fix the deg part here too
    rollTheta = x
    rollDeg = radtodeg(rollTheta)

    pitchTheta = y
    pitchDeg = radtodeg(pitchTheta)

    yawTheta = z
    yawDeg = radtodeg(yawTheta)

    return rollDeg, pitchDeg, yawDeg


def get_wheel_velocities(x,y,z):
    inverse_jacobian = [
        [0.1, 0, -0.05],
        [-0.05, -0.09, -0.05],
        [-0.05, 0.09, -0.05]
    ]
    sphere_angles = angles(x, y, z)

    wheel_angles = np.matmul(inverse_jacobian, sphere_angles)

    return wheel_angles


# motor functions
def run_motor(mid, velocity):
    if velocity > 0:
        mid.setSpeed(int(abs(velocity)) + 13)
        mid.run(Raspi_MotorHAT.FORWARD)
    elif velocity < 0:
        mid.setSpeed(int(abs(velocity)) + 13)
        mid.run(Raspi_MotorHAT.BACKWARD)
    else:
        mid.setSpeed(0)
        mid.run(Raspi_MotorHAT.FORWARD)


def motor_thread(mid, q):
    while True:
        velocity = q.get()
        if mid == 1:
            run_motor(m1, velocity)
        elif mid == 2:
            run_motor(m2, velocity)
        elif mid == 3:
            run_motor(m3, velocity)
        q.task_done()


def qput(velocities):
    m1_queue.put(velocities[0])
    m2_queue.put(velocities[1])
    m3_queue.put(velocities[2])


# image processing functions
def frame_calc(contours, tx, ty, frame, bframe, h, w):
    proc_count = 0 # should be 1 (the bug) or 0 (none)
    for c in contours:
        c_size = cv2.contourArea(c)
        # print(" ")
        if 200 < c_size < 3000 and c_size != 666:
            # figure out the 666 thing later
            proc_count += 1
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
            # row then column for cX and cY
            dx = cX - tx
            dy = -(cY - ty)
            d = math.sqrt((dx ** 2) + (dy ** 2))
            x = float(dx / float(tx)) * x_y_constant
            y = float(dy / float(ty)) * x_y_constant
            if mu20p != mu02p and orientation == True:
                z = 0.5 * math.atan(2 * mu11p/(mu20p - mu02p)) * z_constant
            else:
                z = 0
            # print(x, y, z)

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            # print(c_size, cX, cY, dx, dy)
            cv2.circle(bframe, (cX, cY), 5, (0, 0, 255), -1)

            if d >= Boundary:
                wheel_angles = get_wheel_velocities(x, y, z)
                wheel_velocities = [0] * 3
                for j in range(3):
                    wheel_velocities[j] = wheel_angles[j] * velocity_constant
                qput(wheel_velocities)
                wheel_velocities_p = [0] * 3
                for j in range(3):
                    wheel_velocities_p[j] = abs(wheel_velocities[j]) + 13
                #print(wheel_velocities_p)

        else:
            velocities = [0,0,0]
            qput(velocities)
    # print(proc_count) used for counting number of allowed centroids


def contours_proc(frame, h, w, tx, ty):
    blur_frame = cv2.GaussianBlur(frame[:,:,1], (11,11), 0)
    ret, bframe = cv.threshold(blur_frame, 42, 255, cv.THRESH_BINARY_INV)
    cv2.circle(frame, (int(w / 2), int(h / 2)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (320,240), Boundary, (255, 0, 255), 0)
    cv2.circle(bframe, (320,240), Boundary, (255, 0, 255), 0)
    contours, hierarchy = cv2.findContours(bframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        frame_calc(contours, tx, ty, frame, bframe, h, w)
    else:
        velocities = [0,0,0]
        qput(velocities)


def mouse_track(q):
    # canvas = make_canvas(300, 200, 'Trajectory')
    m_x = 0
    m_y = 0
    mouse_file = open( "/dev/input/mouse0", "rb" )
    coords_file = open("./olf_test5/coords.txt", "w")

    while True:
        frame_num = q.get()
        data = mouse_file.read(3)
        (e, dx, dy) = struct.unpack('3b', data)
        m_x += dx
        m_y += dy
        text = str(frame_num) + ',' + str(m_x) + ',' + str(m_y) + '\n'
        coords_file.write(text)
        print(text)
        q.task_done()



# show image function
def show_image(q):
    img_seq = 1000
    while True:
        frame = q.get()
        # show frame
        if show:
            cv2.imshow("Binary Frame", frame)
        # write frame
        filename = './olf_test5/vid_frames/frame-' + str(img_seq) + '.jpg'
        if write:
            cv2.imwrite(filename, frame)
        img_seq+=1
        mouse_queue.put(img_seq)
        q.task_done()


def make_canvas(width, height, title=None):
    top = tkinter.Tk()
    top.minsize(width=width, height=height)
    if title:
        top.title(title)
    canvas = tkinter.Canvas(top, width=width + 1, height=height + 1)
    canvas.pack()
    return canvas


# ending processes and calculations
def end_proc_count(start_time, processing_times, i):
    processing_end = time.time()
    processing_times[i] = (processing_end - start_time)

    return processing_times


def end_count(frames, processing_times):
    for i in range(3):
        frames.pop(i)
        processing_times.pop(i)

    total = 0.0
    processed = 0.0

    for i in range(len(frames)):
        total += frames[i]
        processed += processing_times[i]

    average = total/len(frames)
    avg_proc = processed/len(frames)
    print(average, 1.0/average, avg_proc * 1000)


# main
def main(number_of_frames):
    vs = init_video()
    time.sleep(2.0)
    count, frames, start_time, end_time, processing_end, processing_times, tx, ty, frame, prev_id, (h,w) = init_frame_and_count(vs, number_of_frames)
    print(1)


    threading.Thread(target = show_image, args = (t_queue, ), daemon = True).start()
    threading.Thread(target = mouse_track, args = (mouse_queue, ), daemon = True).start()
    threading.Thread(target = motor_thread, args = (1, m1_queue, ), daemon = True).start()
    threading.Thread(target = motor_thread, args = (2, m2_queue, ), daemon = True).start()
    threading.Thread(target = motor_thread, args = (3, m3_queue, ), daemon = True).start()

    for i in range(number_of_frames):
        frame = vs.read()
        frame = cv2.flip(frame, -1)

        prev_id, time_diff, start_time, frames = start_count(frame, frames, i, start_time)

        if frame is None:
            break

        contours_proc(frame, h, w, tx, ty)

        t_queue.put(frame)

        processing_times = end_proc_count(start_time, processing_times, i)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    t_queue.join()

    '''if not args.get("video", False):
        vs.stop()
    else:
        vs.release()'''
    vs.stop()
    cv2.destroyAllWindows()

    end_count(frames, processing_times)

if __name__ == '__main__':
    main(500)
