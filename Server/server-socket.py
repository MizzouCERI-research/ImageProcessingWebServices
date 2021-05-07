#!/usr/bin/python3
import pickle
import struct
#import socket
from socket import *
from time import perf_counter as tpf
import cv2 as cv
import requests
import os
#import io
import random
import numpy as np
import time
import json
from collections import OrderedDict
import darknet as dn
from PIL import Image
from ctypes import *
import math

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 40  #Adjust ths value according to tweak the size of the moving object found
binarizationThreshold = 30  #Adjust ths value to tweak the binarization
offsetEntranceLine = 30  #offset of the entrance line above the center of the image
offsetExitLine = 60
yolodir = "../YOLO"
outputFile = "../output/server/output.txt"
referenceFrame = 0
dilatedFrame = 0

#Prep the DNN
labelsPath = os.path.sep.join([yolodir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([yolodir, "yolov3.weights"])
configPath = os.path.sep.join([yolodir, "yolov3.cfg"])
metaPath = os.path.sep.join([yolodir, "coco.data"])
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
dn.set_gpu(0)
net = dn.load_net(configPath.encode("ascii"), weightsPath.encode("ascii"), 0)
meta = dn.load_meta(metaPath.encode("ascii"))

frameCount = 1
peakFPS = 0
minFPS = 10000
averageFPS =0
FPS = []
startTime = time.time()

def array_to_image(arr):
    time1 = time.time()
    arr = arr.transpose(2,0,1)
    time2 = time.time()
    print("transpose took: ", time2-time1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    time3 = time.time()
    print("flattern took: ", time3-time2)
    data = dn.c_array(dn.c_float, arr)
    time4 = time.time()
    print("change to c_array took: ", time4-time3)
    im = dn.IMAGE(w,h,c,data)
    time5 = time.time()
    print("final loading IMAGE from array took: ", time5-time4)
    return im

if __name__=="__main__":
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(("172.17.0.3", 9090))
    print("connected to server...")

    # buffer to save frame
    data = bytearray()

    # size of number used to designate payload size
    # in this case unsigned long long so 8 bytes
    payload_size_size = struct.calcsize("Q")

    # read in frames
    print("receiving data from server")
    while True:
        # 1 iteration = receiving 1 frame (1 + a little more in some cases)
        start = tpf()
        frameStartTime = time.time()
        frameCount+=1
        # first figure out the size of a payload, then build up
        while len(data) < payload_size_size:
            received = s.recv(4096)

            # connection has been closed
            if len(received) == 0:
                break

            data.extend(received)


        # get payload info; which is in the first 8 bytes of the payload
        packed_msg_size = data[:payload_size_size]

        # get the message content (everything after the first 8 bytes), which is our frame
        data = data[payload_size_size:]

        # get payload size so we know how much data to read in before we have a full frame
        msg_size = struct.unpack("Q",packed_msg_size)[0]

        # get the actual frame (we know from msg_size how much bytes to recv)
        # might read a little extra bytes, but will handle that
        while len(data) < msg_size:
            data.extend(s.recv(4096))

        # frame will be data all the way up to msg_size
        frame = data[:msg_size]
 #       print("socket frame type: ", type(frame))
        frame1 = pickle.loads(frame)
        time1 = time.time()
        print(" time used to load frame from socket stream: ", time1-frameStartTime)
        print(frame1.shape)
#        print(frame1[1:100])
#        frame2 = np.reshape(frame1, (360, 640, 3))
 #       print(frame2.shape)
  #      print(frame2[1:100])
        # overwrite data, with left over "data" which includes the next frame
        data = data[msg_size:]
        end = tpf()

        # get recv time before displaying image (likely faster than
        # shown on the server side because data has already arrived buffered
        # before calling recv
        #print("recv frame time: {:0.3f} seconds".format(end - start))

        # Swap the next line with the ndarray lines to switch between ndarray or not using ndarray
#        im = array_to_image(frame1)

        np_data = frame1.ctypes.data_as(POINTER(c_ubyte))
        im = dn.ndarray_image(np_data, frame1.ctypes.shape, frame1.ctypes.strides)

        dn.rgbgr_image(im)
        time3 = time.time()
        r = dn.detect(net, meta, im)
        time4 = time.time()
        print(" time used to detect object from image: ", time4-time3)
        print(r)
#        cv.imshow("frame", pickle.loads(frame))
        currentFPS = 1.0/(time.time() - frameStartTime)
        FPS.append(currentFPS)
        print("response = {}, frame = {}, fps = {} ".format("", frameCount, round(currentFPS, 3)))
        if cv.waitKey(40) == ord("q"):
            print("you pressed 'q' and we will exit")
            break

    s.close()