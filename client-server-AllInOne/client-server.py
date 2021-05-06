from flask import Flask, request, url_for, send_file, Response, jsonify
from flask_restful import Resource, Api
import cv2
import requests
import os
#import io
import random
import math
import numpy as np
import time
import json
from collections import OrderedDict
import darknet as dn
from PIL import Image

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 40  #Adjust ths value according to tweak the size of the moving object found
binarizationThreshold = 30  #Adjust ths value to tweak the binarization
offsetEntranceLine = 30  #offset of the entrance line above the center of the image
offsetExitLine = 60
yolodir = "./YOLO"
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

app = Flask(__name__)
api = Api(app)

def getNextServer():
    file = open("../NextServer.txt", "r")
    return file.read()[:]

@app.route("/")
def index():
    return "Image processing functional decomposition"

@app.route("/init", methods = ["POST"])
def init():
    """
    Re-initialize the object counter to 0 for all objects.
    """
    print("Initializing Application")
    labelDict = {}
    labelsPath = os.path.sep.join([yolodir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    for label in LABELS:
        labelDict[label] = 0
    json.dump(labelDict, open(outputFile, "w"))
    return "Output Counter Initialized"

@app.route("/sendFrame", methods = ["POST"])
def sendFrame():
    frameCount = 1
    peakFPS = 0
    minFPS = 10000
    averageFPS =0
    FPS = []
    startTime = time.time()
    video = cv2.VideoCapture("./road-traffic.mp4")

    uri_init = "http://" + getNextServer() + "/init"
    #uri = "http://" + getNextServer() + "/frameProcessing"
    headers = {"enctype" : "multipart/form-data"}
    #uri = requests.post("http://" + getNextServer() + "/objectClassifier")
    #force 640x480 webcam resolution
    video.set(3,360)
    video.set(4,640)
    response = requests.post(uri_init, json = {})

    for i in range(0,20):
        (grabbed, frame) = video.read()

    while True:
        frameStartTime = time.time()
        frameCount+=1
        (grabbed, frame) = video.read()
        time1 = time.time()
        print("time taken to load image frame ", time1-frameStartTime)
        height = np.size(frame,0)
        width = np.size(frame,1)

        #if cannot grab a frame, this program ends here.
        if not grabbed:
            break
        #cv2.imwrite("Frame.jpg", frame)
        #print(frame.shape)
        data = {"Frame":frame.tolist()}
        time2 = time.time()
        print("time to transform image to list ", time2-frameStartTime)
    #    print("Type of data is ", type(data.get("Frame")))
    #   print("current image is ", data.get("Frame"))
        print("size of image is ", width, " and ", height)
        r = requests.post("http://" + getNextServer() + "/objectClassifier", headers = headers, json = data)
        #r = requests.post(uri, json = data)
        time3 = time.time()
        print("time to post request ", time3-frameStartTime)
        currentFPS = 1.0/(time.time() - frameStartTime)
        print("Total time taken for this frame ", time.time() - frameStartTime)
        FPS.append(currentFPS)
        print("response = {}, frame = {}, fps = {} ".format(r, frameCount, round(currentFPS, 3)))
        file2 = open("../output/client/result.txt", "a")
        file2.write("response = {}, frame = {}, fps = {} ".format(r, frameCount, round(currentFPS, 3)))
        file2.close
        if r == "<Response [500]>":
            break
    print("Average FPS = {}".format(round(np.mean(FPS), 3)))
    print("RunTimeInSeconds = {}".format(round(frameStartTime - startTime, 3)))

@app.route("/frameProcessing", methods = ["POST"])
def frameProcessing():
    """
    Receive the frames from a video sequentially and count the number of objects. Each object is sent to a classifier which saves the count of objects that have been seen.
    """
    global referenceFrame
    global dilatedFrame
    #receive the image from the request.
    file = request.json
    frame = np.array(file["Frame"], dtype = "uint8")
    headers = {"enctype" : "multipart/form-data"}
    r = requests.post("http://" + getNextServer() + "/objectClassifier", headers = headers, json = {"Frame":frame.tolist()} )
    return Response(status=200)

@app.route("/objectClassifier", methods = ["POST"])
def classifier():
    """
    Classify an object and update the counter maintained at output.txt.
    """
    print("Classifying")
    #initialize important variables
    minConfidence = 0.5
    thresholdValue = 0.3
    file = request.json
    time1 = time.time();
    frame = np.array(file["Frame"], dtype = "uint8")
    time2 = time.time();
    print("Time to transform current frame to np array ", time2-time1)
    # width, height = frame.size
    frame = np.reshape(frame, (360, 640, 3))
    time3 = time.time();
    print("Time to transform current frame to reshape np array ", time3-time2)
    # print(frame.shape)
    # print(frame)
    # data = Image.fromarray(frame)
    # print("dimension of frame ", frame.shape[0], frame.shape[1], frame.shape[2])
    im = array_to_image(frame)
    time4 = time.time();
    print("Time to transform array back to image ", time4-time3)
    dn.rgbgr_image(im)
    time5 = time.time();
    print("Time to change image to rgb ", time5-time4)
    r = dn.detect(net, meta, im)
    time6 = time.time();
    print("Time to detect image ", time6-time5)
    print(r)
    return Response(status=200)

@app.route("/getCounts", methods = ["GET"])
def getCounts():
    retval ={}
    output = json.load(open(outputFile))
    for key, val in output.items():
        if val >=1:
            retval[key] = val
    return jsonify(retval)

@app.route("/setNextServer", methods = ["POST"])
def setNextServer():
    data = request.get_json()
    #json.dump(data["server"], open("../NextServer.txt", "w"))
    text_file=open("../NextServer.txt", "w")
    text_file.write(data["server"])
    text_file.close()

def getContourCentroid(x, y, w, h):
    """
    Get the centroid/center of the countours you have
    @return: The coordinates of the  center points
    """
    coordXCentroid = (x+x+w)/2
    coordYCentroid = (y+y+h)/2
    objectCentroid = (int(coordXCentroid),int(coordYCentroid))
    return objectCentroid

#Check if an object in entering in monitored zone
def checkEntranceLineCrossing(y, coorYEntranceLine, coorYExitLine):
    absDistance = abs(y - coorYEntranceLine)

    if ((absDistance <= 2) and (y < coorYExitLine)):
        return 1
    else:
        return 0

def getContours(frame):
    """
    Get the contours in the frame
    @return: contours
    """
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getContourBound(contour):
    """
    Returns a rectangle that is a bound around the contour.
    @return: contour Bounds
    """
    (x,y,w,h) = cv2.boundingRect(contour)
    return (x,y,w,h)

def thresholdImage(frame, binarizationThreshold=30):
    """
    Threshold the image to make it black and white. values above binarizationThreshold are made white and the rest
    is black
    """
    return cv2.threshold(frame, binarizationThreshold, 255, cv2.THRESH_BINARY)[1]

def getImageDiff(referenceFrame, frame):
    """
    Get the difference between 2 frames to isolate and retrieve only the moving object
    """
    return cv2.absdiff(referenceFrame, frame)

def gaussianBlurring(frame):
    """
    Preprocess the image by applying a gaussian blur
    """
    return cv2.GaussianBlur(frame, ksize =(11, 11), sigmaX = 0)

def greyScaleConversion(frame):
    """
    Convert the image from 3 channels to greyscale to reduce the compute required to run it.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def dilateImage(frame, interations=2 ):
    """
    Dilate the image to prevent spots that are black inside an image from being counted as individual objects
    """
    return cv2.dilate(frame, None, iterations=2)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)