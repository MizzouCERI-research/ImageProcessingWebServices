# used this to build docker image wangso/python3-opencv-ffmpeg:V1
# The base image will be used to build both the Server and Client images
# Base images has been uploaded to Dockerhub and is ready for download

FROM ubuntu20.04
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN apt-get -y update && apt-get -y install software-properties-common apt-utils python3-pip python3-dev libopencv-dev python3-opencv build-essential ffmpeg unzip nano curl git wget iproute2 iperf net-tools
RUN pip3 install cmake flask flask_restful scikit-build opencv_contrib_python numpy requests pillow

