FROM ubuntu20.04
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN apt-get -y update && apt-get -y install software-properties-common apt-utils python3-pip python3-dev libopencv-dev python3-opencv build-essential unzip nano curl git
COPY ./requirement.txt /requirment.txt
RUN pip3 install -r /requirements.txt
