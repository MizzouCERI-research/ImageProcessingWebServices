FROM ubuntu20.04
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN apt-get -y update && apt-get install software-properties-common apt-utils python3-pip python3-dev build-essential unzip nano curl git
COPY ./requirement.txt /requirment.txt
RUN pip3 install /requirements.txt