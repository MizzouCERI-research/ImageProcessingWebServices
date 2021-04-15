FROM wangso/docker-python-opencv-ffmpeg
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN apt-get -y update && apt-get -y install apt-utils python3 python3-pip python3-dev build-essential unzip nano curl
RUN pip install flask flask_restful opencv_contrib_python numpy requests
RUN wget https://mizzouceri-s3.s3.amazonaws.com/yolo_weights.zip
RUN unzip yolo_weights.zip
RUN git clone https://github.com/wangso/ImageProcessingWebServices.git
WORKDIR /ImageProcessingWebServices/Server
RUN cp /yolo-object-detection/yolo-coco/yolov3.weights ./YOLO/
RUN rm -r /yolo-object-detection/ /yolo_weights.zip
RUN pip install --upgrade pip

ENTRYPOINT ["python", "server.py"]
