FROM wangso/python3-opencv-ffmpeg:V1
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN wget https://mizzouceri-s3.s3.amazonaws.com/yolo_weights.zip
RUN unzip yolo_weights.zip
RUN git clone https://github.com/wangso/ImageProcessingWebServices.git
WORKDIR /ImageProcessingWebServices/Server
RUN cp /yolo-object-detection/yolo-coco/yolov3.weights ./YOLO/
RUN rm -r /yolo-object-detection/ /yolo_weights.zip
ENTRYPOINT ["python", "server.py"]
