FROM wangso/python3-opencv-ffmpeg:V1
MAINTAINER Songjie Wang "wangso@missouri.edu"
RUN apt-get -y update && apt-get -y install unzip nano curl git wget iproute2 iperf net-tools
RUN pip3 install cmake flask flask_restful scikit-build opencv_contrib_python numpy requests
RUN git clone https://github.com/wangso/ImageProcessingWebServices.git
WORKDIR /ImageProcessingWebServices/Client
RUN wget https://wangso-flynet.s3.amazonaws.com/road-traffic.mp4
RUN wget https://wangso-flynet.s3.amazonaws.com/360p_100KB.mp4
ENTRYPOINT ["python3", "client.py"]
