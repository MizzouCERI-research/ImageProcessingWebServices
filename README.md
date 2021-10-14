# ImageProcessingWebServices
## ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) The master branch is for non-gpu version, for gpu version, checkout the gpu branch


The current version includes GPU capabilities implemented on CUDA image, and will run on nVidia V100 GPU only. If different GPU is used, you need to modify the Darknet build inside the container, and recompile using the specific setting for that type of GPU. 

## Currently available Docker images on Dockerhub (these images still use HTTP POST request to send video streams):

$ docker pull wangso/imgproc-server:gpu

$ docker pull wangso/imgproc-client:gpu

Both images were built in two stages: 1). building base image using Dockerfile in the [Base-Docker-image](https://github.com/wangso/ImageProcessingWebServices/blob/master/Base-Docker-image/) folder. 2). Build final working images (above) using the base images and adding entrypoint commands. 

Both Images can be run with the following settings (we included cap-add to allow modifying network settings). Example commands use resolution at 1080p. The other option currently available is '360p'. These two settings will allow streaming and processing of videos at these two resolutions ONLY!

## Testing with Docker containers only

#### Step 1. Run Server container on Server machine: 

    $ docker pull wangso:imgproc-server:V2 
    
    $ docker run --rm -it --name server --cap-add=all -v /root/server:/ImageProcessingWebServices/output/server --env server=**server_IP**:5000 -p 5000:5000 wangso/imgproc-server:V2
    
<!-- #### Step 2: Before running client, we need to update the server address inside the Server container (replace both server_IP with the IP of the Server) 
#### This command can be run from anywhere where you have curl installed:
    
    $ curl -X POST -H 'Content-Type: application/json' http://**Server_IP**:5000/setNextServer -d '{"server":"**server_IP**:5000"}'
     -->
#### Step 2: Start Client container on Client machine (replace server_IP with the IP of the Server):
    
    $ docker pull wangso:imgproc-client:V2 
    
    $ docker run --rm -it --name client --cap-add=all -v /root/client:/ImageProcessingWebServices/output/client --env server=**server_IP**:5000 wangso/imgproc-client:V2

    For Example: 
    $ docker run --rm -it --name client --cap-add=all -v /root/client:/ImageProcessingWebServices/output/client --env server=3.237.180.140:5000 wangso/imgproc-client:V2


#### To retrieve object types and counts: 
    $ curl -X GET -H 'Content-Type: application/json' http://**server_IP**:5000/getCounts   

    For Example:  
    $ curl -X GET -H 'Content-Type: application/json' http://3.237.180.140:5000/getCounts
    

## Testing with Kubernetes cluster

#### Step 1. Run server deployment:

    $ kubectl apply -f server-deployment.yaml
    
#### Step 2. Monitor server output:

    $ kubectl logs -f server_pod_ID

#### Step 3. Update server address to listen to (URL is depending on if the host machine has TLS, change to HTTP if no TLS on the host)

    $ curl -X POST -H 'Content-Type: application/json' https://Server_IP:Port/setNextServer -d '{"server":"Server_IP:Port"}'
    
#### Step 4. Run client deployment (first modify the env inside yaml file to update the server_IP:port): 

     $ kubectl apply -f client-deployment.yaml
     
     
     
------------------------------------------------------------------------
# Description of the Image processing app components: 

### Server 
[server.py](https://github.com/wangso/ImageProcessingWebServices/blob/master/Server/server.py) implements a RESTFUL webservice for object detection.
The following endpoints currently exist.
1. /init: This initializes all counters to 0
2. /setNextServer: Set the IP address:port (or DNS) of the Server inside the NextServer.txt 
3. /frameProcessing: This uses openCv methods to find the objects that have changed from a preceding frame. It sends each object to /classifier.
4. /objectClassifier: This is a deep learning model(Yolo-V3 trained on COCO) that attempts to classify an image into a fixed set of classes. It increments the [counter](https://github.com/wangso/ImageProcessingWebServices/blob/master/output/server/output.txt) for that object class.
5. /getCounts: This retrieves the current value of the counters.

[Dockerfile](https://github.com/wangso/ImageProcessingWebServices/blob/master/Server/Dockerfile) for starting up a container running the server.

[server-deployment.yaml](https://github.com/wangso/ImageProcessingWebServices/blob/master/Kubenetes-manifest/server-deployment.yaml) for starting up a k8s pod running the server.

### Client 
[client.py](https://github.com/wangso/ImageProcessingWebServices/blob/master/Client/client.py) reads a video and sends frames to server.py.

[Dockerfile](https://github.com/wangso/ImageProcessingWebServices/blob/master/Client/Dockerfile) for starting up a container running the client.

[client-deployment.yaml](https://github.com/wangso/ImageProcessingWebServices/blob/master/Kubenetes-manifest/client-deployment.yaml) for starting up a k8s pod running the client.


