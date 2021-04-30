# Server side: 
docker run -p 5000:5000 Docker_image_ID
docker logs -f container_ID

# Server with gpu capability
docker run --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu

# Command line to interact with Server 
curl -X POST -H 'Content-Type: application/json' http://server_IP:port/setNextServer -d '{"server":"mylongerurllocalhost1"}'
curl -X POST http://server_IP:port/init

# or on machines that have HTTPS enabled:
curl -X POST -H 'Content-Type: application/json' https://server_IP:port/setNextServer -d '{"server":"mylongerurllocalhost1"}'
curl -X POST https://server_IP:port/init

# client side:

# to overwrite entrypoint command
docker run -it --env server=server_IP:port --entrypoint bash Docker_image_ID

# to use pre-defined entrypoint command
docker run -it --env server=server_IP:port Docker_image_ID