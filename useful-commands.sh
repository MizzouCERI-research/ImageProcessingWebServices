# Server side: 
docker run -p 5000:5000 Docker_image_ID
docker logs -f container_ID

# Server with gpu capability
docker run --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu
# serve without gpu capability
docker run --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V2


# Command line to interact with Server 
curl -X POST -H 'Content-Type: application/json' http://server_IP:port/setNextServer -d '{"server":"mylongerurllocalhost1"}'
curl -X POST http://server_IP:port/init

# or on machines that have HTTPS enabled:
curl -X POST -H 'Content-Type: application/json' https://server_IP:port/setNextServer -d '{"server":"mylongerurllocalhost1"}'
curl -X POST https://server_IP:port/init

# client side:

# to overwrite entrypoint command
docker run -it --env server=server_IP:port --entrypoint bash Docker_image_ID

# example
docker run --rm --name client -v /root/client:/ImageProcessingWebServices/output/client --env server=192.5.86.238:5000 wangso/imgproc-client:gpu

# to use pre-defined entrypoint command
docker run -it --env server=server_IP:port Docker_image_ID


# Limit the number of CPUs and amount of memories on the docker container
docker run --rm -m 1000m --cpus=1 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1
docker run --rm -m 4000m --cpus=4 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1
docker run --rm -m 8000m --cpus=8 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1
docker run --rm -m 16000m --cpus=16 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V
docker run --rm -m 16000m --cpus=16 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1
docker run --rm -m 32000m --cpus=32 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1
docker run --rm -m 64000m --cpus=64 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V1

docker run --rm -m 4000m --cpus=4 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu
docker run --rm -m 64000m --cpus=64 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu