# Server side: 
docker run -p 5000:5000 Docker_image_ID
docker logs -f container_ID

# Server with gpu capability
docker run --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu
# serve without gpu capability
docker run --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:V2


# Command line to interact with Server 
curl -X POST -H 'Content-Type: application/json' http://129.114.109.185:5000/setNextServer -d '{"server":"129.114.109.185:5000"}'
curl -X POST -H 'Content-Type: application/json' http://localhost:5000/setNextServer -d '{"server":"localhost:5000"}'
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:5000/setNextServer -d '{"server":"127.0.0.1:5000"}'
curl -X POST http://127.0.0.1:5000/sendFrame


# or on machines that have HTTPS enabled:
# curl -X POST -H 'Content-Type: application/json' https://server_IP:port/setNextServer -d '{"server":"server_IP:port"}'
# curl -X POST https://server_IP:port/init

# curl -X POST -H 'Content-Type: application/json' https://localhost:5000/setNextServer -d '{"server":"localhost:5000"}'
# curl -X POST -H 'Content-Type: application/json' https://127.0.0.1:5000/setNextServer -d '{"server":"127.0.0.1:5000"}'
# client side:

# run server and client with gpu with standard output on 1080p
docker run --rm -it --name server --cap-add=all -v /root/server:/ImageProcessingWebServices/output/server --env resolution='1080p' --gpus all -p 5000:5000 wangso/imgproc-server:gpu
docker run --rm -it --name client --cap-add=all -v /root/client:/ImageProcessingWebServices/output/client --env resolution='1080p' --env server=129.114.109.185:5000 wangso/imgproc-client:gpu

# run interactive containers and skip default entrypoint on 1080p
docker run --rm -it --entrypoint bash --name server --cap-add=all -v /root/server:/ImageProcessingWebServices/output/server --env resolution='1080p' --gpus all -p 5000:5000 wangso/imgproc-server:gpu
docker run --rm -it --entrypoint bash --name client --cap-add=all -v /root/client:/ImageProcessingWebServices/output/client --env resolution='1080p' --env server=129.114.109.185:5000 wangso/imgproc-client:gpu

# to use pre-defined entrypoint command
docker run -it --env server=server_IP:port Docker_image_ID


# Limit the number of CPUs and amount of memories on the docker container
docker run --rm -m 64000m --cpus=64 --name server -v /root/server:/ImageProcessingWebServices/output/server --gpus all -p 5000:5000 wangso/imgproc-server:gpu

TO add network delays/latencies:
	$ tc qdisc add dev eth1 root netem delay 5ms

To control the number of CPUs a Docker container can use: 
	$ docker run --cpuset-cpus="0-2". ….. 
	$ docker run --cpus 2.5 ….  
TO check number cpus on the container: 
	$ nproc
To control the number of memory to use for a Docker container:
	$ docker run -m 512m ….


# modify client for python3 and 1080p video
wget https://emmy8.casa.umass.edu/flynetDemo/drone/video_data/360p_100KB.mp4
mv 360p_100KB.mp4 road-traffic.mp4
export server=129.114.109.185:5000
python3 client.py

# Monitor GPU usage:
watch -d -n 0.5 nvidia-smi

# install tc tools
apt-get install -y iproute2 iperf net-tools

# tc tool to limit network bandwidth
tc qdisc add dev eth0 root tbf rate 50mbit latency 5ms burst 80000
# to remove a tc policy
tc qdisc delete dev eth0 root









